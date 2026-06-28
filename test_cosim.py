"""Compact CARLA + METS-R + Simu5G C-V2X co-simulation smoke scenario.

This is the Simu5G-backed version of
examples/BSM_attack/V2V_BSM_PosFalsifi_intersec_multiVeh.py.  The controller,
route synchronization, C-V2X exchange, and route completion handling live in
V2VCoSimClientMaster.step().
"""

import sys
import time
import traceback
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from clients.V2V_CoSimClient_master import V2VCoSimClientMaster
from cosim_utils.attack_manager import ReplayAttack, V2XAttackManager, assign_attack_vehicle_ids
from cosim_utils.dataset_saver import DatasetSaver
from cosim_utils.helpers import is_port_open, load_scenario, scenario_trip_specs, set_random_seed
from utils.carla_util import open_carla
from utils.simu5g_v2x_util import start_simu5g_bridge_in_terminal
from utils.util import prepare_sim_dirs, read_run_config, run_simulation_in_docker


SCENARIO_PATH = ROOT_DIR / "V2V-Attack-Dataset/scenarios/intersection_4way/intersection_4way_scenario_01.yaml"
DATASET_ROOT = ROOT_DIR / "V2V-Attack-Dataset"
REPLAY_RECORD_PATH = DATASET_ROOT / "replay_records/intersection_4way/intersection_4way_scenario_01/west_to_north.jsonl"
MOVEMENT_NAMES = ("west_to_east", "east_to_west", "north_to_south", "south_to_north")
CAMERA_HEIGHT = 120
RANDOM_SEED = 42
TRAFFIC_DENSITY = "low"
WEATHER = "clear"
MAX_STEPS = 300
WARMUP_STEPS = 0
ATTACK_START_TICK = 40


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_output_log():
    log_dir = ROOT_DIR / "output"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "test_cosim_latest.log"
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)
    print(f"Writing console output to {log_path}", flush=True)
    return log_file, original_stdout, original_stderr


def count_jsonl_records(path):
    with open(path, "r", encoding="utf-8") as fp:
        return sum(1 for line in fp if line.strip())


def attack_metadata(attack, vehicle_ids, dt, tick_offset=0):
    attack_type = getattr(attack, "attack_type", "none")
    attacker_id = getattr(attack, "attacker_vehicle_id", None)
    if attack_type in {"static_ghost_vehicle", "obstacle_ghost_vehicle"}:
        attack_target = getattr(attack, "target_vehicle_id", None)
    else:
        attack_target = "all"
    if attack_type in {"false_information_injection", "sybil"} and attacker_id is not None:
        attack_receiver = [vid for vid in vehicle_ids if str(vid) != str(attacker_id)]
    else:
        attack_receiver = list(vehicle_ids)
    start_tick = getattr(attack, "start_tick", 0)
    end_tick = getattr(attack, "end_tick", None)
    return {
        "attack_type": attack_type,
        "attack_target": attack_target,
        "attack_receiver": attack_receiver,
        "start_time": max(0, start_tick - tick_offset) * dt,
        "end_time": None if end_tick is None else max(0, end_tick - tick_offset) * dt,
        "parameters": attack_parameters(attack),
    }


def attack_parameters(attack):
    skip = {"records", "limiter", "enabled"}
    params = {}
    for key, value in vars(attack).items():
        if key.startswith("_") or key in skip:
            continue
        if isinstance(value, Path):
            value = str(value)
        params[key] = value
    return params


def main():
    log_file, original_stdout, original_stderr = setup_output_log()
    data_saver = None
    cosim_client = None
    duration_sec = None

    # Load the scenario definition and simulation configuration.
    scenario = load_scenario(SCENARIO_PATH)
    config_file = scenario["config_file"]
    config = read_run_config(config_file)
    set_random_seed(RANDOM_SEED, config=config)

    # Configure simulation parameters.
    config.verbose = False
    config.display_all = False
    config.enable_debug_draw = False
    config.draw_route_plan = False
    config.v2v_position_mode = "local"
    config.cv2x_communication_range_m = 500.0
    config.carla_tick_timeout = 5.0
    config.metsr_tick_timeout = 5.0
    config.release_queued_cosim_vehicles = True
    config.camera_layout = "front_rear"
    config.camera_interval_ticks = 5
    config.lidar_interval_ticks = 10

    metsr_roads = list(scenario.get("cosim_roads") or [])
    trip_specs = scenario_trip_specs(scenario, MOVEMENT_NAMES)
    controller_vids = [vid for vid, _, _, _ in trip_specs]
    vehicle_route = {str(vid): movement_name for vid, _, _, movement_name in trip_specs}
    config.metsr_road = metsr_roads
    config.controller_vids = controller_vids
    if scenario.get("network_file"):
        config.network_file = scenario["network_file"]
    if scenario.get("town"):
        config.carla_map = scenario["town"]
    print(f"Loaded scenario {scenario.get('scenario_id', SCENARIO_PATH.name)} from {SCENARIO_PATH}")
    print(f"Using config file: {config_file}")
    print(f"Using METS-R co-sim roads: {metsr_roads}")
    print(f"Using random seed: {RANDOM_SEED}")

    # Start or reuse the Simu5G/Veins C-V2X bridge.
    veins_port = int(getattr(config, "veins_port", 9099))
    if not is_port_open(veins_port):
        print(f"Starting Simu5G bridge on port {veins_port}...")
        start_simu5g_bridge_in_terminal(ROOT_DIR, wait_seconds=5.0)
    else:
        print(f"Simu5G bridge already listening on port {veins_port}.")

    prepare_sim_dirs(config)

    # Start CARLA and connect the CARLA client.
    carla_client, carla_tm = open_carla(config)
    set_random_seed(RANDOM_SEED, traffic_manager=carla_tm)
    print("CARLA server started successfully")

    # Start or reuse the METS-R simulation backend.
    metsr_port = config.metsr_port[0] if hasattr(config, "metsr_port") else config.ports[0]
    metsr_reused = is_port_open(metsr_port)
    if not metsr_reused:
        run_simulation_in_docker(config)
    else:
        print(f"METS-R already running on port {metsr_port}; reusing existing instance.")

    # Build the V2V co-simulation client that coordinates CARLA, METS-R, and Simu5G.
    cosim_client = V2VCoSimClientMaster(
        config,
        carla_client,
        carla_tm,
        controller_vids=controller_vids,
        require_simu5g_uu=True,
    )
    print("V2VCoSimClientMaster created successfully")
    for vid in controller_vids:
        cosim_client.enable_vehicle_sensor(vid)
    if (scenario.get("traffic_lights") or {}).get("enabled") is False:
        light_count = cosim_client.set_unsignalized_intersection_lights()
        print(f"Set {light_count} CARLA traffic lights to frozen yellow.")

    if metsr_reused:
        print("Resetting reused METS-R instance for a clean run.")
        cosim_client.metsr.reset()
        for road in getattr(config, "metsr_road", []):
            cosim_client.metsr.set_cosim_road(road)

    if WARMUP_STEPS > 0:
        cosim_client.metsr.tick(WARMUP_STEPS, max_wait_seconds=10, poll_timeout=1)
    run_start_tick = cosim_client.current_tick
    dt = float(getattr(config, "sim_step_size", 0.1))
    replay_count = count_jsonl_records(REPLAY_RECORD_PATH)
    attack = ReplayAttack(
        replay_name="west_to_north",
        replay_path=REPLAY_RECORD_PATH,
        replay_sender_id=None,
        start_tick=run_start_tick + ATTACK_START_TICK,
        end_tick=run_start_tick + ATTACK_START_TICK + replay_count - 1,
        loop=False,
        use_limiter=False,
    )
    assign_attack_vehicle_ids(attack, controller_vids)
    attacks = V2XAttackManager([attack])
    meta_info = {
        "map": getattr(config, "carla_map", scenario.get("town", "")),
        "scenario": str(SCENARIO_PATH),
        "traffic_density": TRAFFIC_DENSITY,
        "vehicle_route": vehicle_route,
        "weather": WEATHER,
        "random_seed": RANDOM_SEED,
        "sim_step_size": dt,
        "sim_fps": 1.0 / dt,
        "max_steps": MAX_STEPS,
        "planned_duration_sec": MAX_STEPS * dt,
    }
    data_saver = DatasetSaver(
        DATASET_ROOT,
        meta_info,
        attack_metadata(attack, controller_vids, dt, tick_offset=run_start_tick),
    )
    data_saver.log_event(0.0, "run initialized")

    # Generate the scenario vehicles and their METS-R trips.
    for vid, road_from, road_to, movement_name in trip_specs:
        print(f"Generating trip veh={vid} movement={movement_name}: {road_from} -> {road_to}")
        cosim_client.metsr.generate_trip_between_roads([vid], road_from, road_to)
        cosim_client.metsr.update_vehicle_sensor_type([vid], "cv2x", True)

    camera_x, camera_y = scenario.get("intersection_center_xy_carla", [-50, 0])
    cosim_client.set_custom_camera(camera_x, camera_y, CAMERA_HEIGHT)
    print(f"Enabled replay attack from {REPLAY_RECORD_PATH}")

    try:
        previous_entered = set()
        attack_started = False
        attack_ended = False
        # Run the synchronized co-simulation loop.
        for i in range(MAX_STEPS):
            predicted_tick = cosim_client.current_tick + 1
            relative_tick = predicted_tick - run_start_tick
            relative_time = relative_tick * dt
            attack_active = any(attack.active(predicted_tick) for attack in attacks.attacks)
            phase = "replay_attack" if attack_active else "normal"
            if attack_active and not attack_started:
                data_saver.log_event(relative_time, "replay attack started")
                attack_started = True
            if not attack_active and attack_started and not attack_ended:
                data_saver.log_event(relative_time, "replay attack ended")
                attack_ended = True
            step_result = cosim_client.step(extra_v2x_messages=attacks, phase=phase)
            current_entered = {
                vid for vid in controller_vids if cosim_client.carla_entered.get(vid, False)
            }
            for vid in sorted(current_entered - previous_entered):
                data_saver.log_event(relative_time, f"Vehicle {vid} entered co-sim region")
            previous_entered = current_entered
            for vid in step_result.get("done_vids", []):
                data_saver.log_event(relative_time, f"Vehicle {vid} completed route or left co-sim region")
            tick = int(step_result.get("tick", cosim_client.current_tick)) - run_start_tick
            sim_time = tick * dt
            data_saver.record_step(tick, sim_time, cosim_client, step_result, vehicle_ids=controller_vids)
            duration_sec = sim_time
            time.sleep(0.08)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception:
        print("\nSimulation failed with exception:", flush=True)
        traceback.print_exc()
    finally:
        if data_saver is not None:
            data_saver.log_event(duration_sec or 0.0, "run finalized")
            data_saver.finalize(duration_sec=duration_sec)
        if cosim_client is not None:
            cosim_client.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
