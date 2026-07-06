"""Scenario 03 middle-density complicated interaction: fixed random movements from four directions with static ghost vehicle attack."""

import sys
import time
import traceback
from pathlib import Path

import carla


ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from clients.V2V_CoSimClient_master import V2VCoSimClientMaster
from cosim_utils.attack_manager import StaticGhostVehicleAttack, V2XAttackManager, assign_attack_vehicle_ids
from cosim_utils.dataset_saver import DatasetSaver
from cosim_utils.helpers import is_port_open, load_scenario, scenario_trip_specs, set_random_seed
from utils.carla_util import open_carla
from utils.simu5g_v2x_util import start_simu5g_bridge_in_terminal
from utils.util import prepare_sim_dirs, read_run_config, run_simulation_in_docker


CASE_NAME = "RandomDirectionsA_5_StaticGhostVehicleAttack"
SCENARIO_PATH = ROOT_DIR / "V2V-Attack-Dataset/scenarios/intersection_4way/intersection_4way_scenario_03.yaml"
DATASET_ROOT = ROOT_DIR / "V2V-Attack-Dataset"
MOVEMENT_NAMES = ("north_to_west", "east_to_south", "south_to_north", "west_to_east", "north_to_east")
CAMERA_HEIGHT = 120
RANDOM_SEED = 42
TRAFFIC_DENSITY = "middle_density"
WEATHER = "clear"
MAX_STEPS = 400
WARMUP_STEPS = 0
SAME_ROUTE_DEPARTURE_GAP_TICKS = 20
ATTACK_START_TICK = 40


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
    params = {}
    for key, value in vars(attack).items():
        if key.startswith("_") or key in {"records", "limiter", "enabled"}:
            continue
        params[key] = str(value) if isinstance(value, Path) else value
    return params


def build_attack(controller_vids, run_start_tick):
    start_tick = run_start_tick + ATTACK_START_TICK
    attack = StaticGhostVehicleAttack(
        target_vehicle_id=1,
        ghost_id=None,
        scenario_type="intersection",
        speed_mps=0.0,
        start_tick=start_tick,
        end_tick=start_tick + 180,
    )
    assign_attack_vehicle_ids(attack, controller_vids)
    return attack


def configure_run(scenario):
    config = read_run_config(scenario["config_file"])
    set_random_seed(RANDOM_SEED, config=config)
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
    config.metsr_road = list(scenario.get("cosim_roads") or [])
    if scenario.get("network_file"):
        config.network_file = scenario["network_file"]
    if scenario.get("town"):
        config.carla_map = scenario["town"]
    return config


def run_one_step(cosim_client, data_saver, attacks, attack, controller_vids, run_start_tick, dt, state):
    predicted_tick = cosim_client.current_tick + 1
    relative_tick = predicted_tick - run_start_tick
    relative_time = relative_tick * dt
    attack_active = attack.active(predicted_tick)
    phase = attack.attack_type if attack_active else "normal"
    if attack_active and not state["attack_started"]:
        data_saver.log_event(relative_time, f"{attack.attack_type} started")
        state["attack_started"] = True
    if not attack_active and state["attack_started"] and not state["attack_ended"]:
        data_saver.log_event(relative_time, f"{attack.attack_type} ended")
        state["attack_ended"] = True

    step_result = cosim_client.step(extra_v2x_messages=attacks, phase=phase)
    current_entered = {vid for vid in controller_vids if cosim_client.carla_entered.get(vid, False)}
    for vid in sorted(current_entered - state["previous_entered"]):
        data_saver.log_event(relative_time, f"Vehicle {vid} entered co-sim region")
    state["previous_entered"] = current_entered
    for vid in step_result.get("done_vids", []):
        data_saver.log_event(relative_time, f"Vehicle {vid} completed route or left co-sim region")

    tick = int(step_result.get("tick", cosim_client.current_tick)) - run_start_tick
    sim_time = tick * dt
    data_saver.record_step(tick, sim_time, cosim_client, step_result, vehicle_ids=controller_vids)
    state["duration_sec"] = sim_time
    time.sleep(0.08)


def generate_trips_with_gaps(cosim_client, trip_specs, data_saver, attacks, attack, controller_vids, run_start_tick, dt, state):
    last_generated_tick_by_origin = {}
    for vid, road_from, road_to, movement_name in trip_specs:
        origin = road_from
        while (
            origin in last_generated_tick_by_origin
            and cosim_client.current_tick - last_generated_tick_by_origin[origin] < SAME_ROUTE_DEPARTURE_GAP_TICKS
        ):
            run_one_step(cosim_client, data_saver, attacks, attack, controller_vids, run_start_tick, dt, state)
        print(f"Generating trip veh={vid} movement={movement_name}: {road_from} -> {road_to}")
        cosim_client.metsr.generate_trip_between_roads([vid], road_from, road_to)
        cosim_client.metsr.update_vehicle_sensor_type([vid], "cv2x", True)
        last_generated_tick_by_origin[origin] = cosim_client.current_tick


def main():
    scenario = load_scenario(SCENARIO_PATH)
    config = configure_run(scenario)
    trip_specs = scenario_trip_specs(scenario, MOVEMENT_NAMES)
    controller_vids = [vid for vid, _, _, _ in trip_specs]
    vehicle_route = {str(vid): movement_name for vid, _, _, movement_name in trip_specs}
    config.controller_vids = controller_vids

    veins_port = int(getattr(config, "veins_port", 9099))
    if not is_port_open(veins_port):
        print(f"Starting Simu5G bridge on port {veins_port}...")
        start_simu5g_bridge_in_terminal(ROOT_DIR, wait_seconds=5.0)

    prepare_sim_dirs(config)
    carla_client, carla_tm = open_carla(config)
    carla_client.get_world().set_weather(carla.WeatherParameters.ClearNoon)
    set_random_seed(RANDOM_SEED, traffic_manager=carla_tm)

    metsr_port = config.metsr_port[0] if hasattr(config, "metsr_port") else config.ports[0]
    metsr_reused = is_port_open(metsr_port)
    if not metsr_reused:
        run_simulation_in_docker(config)
    else:
        print(f"METS-R already running on port {metsr_port}; reusing existing instance.")

    cosim_client = None
    data_saver = None
    state = {"duration_sec": None}
    try:
        cosim_client = V2VCoSimClientMaster(config, carla_client, carla_tm, controller_vids=controller_vids, require_simu5g_uu=True)
        for vid in controller_vids:
            cosim_client.enable_vehicle_sensor(vid)
        if (scenario.get("traffic_lights") or {}).get("enabled") is False:
            cosim_client.set_unsignalized_intersection_lights()
        if metsr_reused:
            cosim_client.metsr.reset()
            for road in getattr(config, "metsr_road", []):
                cosim_client.metsr.set_cosim_road(road)
        if WARMUP_STEPS > 0:
            cosim_client.metsr.tick(WARMUP_STEPS, max_wait_seconds=10, poll_timeout=1)

        run_start_tick = cosim_client.current_tick
        dt = float(getattr(config, "sim_step_size", 0.1))
        attack = build_attack(controller_vids, run_start_tick)
        attacks = V2XAttackManager([attack])
        data_saver = DatasetSaver(
            DATASET_ROOT,
            {
                "map": getattr(config, "carla_map", scenario.get("town", "")),
                "scenario": str(SCENARIO_PATH),
                "scenario_variant": CASE_NAME,
                "traffic_density": TRAFFIC_DENSITY,
                "vehicle_route": vehicle_route,
                "weather": WEATHER,
                "random_seed": RANDOM_SEED,
                "sim_step_size": dt,
                "sim_fps": 1.0 / dt,
                "max_steps": MAX_STEPS,
                "planned_duration_sec": MAX_STEPS * dt,
            },
            attack_metadata(attack, controller_vids, dt, tick_offset=run_start_tick),
        )
        data_saver.log_event(0.0, f"run initialized: {CASE_NAME}")
        camera_x, camera_y = scenario.get("intersection_center_xy_carla", [0, 0])
        cosim_client.set_custom_camera(camera_x, camera_y, CAMERA_HEIGHT)

        state = {"previous_entered": set(), "attack_started": False, "attack_ended": False, "duration_sec": None}
        generate_trips_with_gaps(cosim_client, trip_specs, data_saver, attacks, attack, controller_vids, run_start_tick, dt, state)
        while cosim_client.current_tick - run_start_tick < MAX_STEPS:
            run_one_step(cosim_client, data_saver, attacks, attack, controller_vids, run_start_tick, dt, state)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception:
        print("\nSimulation failed with exception:", flush=True)
        traceback.print_exc()
    finally:
        if data_saver is not None:
            data_saver.log_event((state or {}).get("duration_sec") or 0.0, "run finalized")
            data_saver.finalize(duration_sec=(state or {}).get("duration_sec"))
        if cosim_client is not None:
            cosim_client.close()


if __name__ == "__main__":
    main()
