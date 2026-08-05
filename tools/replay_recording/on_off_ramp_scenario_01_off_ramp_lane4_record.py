"""Record on/off-ramp scenario 01 mainline-to-off-ramp movement as replayable BSM data."""

import json
import sys
import time
import traceback
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from clients.V2V_CoSimClient_master import V2VCoSimClientMaster
from cosim_utils.C_V2X_manager import (
    bridge_message_from_bsm,
    make_bsm_message,
    make_geo_converter,
    normalize_vehicle,
)
from cosim_utils.helpers import is_port_open, load_scenario, set_random_seed
from utils.carla_util import open_carla
from utils.util import prepare_sim_dirs, read_run_config, run_simulation_in_docker


SCENARIO_PATH = ROOT_DIR / "V2V-Attack-Dataset/scenarios/on_ramp/on_off_ramp_scenario_01.yaml"
MOVEMENT_NAME = "off_ramp"
VEHICLE_ID = 1
LANE_PATHS = {VEHICLE_ID: ["-17_4", "-18_4", "-18_5", "-18_6", "-9_2"]}
MAX_STEPS = 700
WARMUP_STEPS = 0
TARGET_SPEED_MPS = 10.0
SCENE_CENTER_XY_CARLA = [-143.0, 239.0]
CAMERA_HEIGHT = 160
OUTPUT_DIR = ROOT_DIR / "V2V-Attack-Dataset/replay_records/on_ramp/on_off_ramp_scenario_01"
OUTPUT_NAME = "off_ramp_lane4.jsonl"
REUSE_METSR = False
RANDOM_SEED = 42
DEBUG_INTERVAL_TICKS = 10


def movement_spec(scenario, movement_name):
    movement = (scenario.get("movements") or {}).get(movement_name)
    if not isinstance(movement, dict):
        raise ValueError(f"Scenario movement is missing or invalid: {movement_name}")
    return movement["origin"], movement["destination"], movement.get("maneuver", "")


def configure_run(scenario, target_speed_mps):
    config = read_run_config(scenario["config_file"])
    set_random_seed(RANDOM_SEED, config=config)
    config.verbose = False
    config.display_all = False
    config.enable_debug_draw = False
    config.draw_route_plan = False
    config.v2v_position_mode = "local"
    config.carla_tick_timeout = 5.0
    config.metsr_tick_timeout = 5.0
    config.release_queued_cosim_vehicles = True
    config.v2v_target_speed_mps = float(target_speed_mps)
    config.controller_kwargs = {"route_project_to_carla_map": False}
    config.controller_lane_paths = LANE_PATHS
    config.metsr_road = list(scenario.get("cosim_roads") or [])
    if scenario.get("network_file"):
        config.network_file = scenario["network_file"]
    if scenario.get("town"):
        config.carla_map = scenario["town"]
    return config


def reset_metsr_if_reused(cosim_client):
    cosim_client.metsr.reset()
    for road in getattr(cosim_client.config, "metsr_road", []):
        cosim_client.metsr.set_cosim_road(road)


def current_sender_state(cosim_client, vehicle_id):
    for vehicle in cosim_client._current_v2x_vehicles():
        if str(vehicle.get("ID")) == str(vehicle_id):
            return vehicle
    return None


def is_synchronized_state(state):
    if state is None:
        return False
    x = float(state.get("x") or 0.0)
    y = float(state.get("y") or 0.0)
    z = float(state.get("z") or 0.0)
    speed = float(state.get("speed") or state.get("speed_mps") or 0.0)
    bearing = float(state.get("bearing") or state.get("heading_deg") or 0.0)
    return any(abs(value) > 1e-6 for value in (x, y, z, speed, bearing))


def enrich_for_bsm(config, geo_converter, state):
    state = normalize_vehicle(
        state,
        private_veh=True,
        role=state.get("role", "recorded_vehicle"),
        map_name=getattr(config, "carla_map", None),
    )
    if geo_converter is not None and state.get("latitude") is None and state.get("longitude") is None:
        latitude, longitude = geo_converter.to_lat_lon(state.get("x"), state.get("y"))
        if latitude is not None and longitude is not None:
            state["latitude"] = latitude
            state["longitude"] = longitude
    return state


def build_replay_record(config, geo_converter, state, source_tick, record_index):
    dt = float(getattr(config, "sim_step_size", 0.1))
    sim_time = int(source_tick) * dt
    message_count = int(record_index) % 128
    sec_mark_ms = int(round(sim_time * 1000.0)) % 60000
    sender = enrich_for_bsm(config, geo_converter, state)
    message = make_bsm_message(
        tick=source_tick,
        sequence=1,
        sender=sender,
        receiver=None,
        message_id=f"record:{source_tick}:{sender.get('ID')}:bsm",
        payload_bytes=300,
        tx_time_s=sim_time,
        message_count=message_count,
        sec_mark_ms=sec_mark_ms,
        content="recorded BSM for replay",
    )
    bridge_record = bridge_message_from_bsm(message)
    return {
        "record_index": int(record_index),
        "source_tick": int(source_tick),
        "sim_time": sim_time,
        "ID": sender.get("ID"),
        "vehicle_id": sender.get("ID"),
        "vid": sender.get("ID"),
        "role": "recorded_vehicle",
        "x": sender.get("x"),
        "y": sender.get("y"),
        "z": sender.get("z", 0.0),
        "latitude": sender.get("latitude"),
        "longitude": sender.get("longitude"),
        "speed": sender.get("speed", 0.0),
        "speed_mps": sender.get("speed", 0.0),
        "bearing": sender.get("bearing", 0.0),
        "heading_deg": sender.get("bearing", 0.0),
        "map_name": sender.get("map_name"),
        "bsm_core_data": bridge_record.get("bsm_core_data"),
        "transport_envelope": bridge_record.get("transport_envelope"),
        "operational_data": bridge_record.get("operational_data"),
        "bsm_speed_mps": bridge_record.get("bsm_speed_mps"),
        "bsm_heading_deg": bridge_record.get("bsm_heading_deg"),
        "bsm_latitude_deg": bridge_record.get("bsm_latitude_deg"),
        "bsm_longitude_deg": bridge_record.get("bsm_longitude_deg"),
    }


def write_meta(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_record(fp, record):
    fp.write(json.dumps(record, separators=(",", ":")) + "\n")
    fp.flush()


def print_controller_debug(cosim_client, vehicle_id, source_tick):
    controller = getattr(cosim_client, "controllers", {}).get(vehicle_id)
    control = (getattr(cosim_client, "last_controls", {}) or {}).get(vehicle_id)
    route_synced = (getattr(cosim_client, "route_synced", {}) or {}).get(vehicle_id, False)
    queue_len = None
    desired_speed = None
    ego_speed = None
    if controller is not None:
        local_planner = controller.agent.get_local_planner()
        queue_len = len(local_planner.get_plan())
        debug_state = controller.get_last_debug_state()
        desired_speed = debug_state.get("desired_speed")
        ego_speed = debug_state.get("ego_speed")
    if control is None:
        control_text = "control=None"
    else:
        control_text = (
            f"throttle={control.throttle:.3f} "
            f"brake={control.brake:.3f} "
            f"steer={control.steer:.3f}"
        )
    print(
        f"[record-debug] tick={source_tick} "
        f"route_synced={route_synced} queue_len={queue_len} "
        f"ego_speed={ego_speed} desired_speed={desired_speed} {control_text}"
    )


def main():
    scenario_path = Path(SCENARIO_PATH).resolve()
    scenario = load_scenario(scenario_path)
    road_from, road_to, maneuver = movement_spec(scenario, MOVEMENT_NAME)
    config = configure_run(scenario, TARGET_SPEED_MPS)
    config.controller_vids = [VEHICLE_ID]

    output_path = OUTPUT_DIR / OUTPUT_NAME
    meta_path = output_path.with_suffix(".meta.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Recording movement {MOVEMENT_NAME}: {road_from} -> {road_to}")
    print(f"Scenario: {scenario_path}")
    print(f"Output: {output_path}")

    prepare_sim_dirs(config)
    carla_client, carla_tm = open_carla(config)
    set_random_seed(RANDOM_SEED, traffic_manager=carla_tm)
    print("CARLA server started successfully")

    metsr_port = config.metsr_port[0] if hasattr(config, "metsr_port") else config.ports[0]
    metsr_reused = is_port_open(metsr_port)
    if not metsr_reused:
        run_simulation_in_docker(config)
    else:
        print(f"METS-R already running on port {metsr_port}.")

    cosim_client = None
    record_count = 0
    first_source_tick = None
    last_source_tick = None
    started = False
    geo_converter = make_geo_converter(config)
    try:
        cosim_client = V2VCoSimClientMaster(
            config,
            carla_client,
            carla_tm,
            controller_vids=[VEHICLE_ID],
            enable_v2x=False,
            require_simu5g_uu=False,
        )
        print("V2VCoSimClientMaster created successfully")
        if metsr_reused and not REUSE_METSR:
            print("Resetting reused METS-R instance for a clean recording run.")
            reset_metsr_if_reused(cosim_client)

        if WARMUP_STEPS > 0:
            cosim_client.metsr.tick(WARMUP_STEPS, max_wait_seconds=10, poll_timeout=1)
        print(f"Generating trip veh={VEHICLE_ID}: {road_from} -> {road_to} lane_path={LANE_PATHS[VEHICLE_ID]}")
        cosim_client.metsr.generate_trip_between_roads([VEHICLE_ID], road_from, road_to)
        cosim_client.metsr.update_vehicle_sensor_type([VEHICLE_ID], "cv2x", True)

        camera_x, camera_y = scenario.get("scene_center_xy_carla", SCENE_CENTER_XY_CARLA)
        cosim_client.set_custom_camera(camera_x, camera_y, CAMERA_HEIGHT)

        with output_path.open("w", encoding="utf-8") as fp:
            for _ in range(MAX_STEPS):
                step_result = cosim_client.step(phase="trajectory_record")
                source_tick = cosim_client.current_tick
                state = current_sender_state(cosim_client, VEHICLE_ID)
                if is_synchronized_state(state) and VEHICLE_ID in cosim_client.carla_vehs:
                    if not started:
                        print(f"Started recording at source tick {source_tick}")
                        first_source_tick = source_tick
                        started = True
                    record = build_replay_record(
                        config,
                        geo_converter,
                        state,
                        source_tick=source_tick,
                        record_index=record_count,
                    )
                    write_record(fp, record)
                    record_count += 1
                    last_source_tick = source_tick
                if (
                    VEHICLE_ID in cosim_client.carla_vehs
                    and (source_tick <= 50 or source_tick % DEBUG_INTERVAL_TICKS == 0)
                ):
                    print_controller_debug(cosim_client, VEHICLE_ID, source_tick)
                if VEHICLE_ID in step_result.get("done_vids", []) and started:
                    print(f"Vehicle {VEHICLE_ID} completed route at tick {source_tick}")
                    break
                time.sleep(0.02)

        meta = {
            "record_type": "on_off_ramp_replay_bsm_trajectory",
            "scenario_id": scenario.get("scenario_id", scenario_path.stem),
            "scenario_path": str(scenario_path),
            "movement": MOVEMENT_NAME,
            "maneuver": maneuver,
            "origin": road_from,
            "destination": road_to,
            "lane_path": LANE_PATHS[VEHICLE_ID],
            "vehicle_id": VEHICLE_ID,
            "target_speed_mps": TARGET_SPEED_MPS,
            "sim_step_size": float(getattr(config, "sim_step_size", 0.1)),
            "record_count": record_count,
            "first_source_tick": first_source_tick,
            "last_source_tick": last_source_tick,
            "output": str(output_path),
        }
        write_meta(meta_path, meta)
        print(f"Saved {record_count} replay BSM records")
        print(f"Metadata: {meta_path}")
    except KeyboardInterrupt:
        print()
        print("Recording interrupted by user")
    except Exception:
        print()
        print("Recording failed with exception:", flush=True)
        traceback.print_exc()
        raise
    finally:
        if cosim_client is not None:
            cosim_client.close()


if __name__ == "__main__":
    main()
