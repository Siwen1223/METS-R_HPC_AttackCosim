"""
Two-flow extension of the Town05 intersection BSM position-falsification scenario.
This script keeps the same co-simulation/control structure as the multi-vehicle example,
but replaces the four single vehicles with two directional flows of ten vehicles each.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import time
import socket
import subprocess
import signal
import carla

from utils.util import read_run_config, prepare_sim_dirs, run_simulation_in_docker
from utils.carla_util import open_carla
from clients.CoSimClient import CoSimClient
from clients.KafkaDataProcessor import KafkaDataProcessor
from clients.KafkaDataSender import KafkaDataSender
# from cosim_utils.run_data_saver import RunDataSaver
from cosim_utils.v2v_controller_carla import V2VControllerCarla
from tools.derive_carla_roads_from_metsr import derive_carla_road_from_metsr


def is_port_open(port, host="127.0.0.1", timeout=0.5):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def stop_previous_metsr_containers():
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", "ancestor=ennuilei/mets-r_sim"],
            capture_output=True,
            text=True,
        )
        container_ids = [cid for cid in result.stdout.strip().split("\n") if cid]
        if container_ids:
            print(f"Stopping containers: {container_ids}")
            subprocess.run(["docker", "stop"] + container_ids)
        else:
            print("No METS-R containers running.")
    except Exception as e:
        print("Error stopping METS-R containers:", e)


def kill_process_on_port(port=8000):
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}"],
            capture_output=True,
            text=True,
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) > 1:
            for line in lines[1:]:
                parts = line.split()
                pid = int(parts[1])
                print(f"Killing process {pid} using port {port}")
                os.kill(pid, signal.SIGKILL)
        else:
            print(f"No process is using port {port}")
    except Exception as e:
        print(f"Error killing process on port {port}:", e)


def print_controller_debug(tick, controllers, cosim_client, min_pair_distance=8.0):
    active_vids = sorted(
        vid for vid in controllers
        if cosim_client.carla_vehs.get(vid) is not None
    )
    if not active_vids:
        return

    print(f"\n[debug] tick={tick}")
    for vid in active_vids:
        vehicle = cosim_client.carla_vehs.get(vid)
        state = controllers[vid].get_last_debug_state()
        loc = vehicle.get_location()
        yaw = vehicle.get_transform().rotation.yaw
        bearing = (yaw + 90.0) % 360.0
        speed = vehicle.get_velocity()
        speed_mps = (speed.x * speed.x + speed.y * speed.y + speed.z * speed.z) ** 0.5
        print(
            f"  veh={vid} loc=({loc.x:.2f},{loc.y:.2f}) bearing={bearing:.2f} speed={speed_mps:.2f} "
            f"lead={state.get('lead_vid')} lead_d={state.get('lead_distance')} "
            f"conflict={state.get('conflict_vid')} conflict_d={state.get('conflict_distance')} "
            f"factor={state.get('conflict_speed_factor')} junction_blocked={state.get('junction_blocked')} "
            f"ctrl=(thr={state.get('control_throttle')}, brk={state.get('control_brake')}, steer={state.get('control_steer')})"
        )

    for i, vid_a in enumerate(active_vids):
        veh_a = cosim_client.carla_vehs.get(vid_a)
        if veh_a is None:
            continue
        loc_a = veh_a.get_location()
        for vid_b in active_vids[i + 1:]:
            veh_b = cosim_client.carla_vehs.get(vid_b)
            if veh_b is None:
                continue
            loc_b = veh_b.get_location()
            dist = loc_a.distance(loc_b)
            if dist <= min_pair_distance:
                print(f"  [pair-close] veh={vid_a} veh={vid_b} dist={dist:.2f} m")


if __name__ == "__main__":
    force_restart = True
    if force_restart:
        stop_previous_metsr_containers()
        kill_process_on_port(2000)
        kill_process_on_port(2001)
    kill_process_on_port(8000)

    config = read_run_config("configs/run_cosim_CARLAT5.json")
    config.verbose = False

    if not is_port_open(29092):
        os.chdir("docker")
        os.system("docker-compose up -d")
        time.sleep(10)
        os.chdir("..")
    else:
        print("Kafka already running; reusing existing containers.")

    metsr_roads = ["-0", "-1", "0", "1", "-18", "40", "17", "-47"]
    config.metsr_road = metsr_roads

    kafkaDataProcessor = KafkaDataProcessor(config)
    kafkaDataSender = KafkaDataSender(config)
    kafkaDataProcessor.clear()

    config.display_all = True
    prepare_sim_dirs(config)

    carla_client, carla_tm = open_carla(config)
    print("CARLA server started successfully")

    carla_roads = derive_carla_road_from_metsr(
        metsr_road_ids=metsr_roads,
        config_path="configs/run_cosim_CARLAT5.json",
    )
    config.carla_road = carla_roads
    print(f"Derived carla_road from XML: {config.carla_road}")

    metsr_port = config.metsr_port[0] if hasattr(config, "metsr_port") else 4000
    metsr_reused = is_port_open(metsr_port)
    if not metsr_reused:
        run_simulation_in_docker(config)
    else:
        print(f"METS-R already running on port {metsr_port}; reusing existing instance.")

    cosim_client = CoSimClient(config, carla_client, carla_tm)
    print("CoSimClient created successfully")
    if metsr_reused and cosim_client.metsr.current_tick is None:
        print("METS-R current_tick is None; resetting simulation for reuse.")
        cosim_client.metsr.reset()
        for road in getattr(config, "metsr_road", []):
            cosim_client.metsr.set_cosim_road(road)
    cosim_client.metsr.tick(10)
    '''signal_ids = cosim_client.metsr.query_signal()['id_list']
    if signal_ids:
        cosim_client.metsr.set_signal_phase_plan_ticks(
            signal_ids,
            greenTicks=4000,
            yellowTicks=1,
            redTicks=1,
            startPhase=0,
            tickOffset=0,
        )
        print(f"Forced {len(signal_ids)} METS-R signals to stay green.")'''

    flow_a_vids = list(range(1, 4))
    #flow_b_vids = list(range(11, 16))
    cosim_client.metsr.generate_trip_between_roads(flow_a_vids, "41", "17")
    #cosim_client.metsr.generate_trip_between_roads(flow_b_vids, "48", "-1")
    cosim_client.metsr.update_vehicle_sensor_type(flow_a_vids, 1, True)
    #cosim_client.metsr.update_vehicle_sensor_type(flow_b_vids, 1, True)

    #cosim_client.set_custom_camera(-50, 0, 250)
    cosim_client.set_custom_camera(-50, 100, 100)

    replay_data = [[{
        "qty_SV_in_view": 9,
        "altitude": 0.0,
        "SemiMinorAxisAccuracy": 2.0,
        "elevation_confidence": 3.0,
        "heading": 90.0,
        "leap_seconds": 18,
        "SemiMajorAxisAccuracy": 2.0,
        "latitude": -0.0004018664573108967,
        "qty_SV_used": 9,
        "velocity": 0.0,
        "GNSS_unavailable": False,
        "vid": 0,
        "SemiMajorAxisOrientation": 0.0,
        "climb": 0.0,
        "time_confidence": 0.0,
        "utc_time": 126.0,
        "GNSS_networkCorrectionsPresent": False,
        "GNSS_localCorrectionsPresent": False,
        "GNSS_aPDOPofUnder5": False,
        "GNSS_inViewOfUnder5": False,
        "utc_fix_mode": 3,
        "longitude": -5.822375874899527e-05,
        "velocity_confidence": 0.5,
    }]]

    controller_vids = flow_a_vids# + flow_b_vids
    # vehicle_with_sensors = list(controller_vids)
    # for vid in vehicle_with_sensors:
    #     cosim_client.enable_vehicle_sensor(vid)

    controllers = {}
    route_synced = {}
    last_stream = []
    dt = getattr(config, "sim_step_size", 0.1)
    net_path = (ROOT_DIR / config.network_file).resolve()
    debug_every_n = 5
    debug_pair_distance = 10.0
    max_steps = 800

    # dataset_root = ROOT_DIR / "V2X-Attack-Dataset"
    # attack_info = {
    #     "attack_type": "BSM_replay",
    #     "attack_enabled": True,
    #     "start_time": 0.0,
    #     "end_time": 200 * dt,
    #     "parameters": {
    #         "fake_vehicle_id": 0,
    #     },
    # }
    # meta_info = {
    #     "scenario": "V2V_BSM_PosFalsifi_intersec_flow",
    #     "map": getattr(config, "carla_map", None),
    #     "random_seed": config.random_seeds[0] if getattr(config, "random_seeds", None) else None,
    #     "sim_step_size": dt,
    #     "sim_fps": 1.0 / dt if dt else None,
    #     "max_steps": max_steps,
    #     "planned_duration_sec": max_steps * dt,
    #     "vehicle_ids": controller_vids,
    #     "sensor_vehicle_ids": vehicle_with_sensors,
    # }
    # data_saver = RunDataSaver(dataset_root, meta_info, attack_info, sensor_every_n=5)
    # data_saver.log_event(0.0, "Simulation started")

    def init_controller_for_vid(vid):
        if vid in controllers:
            return
        carla_vehicle = cosim_client.carla_vehs.get(vid)
        if carla_vehicle is None:
            return
        carla_vehicle.set_autopilot(False)
        controller = V2VControllerCarla(
            vehicle=carla_vehicle,
            ego_vid=vid,
            net_path=net_path,
            target_speed_mps=10.0,
            enable_debug_draw=True,
        )
        controllers[vid] = controller
        route_synced[vid] = False

    try:
        sim_time = 0.0
        for i in range(max_steps):
            sim_time = i * dt
            for vid in controller_vids:
                init_controller_for_vid(vid)

            for vid, controller in controllers.items():
                if route_synced.get(vid, False):
                    continue
                carla_vehicle = cosim_client.carla_vehs.get(vid)
                if carla_vehicle is None:
                    continue
                start_loc = cosim_client.carla_handoff_locs.get(vid, carla_vehicle.get_location())
                start_yaw = cosim_client.carla_handoff_yaws.get(vid)
                cosim_client.carla.debug.draw_point(
                    start_loc,
                    size=0.25,
                    color=carla.Color(255, 0, 0),
                    life_time=30.0,
                    persistent_lines=True,
                )
                print(f"Vehicle {vid} route planning start loc=({start_loc.x:.2f},{start_loc.y:.2f},{start_loc.z:.2f})")
                route_ids = cosim_client.carla_route.get(vid, [])
                if route_ids and controller.set_route_from_metsr_route(
                    route_ids,
                    stop_waypoint_creation=True,
                    draw_plan=False,
                    start_point_carla=start_loc,
                    start_yaw_carla=start_yaw,
                ):
                        controller.path_planner.draw_lane_points()
                        print(f"Vehicle {vid} route synchronized with METS-R route: {route_ids}")
                        route_synced[vid] = True

            for vid, controller in controllers.items():
                carla_vehicle = cosim_client.carla_vehs.get(vid)
                if carla_vehicle is None:
                    continue
                if not route_synced.get(vid, False):
                    continue
                control = controller.run_step(last_stream, dt=dt)
                carla_vehicle.apply_control(control)
                # data_saver.record_control(i, sim_time, vid, control)

            '''if i <= 500:
                for data in replay_data[0]:
                    kafkaDataSender.send("bsm", data)'''

            cosim_client.step()

            done_vids = []
            inactive_vids = []
            for vid, controller in controllers.items():
                carla_vehicle = cosim_client.carla_vehs.get(vid)
                if carla_vehicle is None:
                    inactive_vids.append(vid)
                    continue
                if not route_synced.get(vid, False):
                    continue
                if not controller.is_route_complete():
                    continue
                private_flag = cosim_client.carla_private_flags.get(vid, False)
                res = cosim_client.metsr.reach_dest(vid, private_flag)
                if res["DATA"][0]["STATUS"] == "OK":
                    cosim_client.destroy_carla_vehicle(vid)
                    done_vids.append(vid)
            for vid in inactive_vids + done_vids:
                controllers.pop(vid, None)
                route_synced.pop(vid, None)
                # data_saver.log_event(sim_time, f"Vehicle {vid} finished route")

            # if i % data_saver.sensor_every_n == 0:
            #     data_saver.save_sensors(cosim_client)

            # data_saver.record_vehicle_state(i, sim_time, cosim_client)

            data_stream = kafkaDataProcessor.process()
            if data_stream is not None:
                last_stream = data_stream
                # data_saver.record_bsm(i, sim_time, data_stream)
            #if i % debug_every_n == 0:
            print_controller_debug(i, controllers, cosim_client, min_pair_distance=debug_pair_distance)
            time.sleep(0.08)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        # data_saver.log_event(sim_time, "Simulation ended")
        # data_saver.finalize(duration_sec=sim_time)
        pass

    cosim_client.metsr.terminate()
    os.chdir("docker")
    os.system("docker-compose down")
    os.chdir("..")
