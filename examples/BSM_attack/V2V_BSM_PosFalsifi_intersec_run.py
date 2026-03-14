"""
Dataset-oriented version of the single-intersection BSM position-falsification scenario.
This script extends V2V_BSM_PosFalsifi_intersec.py by keeping the same attack and driving setup while additionally saving run metadata, BSM logs, sensor outputs, vehicle states, and events.
Like the other current mainline co-simulation examples, it uses V2VControllerCarla and CARLA-side route tracking/control inside the co-simulation area.
Compared with the other V2V_BSM_* scripts, this file is intended for structured dataset generation rather than only interactive simulation and visualization.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import time
import socket

from utils.util import read_run_config, prepare_sim_dirs, run_simulation_in_docker
from utils.carla_util import open_carla
from clients.CoSimClient import CoSimClient
from clients.KafkaDataProcessor import KafkaDataProcessor
from clients.KafkaDataSender import KafkaDataSender
from cosim_utils.v2v_controller_carla import V2VControllerCarla
from cosim_utils.run_data_saver import RunDataSaver

import subprocess
import signal


def is_port_open(port, host="127.0.0.1", timeout=0.5):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def stop_previous_metsr_containers():
    """Stop all running METS-R SIM containers."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", "ancestor=ennuilei/mets-r_sim"],
            capture_output=True,
            text=True,
        )
        container_ids = result.stdout.strip().split('\n')
        container_ids = [cid for cid in container_ids if cid]

        if container_ids:
            print(f"Stopping containers: {container_ids}")
            subprocess.run(["docker", "stop"] + container_ids)
        else:
            print("No METS-R containers running.")
    except Exception as e:
        print("Error stopping METS-R containers:", e)


def kill_process_on_port(port=8000):
    """Kill the process that is using the given port."""
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}"],
            capture_output=True,
            text=True,
        )
        lines = result.stdout.strip().split('\n')
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


if __name__ == '__main__':
    force_restart = True
    if force_restart:
        stop_previous_metsr_containers()
        #kill_process_on_port(8000)
        kill_process_on_port(2000)
        kill_process_on_port(2001)
    kill_process_on_port(8000)

    # CARLA simulator configuration
    config = read_run_config("configs/run_cosim_CARLAT5.json")
    config.verbose = False

    # Start docker
    if not is_port_open(29092):
        os.chdir("docker")
        os.system("docker-compose up -d")
        time.sleep(10) # wait 10s for the Kafka servers to be up
        os.chdir("..")
    else:
        print("Kafka already running; reusing existing containers.")

    to_add_config = {"metsr_road": ["-39", "39", "0", "-0", "-18", "40", "-41", "41"],
                     "carla_road": [0, 18, 39, 40, 41, 243, 245, 249, 269, 281, 285, 293, 296, 306, 350, 351, 742, 746, 790, 791, 797, 824, 831, 1438, 1439, 1464, 1473, 1481, 1489, 1504, 1512, 1551, 1552, 1575, 1581, 1597, 1605, 1615, 1623, 1631, 1639, 1674, 1675, 1693, 1697, 1707, 1715, 2241, 2242, 2280, 2281]}
    for key, value in to_add_config.items():
            setattr(config, key, value)

    # Kafka configuration
    kafkaDataProcessor = KafkaDataProcessor(config)
    kafkaDataSender = KafkaDataSender(config)
    kafkaDataProcessor.clear()

    # CARLA visualization configuration
    config.display_all = True  # Enable display of all vehicles
    # Prepare simulation directories
    prepare_sim_dirs(config)
    # Start CARLA server
    carla_client, carla_tm = open_carla(config)
    print("CARLA server started successfully")
    # Launch METS-R simulation in Docker if not already running
    metsr_port = config.metsr_port[0] if hasattr(config, "metsr_port") else 4000
    metsr_reused = is_port_open(metsr_port)
    if not metsr_reused:
        run_simulation_in_docker(config)
    else:
        print(f"METS-R already running on port {metsr_port}; reusing existing instance.")

    # Create CoSimClient
    cosim_client = CoSimClient(config, carla_client, carla_tm)
    print("CoSimClient created successfully")
    if metsr_reused and cosim_client.metsr.current_tick is None:
        print("METS-R current_tick is None; resetting simulation for reuse.")
        cosim_client.metsr.reset()
        for road in getattr(config, "metsr_road", []):
            cosim_client.metsr.set_cosim_road(road)
    cosim_client.metsr.tick(10)

    # Generate trips
    cosim_client.metsr.generate_trip_between_roads([1], "-39", "-18")
    cosim_client.metsr.update_vehicle_sensor_type([1], 1, True)
    cosim_client.set_custom_camera(-50, 0, 100)

    cosim_client.metsr.generate_trip_between_roads([2], "41", "0")
    cosim_client.metsr.update_vehicle_sensor_type([2], 1, True)

    replay_data = [[{'qty_SV_in_view': 9, 'altitude': 0.0, 'SemiMinorAxisAccuracy': 2.0, 'elevation_confidence': 3.0, 'heading': 90.0, 'leap_seconds': 18, 'SemiMajorAxisAccuracy': 2.0, 'latitude': -0.0004018664573108967, 'qty_SV_used': 9, 'velocity': 0.0, 'GNSS_unavailable': False, 'vid': 0, 'SemiMajorAxisOrientation': 0.0, 'climb': 0.0, 'time_confidence': 0.0, 'utc_time': 126.0, 'GNSS_networkCorrectionsPresent': False, 'GNSS_localCorrectionsPresent': False, 'GNSS_aPDOPofUnder5': False, 'GNSS_inViewOfUnder5': False, 'utc_fix_mode': 3, 'longitude': -5.822375874899527e-05, 'velocity_confidence': 0.5}]]

    vehicle_with_sensors = [1, 2]
    for vid in vehicle_with_sensors:
         cosim_client.enable_vehicle_sensor(vid)

    controller_vids = [1, 2]
    controllers = {}
    route_synced = {}
    last_stream = []
    dt = getattr(config, "sim_step_size", 0.1)
    net_path = (ROOT_DIR / config.network_file).resolve()
    max_steps = 400

    dataset_root = ROOT_DIR / "V2X-Attack-Dataset"
    attack_info = {
        "attack_type": "BSM_pos_falsification",
        "attack_target": "intersection",
        "start_time": 0.0,
        "end_time": 265 * dt,
        "parameters": {
            "fake_vehicle_id": 0,
            "replay_delay_ms": 0,
        },
    }
    meta_info = {
        "map": getattr(config, "carla_map", None),
        "random_seed": config.random_seeds[0] if getattr(config, "random_seeds", None) else None,
        "sim_step_size": dt,
        "sim_fps": 1.0 / dt if dt else None,
        "max_steps": max_steps,
        "planned_duration_sec": max_steps * dt,
    }
    data_saver = RunDataSaver(dataset_root, meta_info, attack_info, sensor_every_n=5)
    data_saver.log_event(0.0, "Simulation started")

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

    attack_started = False
    attack_ended = False

    try:
        for i in range(max_steps):
            sim_time = i * dt
            for vid in controller_vids:
                init_controller_for_vid(vid)

            for vid, controller in controllers.items():
                if not route_synced.get(vid, False) and cosim_client.carla_entered.get(vid, False):
                    route_ids = cosim_client.carla_route.get(vid, [])
                    if route_ids:
                        if controller.set_route_from_metsr_route(route_ids, stop_waypoint_creation=True):
                            route_synced[vid] = True

            for vid, controller in controllers.items():
                carla_vehicle = cosim_client.carla_vehs.get(vid)
                if carla_vehicle is None:
                    continue
                if not cosim_client.carla_entered.get(vid, False):
                    continue
                control = controller.run_step(last_stream, dt=dt)
                carla_vehicle.apply_control(control)
                data_saver.record_control(i, sim_time, vid, control)

            # send attack
            if i <= 265:
                if not attack_started:
                    data_saver.log_event(sim_time, "BSM pos-falsification attack started")
                    attack_started = True
                for data in replay_data[0]:
                    kafkaDataSender.send("bsm", data)
            elif not attack_ended:
                data_saver.log_event(sim_time, "BSM pos-falsification attack ended")
                attack_ended = True

            # Step simulation with CARLA visualization
            cosim_client.step()

            done_vids = []
            for vid, controller in controllers.items():
                if not cosim_client.carla_entered.get(vid, False):
                    continue
                if not controller.is_route_complete():
                    continue
                private_flag = cosim_client.carla_private_flags.get(vid, False)
                dest_road = cosim_client.carla_destRoad.get(vid)
                if dest_road in getattr(config, "metsr_road", []):
                    res = cosim_client.metsr.reach_dest(vid, private_flag)
                    if res["DATA"][0]["STATUS"] == "OK":
                        cosim_client.destroy_carla_vehicle(vid)
                        done_vids.append(vid)
                else:
                    carla_vehicle = cosim_client.carla_vehs.get(vid)
                    if carla_vehicle is not None:
                        loc = carla_vehicle.get_location()
                        res = cosim_client.metsr.exit_cosim_region(vid, loc.x, -loc.y, private_flag, True)
                        if res["DATA"][0]["STATUS"] == "OK":
                            cosim_client.destroy_carla_vehicle(vid)
                            done_vids.append(vid)
            for vid in done_vids:
                controllers.pop(vid, None)
                route_synced.pop(vid, None)
                data_saver.log_event(sim_time, f"Vehicle {vid} finished route")

            if i % data_saver.sensor_every_n == 0:
                data_saver.save_sensors(cosim_client)

            data_saver.record_vehicle_state(i, sim_time, cosim_client)

            data_stream = kafkaDataProcessor.process()
            if data_stream is not None:
                last_stream = data_stream
                data_saver.record_bsm(i, sim_time, data_stream)

            time.sleep(0.08)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        data_saver.log_event(sim_time, "Simulation ended")
        data_saver.finalize(duration_sec=sim_time)

    # Terminate METS-R simulation
    cosim_client.metsr.terminate()
    os.chdir("docker")
    os.system("docker-compose down")
    os.chdir("..")
