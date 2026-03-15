"""
Main no-attack baseline for the Town05 intersection co-simulation scenario.
This script runs the same two-vehicle intersection setup as the attack versions, but without injecting falsified V2V messages.
Unlike the older historical examples, it uses the CARLA-driven V2VControllerCarla and CARLA-side path planning/control inside the co-simulation area.
Compared with the other V2V_BSM_* scripts, this file is the clean reference run used to observe nominal behavior under the same routing and co-simulation configuration.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import time
import socket
import xml.etree.ElementTree as ET

from utils.util import read_run_config, prepare_sim_dirs, run_simulation_in_docker
from utils.carla_util import open_carla
from clients.CoSimClient import CoSimClient
from clients.KafkaDataProcessor import KafkaDataProcessor 
from cosim_utils.v2v_controller_carla import V2VControllerCarla

import subprocess
import signal

#from carla import TrafficLightState


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
    kafkaDataProcessor.clear()

    # CARLA visualization configuration
    config.display_all = True  # Enable display of all vehicles
    # Prepare simulation directories
    dest_data_dirs = prepare_sim_dirs(config)
    # Start CARLA server
    carla_client, carla_tm = open_carla(config)
    print("CARLA server started successfully")
    # Launch METS-R simulation in Docker if not already running
    metsr_port = config.metsr_port[0] if hasattr(config, "metsr_port") else 4000
    metsr_reused = is_port_open(metsr_port)
    if not metsr_reused:
        container_ids = run_simulation_in_docker(config)
    else:
        print(f"METS-R already running on port {metsr_port}; reusing existing instance.")

    ## Create CoSimClient (no co-simulation roads here)
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


    vehicle_with_sensors = [1, 2]
    vehicle_with_sensors = []
    for vid in vehicle_with_sensors:
         cosim_client.enable_vehicle_sensor(vid)
    save_path = "out_0105"
    #cosim_client.metsr.set_cosim_road(["-39", "39", "0", "-0", "-18", "40", "-41", "41"])

    controller_vids = [1, 2]
    #controller_vids = [1]
    controllers = {}
    route_synced = {}
    last_stream = []
    dt = getattr(config, "sim_step_size", 0.1)
    net_path = (ROOT_DIR / config.network_file).resolve()

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
    for vid in controller_vids:
        init_controller_for_vid(vid)


    try:
        for i in range(6000):
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

            if i == 1:
                '''# All green traffic lights (in Carla)
                for traffic_light in cosim_client.carla.get_actors().filter('traffic.traffic_light'):
                    traffic_light.set_state(TrafficLightState.Green)
                    traffic_light.freeze(True)'''
                
                '''# All green traffic lights (in Metsrsim)
                signal_ids = cosim_client.metsr.query_signal()['id_list']
                for sid in signal_ids:
                    cosim_client.metsr.update_signal_timing(sid, greenTime=300, yellowTime=0, redTime=0)'''

                

            if i % 5 == 0:
                for vid in vehicle_with_sensors:
                    cosim_client.collect_sensor_data(save_path)

            data_stream = kafkaDataProcessor.process()
            if data_stream is not None:
                last_stream = data_stream
            '''if i>=90 and i%5 == 0:
                vehicle_states = cosim_client.metsr.query_vehicle([1], True)['DATA']
                print('time', i)
                print('veh1:', vehicle_states)
                print('v2v:', data_stream)
                input("Press Enter to continue...")'''
            '''if i==115:
                # generate position falsification attack bsm data
                print('Attack data:', data_stream)
                replay_data = []
                replay_data.append(data_stream)
                with open("position_falsi_replay.pkl", "wb") as f:
                    pickle.dump(replay_data, f)'''
            time.sleep(0.08)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")

    ## Terminate METS-R simulation
    cosim_client.metsr.terminate()
    os.chdir("docker")
    os.system("docker-compose down")
    os.chdir("..")
