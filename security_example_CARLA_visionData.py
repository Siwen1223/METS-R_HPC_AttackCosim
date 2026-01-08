#import sys
import os

import time
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import pickle

# METS-R imports
from clients.METSRClient import METSRClient
from clients.CoSimClient import CoSimClient
from utils.util import read_run_config, prepare_sim_dirs, run_simulation_in_docker
from utils.carla_util import open_carla
from clients.KafkaDataProcessor import KafkaDataProcessor 
from clients.KafkaDataSender import KafkaDataSender

import subprocess
import signal

import math

class HeadingAwareController:
    def __init__(self, ego_vid, target_velocity=10.0, max_acceleration=2.0, kp=0.5, min_gap=5.0, cone_angle_deg=60):
        self.ego_vid = ego_vid
        self.target_velocity = target_velocity
        self.max_acceleration = max_acceleration
        self.kp = kp
        self.min_gap = min_gap
        self.cone_angle = math.radians(cone_angle_deg / 2)

    def haversine_meters(self, lat1, lon1, lat2, lon2):
        """Approximate small distance in meters (equirectangular)"""
        R = 6371000  # radius of Earth in meters
        x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
        y = math.radians(lat2 - lat1)
        return R * math.sqrt(x*x + y*y)

    def compute_acceleration(self, data_stream):
        ego = next((v for v in data_stream if v['vid'] == self.ego_vid), None)
        if not ego:
            raise ValueError("Ego vehicle not found")

        ego_lat = ego['latitude']
        ego_lon = ego['longitude']
        ego_vel = ego['velocity']
        ego_heading = ego['heading']

        # Find valid front vehicles in heading cone
        front_candidates = []
        for v in data_stream:
            if v['vid'] == self.ego_vid:
                continue
            
            dist = self.haversine_meters(ego_lat, ego_lon, v['latitude'], v['longitude'])
            front_candidates.append((dist, v))

        if front_candidates:
            front_dist, front = min(front_candidates, key=lambda x: x[0])
            relative_speed = ego_vel - front['velocity']
        else:
            front = None
            front_dist = float('inf')
            relative_speed = 0.0

        # --- Control Logic ---
        if front and front_dist < self.min_gap:
            acceleration = -self.max_acceleration
        elif front and front_dist < 2 * self.min_gap and relative_speed > 0:
            decel = min(self.max_acceleration, relative_speed ** 2 / (2 * (front_dist - self.min_gap)))
            acceleration = -decel
        else:
            error = self.target_velocity - ego_vel
            acceleration = self.kp * error

        return max(-self.max_acceleration, min(self.max_acceleration, acceleration))
    
def stop_previous_metsr_containers():
    """Stop all running METS-R SIM containers."""
    try:
        # 获取运行中的 ennuilei/mets-r_sim 容器ID列表
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", "ancestor=ennuilei/mets-r_sim"],
            capture_output=True,
            text=True,
        )
        container_ids = result.stdout.strip().split('\n')
        container_ids = [cid for cid in container_ids if cid]  # 去掉空字符串

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
        # 查找占用端口的进程
        result = subprocess.run(
            ["lsof", "-i", f":{port}"],
            capture_output=True,
            text=True,
        )
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            # 第一行是表头，第二行开始才是进程
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
    # docker stop $(docker ps -q --filter ancestor=ennuilei/mets-r_sim)
    # sudo lsof -i :8000
    # kill -9 ******
    stop_previous_metsr_containers()
    kill_process_on_port(8000)
    kill_process_on_port(2000)
    kill_process_on_port(2001)


    # CARLA simulator configuration
    config = read_run_config("configs/run_cosim_CARLAT7.json")
    config.verbose = False

    # Start docker
    os.chdir("docker")
    os.system("docker-compose up -d")
    time.sleep(10) # wait 10s for the Kafka servers to be up
    os.chdir("..")

    # Kafka configuration
    kafkaDataProcessor = KafkaDataProcessor(config)
    kafkaDataSender = KafkaDataSender(config)
    kafkaDataProcessor.clear()

    '''# Run simulation
    sim_dirs = prepare_sim_dirs(config)
    run_simulation_in_docker(config)
    print("Run simulation:", sim_dirs)

    #sim_client = METSRClient(host="localhost", sim_folder=sim_dirs[0], port=4000, verbose=False)
    #sim_client.terminate()
    sim_client = METSRClient(host="localhost", sim_folder=sim_dirs[0], port=4000, verbose=False)

    ## Collect the BSM used for attacking
    veh_num = 1
    sim_client.generate_trip_between_roads(list(range(1)), "-20", "-20")
    # Set up 1 vehicle as V2X vehicle (sensorType = 1)
    sim_client.update_vehicle_sensor_type(list(range(1)), 1, True)
    sim_client.query_vehicle(0, True)
    # create a very slow vehicle (probably by driving a real vehicle), record the information from the data stream
    replay_data = []
    for i in range(3000):
        data = sim_client.query_vehicle(list(range(1)), True)['DATA'][0]
        if data['speed'] > 1:
                sim_client.control_vehicle(0, -0.5, True)
        sim_client.tick(1)
        res = kafkaDataProcessor.process()
        replay_data.append(res)
    print("Time: " , 10)
    for key, value in replay_data[10][0].items():
        print(key, value)
    sim_client.terminate()
    with open("slow_vehicle_replay.pkl", "wb") as f:
        pickle.dump(replay_data, f)'''

    with open("slow_vehicle_replay.pkl", "rb") as f:
        replay_data = pickle.load(f)

    ## Lauch an attack
    # CARLA visualization configuration - Use CARLA T7 map as in security example
    config.display_all = True  # Enable display of all vehicles
    # Prepare simulation directories
    dest_data_dirs = prepare_sim_dirs(config)
    print(f"Simulation directories prepared: {dest_data_dirs}")
    # Start CARLA server
    carla_client, carla_tm = open_carla(config)
    print("CARLA server started successfully")
    # Launch METS-R simulation in Docker
    print("Launching METS-R simulation...")
    container_ids = run_simulation_in_docker(config)

    ## Create CoSimClient (no co-simulation roads needed for visualization-only mode)
    cosim_client = CoSimClient(config, carla_client, carla_tm)
    print("CoSimClient created successfully")
    cosim_client.metsr.tick(10)
    # Generate trips between roads as in security example
    print("Generating initial trips between roads...")
    # Following security_example.ipynb pattern: generate trips between specific roads
    # Vehicle 0 travels from road "-48#0" to road "-50"
    cosim_client.metsr.generate_trip_between_roads(list(range(1, 11)), "-20", "-20")
    controller = HeadingAwareController(1)
    cosim_client.metsr.update_vehicle_sensor_type(1, 1, True)
    cosim_client.set_custom_camera(80, -110, 180)

    # deploy sensors on vehicle 1 and 5
    vehicle_with_sensors = [1, 5]
    for vid in vehicle_with_sensors:
         cosim_client.enable_vehicle_sensor(vid)

    ## Run simulation with BSM attack active and CARLA visualization
    print("Running simulation with BSM attack and CARLA visualization...")
    print("Vehicles with compromised sensors should exhibit different behavior patterns")

    # Save BSM messages
    log_file = f"_out/bsm_{time.strftime('%Y%m%d_%H%M%S')}.txt"

    attacked_data = {}
    try:
        # Run simulation for specified duration
        for i in range(6000):
            # Step simulation with CARLA visualization
            cosim_client.step()
            # collect sensor data every 5 ticks (0.5s)
            if i % 5 == 0:
                for vid in vehicle_with_sensors:
                    cosim_client.collect_sensor_data()
            
            # Collect data every 10 ticks for analysis
            if i + 10 < 3000:
                if replay_data[i+10] is not None:
                    for data in replay_data[i+10]:
                            kafkaDataSender.send("bsm", data)
                            with open(log_file, "a") as f:
                                f.write(str(data) + "\n\n")
            data_stream = kafkaDataProcessor.process()
            if data_stream is not None:
                    vids = [data['vid'] for data in data_stream]
                    if 1 in vids:
                        acc = controller.compute_acceleration(data_stream)
                        cosim_client.metsr.control_vehicle(1, acc, True)
                    else:
                        cosim_client.metsr.control_vehicle(1, 0, True)  # no signal then keep the same speed
            else:
                    cosim_client.metsr.control_vehicle(1, 0, True)  # no signal then keep the same speed
            vehicle_states = cosim_client.metsr.query_vehicle(list(range(1, 101)), True)['DATA']
            for vid, vehicle_state in zip(list(range(1,101)), vehicle_states):
                    if vid not in attacked_data:
                        attacked_data[vid] = []
                    # Append the vehicle state to the list for this vid
                    attacked_data[vid].append(vehicle_state)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")

    ## Terminate METS-R simulation
    cosim_client.metsr.terminate()
    os.chdir("docker")
    os.system("docker-compose down")
    os.chdir("..")

    '''fig, axs = plt.subplots(1,2,figsize=(10, 3), sharex=True, sharey=False)

    for vid in range(1, 11):
        attacked_speed = [data['speed'] for data in attacked_data[vid]]
        attacked_acc = [data['acc'] for data in attacked_data[vid]]

    axs[0].plot(attacked_speed, lw=0.5)
    axs[1].plot(attacked_acc, lw=0.5)

    axs[0].set_ylabel(r"Vehicle speed (m/s)")
    axs[1].set_ylabel(r"Vehicle acceleration ($m/s^2$)")

    axs[0].set_xticks(np.arange(0, 6001, 600))

    axs[0].set_xticklabels(np.arange(0, 601, 60))

    axs[0].set_xlabel("Second")
    axs[1].set_xlabel("Second")

    axs[0].set_title("Vehicle speed")
    axs[1].set_title("Vehicle acc")

    axs[0].set_xlim([0, 6000])
    axs[0].set_ylim([0, 17])
    axs[1].set_xlim([0, 6000])
    axs[1].set_ylim([-3, 3])
    plt.show()'''




                   