"""
Historical version from November 2025.
This script belongs to the early single-scenario prototype line and uses the in-file HeadingAwareController instead of the newer CARLA-side V2VControllerCarla.
Its control logic is a simple V2V-based acceleration controller with heading-cone filtering, but it still does not use CARLA lane-level path planning, conflict handling, or CARLA-driven route tracking.
This example focuses on a BSM position-falsification attack at the intersection scenario that later evolved into the newer V2V_BSM_* CARLA-controlled scripts.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import time

from utils.util import read_run_config, prepare_sim_dirs, run_simulation_in_docker
from utils.carla_util import open_carla
from clients.CoSimClient import CoSimClient
from clients.KafkaDataProcessor import KafkaDataProcessor 
from clients.KafkaDataSender import KafkaDataSender

import subprocess
import signal
import math
from carla import TrafficLightState
import pickle


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

class HeadingAwareController:
    def __init__(self, ego_vid, target_velocity=10.0, max_acceleration=2.0, max_decceleration=-6.0, kp=0.5, min_gap=8.0, cone_angle_deg=60):
        self.ego_vid = ego_vid
        self.target_velocity = target_velocity
        self.max_acceleration = max_acceleration
        self.max_decceleration = max_decceleration
        self.kp = kp
        self.min_gap = min_gap
        self.cone_angle = math.radians(cone_angle_deg / 2)

    def haversine_meters(self, lat1, lon1, lat2, lon2):
        """Approximate small distance in meters (equirectangular)"""
        R = 6371000  # radius of Earth in meters
        x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
        y = math.radians(lat2 - lat1)
        return R * math.sqrt(x*x + y*y)
    
    def bearing_radians(self, lat1, lon1, lat2, lon2):
        """
        Compute relative bearing from (lat1, lon1) to (lat2, lon2).
        latitude => East-West  local x
        longitude => North-South  local y
        """
        dN = lon2 - lon1   # North from longitude
        dE = lat2 - lat1   # East  from latitude
        return math.atan2(dE, dN)
    
    def bearing_degrees(self, lat1, lon1, lat2, lon2):
        """
        Same as bearing_radians, but return degrees.
        """
        return math.degrees(self.bearing_radians(lat1, lon1, lat2, lon2))


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
            bearing = self.bearing_radians(ego_lat, ego_lon, v['latitude'], v['longitude'])
            heading = math.radians((ego_heading + 180) % 360 - 180)
            angle_diff = (bearing - heading + math.pi) % (2 * math.pi) - math.pi
            if abs(angle_diff) < self.cone_angle:
                front_candidates.append((dist, v, angle_diff))

        if front_candidates:
            front_dist, front, angle_diff = min(front_candidates, key=lambda x: x[0])
            relative_speed = ego_vel - front['velocity']
        else:
            front = None
            front_dist = float('inf')
            relative_speed = 0.0

        # --- Control Logic ---
        if front and front_dist < self.min_gap:
            acceleration = self.max_decceleration
        #elif front and front_dist < 2 * self.min_gap and relative_speed > 0: # 还要改
        elif front and front_dist < 3 * self.min_gap and relative_speed > 0:
            decel = min(-self.max_decceleration, relative_speed ** 2 / (2 * (front_dist - self.min_gap)))
            acceleration = -decel
        else:
            error = self.target_velocity - ego_vel
            acceleration = self.kp * error

        return max(self.max_decceleration, min(self.max_acceleration, acceleration))

if __name__ == '__main__':
    stop_previous_metsr_containers()
    kill_process_on_port(8000)
    kill_process_on_port(2000)
    kill_process_on_port(2001)

    # CARLA simulator configuration
    config = read_run_config("configs/run_cosim_CARLAT5.json")
    config.verbose = False

     # Start docker
    os.chdir("docker")
    os.system("docker-compose up -d")
    time.sleep(10) # wait 10s for the Kafka servers to be up
    os.chdir("..")

    '''to_add_config = {"metsr_road": ["-39", "39", "0", "-0", "-18", "40", "-41", "41"],
                     "carla_road": [39, 1631, 1639, 1674, 1623, 0, 1552, 1489, 1481, \
                                    17, 47, 46, 1211, 1174, 1140, 1182, 1222, 1290, 1141, 1291, 1236, 1193, 1231, 1201]}
    '''
    # Kafka configuration
    kafkaDataProcessor = KafkaDataProcessor(config)
    kafkaDataSender = KafkaDataSender(config)
    kafkaDataProcessor.clear()

    # CARLA visualization configuration
    config.display_all = True  # Enable display of all vehicles
    # Prepare simulation directories
    dest_data_dirs = prepare_sim_dirs(config)
    # Start CARLA server
    carla_client, carla_tm = open_carla(config)
    print("CARLA server started successfully")
    # Launch METS-R simulation in Docker
    container_ids = run_simulation_in_docker(config)

    ## Create CoSimClient (no co-simulation roads here)
    cosim_client = CoSimClient(config, carla_client, carla_tm)
    print("CoSimClient created successfully")
    cosim_client.metsr.tick(10)

    # Generate trips
    cosim_client.metsr.generate_trip_between_roads([1], "-39", "-18")
    controller = HeadingAwareController(1)
    cosim_client.metsr.update_vehicle_sensor_type([1], 1, True)

    cosim_client.metsr.generate_trip_between_roads([2], "41", "0")
    controller1 = HeadingAwareController(2)
    cosim_client.metsr.update_vehicle_sensor_type([2], 1, True)
    cosim_client.set_custom_camera(-50, 0, 100)

    #replay_data = [[{'qty_SV_in_view': 9, 'altitude': 0.0, 'SemiMinorAxisAccuracy': 2.0, 'elevation_confidence': 3.0, 'heading': 90.0, 'leap_seconds': 18, 'SemiMajorAxisAccuracy': 2.0, 'latitude': -0.0004018664573108967, 'qty_SV_used': 9, 'velocity': 0.0, 'GNSS_unavailable': False, 'vid': 0, 'SemiMajorAxisOrientation': 0.0, 'climb': 0.0, 'time_confidence': 0.0, 'utc_time': 126.0, 'GNSS_networkCorrectionsPresent': False, 'GNSS_localCorrectionsPresent': False, 'GNSS_aPDOPofUnder5': False, 'GNSS_inViewOfUnder5': False, 'utc_fix_mode': 3, 'longitude': -5.822375874899527e-05, 'velocity_confidence': 0.5}]]
    '''with open("position_falsi_replay.pkl", "rb") as f:
        replay_data = pickle.load(f)'''
    
    vehicle_with_sensors = [1, 2]
    vehicle_with_sensors = []
    for vid in vehicle_with_sensors:
         cosim_client.enable_vehicle_sensor(vid)
    save_path = "out_1221"
    #cosim_client.metsr.set_cosim_road(["-39", "39", "0", "-0", "-18", "40", "-41", "41"])



    # 采集数据、视频

    try:
        for i in range(6000):
            # Step simulation with CARLA visualization
            cosim_client.step()
            if i == 1:
                '''# All green traffic lights (in Carla)
                for traffic_light in cosim_client.carla.get_actors().filter('traffic.traffic_light'):
                    traffic_light.set_state(TrafficLightState.Green)
                    traffic_light.freeze(True)'''
                
                # All green traffic lights (in Metsrsim)
                signal_ids = cosim_client.metsr.query_signal()['id_list']
                for sid in signal_ids:
                    cosim_client.metsr.set_signal_phase_plan_ticks(sid, greenTicks=4000, yellowTicks=1, 
                                                 redTicks=1, startPhase=0, tickOffset=0)


                

            if i % 5 == 0:
                for vid in vehicle_with_sensors:
                    cosim_client.collect_sensor_data(save_path)

            # send attack
            '''if i<= 265: 
                for data in replay_data[0]:
                    kafkaDataSender.send("bsm", data)'''

            data_stream = kafkaDataProcessor.process()
            if data_stream is not None:
                    vids = [data['vid'] for data in data_stream]
                    if 1 in vids:
                        acc = controller.compute_acceleration(data_stream)
                        cosim_client.metsr.control_vehicle(1, acc, True)
                    else:
                        cosim_client.metsr.control_vehicle(1, 0, True)  # no signal then keep the same speed
                    if 2 in vids:
                        acc = controller1.compute_acceleration(data_stream)
                        cosim_client.metsr.control_vehicle(2, acc, True)
                    else:
                        cosim_client.metsr.control_vehicle(2, 0, True)  # no signal then keep the same speed
            else:
                    cosim_client.metsr.control_vehicle(1, 0, True)  # no signal then keep the same speed
                    cosim_client.metsr.control_vehicle(2, 0, True)  # no signal then keep the same speed
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
