import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import os
import time
import xml.etree.ElementTree as ET

from utils.util import read_run_config, prepare_sim_dirs, run_simulation_in_docker
from utils.carla_util import open_carla
from clients.CoSimClient import CoSimClient
from clients.KafkaDataProcessor import KafkaDataProcessor 
from clients.KafkaDataSender import KafkaDataSender
from attack_data_collect_sim.v2v_controller_carla import V2VControllerCarla

import subprocess
import signal
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


def load_sumo_net(net_path):
    tree = ET.parse(net_path)
    root = tree.getroot()
    location = root.find("location")
    net_offset = (0.0, 0.0)
    if location is not None:
        offset_str = location.get("netOffset", "0,0")
        parts = offset_str.split(",")
        if len(parts) >= 2:
            net_offset = (float(parts[0]), float(parts[1]))
    nodes = {}
    for node in root.findall("node"):
        node_id = node.get("id")
        if node_id is None:
            continue
        nodes[node_id] = (float(node.get("x", "0")), float(node.get("y", "0")))
    edges = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if edge_id is None:
            continue
        edges[edge_id] = edge
    return nodes, edges, net_offset


def edge_shape_points(edge, nodes):
    shape = edge.get("shape")
    if shape:
        points = []
        for pair in shape.strip().split(" "):
            parts = pair.split(",")
            if len(parts) < 2:
                continue
            x_str, y_str = parts[0], parts[1]
            points.append((float(x_str), float(y_str)))
        return points
    from_id = edge.get("from")
    to_id = edge.get("to")
    if from_id in nodes and to_id in nodes:
        return [nodes[from_id], nodes[to_id]]
    return []


def sample_polyline(points, step):
    if len(points) < 2:
        return points
    sampled = [points[0]]
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        dx = x1 - x0
        dy = y1 - y0
        seg_len = (dx * dx + dy * dy) ** 0.5
        if seg_len == 0:
            continue
        num = max(1, int(seg_len // step))
        for i in range(1, num + 1):
            t = min(1.0, i * step / seg_len)
            sampled.append((x0 + t * dx, y0 + t * dy))
    return sampled


def build_route_coords(route_ids, edges, nodes, net_offset, step=5.0):
    coords = []
    for road_id in route_ids:
        edge = edges.get(str(road_id))
        if edge is None:
            continue
        points = edge_shape_points(edge, nodes)
        for x, y in sample_polyline(points, step):
            coords.append((x - net_offset[0], y - net_offset[1]))
    return coords

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

    to_add_config = {"metsr_road": ["-39", "39", "0", "-0", "-18", "40", "-41", "41"],
                     "carla_road": [0, 18, 39, 40, 41, 350, 351, 797, 1464, 1489, 1575, 1697, 2280]}
    for key, value in to_add_config.items():
            setattr(config, key, value)
        
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
    cosim_client.metsr.update_vehicle_sensor_type([1], 1, True)

    cosim_client.metsr.generate_trip_between_roads([2], "41", "0")
    cosim_client.metsr.update_vehicle_sensor_type([2], 1, True)
    cosim_client.set_custom_camera(-50, 0, 100)

    replay_data = [[{'qty_SV_in_view': 9, 'altitude': 0.0, 'SemiMinorAxisAccuracy': 2.0, 'elevation_confidence': 3.0, 'heading': 90.0, 'leap_seconds': 18, 'SemiMajorAxisAccuracy': 2.0, 'latitude': -0.0004018664573108967, 'qty_SV_used': 9, 'velocity': 0.0, 'GNSS_unavailable': False, 'vid': 0, 'SemiMajorAxisOrientation': 0.0, 'climb': 0.0, 'time_confidence': 0.0, 'utc_time': 126.0, 'GNSS_networkCorrectionsPresent': False, 'GNSS_localCorrectionsPresent': False, 'GNSS_aPDOPofUnder5': False, 'GNSS_inViewOfUnder5': False, 'utc_fix_mode': 3, 'longitude': -5.822375874899527e-05, 'velocity_confidence': 0.5}]]
    '''with open("position_falsi_replay.pkl", "rb") as f:
        replay_data = pickle.load(f)'''
    
    vehicle_with_sensors = [1, 2]
    vehicle_with_sensors = []
    for vid in vehicle_with_sensors:
         cosim_client.enable_vehicle_sensor(vid)
    save_path = "out_0105"
    #cosim_client.metsr.set_cosim_road(["-39", "39", "0", "-0", "-18", "40", "-41", "41"])

    controller_vids = [1, 2]
    controllers = {}
    route_synced = {}
    last_stream = []
    dt = getattr(config, "sim_step_size", 0.1)
    net_path = (ROOT_DIR / config.network_file).resolve()
    net_nodes, net_edges, net_offset = load_sumo_net(net_path)
    route_step = 5.0

    def init_controller_for_vid(vid):
        if vid in controllers:
            return
        carla_vehicle = cosim_client.carla_vehs.get(vid)
        if carla_vehicle is None:
            return
        carla_vehicle.set_autopilot(False)
        controller = V2VControllerCarla(vehicle=carla_vehicle, ego_vid=vid, target_speed_mps=10.0)
        controllers[vid] = controller
        route_synced[vid] = False



    # 采集数据、视频

    try:
        for i in range(6000):
            for vid in controller_vids:
                init_controller_for_vid(vid)

            for vid, controller in controllers.items():
                if not route_synced.get(vid, False) and cosim_client.carla_entered.get(vid, False):
                    route_ids = cosim_client.carla_route.get(vid, [])
                    if route_ids:
                        coord_map = build_route_coords(route_ids, net_edges, net_nodes, net_offset, step=route_step)
                        if coord_map:
                            controller.set_route_from_metsr_coords(coord_map, stop_waypoint_creation=True)
                            route_synced[vid] = True

            for vid, controller in controllers.items():
                carla_vehicle = cosim_client.carla_vehs.get(vid)
                if carla_vehicle is None:
                    continue
                if not cosim_client.carla_entered.get(vid, False):
                    continue
                control = controller.run_step(last_stream, dt=dt)
                carla_vehicle.apply_control(control)

            # send attack
            if i<= 265: 
                for data in replay_data[0]:
                    kafkaDataSender.send("bsm", data)

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
                    cosim_client.metsr.update_signal_timing(sid, greenTime=300, yellowTime=0, redTime=0)

                

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
