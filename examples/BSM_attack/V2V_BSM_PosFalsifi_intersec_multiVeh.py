"""
Multi-vehicle extension of the Town05 intersection BSM position-falsification scenario.
This script generalizes the single-intersection attack setup to multiple vehicles simultaneously traversing the same intersection from different directions under co-simulation.
Like the other current mainline co-simulation examples, it uses V2VControllerCarla and CARLA-side route tracking/control within the co-simulation area.
Compared with V2V_BSM_PosFalsifi_intersec.py and V2V_BSM_PosFalsifi_intersec_run.py, this file focuses on denser vehicle interaction rather than the minimal two-vehicle case or dataset saving.
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

import carla

from utils.util import read_run_config, prepare_sim_dirs, run_simulation_in_docker
from utils.carla_util import open_carla
from clients.CoSimClient import CoSimClient
from clients.KafkaDataProcessor import KafkaDataProcessor
from clients.KafkaDataSender import KafkaDataSender
from cosim_utils.run_data_saver import RunDataSaver
from cosim_utils.v2v_controller_carla import V2VControllerCarla

import subprocess
import signal


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


def load_sumo_net(net_path: Path):
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
            points.append((float(parts[0]), float(parts[1])))
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


def derive_carla_roads(carla_world, network_file, metsr_roads, sample_step=5.0, coverage_threshold=5.0, carla_step=2.0):
    net_path = (ROOT_DIR / network_file).resolve()
    nodes, edges, net_offset = load_sumo_net(net_path)
    carla_map = carla_world.get_map()

    def transform_point(x, y, mode):
        if mode == "flip":
            return x, -y
        if mode == "flip+add":
            return x + net_offset[0], -(y + net_offset[1])
        if mode == "flip+sub":
            return x - net_offset[0], -(y - net_offset[1])
        raise ValueError(mode)

    def eval_transform(mode):
        distances = []
        for road_id in metsr_roads:
            edge = edges.get(road_id)
            if edge is None:
                continue
            points = edge_shape_points(edge, nodes)
            for x, y in sample_polyline(points, sample_step):
                tx, ty = transform_point(x, y, mode)
                loc = carla.Location(x=tx, y=ty, z=0.0)
                wp = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                distances.append(loc.distance(wp.transform.location))
        return sum(distances) / max(1, len(distances))

    best_mode = min(["flip", "flip+add", "flip+sub"], key=eval_transform)

    sumo_points = []
    for road_id in metsr_roads:
        edge = edges.get(road_id)
        if edge is None:
            continue
        points = edge_shape_points(edge, nodes)
        for x, y in sample_polyline(points, sample_step):
            tx, ty = transform_point(x, y, best_mode)
            sumo_points.append((tx, ty))

    threshold_sq = coverage_threshold * coverage_threshold
    road_ids = set()

    def sample_carla_segment_points(wp_start, wp_end, step):
        pts = []
        wp = wp_start
        end_loc = wp_end.transform.location
        for _ in range(2000):
            loc = wp.transform.location
            pts.append((loc.x, loc.y))
            if loc.distance(end_loc) < step * 1.5:
                pts.append((end_loc.x, end_loc.y))
                break
            nxt = wp.next(step)
            if not nxt:
                break
            wp = nxt[0]
            if wp.road_id != wp_start.road_id or wp.lane_id != wp_start.lane_id:
                break
        return pts

    for wp_start, wp_end in carla_map.get_topology():
        pts = sample_carla_segment_points(wp_start, wp_end, carla_step)
        keep = False
        for px, py in pts:
            min_d2 = min((px - sx) ** 2 + (py - sy) ** 2 for sx, sy in sumo_points)
            if min_d2 <= threshold_sq:
                keep = True
                break
        if keep:
            road_ids.add(wp_start.road_id)

    return sorted(road_ids), best_mode


def print_controller_debug(tick, controllers, cosim_client, min_pair_distance=8.0):
    active_vids = sorted(
        vid for vid in controllers
        if cosim_client.carla_entered.get(vid, False) and cosim_client.carla_vehs.get(vid) is not None
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

    carla_roads, transform_mode = derive_carla_roads(
        carla_client.get_world(),
        config.network_file,
        metsr_roads,
        sample_step=5.0,
        coverage_threshold=5.0,
        carla_step=2.0,
    )
    config.carla_road = carla_roads
    print(f"Derived carla_road with transform={transform_mode}: {config.carla_road}")

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

    # Four vehicles, four directions, go straight through node 1427 intersection.
    trip_specs = [
        (1, "-0", "-1"),
        (2, "1", "0"),
        (3, "-47", "-18"),
        (4, "40", "17"),
    ]
    for vid, road_from, road_to in trip_specs:
        cosim_client.metsr.generate_trip_between_roads([vid], road_from, road_to)
        cosim_client.metsr.update_vehicle_sensor_type([vid], 1, True)

    cosim_client.set_custom_camera(-50, 0, 100)

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

    controller_vids = [1, 2, 3, 4]
    vehicle_with_sensors = list(controller_vids)
    for vid in vehicle_with_sensors:
        cosim_client.enable_vehicle_sensor(vid)

    controllers = {}
    route_synced = {}
    last_stream = []
    dt = getattr(config, "sim_step_size", 0.1)
    net_path = (ROOT_DIR / config.network_file).resolve()
    debug_every_n = 5
    debug_pair_distance = 10.0
    max_steps = 300

    dataset_root = ROOT_DIR / "V2X-Attack-Dataset"
    attack_info = {
        "attack_type": "BSM_replay",
        "attack_enabled": True,
        "start_time": 0.0,
        "end_time": 200 * dt,
        "parameters": {
            "fake_vehicle_id": 0,
        },
    }
    meta_info = {
        "scenario": "V2V_BSM_PosFalsifi_intersec_multiVeh",
        "map": getattr(config, "carla_map", None),
        "random_seed": config.random_seeds[0] if getattr(config, "random_seeds", None) else None,
        "sim_step_size": dt,
        "sim_fps": 1.0 / dt if dt else None,
        "max_steps": max_steps,
        "planned_duration_sec": max_steps * dt,
        "vehicle_ids": controller_vids,
        "sensor_vehicle_ids": vehicle_with_sensors,
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

    try:
        sim_time = 0.0
        for i in range(max_steps):
            sim_time = i * dt
            for vid in controller_vids:
                init_controller_for_vid(vid)

            for vid, controller in controllers.items():
                if not route_synced.get(vid, False) and cosim_client.carla_entered.get(vid, False):
                    route_ids = cosim_client.carla_route.get(vid, [])
                    if route_ids and controller.set_route_from_metsr_route(route_ids, stop_waypoint_creation=True):
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

            # Same attack pattern as single-intersection demo.
            if i <= 200:
                for data in replay_data[0]:
                    kafkaDataSender.send("bsm", data)

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
            if i % debug_every_n == 0:
                print_controller_debug(i, controllers, cosim_client, min_pair_distance=debug_pair_distance)
            time.sleep(0.08)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        data_saver.log_event(sim_time, "Simulation ended")
        data_saver.finalize(duration_sec=sim_time)

    cosim_client.metsr.terminate()
    os.chdir("docker")
    os.system("docker-compose down")
    os.chdir("..")
