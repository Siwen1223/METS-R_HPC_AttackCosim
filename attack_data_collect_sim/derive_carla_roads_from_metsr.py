import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import carla

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.util import read_run_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive CARLA road_id list from METS-R/SUMO edge IDs."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/run_cosim_CARLAT5.json",
        help="Run config JSON (for network_file and CARLA host/port).",
    )
    parser.add_argument(
        "-r",
        "--metsr-roads",
        required=True,
        help="Comma-separated SUMO edge IDs, e.g. -39,39,0,-0,-18,40,-41,41",
    )
    parser.add_argument(
        "--sample-step",
        type=float,
        default=5.0,
        help="Sample interval in meters along each edge polyline.",
    )
    parser.add_argument(
        "--transform",
        choices=["auto", "flip", "flip+add", "flip+sub"],
        default="auto",
        help="Coordinate transform: flip=(x, -y), add/sub uses netOffset with flip.",
    )
    return parser.parse_args()


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


def main():
    args = parse_args()
    config = read_run_config(args.config)
    net_path = (ROOT_DIR / config.network_file).resolve()
    if not net_path.exists():
        print(f"ERROR: network_file not found: {net_path}")
        sys.exit(1)

    nodes, edges, net_offset = load_sumo_net(net_path)
    metsr_roads = [r.strip() for r in args.metsr_roads.split(",") if r.strip()]

    client = carla.Client(config.carla_host, config.carla_port)
    client.set_timeout(10.0)
    if getattr(config, "carla_map", None):
        world = client.load_world(config.carla_map)
    else:
        world = client.get_world()
    carla_map = world.get_map()

    def transform_point(x, y, mode):
        if mode == "flip":
            return x, -y
        if mode == "flip+add":
            return x + net_offset[0], -(y + net_offset[1])
        if mode == "flip+sub":
            return x - net_offset[0], -(y - net_offset[1])
        raise ValueError(f"Unknown transform mode: {mode}")

    def eval_transform(mode):
        road_ids = set()
        distances = []
        missing_edges = []
        for road_id in metsr_roads:
            edge = edges.get(road_id)
            if edge is None:
                missing_edges.append(road_id)
                continue
            points = edge_shape_points(edge, nodes)
            for x, y in sample_polyline(points, args.sample_step):
                tx, ty = transform_point(x, y, mode)
                location = carla.Location(x=tx, y=ty, z=0.0)
                waypoint = carla_map.get_waypoint(
                    location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                road_ids.add(waypoint.road_id)
                distances.append(location.distance(waypoint.transform.location))
        avg_dist = sum(distances) / max(1, len(distances))
        max_dist = max(distances) if distances else 0.0
        return road_ids, missing_edges, avg_dist, max_dist

    modes = ["flip", "flip+add", "flip+sub"] if args.transform == "auto" else [args.transform]
    results = {}
    for mode in modes:
        results[mode] = eval_transform(mode)

    if args.transform == "auto":
        best_mode = min(results.items(), key=lambda item: item[1][2])[0]
    else:
        best_mode = args.transform

    road_ids, missing_edges, avg_dist, max_dist = results[best_mode]

    if missing_edges:
        print(f"WARNING: edges not found in SUMO net: {missing_edges}")

    road_list = sorted(road_ids)
    print(f"Selected transform: {best_mode}")
    print(f"Average projection distance: {avg_dist:.2f} m (max {max_dist:.2f} m)")
    print("Derived carla_road list:")
    print(road_list)
    print("\nPython snippet:")
    print(f"config.carla_road = {road_list}")


if __name__ == "__main__":
    main()
