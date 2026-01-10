import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import carla

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.util import read_run_config
from agents.navigation.global_route_planner import GlobalRoutePlanner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw CARLA start/end points for METS-R road IDs."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/run_cosim_CARLAT5.json",
        help="Run config JSON (for network_file and CARLA host/port).",
    )
    parser.add_argument("--origin-road", default="-39")
    parser.add_argument("--dest-road", default="-18")
    parser.add_argument(
        "--route",
        default="-39,-0,-18",
        help="Comma-separated METS-R road IDs (e.g., -39,-0,-18).",
    )
    parser.add_argument("--life-time", type=float, default=30.0)
    parser.add_argument(
        "--half-road-width",
        type=float,
        default=3.5,
        help="Offset distance in meters to separate opposite directions.",
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
    edges = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if edge_id is None:
            continue
        edges[edge_id] = edge
    return edges, net_offset


def edge_shape_points(edge):
    shape = edge.get("shape")
    if not shape:
        return []
    points = []
    for pair in shape.strip().split(" "):
        parts = pair.split(",")
        if len(parts) < 2:
            continue
        points.append((float(parts[0]), float(parts[1])))
    return points


def start_point(points):
    if not points:
        return None, None
    if len(points) < 2:
        return points[0], None
    return points[0], points[1]


def end_point(points):
    if not points:
        return None, None
    if len(points) < 2:
        return points[-1], None
    return points[-2], points[-1]


def offset_point(p0, p1, offset, direction='right'):
    if p0 is None or p1 is None or offset == 0.0:
        return p0
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    norm = (dx * dx + dy * dy) ** 0.5
    if norm == 0.0:
        return p0
    # Clockwise 90-degree perpendicular: (dx, dy) -> (dy, -dx)
    if direction=='right':
        ox = dy / norm * offset
        oy = -dx / norm * offset
    else:
        ox = -dy / norm * offset
        oy = dx / norm * offset
    return (p0[0] + ox, p0[1] + oy)


def sumo_to_metsr(point, net_offset):
    return point[0] - net_offset[0], point[1] - net_offset[1]


def metsr_to_carla(point):
    return carla.Location(x=point[0], y=-point[1], z=0.5)


def main():
    args = parse_args()
    config = read_run_config(args.config)
    net_path = (ROOT_DIR / config.network_file).resolve()
    if not net_path.exists():
        raise FileNotFoundError(net_path)

    edges, net_offset = load_sumo_net(net_path)
    if args.route:
        route_ids = [r.strip() for r in args.route.split(",") if r.strip()]
    else:
        route_ids = [args.origin_road, args.dest_road]

    route_waypoints = []
    missing = []
    for i, road_id in enumerate(route_ids):
        edge = edges.get(road_id)
        if edge is None:
            missing.append(road_id)
            continue
        points = edge_shape_points(edge)
        if not points:
            missing.append(road_id)
            continue
        p0, p1 = start_point(points)
        if p0 is None:
            missing.append(road_id)
            continue
        start_off = offset_point(p0, p1, args.half_road_width)
        route_waypoints.append(sumo_to_metsr(start_off, net_offset))
        if i == len(route_ids) - 1:
            pend0, pend1 = end_point(points)
            end_off = offset_point(pend1, pend0, args.half_road_width, direction='left')
            route_waypoints.append(sumo_to_metsr(end_off, net_offset))

    if missing:
        print(f"Missing edges: {missing}")
    if len(route_waypoints) < 2:
        print("Not enough route waypoints to draw.")
        return

    client = carla.Client(config.carla_host, config.carla_port)
    client.set_timeout(10.0)
    if getattr(config, "carla_map", None):
        world = client.load_world(config.carla_map)
    else:
        world = client.get_world()
    spectator = world.get_spectator()
    transform = carla.Transform()
    transform.location.x = -40
    transform.location.y = 0
    transform.location.z = 200
    transform.rotation.yaw = -90
    transform.rotation.pitch = -90
    spectator.set_transform(transform)

    coarse_points = [metsr_to_carla(p) for p in route_waypoints]

    for loc in coarse_points:
        world.debug.draw_point(
            loc,
            size=0.35,
            color=carla.Color(0, 255, 255),
            life_time=0.0,
            persistent_lines=True,
        )

    grp = GlobalRoutePlanner(world.get_map(), sampling_resolution=2.0)
    lane_waypoints = []
    for cur, nxt in zip(coarse_points[:-1], coarse_points[1:]):
        segment = grp.trace_route(cur, nxt)
        lane_waypoints.extend(segment)

    '''for wp, _ in lane_waypoints:
        world.debug.draw_point(
            wp.transform.location,
            size=0.12,
            color=carla.Color(255, 255, 0),
            life_time=0.0,
            persistent_lines=True,
        )'''

    world.debug.draw_point(coarse_points[0], size=0.45, color=carla.Color(0, 255, 0), life_time=0.0, persistent_lines=True)
    world.debug.draw_point(coarse_points[-1], size=0.45, color=carla.Color(255, 0, 0), life_time=0.0, persistent_lines=True)

    print("Route waypoints (METS-R):", route_waypoints)
    print("Drawn in CARLA. Cyan=route waypoints, Yellow=lane waypoints, Green=origin, Red=dest.")


if __name__ == "__main__":
    main()
