from pathlib import Path
import xml.etree.ElementTree as ET

import carla

from agents.navigation.global_route_planner import GlobalRoutePlanner


class CosimPathPlanner:
    def __init__(self, world, net_path, half_road_width=3.5, sampling_resolution=2.0):
        self.world = world
        self.map = world.get_map() if world is not None else None
        self.half_road_width = half_road_width
        self.net_path = Path(net_path)
        self.edges, self.net_offset = self._load_sumo_net(self.net_path)
        self.grp = None
        if self.map is not None:
            self.grp = GlobalRoutePlanner(self.map, sampling_resolution=sampling_resolution)
        self.coarse_points_metsr = []
        self.coarse_points_carla = []
        self.lane_waypoints = []
        self.missing_edges = []

    def build_coarse_points(self, route_ids):
        self.coarse_points_metsr = []
        self.coarse_points_carla = []
        self.missing_edges = []
        if not route_ids:
            return []
        for i, road_id in enumerate(route_ids):
            edge = self.edges.get(str(road_id))
            if edge is None:
                self.missing_edges.append(road_id)
                continue
            points = self._edge_shape_points(edge)
            if not points:
                self.missing_edges.append(road_id)
                continue
            for frac in (0.2, 0.5):
                p0, p1 = self._point_at_fraction(points, frac)
                if p0 is None or p1 is None:
                    self.missing_edges.append(road_id)
                    continue
                off = self._offset_point(p0, p1, self.half_road_width, direction="right")
                self.coarse_points_metsr.append(self._sumo_to_metsr(off))
            if i == len(route_ids) - 1:
                pend0, pend1 = self._end_point(points)
                end_off = self._offset_point(pend1, pend0, self.half_road_width, direction="left")
                self.coarse_points_metsr.append(self._sumo_to_metsr(end_off))

        self.coarse_points_carla = [self._metsr_to_carla(p) for p in self.coarse_points_metsr]
        return list(self.coarse_points_metsr)

    def build_lane_points(self, route_ids):
        if self.grp is None:
            return []
        self.build_coarse_points(route_ids)
        self.lane_waypoints = []
        if len(self.coarse_points_carla) < 2:
            return []
        for cur, nxt in zip(self.coarse_points_carla[:-1], self.coarse_points_carla[1:]):
            segment = self.grp.trace_route(cur, nxt)
            self.lane_waypoints.extend(segment)
        return [wp.transform.location for wp, _ in self.lane_waypoints]

    def draw_coarse_points(self, color=None, size=0.35, life_time=0.0):
        if self.world is None:
            return
        if color is None:
            color = carla.Color(0, 255, 255)
        for loc in self.coarse_points_carla:
            self.world.debug.draw_point(
                loc,
                size=size,
                color=color,
                life_time=life_time,
                persistent_lines=True,
            )

    def draw_lane_points(self, color=None, size=0.12, life_time=0.0):
        if self.world is None:
            return
        if color is None:
            color = carla.Color(255, 255, 0)
        for wp, _ in self.lane_waypoints:
            self.world.debug.draw_point(
                wp.transform.location,
                size=size,
                color=color,
                life_time=life_time,
                persistent_lines=True,
            )

    def _load_sumo_net(self, net_path):
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

    def _edge_shape_points(self, edge):
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

    def _point_at_fraction(self, points, fraction):
        if not points:
            return None, None
        if len(points) < 2:
            return points[0], None
        total = 0.0
        seg_lengths = []
        for a, b in zip(points[:-1], points[1:]):
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            length = (dx * dx + dy * dy) ** 0.5
            seg_lengths.append(length)
            total += length
        if total == 0.0:
            return points[0], points[1]
        target = total * fraction
        walked = 0.0
        for idx, length in enumerate(seg_lengths):
            if walked + length >= target:
                a = points[idx]
                b = points[idx + 1]
                t = (target - walked) / length if length > 0.0 else 0.0
                px = a[0] + (b[0] - a[0]) * t
                py = a[1] + (b[1] - a[1]) * t
                return (px, py), b
            walked += length
        return points[-2], points[-1]

    def _end_point(self, points):
        if not points:
            return None, None
        if len(points) < 2:
            return points[-1], None
        return points[-2], points[-1]

    def _offset_point(self, p0, p1, offset, direction="right"):
        if p0 is None or p1 is None or offset == 0.0:
            return p0
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        norm = (dx * dx + dy * dy) ** 0.5
        if norm == 0.0:
            return p0
        if direction == "right":
            ox = dy / norm * offset
            oy = -dx / norm * offset
        else:
            ox = -dy / norm * offset
            oy = dx / norm * offset
        return (p0[0] + ox, p0[1] + oy)

    def _sumo_to_metsr(self, point):
        return point[0] - self.net_offset[0], point[1] - self.net_offset[1]

    def _metsr_to_carla(self, point):
        return carla.Location(x=point[0], y=-point[1], z=0.5)

    def _carla_to_metsr(self, location):
        return (location.x, -location.y)
