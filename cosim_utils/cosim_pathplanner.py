from collections import defaultdict
from heapq import heappop, heappush
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import carla

from cosim_utils.agents.navigation.global_route_planner import GlobalRoutePlanner


class CosimPathPlanner:
    """
    Build a lane-level CARLA path from the handoff pose and a METS-R road-id sequence.

    The planner works directly on the SUMO net lane graph:
    1. choose the most plausible current driving lane from the handoff pose,
    2. walk the lane/connection graph so the path visits the requested road ids in order,
    3. sample those lane centerlines into a dense CARLA route for the controller.
    """

    def __init__(self, world, net_path, half_road_width=3.5, sampling_resolution=2.0, gap_trace_threshold=5.0):
        self.world = world
        self.map = world.get_map() if world is not None else None
        self.half_road_width = half_road_width
        self.gap_trace_threshold = float(gap_trace_threshold)
        self.net_path = Path(net_path)
        (
            self.edges,
            self.lanes,
            self.lane_successors,
            self.lane_predecessors,
            self.edge_successors,
            self.net_offset,
        ) = self._load_sumo_net(self.net_path)
        self.grp = None
        if self.map is not None:
            self.grp = GlobalRoutePlanner(self.map, sampling_resolution=sampling_resolution)
        self.coarse_points_metsr = []
        self.coarse_points_carla = []
        self.lane_waypoints = []
        self.missing_edges = []
        self.selected_lane_path = []
        self.project_to_carla_map = True

    def build_coarse_points(self, route_ids):
        """
        Build simple debug anchors from the first driving lane of each requested road.
        """
        self.coarse_points_metsr = []
        self.coarse_points_carla = []
        self.missing_edges = []
        for road_id in route_ids:
            edge = self.edges.get(str(road_id))
            if edge is None or not edge["driving_lane_ids"]:
                self.missing_edges.append(road_id)
                continue
            lane = self.lanes[edge["driving_lane_ids"][0]]
            for fraction in (0.2, 0.5):
                point, _ = self._point_at_fraction(lane["shape_metsr"], fraction)
                if point is not None:
                    self.coarse_points_metsr.append(point)
        self.coarse_points_carla = [self._metsr_to_carla(point) for point in self.coarse_points_metsr]
        return list(self.coarse_points_metsr)

    def build_lane_points(self, route_ids, start_point_carla=None, start_yaw_carla=None, preferred_lane_ids=None):
        """
        Generate a lane-level route by searching the lane graph instead of offsetting edge shapes.
        """
        self.lane_waypoints = []
        self.selected_lane_path = []
        self.coarse_points_metsr = []
        self.coarse_points_carla = []
        self.missing_edges = []

        route_ids = [str(route_id) for route_id in route_ids if str(route_id) in self.edges]
        if not route_ids or start_point_carla is None:
            return []

        start_lane_id = self._preferred_start_lane(preferred_lane_ids, route_ids)
        if start_lane_id is None:
            start_lane_id = self._select_start_lane(start_point_carla, start_yaw_carla, route_ids)
        if start_lane_id is None:
            return []

        preferred_lane_path = self._preferred_lane_path(
            preferred_lane_ids,
            route_ids,
            start_lane_id=start_lane_id,
        )
        lane_path = preferred_lane_path
        if not lane_path:
            lane_path = self._build_lane_path(start_lane_id, route_ids)
        if not lane_path:
            return []

        self.selected_lane_path = lane_path
        if preferred_lane_path:
            lane_points_carla = self._sample_preferred_lane_path_locations(
                lane_path,
                self._carla_to_metsr(start_point_carla),
            )
        else:
            lane_points_carla = self._sample_lane_path_locations(
                lane_path,
                self._carla_to_metsr(start_point_carla),
            )
        if not lane_points_carla:
            return []

        self.coarse_points_metsr = self._build_route_markers_from_lane_path(lane_path)
        self.coarse_points_carla = [self._metsr_to_carla(point) for point in self.coarse_points_metsr]

        self.lane_waypoints = []
        route_points = []
        for loc in lane_points_carla:
            if self.map is not None and self.project_to_carla_map:
                wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp is not None:
                    route_points.append(wp.transform.location)
                    self.lane_waypoints.append((wp, None))
                    continue
            self.lane_waypoints.append((loc, None))
            route_points.append(loc)
        return self._dedupe_locations(route_points)

    def route_handoff_pose(self, route_ids, reference_carla_location=None, preferred_lane_ids=None):
        """
        Choose a route-compatible starting lane and project the handoff pose onto that lane.
        Inputs: METS-R edge route and optional current CARLA handoff location.
        Outputs: Returns (CARLA location, CARLA yaw, lane path) or None.
        """
        route_ids = [str(route_id) for route_id in route_ids if str(route_id) in self.edges]
        if not route_ids:
            return None
        reference_metsr = self._carla_to_metsr(reference_carla_location) if reference_carla_location is not None else None
        start_lane_id = self._preferred_start_lane(preferred_lane_ids, route_ids)
        lane_path = self._preferred_lane_path(preferred_lane_ids, route_ids)
        if not lane_path and start_lane_id is not None:
            lane_path = self._build_lane_path(start_lane_id, route_ids)
        if not lane_path:
            lane_path = self._select_route_lane_path(route_ids, reference_metsr=reference_metsr)
        if not lane_path:
            return None

        lane = self.lanes[lane_path[0]]
        shape = lane["shape_metsr"]
        if len(shape) < 2:
            return None
        if reference_metsr is not None:
            _, point_metsr, seg_idx, seg_t = self._distance_to_polyline(reference_metsr, shape)
        else:
            point_metsr = shape[0]
            seg_idx = 0
            seg_t = 0.0
        yaw = self._segment_yaw_carla(shape, seg_idx, seg_t)
        return self._metsr_to_carla(point_metsr), yaw, lane_path

    def _preferred_start_lane(self, preferred_lane_ids, route_ids):
        if not preferred_lane_ids:
            return None
        route_set = set(str(route_id) for route_id in route_ids)
        for lane_id in preferred_lane_ids:
            lane_id = str(lane_id)
            lane = self.lanes.get(lane_id)
            if lane is None or not lane["is_driving"]:
                continue
            if lane["edge_id"] in route_set:
                return lane_id
        return None

    def _preferred_lane_path(self, preferred_lane_ids, route_ids, start_lane_id=None):
        if not preferred_lane_ids:
            return []

        expanded_route_edges = self._expanded_route_edges(route_ids)
        if not expanded_route_edges:
            return []
        route_index = {edge_id: index for index, edge_id in enumerate(expanded_route_edges)}

        lane_path = []
        lane_edge_indices = []
        for raw_lane_id in preferred_lane_ids:
            lane_id = str(raw_lane_id)
            lane = self.lanes.get(lane_id)
            if lane is None or lane["is_internal"] or not lane["is_driving"]:
                continue
            edge_id = lane["edge_id"]
            if edge_id not in route_index:
                continue
            if lane_path and lane_path[-1] == lane_id:
                continue
            lane_path.append(lane_id)
            lane_edge_indices.append(route_index[edge_id])

        if len(lane_path) < 2:
            return []
        if start_lane_id is not None and lane_path[0] != start_lane_id:
            return []
        if lane_edge_indices[0] != 0 or lane_edge_indices[-1] != len(expanded_route_edges) - 1:
            return []
        if any(next_idx < cur_idx for cur_idx, next_idx in zip(lane_edge_indices, lane_edge_indices[1:])):
            return []
        return lane_path

    def _expanded_route_edges(self, route_ids):
        route_ids = [str(route_id) for route_id in route_ids if str(route_id) in self.edges]
        if len(route_ids) <= 1:
            return route_ids

        expanded = []
        for current_edge, next_edge in zip(route_ids[:-1], route_ids[1:]):
            segment = self._shortest_edge_path(current_edge, next_edge)
            if not segment:
                return []
            if expanded and expanded[-1] == segment[0]:
                expanded.extend(segment[1:])
            else:
                expanded.extend(segment)
        return expanded

    def build_carla_routepoints_from_metsr(self, route_ids, centerline_response, sampling_locs=(0.2, 0.5, 0.8)):
        """
        Build a CARLA lane path directly from METS-R lane centerlines instead of XML edge shapes.
        """
        if self.grp is None:
            return []
        self.coarse_points_metsr = []
        self.coarse_points_carla = []
        self.lane_waypoints = []
        self.missing_edges = []
        if not route_ids:
            return []

        centerline_map = {}
        for item in centerline_response.get("DATA", []):
            centerline_map[str(item.get("ID"))] = item.get("centerline", [])

        for road_id in route_ids:
            points = centerline_map.get(str(road_id), [])
            if len(points) < 2:
                self.missing_edges.append(road_id)
                continue
            for frac in sampling_locs:
                sampled, _ = self._point_at_fraction(points, frac)
                if sampled is None:
                    self.missing_edges.append(road_id)
                    continue
                self.coarse_points_metsr.append((sampled[0], sampled[1]))

        self.coarse_points_carla = [self._metsr_to_carla(point) for point in self.coarse_points_metsr]
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

    def draw_lane_points(self, color=None, size=0.22, life_time=0.0):
        if self.world is None:
            return
        if color is None:
            color = carla.Color(255, 255, 0)
        for wp, _ in self.lane_waypoints:
            loc = wp.transform.location if hasattr(wp, "transform") else wp
            loc = carla.Location(x=loc.x, y=loc.y, z=loc.z + 1.0)
            self.world.debug.draw_point(
                loc,
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
        lanes = {}
        lane_successors = defaultdict(list)
        lane_predecessors = defaultdict(list)
        edge_successors = defaultdict(list)

        for edge in root.findall("edge"):
            edge_id = edge.get("id")
            if edge_id is None:
                continue
            edge_function = edge.get("function")
            type_tokens = edge.get("type", "").split("|") if edge.get("type") else []
            edge_data = {
                "id": edge_id,
                "function": edge_function,
                "lane_ids": [],
                "driving_lane_ids": [],
            }
            for lane in edge.findall("lane"):
                lane_id = lane.get("id")
                if lane_id is None:
                    continue
                lane_index = int(lane.get("index", "0"))
                if edge_function == "internal" or edge_id.startswith(":"):
                    lane_role = "internal"
                    is_driving = True
                else:
                    lane_role = type_tokens[lane_index] if lane_index < len(type_tokens) else None
                    is_driving = lane_role == "driving"
                shape_sumo = self._shape_points_from_attr(lane.get("shape") or edge.get("shape"))
                shape_metsr = [self._sumo_to_metsr(point, net_offset) for point in shape_sumo]
                lane_data = {
                    "id": lane_id,
                    "edge_id": edge_id,
                    "index": lane_index,
                    "role": lane_role,
                    "is_driving": is_driving,
                    "is_internal": edge_function == "internal" or edge_id.startswith(":"),
                    "shape_metsr": shape_metsr,
                    "length": self._polyline_length(shape_metsr),
                }
                lanes[lane_id] = lane_data
                edge_data["lane_ids"].append(lane_id)
                if is_driving:
                    edge_data["driving_lane_ids"].append(lane_id)
            edges[edge_id] = edge_data

        for conn in root.findall("connection"):
            from_edge = conn.get("from")
            to_edge = conn.get("to")
            from_lane = conn.get("fromLane")
            to_lane = conn.get("toLane")
            if from_edge is None or to_edge is None or from_lane is None or to_lane is None:
                continue
            from_lane_id = self._compose_lane_id(from_edge, from_lane)
            to_lane_id = self._compose_lane_id(to_edge, to_lane)
            via_lane_id = conn.get("via")
            if via_lane_id:
                self._add_lane_arc(lane_successors, lane_predecessors, lanes, from_lane_id, via_lane_id)
                self._add_lane_arc(lane_successors, lane_predecessors, lanes, via_lane_id, to_lane_id)
            else:
                self._add_lane_arc(lane_successors, lane_predecessors, lanes, from_lane_id, to_lane_id)
            if from_edge != to_edge and to_edge not in edge_successors[from_edge]:
                edge_successors[from_edge].append(to_edge)

        return edges, lanes, lane_successors, lane_predecessors, edge_successors, net_offset

    def _add_lane_arc(self, successors, predecessors, lanes, from_lane_id, to_lane_id):
        if from_lane_id not in lanes or to_lane_id not in lanes:
            return
        if not lanes[from_lane_id]["is_driving"] or not lanes[to_lane_id]["is_driving"]:
            return
        if to_lane_id not in successors[from_lane_id]:
            successors[from_lane_id].append(to_lane_id)
        if from_lane_id not in predecessors[to_lane_id]:
            predecessors[to_lane_id].append(from_lane_id)

    def _select_start_lane(self, start_point_carla, start_yaw_carla, route_ids):
        start_metsr = self._carla_to_metsr(start_point_carla)
        candidates = []
        for lane_id, lane in self.lanes.items():
            if lane["is_internal"] or not lane["is_driving"]:
                continue
            if len(lane["shape_metsr"]) < 2:
                continue
            distance, _, seg_idx, seg_t = self._distance_to_polyline(start_metsr, lane["shape_metsr"])
            if distance > 20.0:
                continue
            heading_error = 0.0
            if start_yaw_carla is not None:
                heading = self._segment_yaw_carla(lane["shape_metsr"], seg_idx, seg_t)
                heading_error = self._angular_diff_deg(start_yaw_carla, heading)
                if heading_error > 85.0:
                    continue
            edge_path = self._shortest_edge_path(lane["edge_id"], route_ids[0])
            if edge_path is None:
                continue
            path_to_first = self._shortest_path_to_edge(
                lane_id,
                route_ids[0],
                allowed_edges=set(edge_path),
                max_visits=80,
            )
            if path_to_first is None:
                continue
            full_path = self._build_lane_path(lane_id, route_ids)
            if not full_path:
                continue
            hop_penalty = 2.0 * self._count_noninternal_transitions(path_to_first)
            score = distance + 0.08 * heading_error + hop_penalty
            candidates.append((score, lane_id))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _select_route_lane_path(self, route_ids, reference_metsr=None):
        first_edge = self.edges.get(str(route_ids[0]))
        if first_edge is None:
            return []
        candidates = []
        for lane_id in first_edge["driving_lane_ids"]:
            lane_path = self._build_lane_path(lane_id, route_ids)
            if not lane_path:
                continue
            distance = 0.0
            if reference_metsr is not None:
                distance, _, _, _ = self._distance_to_polyline(reference_metsr, self.lanes[lane_id]["shape_metsr"])
            path_length = sum(
                self.lanes[path_lane_id]["length"]
                for path_lane_id in lane_path
                if path_lane_id in self.lanes
            )
            candidates.append((distance, path_length, self.lanes[lane_id]["index"], lane_path))
        if not candidates:
            return []
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return list(candidates[0][3])

    def _build_lane_path(self, start_lane_id, route_ids):
        lane_path = [start_lane_id]
        current_lane_id = start_lane_id

        first_route_idx = 0
        current_edge = self.lanes[current_lane_id]["edge_id"]
        if current_edge in route_ids:
            first_route_idx = route_ids.index(current_edge)
        else:
            edge_path = self._shortest_edge_path(current_edge, route_ids[0])
            if edge_path is None:
                return []
            prefix = self._shortest_path_to_edge(
                current_lane_id,
                route_ids[0],
                allowed_edges=set(edge_path),
                max_visits=80,
            )
            if not prefix:
                return []
            lane_path = self._append_lane_path(lane_path, prefix)
            current_lane_id = lane_path[-1]
            first_route_idx = 0

        for route_idx in range(first_route_idx, len(route_ids) - 1):
            current_edge = route_ids[route_idx]
            next_edge = route_ids[route_idx + 1]
            edge_path = self._shortest_edge_path(current_edge, next_edge)
            if not edge_path:
                return []
            segment = self._shortest_path_to_edge(
                current_lane_id,
                next_edge,
                allowed_edges=set(edge_path),
                max_visits=40,
            )
            if not segment:
                return []
            lane_path = self._append_lane_path(lane_path, segment)
            current_lane_id = lane_path[-1]

        return lane_path

    def _shortest_edge_path(self, start_edge_id, target_edge_id):
        if start_edge_id == target_edge_id:
            return [start_edge_id]
        heap = [(0.0, start_edge_id, [start_edge_id])]
        best_cost = {start_edge_id: 0.0}
        while heap:
            cost, edge_id, path = heappop(heap)
            if cost > best_cost.get(edge_id, float("inf")):
                continue
            for next_edge_id in self.edge_successors.get(edge_id, []):
                next_edge = self.edges.get(next_edge_id)
                if next_edge is None:
                    continue
                next_cost = cost + 1.0
                if next_cost >= best_cost.get(next_edge_id, float("inf")):
                    continue
                next_path = path + [next_edge_id]
                if next_edge_id == target_edge_id:
                    return next_path
                best_cost[next_edge_id] = next_cost
                heappush(heap, (next_cost, next_edge_id, next_path))
        return None

    def _shortest_path_to_edge(self, start_lane_id, target_edge_id, allowed_edges=None, max_visits=80):
        if start_lane_id not in self.lanes:
            return None
        start_edge = self.lanes[start_lane_id]["edge_id"]
        if start_edge == target_edge_id:
            return [start_lane_id]

        heap = [(0.0, start_lane_id, [start_lane_id])]
        best_cost = {start_lane_id: 0.0}
        visits = 0
        while heap and visits < max_visits:
            cost, lane_id, path = heappop(heap)
            visits += 1
            if cost > best_cost.get(lane_id, float("inf")):
                continue
            for next_lane_id in self.lane_successors.get(lane_id, []):
                next_lane = self.lanes[next_lane_id]
                next_edge = next_lane["edge_id"]
                if not next_lane["is_internal"] and allowed_edges is not None and next_edge not in allowed_edges:
                    continue
                lane_change_penalty = 0.0
                current_lane = self.lanes[lane_id]
                if not current_lane["is_internal"] and not next_lane["is_internal"]:
                    lane_change_penalty = abs(next_lane["index"] - current_lane["index"])
                next_cost = cost + next_lane["length"] + 3.0 * lane_change_penalty
                if next_cost >= best_cost.get(next_lane_id, float("inf")):
                    continue
                next_path = path + [next_lane_id]
                if next_edge == target_edge_id:
                    return next_path
                best_cost[next_lane_id] = next_cost
                heappush(heap, (next_cost, next_lane_id, next_path))
        return None

    def _sample_lane_path_locations(self, lane_path, start_point_metsr, step=2.0, bridge_gaps_with_grp=True):
        locations = []
        previous_end = None
        for idx, lane_id in enumerate(lane_path):
            shape = list(self.lanes[lane_id]["shape_metsr"])
            if len(shape) < 2:
                continue
            if idx == 0:
                shape = self._clip_polyline_from_point(shape, start_point_metsr)
            if len(shape) < 2:
                continue

            if previous_end is not None:
                gap = self._distance_2d(previous_end, shape[0])
                if gap > self.gap_trace_threshold and self.grp is not None and bridge_gaps_with_grp:
                    self._append_grp_gap_locations(locations, previous_end, shape[0])
                elif gap > 1e-3:
                    self._append_metsr_locations(
                        locations,
                        self._resample_polyline([previous_end, shape[0]], step=step),
                    )

            self._append_metsr_locations(locations, self._resample_polyline(shape, step=step))
            previous_end = shape[-1]
        return locations

    def _sample_preferred_lane_path_locations(self, lane_path, start_point_metsr, step=2.0):
        locations = []
        groups = self._lane_path_edge_groups(lane_path)
        previous_last_lane_id = None
        for group_idx, group_lane_ids in enumerate(groups):
            if previous_last_lane_id is not None:
                bridge_points = self._lane_connection_bridge_points(previous_last_lane_id, group_lane_ids[0], step=step)
                if bridge_points:
                    self._append_metsr_locations(locations, bridge_points)
                else:
                    previous_shape = self.lanes[previous_last_lane_id]["shape_metsr"]
                    current_shape = self.lanes[group_lane_ids[0]]["shape_metsr"]
                    if previous_shape and current_shape:
                        self._append_metsr_locations(
                            locations,
                            self._resample_polyline([previous_shape[-1], current_shape[0]], step=step),
                        )

            group_start = start_point_metsr if group_idx == 0 else None
            group_points = self._sample_same_edge_lane_group(group_lane_ids, start_point_metsr=group_start, step=step)
            if group_points:
                self._append_metsr_locations(locations, group_points)
            previous_last_lane_id = group_lane_ids[-1]
        return locations

    def _lane_path_edge_groups(self, lane_path):
        groups = []
        current_group = []
        current_edge = None
        for lane_id in lane_path:
            lane = self.lanes.get(lane_id)
            if lane is None:
                continue
            edge_id = lane["edge_id"]
            if current_group and edge_id != current_edge:
                groups.append(current_group)
                current_group = []
            current_group.append(lane_id)
            current_edge = edge_id
        if current_group:
            groups.append(current_group)
        return groups

    def _sample_same_edge_lane_group(self, lane_ids, start_point_metsr=None, step=2.0):
        if not lane_ids:
            return []
        if len(lane_ids) == 1:
            shape = list(self.lanes[lane_ids[0]]["shape_metsr"])
            if start_point_metsr is not None:
                shape = self._clip_polyline_from_point(shape, start_point_metsr)
            return self._resample_polyline(shape, step=step)

        lane_shapes = [self.lanes[lane_id]["shape_metsr"] for lane_id in lane_ids]
        if any(len(shape) < 2 for shape in lane_shapes):
            return []

        base_shape = lane_shapes[0]
        start_fraction = 0.0
        if start_point_metsr is not None:
            _, _, seg_idx, seg_t = self._distance_to_polyline(start_point_metsr, base_shape)
            start_fraction = self._polyline_fraction_at_segment(base_shape, seg_idx, seg_t)

        base_length = max(self._polyline_length(base_shape), 1e-6)
        fraction_step = max(0.01, float(step) / base_length)
        fractions = []
        value = start_fraction
        while value < 1.0:
            fractions.append(value)
            value += fraction_step
        if not fractions or fractions[-1] < 1.0:
            fractions.append(1.0)

        transition_start = min(0.35, start_fraction + 0.05)
        transition_end = max(transition_start + 0.15, 0.88)
        if start_fraction > transition_start:
            transition_start = start_fraction
            transition_end = min(1.0, max(start_fraction + 0.2, transition_end))

        points = []
        max_lane_pos = len(lane_shapes) - 1
        for fraction in fractions:
            if transition_end <= transition_start:
                progress = 1.0
            else:
                progress = (fraction - transition_start) / (transition_end - transition_start)
            progress = max(0.0, min(1.0, progress))
            progress = progress * progress * (3.0 - 2.0 * progress)
            lane_pos = progress * max_lane_pos
            lower_idx = int(math.floor(lane_pos))
            upper_idx = min(max_lane_pos, lower_idx + 1)
            lane_blend = lane_pos - lower_idx
            lower_point, _ = self._point_at_fraction(lane_shapes[lower_idx], fraction)
            upper_point, _ = self._point_at_fraction(lane_shapes[upper_idx], fraction)
            if lower_point is None or upper_point is None:
                continue
            points.append(self._interpolate_point(lower_point, upper_point, lane_blend))
        return self._dedupe_metsr_points(points)

    def _lane_connection_bridge_points(self, from_lane_id, to_lane_id, step=2.0):
        path = self._shortest_path_to_lane(from_lane_id, to_lane_id, max_visits=12)
        if not path:
            return []
        points = []
        for lane_id in path[1:-1]:
            lane = self.lanes.get(lane_id)
            if lane is None or not lane["is_internal"]:
                continue
            points.extend(self._resample_polyline(lane["shape_metsr"], step=step))
        return self._dedupe_metsr_points(points)

    def _shortest_path_to_lane(self, start_lane_id, target_lane_id, max_visits=20):
        if start_lane_id == target_lane_id:
            return [start_lane_id]
        heap = [(0.0, start_lane_id, [start_lane_id])]
        best_cost = {start_lane_id: 0.0}
        visits = 0
        target_edge = self.lanes.get(target_lane_id, {}).get("edge_id")
        while heap and visits < max_visits:
            cost, lane_id, path = heappop(heap)
            visits += 1
            if cost > best_cost.get(lane_id, float("inf")):
                continue
            for next_lane_id in self.lane_successors.get(lane_id, []):
                next_lane = self.lanes.get(next_lane_id)
                if next_lane is None:
                    continue
                if next_lane_id != target_lane_id and not next_lane["is_internal"] and next_lane["edge_id"] == target_edge:
                    continue
                next_cost = cost + max(0.1, next_lane["length"])
                if next_cost >= best_cost.get(next_lane_id, float("inf")):
                    continue
                next_path = path + [next_lane_id]
                if next_lane_id == target_lane_id:
                    return next_path
                best_cost[next_lane_id] = next_cost
                heappush(heap, (next_cost, next_lane_id, next_path))
        return []

    def _append_metsr_locations(self, locations, points):
        for point in points:
            self._append_location(locations, self._metsr_to_carla(point))

    def _append_grp_gap_locations(self, locations, start_point_metsr, end_point_metsr):
        start_loc = self._metsr_to_carla(start_point_metsr)
        end_loc = self._metsr_to_carla(end_point_metsr)
        segment = self.grp.trace_route(start_loc, end_loc)
        if not segment:
            self._append_location(locations, end_loc)
            return
        for waypoint, _ in segment:
            self._append_location(locations, waypoint.transform.location)

    def _append_location(self, locations, loc, eps=0.3):
        if not locations or loc.distance(locations[-1]) > eps:
            locations.append(loc)

    def _build_route_markers_from_lane_path(self, lane_path):
        markers = []
        seen_edges = set()
        for lane_id in lane_path:
            lane = self.lanes[lane_id]
            edge_id = lane["edge_id"]
            if lane["is_internal"] or edge_id in seen_edges or len(lane["shape_metsr"]) < 2:
                continue
            seen_edges.add(edge_id)
            for fraction in (0.2, 0.5):
                point, _ = self._point_at_fraction(lane["shape_metsr"], fraction)
                if point is not None:
                    markers.append(point)
        return markers

    def _append_lane_path(self, base_path, extension_path):
        if not base_path:
            return list(extension_path)
        if not extension_path:
            return list(base_path)
        if base_path[-1] == extension_path[0]:
            return base_path + extension_path[1:]
        return base_path + extension_path

    def _shape_points_from_attr(self, shape_attr):
        if not shape_attr:
            return []
        points = []
        for pair in shape_attr.strip().split(" "):
            parts = pair.split(",")
            if len(parts) < 2:
                continue
            if len(parts) >= 3:
                points.append((float(parts[0]), float(parts[1]), float(parts[2])))
            else:
                points.append((float(parts[0]), float(parts[1])))
        return points

    def _point_at_fraction(self, points, fraction):
        if not points:
            return None, None
        if len(points) < 2:
            return points[0], None
        total = self._polyline_length(points)
        if total == 0.0:
            return points[0], points[1]
        target = total * fraction
        walked = 0.0
        for idx, (a, b) in enumerate(zip(points[:-1], points[1:])):
            length = self._distance_2d(a, b)
            if walked + length >= target:
                t = (target - walked) / length if length > 0.0 else 0.0
                px = a[0] + (b[0] - a[0]) * t
                py = a[1] + (b[1] - a[1]) * t
                if len(a) >= 3 and len(b) >= 3:
                    pz = a[2] + (b[2] - a[2]) * t
                    return (px, py, pz), b
                return (px, py), b
            walked += length
        return points[-2], points[-1]

    def _distance_to_polyline(self, point, polyline):
        best_dist = float("inf")
        best_proj = None
        best_idx = 0
        best_t = 0.0
        for idx, (a, b) in enumerate(zip(polyline[:-1], polyline[1:])):
            proj, t = self._project_point_to_segment(point, a, b)
            dist = self._distance_2d(point, proj)
            if dist < best_dist:
                best_dist = dist
                best_proj = proj
                best_idx = idx
                best_t = t
        return best_dist, best_proj, best_idx, best_t

    def _clip_polyline_from_point(self, polyline, point):
        if len(polyline) < 2:
            return polyline
        _, proj, seg_idx, _ = self._distance_to_polyline(point, polyline)
        clipped = [proj]
        clipped.extend(polyline[seg_idx + 1 :])
        return clipped

    def _polyline_fraction_at_segment(self, polyline, seg_idx, seg_t):
        total = self._polyline_length(polyline)
        if total <= 0.0:
            return 0.0
        walked = 0.0
        for idx, (a, b) in enumerate(zip(polyline[:-1], polyline[1:])):
            segment_length = self._distance_2d(a, b)
            if idx == seg_idx:
                walked += max(0.0, min(1.0, seg_t)) * segment_length
                break
            walked += segment_length
        return max(0.0, min(1.0, walked / total))

    def _resample_polyline(self, polyline, step=2.0):
        if len(polyline) < 2:
            return polyline

        cumulative = [0.0]
        for current, nxt in zip(polyline[:-1], polyline[1:]):
            cumulative.append(cumulative[-1] + self._distance_2d(current, nxt))
        total = cumulative[-1]
        if total <= 0.0:
            return [polyline[0]]

        result = [polyline[0]]
        target = float(step)
        segment_idx = 0
        while target < total:
            while segment_idx < len(cumulative) - 2 and cumulative[segment_idx + 1] < target:
                segment_idx += 1
            start = polyline[segment_idx]
            end = polyline[segment_idx + 1]
            seg_start = cumulative[segment_idx]
            seg_len = cumulative[segment_idx + 1] - seg_start
            t = 0.0 if seg_len <= 0.0 else (target - seg_start) / seg_len
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            if len(start) >= 3 and len(end) >= 3:
                z = start[2] + (end[2] - start[2]) * t
                result.append((x, y, z))
            else:
                result.append((x, y))
            target += float(step)

        if not self._points_close(result[-1], polyline[-1]):
            result.append(polyline[-1])
        deduped = [result[0]]
        for point in result[1:]:
            if not self._points_close(deduped[-1], point):
                deduped.append(point)
        return deduped

    def _count_noninternal_transitions(self, lane_path):
        count = 0
        prev_edge = self.lanes[lane_path[0]]["edge_id"]
        for lane_id in lane_path[1:]:
            edge_id = self.lanes[lane_id]["edge_id"]
            if not edge_id.startswith(":") and edge_id != prev_edge:
                count += 1
                prev_edge = edge_id
        return count

    def _segment_yaw_carla(self, polyline, seg_idx, seg_t):
        a = polyline[seg_idx]
        b = polyline[min(seg_idx + 1, len(polyline) - 1)]
        a_loc = self._metsr_to_carla(a)
        b_loc = self._metsr_to_carla(b)
        dx = b_loc.x - a_loc.x
        dy = b_loc.y - a_loc.y
        if dx == 0.0 and dy == 0.0:
            return 0.0
        return math.degrees(math.atan2(dy, dx))

    def _angular_diff_deg(self, a, b):
        diff = (a - b + 180.0) % 360.0 - 180.0
        return abs(diff)

    def _dedupe_locations(self, locations, eps=0.3):
        if not locations:
            return []
        deduped = [locations[0]]
        for loc in locations[1:]:
            if loc.distance(deduped[-1]) > eps:
                deduped.append(loc)
        return deduped

    def _dedupe_metsr_points(self, points, eps=0.3):
        if not points:
            return []
        deduped = [points[0]]
        for point in points[1:]:
            if self._distance_2d(deduped[-1], point) > eps:
                deduped.append(point)
        return deduped

    def _interpolate_point(self, first, second, fraction):
        fraction = max(0.0, min(1.0, float(fraction)))
        x = first[0] + (second[0] - first[0]) * fraction
        y = first[1] + (second[1] - first[1]) * fraction
        if len(first) >= 3 and len(second) >= 3:
            z = first[2] + (second[2] - first[2]) * fraction
            return (x, y, z)
        return (x, y)

    def _polyline_length(self, points):
        total = 0.0
        for a, b in zip(points[:-1], points[1:]):
            total += self._distance_2d(a, b)
        return total

    def _distance_2d(self, a, b):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        return math.sqrt(dx * dx + dy * dy)

    def _project_point_to_segment(self, point, a, b):
        ax, ay = a[0], a[1]
        bx, by = b[0], b[1]
        px, py = point[0], point[1]
        abx = bx - ax
        aby = by - ay
        denom = abx * abx + aby * aby
        if denom == 0.0:
            return a, 0.0
        t = ((px - ax) * abx + (py - ay) * aby) / denom
        t = max(0.0, min(1.0, t))
        x = ax + abx * t
        y = ay + aby * t
        if len(a) >= 3 and len(b) >= 3:
            return (x, y, a[2] + (b[2] - a[2]) * t), t
        return (x, y), t

    def _points_close(self, a, b, eps=0.05):
        return self._distance_2d(a, b) <= eps

    def _compose_lane_id(self, edge_id, lane_index):
        return f"{edge_id}_{lane_index}"

    def _sumo_to_metsr(self, point, net_offset=None):
        if net_offset is None:
            net_offset = self.net_offset
        if len(point) >= 3:
            return point[0] - net_offset[0], point[1] - net_offset[1], point[2]
        return point[0] - net_offset[0], point[1] - net_offset[1]

    def _metsr_to_carla(self, point):
        z = point[2] + 0.5 if len(point) >= 3 else 0.5
        return carla.Location(x=point[0], y=-point[1], z=z)

    def _carla_to_metsr(self, location):
        return (location.x, -location.y)
