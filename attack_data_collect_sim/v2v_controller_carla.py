import math

import carla
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption
from agents.tools.misc import get_speed

from attack_data_collect_sim.cosim_pathplanner import CosimPathPlanner


class V2VControllerCarla:
    """
    CARLA-driven V2V controller. It plans in CARLA, applies VehicleControl,
    and exposes CARLA state for syncing back to METS-R.
    """

    def __init__(
        self,
        vehicle,
        ego_vid,
        map_inst=None,
        net_path=None,
        path_planner=None,
        target_speed_mps=10.0,
        time_headway=1.2,
        min_gap=2.5,
        lane_half_width=2.0,
        conflict_horizon_s=4.0,
        conflict_time_gap=1.0,
        conflict_time_safe=3.0,
        conflict_ignore_dist=2.0,
        conflict_max_dist=40.0,
        junction_yield_radius=12.0,
        lane_change_lookahead_s=4.0,
        enable_overtake_lane_change=False,
        enable_debug_draw=False,
        v2v_position_mode="geodetic",
        v2v_lat_key="latitude",
        v2v_lon_key="longitude",
        v2v_x_key="x",
        v2v_y_key="y",
    ):
        self.vehicle = vehicle
        self.ego_vid = ego_vid
        self.world = vehicle.get_world()
        self.map = map_inst or self.world.get_map()
        self.target_speed_mps = target_speed_mps
        self.time_headway = time_headway
        self.min_gap = min_gap
        self.lane_half_width = lane_half_width
        self.conflict_horizon_s = conflict_horizon_s
        self.conflict_time_gap = conflict_time_gap
        self.conflict_time_safe = conflict_time_safe
        self.conflict_ignore_dist = conflict_ignore_dist
        self.conflict_max_dist = conflict_max_dist
        self.junction_yield_radius = junction_yield_radius
        self.lane_change_lookahead_s = lane_change_lookahead_s
        self.enable_overtake_lane_change = enable_overtake_lane_change
        self.enable_debug_draw = enable_debug_draw
        self.v2v_position_mode = v2v_position_mode
        self.v2v_lat_key = v2v_lat_key
        self.v2v_lon_key = v2v_lon_key
        self.v2v_x_key = v2v_x_key
        self.v2v_y_key = v2v_y_key
        self._lane_change_cooldown = 0.0
        self.path_planner = path_planner
        if self.path_planner is None and net_path is not None:
            self.path_planner = CosimPathPlanner(self.world, net_path)

        opt_dict = {
            "ignore_traffic_lights": True,
            "ignore_vehicles": True,
        }
        self.agent = BasicAgent(self.vehicle, target_speed=self._to_kmh(target_speed_mps), opt_dict=opt_dict, map_inst=self.map)
        self._route_points = []

    def set_destination_xy(self, end_xy, start_xy=None, clean_queue=True):
        start_loc = None
        if start_xy is not None:
            start_loc = self._metsr_to_carla_location(start_xy[0], start_xy[1])
        end_loc = self._metsr_to_carla_location(end_xy[0], end_xy[1])
        self.agent.set_destination(end_loc, start_location=start_loc, clean_queue=clean_queue)

    def set_route_from_carla_coords(self, coord_map, clean_queue=True, stop_waypoint_creation=True):
        if not coord_map:
            return
        plan = []
        self._route_points = []
        for loc in coord_map:
            if not isinstance(loc, carla.Location):
                loc = carla.Location(x=loc[0], y=loc[1], z=loc[2] if len(loc) > 2 else 0.0)
            wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is not None:
                plan.append((wp, RoadOption.LANEFOLLOW))
                self._route_points.append(wp.transform.location)
        if plan:
            self.agent.set_global_plan(
                plan,
                stop_waypoint_creation=stop_waypoint_creation,
                clean_queue=clean_queue,
            )

    def set_route_from_metsr_route(self, route_ids, clean_queue=True, stop_waypoint_creation=True, draw_plan=False):
        if not route_ids or self.path_planner is None:
            return False
        lane_points = self.path_planner.build_lane_points(route_ids)
        if draw_plan:
            self.path_planner.draw_coarse_points()
            self.path_planner.draw_lane_points()
        if not lane_points:
            return False
        self.set_route_from_carla_coords(
            lane_points,
            clean_queue=clean_queue,
            stop_waypoint_creation=stop_waypoint_creation,
        )
        return True

    def run_step(self, data_stream, dt=0.05):
        ego_speed = max(0.0, get_speed(self.vehicle) / 3.6)
        desired_speed = self.target_speed_mps

        if self._lane_change_cooldown > 0.0:
            self._lane_change_cooldown = max(0.0, self._lane_change_cooldown - dt)

        ego_v2v = self._get_ego_v2v(data_stream)
        path_points = self._get_path_points()

        self._ensure_turn_lane(ego_speed, ego_v2v, data_stream, dt)
        lead = self._find_lead_vehicle(ego_v2v, data_stream)
        if lead is not None:
            desired_speed = min(desired_speed, self._speed_for_gap(ego_speed, lead["distance"]))

        conflict = self._find_conflict_vehicle(ego_v2v, data_stream, path_points, ego_speed)
        if conflict is not None:
            desired_speed = min(desired_speed, ego_speed * conflict["speed_factor"])

        if self._junction_blocked(ego_v2v, data_stream, path_points):
            desired_speed = 0.0

        if self.enable_overtake_lane_change:
            self._maybe_request_lane_change(lead, ego_v2v, data_stream, dt)
        self.agent.set_target_speed(self._to_kmh(desired_speed))
        control = self.agent.run_step()
        if self.enable_debug_draw:
            self._draw_plan_points()
        return control

    def get_metsr_state(self):
        loc = self.vehicle.get_location()
        yaw = self.vehicle.get_transform().rotation.yaw
        bearing = (yaw + 90.0) % 360.0
        return loc.x, -loc.y, bearing

    def _get_ego_v2v(self, data_stream):
        for v in data_stream:
            if v.get("vid") == self.ego_vid:
                return v
        return None

    def _find_lead_vehicle(self, ego_v2v, data_stream):
        if ego_v2v is None:
            return None
        ego_heading = ego_v2v.get("heading", 0.0)
        best = None
        best_dist = float("inf")
        for v in data_stream:
            if v.get("vid") == self.ego_vid:
                continue
            dx, dy = self._relative_xy(ego_v2v, v)
            longitudinal, lateral = self._project_to_heading(ego_heading, dx, dy)
            if longitudinal <= 0.0:
                continue
            if abs(lateral) > self.lane_half_width:
                continue
            if longitudinal < best_dist:
                best_dist = longitudinal
                best = v
        if best is None:
            return None
        return {"vehicle": best, "distance": best_dist}

    def _find_conflict_vehicle(self, ego_v2v, data_stream, path_points, ego_speed):
        # Skip if no ego data or insufficient path geometry.
        if ego_v2v is None or len(path_points) < 2:
            return None
        best = None
        best_gap = float("inf")
        for v in data_stream:
            # Ignore self.
            if v.get("vid") == self.ego_vid:
                continue
            other_loc = self._v2v_to_carla_location(ego_v2v, v)
            # Skip vehicles without a valid position.
            if other_loc is None:
                continue
            other_speed = max(0.1, v.get("velocity", 0.0))
            other_heading = v.get("heading", 0.0)
            other_end = self._project_forward(other_loc, other_heading, other_speed * self.conflict_horizon_s)
            hit = self._path_intersection(path_points, other_loc, other_end)
            # Skip if their projected path does not intersect ours.
            if hit is None:
                continue
            ego_dist, other_dist, hit_loc = hit
            # Ignore if the conflict point is effectively at the ego's current position. Avoid deadlock.
            if ego_dist < self.conflict_ignore_dist:
                continue
            ego_heading = ego_v2v.get("heading", 0.0)
            ego_loc = self.vehicle.get_location()
            dx_e = hit_loc.x - ego_loc.x
            dy_e = hit_loc.y - ego_loc.y
            ego_long, _ = self._project_to_heading(ego_heading, dx_e, dy_e)
            # Skip if the conflict point is behind the ego vehicle.
            if ego_long <= 0.0:
                continue
            dx_o = hit_loc.x - other_loc.x
            dy_o = hit_loc.y - other_loc.y
            other_long, _ = self._project_to_heading(other_heading, dx_o, dy_o)
            # Skip if the conflict point is behind the other vehicle.
            if other_long <= 0.0:
                continue
            # Skip conflicts that are too far away from either vehicle.
            if ego_dist > self.conflict_max_dist or other_dist > self.conflict_max_dist:
                continue
            ego_time = ego_dist / max(0.1, ego_speed)
            other_time = other_dist / other_speed
            # Skip if the other vehicle is clearly earlier than the ego.
            if other_time < ego_time - self.conflict_time_safe:
                continue
            # Skip if the other vehicle is clearly later than the ego.
            if other_time > ego_time + self.conflict_time_safe:
                continue
            gap = abs(ego_time - other_time)
            # Accept conflicts only inside the time-gap window and keep the closest gap.
            if gap < self.conflict_time_gap and gap < best_gap:
                best_gap = gap
                speed_factor = 0.8 if ego_time < other_time else 0.3
                best = {"vehicle": v, "distance": ego_dist, "speed_factor": speed_factor}
        return best

    def _junction_blocked(self, ego_v2v, data_stream, path_points):
        if ego_v2v is None or not path_points:
            return False
        if not any(wp.is_junction for wp in self._path_waypoints(path_points)):
            return False
        for v in data_stream:
            if v.get("vid") == self.ego_vid:
                continue
            other_loc = self._v2v_to_carla_location(ego_v2v, v)
            if other_loc is None:
                continue
            wp = self.map.get_waypoint(other_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is None or not wp.is_junction:
                continue
            if other_loc.distance(path_points[0]) < self.junction_yield_radius:
                return True
        return False

    def _maybe_request_lane_change(self, lead, ego_v2v, data_stream, dt):
        if self._lane_change_cooldown > 0.0:
            return
        if lead is None or lead["distance"] > 15.0:
            return
        ego_loc = self.vehicle.get_location()
        ego_wp = self.map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if ego_wp is None:
            return
        for direction in ("left", "right"):
            target_lane = ego_wp.get_left_lane() if direction == "left" else ego_wp.get_right_lane()
            if target_lane is None or target_lane.lane_type != carla.LaneType.Driving:
                continue
            if self._lane_clear(target_lane, ego_v2v, data_stream):
                plan = self._build_lane_change_plan(target_lane, direction)
                if plan:
                    self.agent.set_global_plan(plan, clean_queue=True)
                    self._lane_change_cooldown = 3.0
                break

    def _ensure_turn_lane(self, ego_speed, ego_v2v, data_stream, dt):
        if self._lane_change_cooldown > 0.0:
            return
        planner = self.agent.get_local_planner()
        sampling = getattr(planner, "_sampling_radius", 2.0)
        horizon_dist = max(10.0, ego_speed * self.lane_change_lookahead_s)
        steps = int(max(3, min(30, horizon_dist / max(0.5, sampling))))
        incoming = planner.get_incoming_waypoint_and_direction(steps=steps)
        if incoming is None:
            return
        _, road_option = incoming
        if road_option not in (RoadOption.LEFT, RoadOption.RIGHT):
            return
        if self._plan_has_lane_change(road_option):
            return
        ego_loc = self.vehicle.get_location()
        ego_wp = self.map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if ego_wp is None:
            return
        direction = "left" if road_option == RoadOption.LEFT else "right"
        target_lane = ego_wp.get_left_lane() if direction == "left" else ego_wp.get_right_lane()
        if target_lane is None or target_lane.lane_type != carla.LaneType.Driving:
            return
        if not self._lane_change_allowed(ego_wp, direction):
            return
        if not self._lane_clear(target_lane, ego_v2v, data_stream):
            return
        plan = self._build_lane_change_plan(target_lane, direction)
        if plan:
            self.agent.set_global_plan(plan, clean_queue=True)
            self._lane_change_cooldown = 3.0

    def _lane_change_allowed(self, waypoint, direction):
        allowed = waypoint.lane_change
        if direction == "left":
            return allowed in (carla.LaneChange.Left, carla.LaneChange.Both)
        return allowed in (carla.LaneChange.Right, carla.LaneChange.Both)

    def _plan_has_lane_change(self, road_option):
        plan = list(self.agent.get_local_planner().get_plan())
        if not plan:
            return False
        option = plan[0][1]
        if road_option == RoadOption.LEFT:
            return option == RoadOption.CHANGELANELEFT
        return option == RoadOption.CHANGELANERIGHT

    def _lane_clear(self, target_lane, ego_v2v, data_stream):
        if ego_v2v is None:
            return False
        for v in data_stream:
            if v.get("vid") == self.ego_vid:
                continue
            other_loc = self._v2v_to_carla_location(ego_v2v, v)
            if other_loc is None:
                continue
            wp = self.map.get_waypoint(other_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is None:
                continue
            if wp.road_id == target_lane.road_id and wp.lane_id == target_lane.lane_id:
                if other_loc.distance(target_lane.transform.location) < 15.0:
                    return False
        return True

    def _build_lane_change_plan(self, target_lane, direction, steps=25, step_dist=2.0):
        plan = []
        wp = target_lane
        option = RoadOption.CHANGELANELEFT if direction == "left" else RoadOption.CHANGELANERIGHT
        plan.append((wp, option))
        for _ in range(steps):
            nxt = wp.next(step_dist)
            if not nxt:
                break
            wp = nxt[0]
            plan.append((wp, RoadOption.LANEFOLLOW))
        return plan

    def _get_path_points(self, count=50):
        plan = list(self.agent.get_local_planner().get_plan())
        points = []
        for wp, _ in plan[:count]:
            points.append(wp.transform.location)
        if not points:
            points.append(self.vehicle.get_location())
        return points

    def _path_waypoints(self, points):
        waypoints = []
        for loc in points:
            wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is not None:
                waypoints.append(wp)
        return waypoints

    def _draw_plan_points(self):
        if not self._route_points:
            return
        color = carla.Color(255, 220, 0)
        for loc in self._route_points:
            self.world.debug.draw_point(
                loc,
                size=0.12,
                color=color,
                life_time=0.0,
                persistent_lines=True,
            )

    def _path_intersection(self, path_points, seg_start, seg_end):
        total = 0.0
        for p0, p1 in zip(path_points[:-1], path_points[1:]):
            hit = self._segment_intersection(p0, p1, seg_start, seg_end)
            seg_len = p0.distance(p1)
            if hit is not None:
                ego_dist = total + p0.distance(hit)
                other_dist = seg_start.distance(hit)
                return ego_dist, other_dist, hit
            total += seg_len
        return None

    def _segment_intersection(self, p0, p1, p2, p3):
        x1, y1 = p0.x, p0.y
        x2, y2 = p1.x, p1.y
        x3, y3 = p2.x, p2.y
        x4, y4 = p3.x, p3.y
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-6:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return carla.Location(x=x1 + t * (x2 - x1), y=y1 + t * (y2 - y1), z=0.0)
        return None

    def _speed_for_gap(self, ego_speed, distance):
        gap = max(0.1, distance - self.min_gap)
        return min(self.target_speed_mps, gap / max(0.1, self.time_headway))

    def _relative_xy(self, ego_v2v, other_v2v):
        if self.v2v_position_mode == "local":
            ex = ego_v2v.get(self.v2v_x_key)
            ey = ego_v2v.get(self.v2v_y_key)
            ox = other_v2v.get(self.v2v_x_key)
            oy = other_v2v.get(self.v2v_y_key)
            if None in (ex, ey, ox, oy):
                return 0.0, 0.0
            return ox - ex, oy - ey
        return self._latlon_delta(ego_v2v, other_v2v)

    def _latlon_delta(self, ego_v2v, other_v2v):
        lat1 = ego_v2v.get(self.v2v_lat_key)
        lon1 = ego_v2v.get(self.v2v_lon_key)
        lat2 = other_v2v.get(self.v2v_lat_key)
        lon2 = other_v2v.get(self.v2v_lon_key)
        if None in (lat1, lon1, lat2, lon2):
            return 0.0, 0.0
        r = 6371000.0
        dx = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2.0)) * r
        dy = math.radians(lat2 - lat1) * r
        return dx, dy

    def _v2v_to_carla_location(self, ego_v2v, other_v2v):
        if self.v2v_position_mode == "local":
            x = other_v2v.get(self.v2v_x_key)
            y = other_v2v.get(self.v2v_y_key)
            if x is None or y is None:
                return None
            return carla.Location(x=x, y=-y, z=0.0)
        dx, dy = self._relative_xy(ego_v2v, other_v2v)
        ego_loc = self.vehicle.get_location()
        return carla.Location(x=ego_loc.x + dx, y=ego_loc.y + dy, z=0.0)

    def _project_to_heading(self, heading_deg, dx, dy):
        heading = math.radians(heading_deg)
        forward_x = math.sin(heading)
        forward_y = math.cos(heading)
        right_x = math.cos(heading)
        right_y = -math.sin(heading)
        longitudinal = dx * forward_x + dy * forward_y
        lateral = dx * right_x + dy * right_y
        return longitudinal, lateral

    def _project_forward(self, origin, heading_deg, distance):
        heading = math.radians(heading_deg)
        dx = math.sin(heading) * distance
        dy = math.cos(heading) * distance
        return carla.Location(x=origin.x + dx, y=origin.y + dy, z=origin.z)

    def _metsr_to_carla_location(self, x, y):
        return carla.Location(x=x, y=-y, z=0.0)

    def _to_kmh(self, speed_mps):
        return speed_mps * 3.6
