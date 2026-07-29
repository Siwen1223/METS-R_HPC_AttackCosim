import math
from types import SimpleNamespace

import carla
from cosim_utils.agents.navigation.basic_agent import BasicAgent
from cosim_utils.agents.navigation.local_planner import RoadOption
from cosim_utils.agents.tools.misc import get_speed

from cosim_utils.cosim_pathplanner import CosimPathPlanner


class V2VControllerCarla:
    """
    Control a CARLA vehicle with a CARLA-side route while incorporating nearby V2V information for car-following, lane changes, and conflict handling.
    Inputs: A CARLA vehicle, route-planning resources, controller parameters, and V2V message settings.
    Outputs: Produces CARLA control commands and exposes synchronized vehicle state for co-simulation.
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
        min_gap=1.5,
        lead_vehicle_length=4.8,
        idm_max_accel=2.0,
        idm_comfort_decel=3.0,
        idm_accel_exponent=4.0,
        lane_half_width=2.0,
        conflict_horizon_s=4.0,
        conflict_time_gap=3.0,
        conflict_time_safe=3.0,
        conflict_arrival_margin_s=0.7,
        conflict_stop_buffer=5.0,
        conflict_yield_factor=0.25,
        conflict_ignore_dist=2.0,
        conflict_max_dist=40.0,
        conflict_min_projection_dist=4.0,
        junction_yield_radius=12.0,
        junction_stop_buffer=3.0,
        path_block_lateral_m=3.0,
        path_block_max_dist=80.0,
        path_block_max_speed_mps=2.0,
        path_block_static_speed_mps=0.3,
        path_block_stop_buffer=7.0,
        curve_speed_cap_mps=7.5,
        junction_curve_speed_cap_mps=6.0,
        sharp_right_curve_speed_cap_mps=5.0,
        curve_lookahead_m=25.0,
        curve_turn_threshold_deg=35.0,
        local_planner_sampling_radius=1.0,
        local_planner_base_min_distance=1.5,
        local_planner_distance_ratio=0.2,
        waypoint_behind_threshold=0.5,
        waypoint_behind_max_prune_distance=8.0,
        control_dt=0.1,
        lane_change_lookahead_s=4.0,
        lane_change_execution_speed_mps=4.0,
        enable_overtake_lane_change=False,
        enable_debug_draw=False,
        route_project_to_carla_map=True,
        preferred_lane_ids=None,
        v2v_position_mode="geodetic",
        v2v_lat_key="true_x", #"latitude", latitude and longitude are noisy data.
        v2v_lon_key="true_y", #"longitude",
        v2v_x_key="x",
        v2v_y_key="y",
    ):
        """
        Initialize the CARLA V2V controller, behavior agent, and optional co-sim path planner.
        Inputs: Ego vehicle, ego ID, optional map/planner resources, control parameters, and V2V field settings.
        Outputs: Sets up the controller state, behavior agent, and cached route containers.
        """
        self.vehicle = vehicle
        self.ego_vid = ego_vid
        self.world = vehicle.get_world()
        self.map = map_inst or self.world.get_map()
        self.target_speed_mps = target_speed_mps
        self.time_headway = time_headway
        self.min_gap = min_gap
        self.lead_vehicle_length = lead_vehicle_length
        self.idm_max_accel = idm_max_accel
        self.idm_comfort_decel = idm_comfort_decel
        self.idm_accel_exponent = idm_accel_exponent
        self.lane_half_width = lane_half_width
        self.conflict_horizon_s = conflict_horizon_s
        self.conflict_time_gap = conflict_time_gap
        self.conflict_time_safe = conflict_time_safe
        self.conflict_arrival_margin_s = conflict_arrival_margin_s
        self.conflict_stop_buffer = conflict_stop_buffer
        self.conflict_yield_factor = conflict_yield_factor
        self.conflict_ignore_dist = conflict_ignore_dist
        self.conflict_max_dist = conflict_max_dist
        self.conflict_min_projection_dist = conflict_min_projection_dist
        self.junction_yield_radius = junction_yield_radius
        self.junction_stop_buffer = junction_stop_buffer
        self.path_block_lateral_m = path_block_lateral_m
        self.path_block_max_dist = path_block_max_dist
        self.path_block_max_speed_mps = path_block_max_speed_mps
        self.path_block_static_speed_mps = path_block_static_speed_mps
        self.path_block_stop_buffer = path_block_stop_buffer
        self.curve_speed_cap_mps = curve_speed_cap_mps
        self.junction_curve_speed_cap_mps = junction_curve_speed_cap_mps
        self.sharp_right_curve_speed_cap_mps = sharp_right_curve_speed_cap_mps
        self.curve_lookahead_m = curve_lookahead_m
        self.curve_turn_threshold_deg = curve_turn_threshold_deg
        self.local_planner_sampling_radius = local_planner_sampling_radius
        self.local_planner_base_min_distance = local_planner_base_min_distance
        self.local_planner_distance_ratio = local_planner_distance_ratio
        self.waypoint_behind_threshold = waypoint_behind_threshold
        self.waypoint_behind_max_prune_distance = waypoint_behind_max_prune_distance
        self.control_dt = control_dt
        self.lane_change_lookahead_s = lane_change_lookahead_s
        self.lane_change_execution_speed_mps = lane_change_execution_speed_mps
        self.enable_overtake_lane_change = enable_overtake_lane_change
        self.enable_debug_draw = enable_debug_draw
        self.route_project_to_carla_map = bool(route_project_to_carla_map)
        self.preferred_lane_ids = list(preferred_lane_ids or [])
        self.v2v_position_mode = v2v_position_mode
        self.v2v_lat_key = v2v_lat_key
        self.v2v_lon_key = v2v_lon_key
        self.v2v_x_key = v2v_x_key
        self.v2v_y_key = v2v_y_key
        self._lane_change_cooldown = 0.0
        self._last_lane_change_debug = {}
        self.path_planner = path_planner
        if self.path_planner is None and net_path is not None:
            self.path_planner = CosimPathPlanner(self.world, net_path)
        if self.path_planner is not None:
            self.path_planner.project_to_carla_map = self.route_project_to_carla_map

        opt_dict = {
            "ignore_traffic_lights": True,
            "ignore_vehicles": True,
            "dt": self.control_dt,
            "sampling_radius": self.local_planner_sampling_radius,
            "base_min_distance": self.local_planner_base_min_distance,
            "distance_ratio": self.local_planner_distance_ratio,
            "waypoint_behind_threshold": self.waypoint_behind_threshold,
            "waypoint_behind_max_prune_distance": self.waypoint_behind_max_prune_distance,
        }
        self.agent = BasicAgent(self.vehicle, target_speed=self._to_kmh(target_speed_mps), opt_dict=opt_dict, map_inst=self.map)
        local_planner = self.agent.get_local_planner()
        vehicle_controller = getattr(local_planner, "_vehicle_controller", None)
        if vehicle_controller is not None:
            vehicle_controller.past_steering = 0.0
        self._route_points = []
        self._route_debug_drawn = False
        self._last_debug_state = {}

    # Route setup API.

    def set_destination_xy(self, end_xy, start_xy=None, clean_queue=True):
        """
        Set a CARLA destination for the agent from METS-R-style XY coordinates.
        Inputs: Destination XY, optional start XY, and whether to clean the current waypoint queue.
        Outputs: Updates the BasicAgent destination and route queue.
        """
        start_loc = None
        if start_xy is not None:
            start_loc = self._metsr_to_carla_location(start_xy[0], start_xy[1])
        end_loc = self._metsr_to_carla_location(end_xy[0], end_xy[1])
        self.agent.set_destination(end_loc, start_location=start_loc, clean_queue=clean_queue)

    def set_route_from_carla_coords(
        self,
        coord_map,
        clean_queue=True,
        stop_waypoint_creation=True,
        start_point_carla=None,
        start_yaw_carla=None,
    ):
        """
        Convert a sequence of CARLA coordinates into a global waypoint plan for the agent.
        Inputs: Route coordinates, queue reset option, and stop_waypoint_creation flag.
        Outputs: Updates the agent global plan and cached route points.
        """
        if not coord_map:
            return
        original_point_count = len(coord_map)
        coord_map = self._path_trim_to_nearest_ahead(
            coord_map,
            start_point_carla=start_point_carla,
            start_yaw_carla=start_yaw_carla,
        )
        if self.enable_debug_draw and start_point_carla is not None and coord_map:
            first_loc = coord_map[0] if isinstance(coord_map[0], carla.Location) else carla.Location(
                x=coord_map[0][0],
                y=coord_map[0][1],
                z=coord_map[0][2] if len(coord_map[0]) > 2 else 0.0,
            )
            second_loc = None
            if len(coord_map) > 1:
                second_loc = coord_map[1] if isinstance(coord_map[1], carla.Location) else carla.Location(
                    x=coord_map[1][0],
                    y=coord_map[1][1],
                    z=coord_map[1][2] if len(coord_map[1]) > 2 else 0.0,
                )

            ref_heading = ((start_yaw_carla if start_yaw_carla is not None else self.vehicle.get_transform().rotation.yaw) + 90.0) % 360.0
            first_dx = first_loc.x - start_point_carla.x
            first_dy = first_loc.y - start_point_carla.y
            first_long, first_lat = self._geom_project_to_heading(ref_heading, first_dx, first_dy)

            second_long = None
            second_lat = None
            tangent_bearing = None
            if second_loc is not None:
                second_dx = second_loc.x - start_point_carla.x
                second_dy = second_loc.y - start_point_carla.y
                second_long, second_lat = self._geom_project_to_heading(ref_heading, second_dx, second_dy)
                seg_dx = second_loc.x - first_loc.x
                seg_dy = second_loc.y - first_loc.y
                tangent_bearing = (math.degrees(math.atan2(seg_dx, -seg_dy)) + 360.0) % 360.0

            print(
                f"[route-trim] veh={self.ego_vid} "
                f"points={original_point_count}->{len(coord_map)} "
                f"start=({start_point_carla.x:.2f},{start_point_carla.y:.2f}) "
                f"first=({first_loc.x:.2f},{first_loc.y:.2f}) "
                f"first_long={first_long:.2f} first_lat={first_lat:.2f} "
                f"second="
                + (
                    f"({second_loc.x:.2f},{second_loc.y:.2f}) second_long={second_long:.2f} "
                    f"second_lat={second_lat:.2f} tangent_bearing={tangent_bearing:.2f}"
                    if second_loc is not None
                    else "None"
                )
            )
        plan = []
        self._route_points = []
        for loc in coord_map:
            if not isinstance(loc, carla.Location):
                loc = carla.Location(x=loc[0], y=loc[1], z=loc[2] if len(loc) > 2 else 0.0)
            if self.route_project_to_carla_map:
                wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp is None:
                    continue
                route_loc = wp.transform.location
            else:
                wp = SimpleNamespace(transform=carla.Transform(loc, carla.Rotation()))
                route_loc = loc
            plan.append((wp, RoadOption.LANEFOLLOW))
            self._route_points.append(route_loc)
        if plan:
            self.agent.set_global_plan(
                plan,
                stop_waypoint_creation=stop_waypoint_creation,
                clean_queue=clean_queue,
            )
            self._route_debug_drawn = False

    def set_route_from_metsr_route(
        self,
        route_ids,
        clean_queue=True,
        stop_waypoint_creation=True,
        draw_plan=False,
        start_point_carla=None,
        start_yaw_carla=None,
    ):
        """
        Build a CARLA route from a METS-R road-id sequence and load it into the agent.
        Inputs: METS-R route IDs, queue reset option, stop_waypoint_creation flag, and optional draw flag.
        Outputs: Returns whether route loading succeeded and updates the agent plan.
        """
        if not route_ids or self.path_planner is None:
            return False
        lane_points = self.path_planner.build_lane_points(
            route_ids,
            start_point_carla=start_point_carla,
            start_yaw_carla=start_yaw_carla,
            preferred_lane_ids=self.preferred_lane_ids,
        )
        if draw_plan:
            self.path_planner.draw_coarse_points()
            self.path_planner.draw_lane_points()
        if not lane_points:
            return False
        self.set_route_from_carla_coords(
            lane_points,
            clean_queue=clean_queue,
            stop_waypoint_creation=stop_waypoint_creation,
            start_point_carla=start_point_carla,
            start_yaw_carla=start_yaw_carla,
        )
        return True

    def set_route_from_metsr_route_with_centerline(
        self,
        route_ids,
        centerline_response,
        clean_queue=True,
        stop_waypoint_creation=True,
        draw_plan=False,
    ):
        """
        Build a CARLA route from METS-R lane centerline query results and load it into the agent.
        Unlike set_route_from_metsr_route(), this path uses query_centerline() output directly instead of
        reconstructing coarse points from XML edge geometry.
        """
        if not route_ids or self.path_planner is None:
            return False
        lane_points = self.path_planner.build_carla_routepoints_from_metsr(route_ids, centerline_response)
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

    # Public control API.

    def run_step(self, data_stream, dt=0.05):
        """
        Compute one control step using V2V-aware following, conflict handling, and route tracking logic.
        Inputs: Current V2V data stream and controller time step.
        Outputs: Returns a CARLA VehicleControl command for the ego vehicle.
        """
        # Start from the nominal cruise target and the current ego speed.
        ego_speed = max(0.0, get_speed(self.vehicle) / 3.6)
        desired_speed = self.target_speed_mps

        # Count down any active lane-change cooldown so we do not keep replanning every tick.
        if self._lane_change_cooldown > 0.0:
            self._lane_change_cooldown = max(0.0, self._lane_change_cooldown - dt)

        # Gather ego state directly from CARLA and the short path segment used for V2V checks.
        ego_v2v = self._ego_state_record(data_stream)
        path_points = self._path_points()
        attack_influenced_speed = False
        attack_influence_source = ""
        self._last_lane_change_debug = {
            "attempted": False,
            "source": None,
            "blocker_vid": None,
            "blocker_distance": None,
            "status": "disabled" if not self.enable_overtake_lane_change else "not_needed",
            "direction": None,
        }
        # Do not snap the CARLA actor to a route point here. Stale waypoints are handled by LocalPlanner
        # queue pruning; direct set_transform calls make lane changes look like discontinuous jumps.

        # If a turn is coming up, try to move into the appropriate turn lane before the junction.
        self._lane_ensure_turn_alignment(ego_speed, ego_v2v, data_stream)

        # Check whether there is a same-lane vehicle ahead that should trigger car-following behavior.
        lead = self._decision_lead_vehicle(ego_v2v, data_stream)
        if lead is not None:
            # A lead vehicle means the ego should follow IDM dynamics instead of using only spacing.
            lead_speed = min(desired_speed, self._decision_speed_from_gap(ego_speed, lead, dt))
            if lead_speed < desired_speed - 1e-3 and (lead["vehicle"] or {}).get("attacked", False):
                attack_influenced_speed = True
                attack_influence_source = "lead_vehicle"
            desired_speed = lead_speed

        # Catch stationary or very slow V2V objects that occupy the planned path, including inside junctions.
        path_blocker = self._decision_path_blocking_vehicle(ego_v2v, data_stream, path_points)
        if path_blocker is not None:
            if path_blocker["speed"] <= self.path_block_static_speed_mps:
                blocker_speed = min(
                    desired_speed,
                    self._decision_speed_to_path_blocker(path_blocker["distance"]),
                )
            else:
                blocker_speed = min(
                    desired_speed,
                    self._decision_speed_from_gap(ego_speed, path_blocker, dt),
                )
            if blocker_speed < desired_speed - 1e-3 and (path_blocker["vehicle"] or {}).get("attacked", False):
                attack_influenced_speed = True
                attack_influence_source = "path_blocker"
            desired_speed = blocker_speed

        # Check for the most relevant crossing vehicle and reduce speed if a time-critical conflict exists.
        conflict = self._decision_conflict_vehicle(ego_v2v, data_stream, path_points, ego_speed)
        if conflict is not None and conflict["speed_factor"] < 1.0:
            # A crossing conflict means we taper toward a stop before the conflict point.
            conflict_speed = min(
                desired_speed,
                ego_speed * conflict["speed_factor"],
                self._decision_speed_to_conflict_point(conflict["distance"]),
            )
            if conflict_speed < desired_speed - 1e-3 and (conflict["vehicle"] or {}).get("attacked", False):
                attack_influenced_speed = True
                attack_influence_source = "crossing_conflict"
            desired_speed = conflict_speed

        # Before entering a junction, stop or slow if another truly intersecting flow is already occupying it.
        junction_blocker = self._decision_junction_blocked(ego_v2v, data_stream, path_points)
        junction_blocked = junction_blocker is not None
        junction_entry_dist = self._path_distance_to_junction_entry(path_points) if junction_blocked else None
        if junction_blocked:
            before_junction_speed = desired_speed
            if junction_entry_dist is None:
                # Fallback: if we cannot estimate a stop line distance, use a full stop.
                desired_speed = 0.0
            else:
                # Otherwise taper speed toward the junction entry so the ego stops near the stop line.
                desired_speed = min(
                    desired_speed,
                    self._decision_speed_to_stop_line(junction_entry_dist),
                )
            if desired_speed < before_junction_speed - 1e-3 and (junction_blocker or {}).get("attacked", False):
                attack_influenced_speed = True
                attack_influence_source = "junction_blocker"

        # Optionally try an overtaking lane change if a slow lead vehicle or path blocker is blocking progress.
        lane_change_planned = False
        if self.enable_overtake_lane_change:
            # Only consider overtaking when this optional behavior is enabled by the scenario.
            overtake_blocker, overtake_source, overtake_max_distance = self._lane_overtake_blocker(lead, path_blocker)
            lane_change_planned = self._lane_try_overtake(
                overtake_blocker,
                ego_v2v,
                data_stream,
                max_distance=overtake_max_distance,
                source=overtake_source,
            )
            if lane_change_planned:
                desired_speed = max(
                    desired_speed,
                    min(self.target_speed_mps, self.lane_change_execution_speed_mps),
                )

        # Cap speed before sharp upcoming turns so the PID follower can stay on the lane centerline.
        curve_speed_cap = self._decision_curve_speed_cap(path_points)
        if curve_speed_cap is not None:
            desired_speed = min(desired_speed, curve_speed_cap)

        # Hand the final speed target to BasicAgent and let it generate the low-level CARLA control.
        self.agent.set_target_speed(self._to_kmh(desired_speed))
        control = self.agent.run_step()
        local_planner = self.agent.get_local_planner()
        '''target_wp = getattr(local_planner, "target_waypoint", None)
        if target_wp is not None:
            target_loc = target_wp.transform.location
            print(
                f"[target-wp] veh={self.ego_vid} "
                f"loc=({target_loc.x:.2f},{target_loc.y:.2f},{target_loc.z:.2f}) "
                f"road={target_wp.road_id} lane={target_wp.lane_id}"
            )
        else:
            print(f"[target-wp] veh={self.ego_vid} None")'''

        # Cache a compact summary so the outer script can print one-line debug state per vehicle.
        self._last_debug_state = {
            "ego_vid": self.ego_vid,
            "ego_speed": ego_speed,
            "desired_speed": desired_speed,
            "lead_vid": lead["vehicle"].get("vid") if lead is not None else None,
            "lead_distance": lead["distance"] if lead is not None else None,
            "path_blocker_vid": path_blocker["vehicle"].get("vid") if path_blocker is not None else None,
            "path_blocker_distance": path_blocker["distance"] if path_blocker is not None else None,
            "path_blocker_lateral": path_blocker["lateral"] if path_blocker is not None else None,
            "conflict_vid": conflict["vehicle"].get("vid") if conflict is not None else None,
            "conflict_distance": conflict["distance"] if conflict is not None else None,
            "conflict_speed_factor": conflict["speed_factor"] if conflict is not None else None,
            "conflict_ego_time": conflict["ego_time"] if conflict is not None else None,
            "conflict_other_time": conflict["other_time"] if conflict is not None else None,
            "junction_blocked": junction_blocked,
            "junction_blocker_vid": junction_blocker.get("vid") if junction_blocker is not None else None,
            "junction_entry_distance": junction_entry_dist,
            "curve_speed_cap": curve_speed_cap,
            "attack_influenced_speed": attack_influenced_speed,
            "attack_influence_source": attack_influence_source,
            "lane_change_attempted": self._last_lane_change_debug.get("attempted"),
            "lane_change_source": self._last_lane_change_debug.get("source"),
            "lane_change_blocker_vid": self._last_lane_change_debug.get("blocker_vid"),
            "lane_change_blocker_distance": self._last_lane_change_debug.get("blocker_distance"),
            "lane_change_status": self._last_lane_change_debug.get("status"),
            "lane_change_direction": self._last_lane_change_debug.get("direction"),
            "path_point_count": len(path_points),
            "control_throttle": control.throttle,
            "control_brake": control.brake,
            "control_steer": control.steer,
        }

        # Keep route points visible in CARLA when debug drawing is enabled.
        if self.enable_debug_draw:
            self._route_draw_points()
        return control

    def is_route_complete(self):
        """
        Check whether the current agent route has been completed.
        Inputs: No additional inputs.
        Outputs: Returns True if the BasicAgent has finished its route, otherwise False.
        """
        if not self.agent.done():
            return False
        if not self._route_points:
            return True
        current_loc = self.vehicle.get_location()
        return current_loc.distance(self._route_points[-1]) <= 8.0

    def get_last_debug_state(self):
        """
        Return the cached debug summary from the most recent control step.
        Inputs: No additional inputs.
        Outputs: Returns a dictionary containing the latest lead/conflict/junction/control diagnostics.
        """
        return dict(self._last_debug_state)

    def get_metsr_state(self):
        """
        Export the ego vehicle state in the coordinate/bearing format expected by METS-R.
        Inputs: No additional inputs.
        Outputs: Returns x, y, and bearing derived from the current CARLA vehicle state.
        """
        loc = self.vehicle.get_location()
        yaw = self.vehicle.get_transform().rotation.yaw
        # Verified with the current Town05 co-sim setup:
        # CARLA yaw 0 deg points along +x, while METS-R/BSM heading 90 deg
        # points along the same world direction. Therefore bearing = yaw + 90.
        bearing = (yaw + 90.0) % 360.0
        return loc.x, -loc.y, bearing

    # Decision helpers.

    def _decision_lead_vehicle(self, ego_v2v, data_stream):
        """
        Find the closest vehicle ahead of the ego on the same CARLA lane.
        Inputs: The ego V2V record and the full V2V data stream.
        Outputs: Returns a lead-vehicle dictionary with distance information or None.
        """
        if ego_v2v is None:
            return None
        ego_wp = self.map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_wp is None or ego_wp.is_junction:
            return None
        ego_heading = ego_v2v.get("heading", 0.0)
        best = None
        best_dist = float("inf")
        for other_v2v in self._v2v_other_records(data_stream):
            other_loc = self._v2v_to_carla_location(ego_v2v, other_v2v)
            if other_loc is None:
                continue
            other_wp = self.map.get_waypoint(
                other_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if other_wp is None or other_wp.is_junction:
                continue
            if other_wp.road_id != ego_wp.road_id or other_wp.lane_id != ego_wp.lane_id:
                continue
            dx, dy = self._v2v_relative_xy(ego_v2v, other_v2v)
            longitudinal, _ = self._geom_project_to_heading(ego_heading, dx, dy)
            if longitudinal <= 0.0:
                continue
            if longitudinal < best_dist:
                best_dist = longitudinal
                best = other_v2v
        if best is None:
            return None
        return {"vehicle": best, "distance": best_dist}

    def _decision_path_blocking_vehicle(self, ego_v2v, data_stream, path_points):
        """
        Find a stationary or very slow V2V object that physically occupies the ego's planned path.
        Inputs: Ego V2V record, full V2V stream, and upcoming CARLA path points.
        Outputs: Returns the closest blocking object dictionary or None.
        """
        if ego_v2v is None or len(path_points) < 1:
            return None

        ego_loc = self.vehicle.get_location()
        path = [ego_loc] + list(path_points)
        if len(path) < 2:
            return None

        best = None
        best_dist = float("inf")
        for other_v2v in self._v2v_other_records(data_stream):
            other_speed = self._v2v_record_speed(other_v2v)
            if other_speed > self.path_block_max_speed_mps:
                continue

            other_loc = self._v2v_to_carla_location(ego_v2v, other_v2v)
            if other_loc is None:
                continue

            projection = self._path_nearest_projection(path, other_loc)
            if projection is None:
                continue

            distance = projection["distance"]
            lateral = projection["lateral"]
            if distance <= self.conflict_ignore_dist:
                continue
            if distance > self.path_block_max_dist:
                continue
            if lateral > self.path_block_lateral_m:
                continue
            if distance < best_dist:
                best_dist = distance
                best = {
                    "vehicle": other_v2v,
                    "distance": distance,
                    "lateral": lateral,
                    "speed": other_speed,
                }
        return best

    def _decision_conflict_vehicle(self, ego_v2v, data_stream, path_points, ego_speed):
        """
        Detect the most relevant crossing-conflict vehicle based on predicted intersection timing along the ego path.
        Inputs: Ego V2V record, full V2V data stream, ego path points, and ego speed.
        Outputs: Returns a conflict dictionary with speed_factor information or None.
        """
        # No conflict search is possible without ego V2V data or a usable path polyline.
        if ego_v2v is None or len(path_points) < 2:
            return None

        best_priority = None
        best_priority_gap = float("inf")
        best_yield = None
        best_yield_score = (float("inf"), float("inf"), float("inf"))
        for other_v2v in self._v2v_other_records(data_stream):
            conflict_state = self._conflict_state(ego_v2v, other_v2v, path_points)
            # Skip vehicles whose projected motion never intersects the ego path.
            if conflict_state is None:
                continue

            ego_dist = conflict_state["ego_dist"]
            other_dist = conflict_state["other_dist"]
            # Ignore a conflict point that is effectively under the ego already; this avoids self-deadlock.
            if ego_dist < self.conflict_ignore_dist:
                continue
            # Reject intersections that lie behind either vehicle instead of ahead along its travel direction.
            if conflict_state["ego_long"] <= 0.0 or conflict_state["other_long"] <= 0.0:
                continue
            # Ignore crossings that are too far away to matter for near-term control.
            if ego_dist > self.conflict_max_dist or other_dist > self.conflict_max_dist:
                continue

            ego_time = ego_dist / max(0.1, ego_speed)
            other_time = other_dist / max(0.01, conflict_state["other_speed"])
            gap = abs(ego_time - other_time)
            # Ignore conflicts whose arrival times are far enough apart to be operationally safe.
            if gap > self.conflict_time_gap:
                continue

            ego_has_priority = self._conflict_ego_has_priority(
                ego_v2v,
                other_v2v,
                ego_time,
                other_time,
            )
            conflict = {
                "vehicle": other_v2v,
                "distance": ego_dist,
                "speed_factor": 1.0 if ego_has_priority else self.conflict_yield_factor,
                "ego_time": ego_time,
                "other_time": other_time,
                "time_gap": gap,
            }
            if ego_has_priority:
                if gap < best_priority_gap:
                    best_priority_gap = gap
                    best_priority = conflict
            else:
                yield_score = (ego_time, ego_dist, gap)
                if yield_score < best_yield_score:
                    best_yield_score = yield_score
                    best_yield = conflict
        return best_yield or best_priority

    def _decision_junction_blocked(self, ego_v2v, data_stream, path_points):
        """
        Check whether a true crossing vehicle is already occupying or about to occupy the ego junction path.
        Inputs: Ego V2V record, full V2V data stream, and ego path points.
        Outputs: Returns the blocking V2V record if blocked, otherwise None.
        """
        if ego_v2v is None or len(path_points) < 2:
            return None

        ego_wp = self.map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_wp is None or ego_wp.is_junction:
            return None

        junction_points = self._path_junction_points(path_points)
        if len(junction_points) < 2:
            return None

        ego_heading = ego_v2v.get("heading", 0.0)
        for other_v2v in self._v2v_other_records(data_stream):
            dx, dy = self._v2v_relative_xy(ego_v2v, other_v2v)
            longitudinal, lateral = self._geom_project_to_heading(ego_heading, dx, dy)
            other_heading = other_v2v.get("heading", 0.0)
            heading_delta = abs((other_heading - ego_heading + 180.0) % 360.0 - 180.0)
            # Same-direction lead vehicles should be handled by car-following, not junction blocking.
            if longitudinal > 0.0 and abs(lateral) <= self.lane_half_width * 2.0 and heading_delta <= 35.0:
                continue

            conflict_state = self._conflict_state(ego_v2v, other_v2v, junction_points)
            if conflict_state is None:
                continue
            if conflict_state["ego_long"] <= 0.0 or conflict_state["other_long"] <= 0.0:
                continue

            other_wp = self.map.get_waypoint(
                conflict_state["other_loc"],
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if other_wp is None:
                continue
            if other_wp.is_junction or conflict_state["other_dist"] < self.junction_yield_radius:
                return other_v2v
        return None

    def _decision_speed_from_gap(self, ego_speed, lead, dt):
        """
        Convert a lead-vehicle state into a target speed using the Intelligent Driver Model.
        Inputs: Ego speed, lead-vehicle dictionary, and controller time step.
        Outputs: Returns a target speed in m/s.
        """
        lead_vehicle = lead.get("vehicle", {})
        # lead["distance"] is measured between reported vehicle reference points; IDM needs a net bumper gap.
        center_gap = max(0.1, float(lead.get("distance", 0.1) or 0.1))
        net_gap = max(0.1, center_gap - float(self.lead_vehicle_length))
        lead_speed = self._v2v_record_speed(lead_vehicle)
        delta_v = ego_speed - lead_speed
        accel = self._idm_acceleration(ego_speed, lead_speed, net_gap, delta_v)
        next_speed = ego_speed + accel * max(0.0, float(dt))
        return max(0.0, min(self.target_speed_mps, next_speed))

    def _idm_acceleration(self, ego_speed, lead_speed, net_gap, delta_v):
        """
        Compute IDM acceleration from ego speed, lead speed, net gap, and closing speed.
        Inputs: Speeds in m/s, net gap in m, and signed closing speed ego-minus-lead in m/s.
        Outputs: Returns acceleration in m/s^2.
        """
        desired_speed = max(0.1, float(self.target_speed_mps))
        max_accel = max(0.1, float(self.idm_max_accel))
        comfort_decel = max(0.1, float(self.idm_comfort_decel))
        exponent = max(1.0, float(self.idm_accel_exponent))
        min_gap = max(0.0, float(self.min_gap))
        time_headway = max(0.0, float(self.time_headway))
        desired_gap = min_gap + max(
            0.0,
            ego_speed * time_headway
            + ego_speed * delta_v / (2.0 * math.sqrt(max_accel * comfort_decel)),
        )
        free_term = (max(0.0, ego_speed) / desired_speed) ** exponent
        interaction_term = (desired_gap / max(0.1, net_gap)) ** 2
        accel = max_accel * (1.0 - free_term - interaction_term)
        return max(-comfort_decel * 2.0, min(max_accel, accel))

    def _decision_speed_to_stop_line(self, distance_to_stop_line):
        """
        Convert remaining distance to a junction stop line into a smooth target speed cap.
        Inputs: Distance from the ego to the junction entry along the planned path.
        Outputs: Returns a target speed in m/s that aims to stop shortly before the junction.
        """
        remaining = max(0.0, distance_to_stop_line - self.junction_stop_buffer)
        return min(self.target_speed_mps, remaining / max(0.1, self.time_headway))

    def _decision_speed_to_conflict_point(self, distance_to_conflict):
        """
        Convert distance to a crossing conflict point into a smooth target speed cap.
        Inputs: Along-path distance to the conflict point.
        Outputs: Returns a target speed in m/s that aims to stop before the conflict point.
        """
        remaining = max(0.0, distance_to_conflict - self.conflict_stop_buffer)
        return min(self.target_speed_mps, remaining / max(0.1, self.time_headway))

    def _decision_speed_to_path_blocker(self, distance_to_blocker):
        """
        Convert distance to a path-blocking object into a smooth speed cap.
        Inputs: Along-path distance to the blocking object.
        Outputs: Returns a target speed in m/s that stops before the object.
        """
        remaining = max(0.0, distance_to_blocker - self.path_block_stop_buffer)
        return min(self.target_speed_mps, remaining / max(0.1, self.time_headway))

    def _conflict_ego_has_priority(self, ego_v2v, other_v2v, ego_time, other_time):
        """
        Resolve an unsignalized crossing conflict with an arrival margin and deterministic tie-break.
        Inputs: Ego/other V2V records and their estimated arrival times at the conflict point.
        Outputs: Returns True when ego should continue, False when ego should yield.
        """
        margin = max(0.0, float(self.conflict_arrival_margin_s))
        if ego_time + margin < other_time:
            return True
        if other_time + margin < ego_time:
            return False

        # Near-simultaneous arrivals: use the common unsignalized rule of yielding to the right.
        ego_heading = ego_v2v.get("heading", 0.0)
        ego_loc = self.vehicle.get_location()
        other_loc = self._v2v_to_carla_location(ego_v2v, other_v2v)
        if other_loc is not None:
            _, lateral = self._geom_project_to_heading(
                ego_heading,
                other_loc.x - ego_loc.x,
                other_loc.y - ego_loc.y,
            )
            if lateral > self.lane_half_width:
                return False
            if lateral < -self.lane_half_width:
                return True

        # Fallback for ambiguous geometry: keep the result deterministic to avoid both vehicles proceeding.
        return self._vehicle_priority_key(self.ego_vid) <= self._vehicle_priority_key(other_v2v.get("vid"))

    # Lane-change helpers.

    def _lane_ensure_turn_alignment(self, ego_speed, ego_v2v, data_stream):
        """
        Move the ego into a required turn lane before an upcoming intersection maneuver.
        Inputs: Ego speed, ego V2V record, and full V2V data stream.
        Outputs: May replace the current agent plan and update the lane-change cooldown.
        """
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
        if self._lane_plan_has_change(road_option):
            return

        ego_wp = self.map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_wp is None:
            return

        direction = "left" if road_option == RoadOption.LEFT else "right"
        target_lane = ego_wp.get_left_lane() if direction == "left" else ego_wp.get_right_lane()
        if target_lane is None or target_lane.lane_type != carla.LaneType.Driving:
            return
        if not self._lane_change_allowed(ego_wp, direction):
            return
        if not self._lane_is_clear(target_lane, ego_v2v, data_stream):
            return

        plan = self._lane_change_plan(ego_wp, direction)
        if plan:
            self.agent.set_global_plan(plan, stop_waypoint_creation=False, clean_queue=True)
            self._lane_change_cooldown = 3.0

    def _lane_overtake_blocker(self, lead, path_blocker):
        """
        Select the blocker that should trigger an optional overtaking lane change.
        Inputs: Same-lane lead vehicle info and path-blocking vehicle info.
        Outputs: Returns blocker info, source label, and trigger distance.
        """
        if path_blocker is not None and (lead is None or lead["distance"] > 15.0):
            return path_blocker, "path_blocker", self.path_block_max_dist
        if lead is not None:
            return lead, "lead", 15.0
        return None, None, 0.0

    def _lane_try_overtake(self, blocker, ego_v2v, data_stream, max_distance=15.0, source=None):
        """
        Attempt an overtaking lane change when enabled and the current blocker is too close.
        Inputs: Lead/path-blocking vehicle info, ego V2V record, full V2V stream, trigger distance, and source label.
        Outputs: May replace the current agent plan and update the lane-change cooldown.
        """
        self._last_lane_change_debug.update({
            "source": source,
            "blocker_vid": (blocker.get("vehicle") or {}).get("vid") if blocker is not None else None,
            "blocker_distance": blocker.get("distance") if blocker is not None else None,
        })
        if self._lane_change_cooldown > 0.0:
            self._last_lane_change_debug["status"] = "cooldown"
            return False
        if blocker is None:
            self._last_lane_change_debug["status"] = "no_blocker"
            return False
        if blocker["distance"] > max_distance:
            self._last_lane_change_debug["status"] = "blocker_too_far"
            return False

        self._last_lane_change_debug["attempted"] = True
        ego_wp = self.map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_wp is None:
            self._last_lane_change_debug["status"] = "no_ego_waypoint"
            return False

        saw_side_lane = False
        for direction in ("left", "right"):
            target_lane = ego_wp.get_left_lane() if direction == "left" else ego_wp.get_right_lane()
            if target_lane is None or target_lane.lane_type != carla.LaneType.Driving:
                continue
            saw_side_lane = True
            if not self._lane_is_clear(target_lane, ego_v2v, data_stream):
                self._last_lane_change_debug["status"] = f"{direction}_not_clear"
                continue
            distance_same_lane = min(1.0, max(0.2, float(blocker.get("distance", 0.0)) - 2.0))
            plan = self._lane_change_plan(
                ego_wp,
                direction,
                distance_same_lane=distance_same_lane,
                check_lane_marking=False,
            )
            if not plan:
                self._last_lane_change_debug["status"] = f"{direction}_plan_empty"
                continue
            if plan:
                self.agent.set_global_plan(plan, stop_waypoint_creation=False, clean_queue=True)
                self._lane_change_cooldown = 3.0
                self._last_lane_change_debug["status"] = "planned"
                self._last_lane_change_debug["direction"] = direction
                return True
        if not saw_side_lane:
            self._last_lane_change_debug["status"] = "no_side_lane"
        return False

    def _lane_change_allowed(self, waypoint, direction):
        """
        Check whether CARLA lane markings permit a lane change in the requested direction.
        Inputs: Current lane waypoint and desired lane-change direction.
        Outputs: Returns True if the lane change is allowed, otherwise False.
        """
        allowed = waypoint.lane_change
        if direction == "left":
            return allowed in (carla.LaneChange.Left, carla.LaneChange.Both)
        return allowed in (carla.LaneChange.Right, carla.LaneChange.Both)

    def _lane_plan_has_change(self, road_option):
        """
        Check whether the current local plan already begins with the requested lane-change maneuver.
        Inputs: Desired RoadOption for the lane change direction.
        Outputs: Returns True if the current plan already contains that lane change, otherwise False.
        """
        plan = list(self.agent.get_local_planner().get_plan())
        if not plan:
            return False
        option = plan[0][1]
        if road_option == RoadOption.LEFT:
            return option == RoadOption.CHANGELANELEFT
        return option == RoadOption.CHANGELANERIGHT

    def _lane_is_clear(self, target_lane, ego_v2v, data_stream):
        """
        Check whether the target lane is free of nearby vehicles around the merge point.
        Inputs: Target lane waypoint, ego V2V record, and full V2V data stream.
        Outputs: Returns True if the target lane is considered clear, otherwise False.
        """
        if ego_v2v is None:
            return False
        for other_v2v in self._v2v_other_records(data_stream):
            other_loc = self._v2v_to_carla_location(ego_v2v, other_v2v)
            if other_loc is None:
                continue
            wp = self.map.get_waypoint(other_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is None:
                continue
            if wp.road_id == target_lane.road_id and wp.lane_id == target_lane.lane_id:
                if other_loc.distance(target_lane.transform.location) < 15.0:
                    return False
        return True

    def _lane_change_plan(
        self,
        ego_waypoint,
        direction,
        distance_same_lane=8.0,
        distance_other_lane=30.0,
        lane_change_distance=25.0,
        step_dist=2.0,
        check_lane_marking=True,
    ):
        """
        Build a smooth CARLA waypoint plan that performs a lane change and then follows the new lane.
        Inputs: Current ego waypoint, lane-change direction, maneuver distances, and step distance.
        Outputs: Returns a waypoint plan list for BasicAgent.set_global_plan().
        """
        plan = self.agent._generate_lane_change_path(
            ego_waypoint,
            direction,
            distance_same_lane,
            distance_other_lane,
            lane_change_distance,
            check_lane_marking,
            1,
            step_dist,
        )
        if plan:
            return plan
        return self._lane_change_plan_direct(
            ego_waypoint,
            direction,
            distance_same_lane=distance_same_lane,
            distance_other_lane=distance_other_lane,
            step_dist=step_dist,
        )

    def _lane_change_plan_direct(
        self,
        ego_waypoint,
        direction,
        distance_same_lane=8.0,
        distance_other_lane=30.0,
        step_dist=2.0,
    ):
        """
        Build a lane-change plan from the nearest adjacent lane when CARLA's helper cannot find a side lane farther ahead.
        Inputs: Current ego waypoint, lane-change direction, same-lane distance, other-lane distance, and step distance.
        Outputs: Returns a waypoint plan list or an empty list if no adjacent driving lane is available.
        """
        plan = [(ego_waypoint, RoadOption.LANEFOLLOW)]
        wp = ego_waypoint
        travelled = 0.0
        while travelled < max(0.0, distance_same_lane):
            next_wps = wp.next(step_dist)
            if not next_wps:
                break
            next_wp = next_wps[0]
            travelled += next_wp.transform.location.distance(wp.transform.location)
            wp = next_wp
            plan.append((wp, RoadOption.LANEFOLLOW))

        side_wp = wp.get_left_lane() if direction == "left" else wp.get_right_lane()
        if side_wp is None or side_wp.lane_type != carla.LaneType.Driving:
            side_wp = ego_waypoint.get_left_lane() if direction == "left" else ego_waypoint.get_right_lane()
        if side_wp is None or side_wp.lane_type != carla.LaneType.Driving:
            return []

        option = RoadOption.CHANGELANELEFT if direction == "left" else RoadOption.CHANGELANERIGHT
        plan.append((side_wp, option))

        wp = side_wp
        travelled = 0.0
        while travelled < max(0.0, distance_other_lane):
            next_wps = wp.next(step_dist)
            if not next_wps:
                break
            next_wp = next_wps[0]
            travelled += next_wp.transform.location.distance(wp.transform.location)
            wp = next_wp
            plan.append((wp, RoadOption.LANEFOLLOW))
        return plan

    # Path helpers.

    def _path_trim_to_nearest_ahead(self, coord_map, start_point_carla=None, start_yaw_carla=None):
        """
        Drop only the stale prefix of a route near the current handoff position.

        The trim is intentionally local: it first finds the earliest route point that is close to
        the ego, then advances within that local neighborhood until the retained start point is no
        longer behind the ego. This avoids jumping to a later route suffix that happens to be
        geometrically close to the current position.
        """
        if not coord_map:
            return []

        points = []
        for loc in coord_map:
            if isinstance(loc, carla.Location):
                points.append(loc)
            else:
                points.append(carla.Location(x=loc[0], y=loc[1], z=loc[2] if len(loc) > 2 else 0.0))

        ego_loc = start_point_carla if start_point_carla is not None else self.vehicle.get_location()
        if start_yaw_carla is not None:
            yaw_rad = math.radians(start_yaw_carla)
            forward = carla.Vector3D(x=math.cos(yaw_rad), y=math.sin(yaw_rad), z=0.0)
        else:
            forward = self.vehicle.get_transform().get_forward_vector()
        trim_radius = 12.0
        cos_angle_threshold = math.cos(math.radians(45.0))
        nearby_idx = None
        for idx, point in enumerate(points):
            if ego_loc.distance(point) <= trim_radius:
                nearby_idx = idx
                break

        if nearby_idx is None:
            return points

        start_idx = nearby_idx
        while start_idx < len(points) - 1:
            point = points[start_idx]
            dx = point.x - ego_loc.x
            dy = point.y - ego_loc.y
            dz = point.z - ego_loc.z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            longitudinal = dx * forward.x + dy * forward.y + dz * forward.z
            cos_angle = 1.0 if distance <= 1e-3 else longitudinal / distance
            if longitudinal > 0.0 and cos_angle >= cos_angle_threshold:
                break
            start_idx += 1

        return points[start_idx:]

    def _path_points(self, count=50):
        """
        Extract a finite set of upcoming path points from the current local planner queue.
        Inputs: Maximum number of points to read from the local plan.
        Outputs: Returns a list of CARLA Locations representing the upcoming path.
        """
        plan = list(self.agent.get_local_planner().get_plan())
        points = [wp.transform.location for wp, _ in plan[:count]]
        if not points:
            points.append(self.vehicle.get_location())
        return points

    def _snap_to_next_path_point_if_needed(self, path_points):
        if len(path_points) < 2:
            return False
        current_loc = self.vehicle.get_location()
        current_yaw = self.vehicle.get_transform().rotation.yaw
        next_loc = path_points[0]
        dist = current_loc.distance(next_loc)
        if dist <= 1e-3 or dist >= 15.0:
            return False

        heading_rad = math.radians(current_yaw)
        forward_x = math.cos(heading_rad)
        forward_y = math.sin(heading_rad)
        to_next_x = (next_loc.x - current_loc.x) / dist
        to_next_y = (next_loc.y - current_loc.y) / dist
        cos_angle = max(-1.0, min(1.0, forward_x * to_next_x + forward_y * to_next_y))
        angle_deg = math.degrees(math.acos(cos_angle))
        if angle_deg <= 60.0:
            return False

        next_next_loc = path_points[1]
        seg_dx = next_next_loc.x - next_loc.x
        seg_dy = next_next_loc.y - next_loc.y
        if abs(seg_dx) < 1e-3 and abs(seg_dy) < 1e-3:
            snap_yaw = current_yaw
        else:
            snap_yaw = math.degrees(math.atan2(seg_dy, seg_dx))

        current_transform = self.vehicle.get_transform()
        snapped_transform = carla.Transform(
            carla.Location(x=next_loc.x, y=next_loc.y, z=next_loc.z),
            carla.Rotation(
                pitch=current_transform.rotation.pitch,
                yaw=snap_yaw,
                roll=current_transform.rotation.roll,
            ),
        )
        self.vehicle.set_transform(snapped_transform)
        return True

    def _path_waypoints(self, path_points):
        """
        Map CARLA locations to driving-lane waypoints along the current path.
        Inputs: A list of CARLA path locations.
        Outputs: Returns the corresponding CARLA waypoint list for valid driving lanes.
        """
        waypoints = []
        for loc in path_points:
            wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is not None:
                waypoints.append(wp)
        return waypoints

    def _path_junction_points(self, path_points):
        """
        Extract only the next contiguous junction segment from the ego path.
        Inputs: A list of upcoming CARLA path locations.
        Outputs: Returns the junction-focused path slice or an empty list.
        """
        junction_flags = []
        for loc in path_points:
            wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            junction_flags.append(wp is not None and wp.is_junction)

        first_idx = next((idx for idx, is_junction in enumerate(junction_flags) if is_junction), None)
        if first_idx is None:
            return []

        last_idx = first_idx
        for idx in range(first_idx + 1, len(junction_flags)):
            if junction_flags[idx]:
                last_idx = idx
            else:
                break

        start_idx = max(0, first_idx - 1)
        end_idx = min(len(path_points), last_idx + 2)
        return path_points[start_idx:end_idx]

    def _path_distance_to_junction_entry(self, path_points):
        """
        Measure path distance from the ego position to the first waypoint that lies inside the next junction.
        Inputs: A list of upcoming CARLA path locations.
        Outputs: Returns the path distance in meters, or None if no junction is ahead.
        """
        if not path_points:
            return None

        ego_loc = self.vehicle.get_location()
        prev_loc = ego_loc
        distance = 0.0
        for loc in path_points:
            distance += prev_loc.distance(loc)
            wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is not None and wp.is_junction:
                return distance
            prev_loc = loc
        return None

    def _decision_curve_speed_cap(self, path_points):
        """
        Return a speed cap when the upcoming path bends sharply within the near lookahead window.
        Inputs: Upcoming CARLA path locations.
        Outputs: Speed cap in m/s, or None when no curve cap is needed.
        """
        ego_wp = self.map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        junction_cap = None
        if ego_wp is not None and ego_wp.is_junction and self.junction_curve_speed_cap_mps > 0.0:
            junction_cap = min(self.curve_speed_cap_mps, self.junction_curve_speed_cap_mps)

        if len(path_points) < 3 or self.curve_speed_cap_mps <= 0.0:
            return junction_cap

        points = [self.vehicle.get_location()] + path_points
        prev_loc = points[0]
        prev_heading = None
        traveled = 0.0
        cumulative_turn = 0.0
        cumulative_signed_turn = 0.0
        turn_threshold_rad = math.radians(self.curve_turn_threshold_deg)

        for loc in points[1:]:
            segment = prev_loc.distance(loc)
            if segment < 0.2:
                continue
            traveled += segment
            heading = math.atan2(loc.y - prev_loc.y, loc.x - prev_loc.x)
            if prev_heading is not None:
                signed_delta = (heading - prev_heading + math.pi) % (2.0 * math.pi) - math.pi
                cumulative_signed_turn += signed_delta
                cumulative_turn += abs(signed_delta)
                if (
                    self.sharp_right_curve_speed_cap_mps > 0.0
                    and cumulative_signed_turn >= turn_threshold_rad
                ):
                    return self.sharp_right_curve_speed_cap_mps
                if cumulative_turn >= turn_threshold_rad:
                    return self.curve_speed_cap_mps
            if traveled >= self.curve_lookahead_m:
                break
            prev_heading = heading
            prev_loc = loc

        return junction_cap

    def _route_draw_points(self):
        """
        Draw the cached global route points for debugging in the CARLA world.
        Inputs: No additional inputs.
        Outputs: Renders the cached route points in the CARLA debug layer.
        """
        if not self._route_points:
            return
        if self._route_debug_drawn:
            return
        color = carla.Color(255, 220, 0)
        draw_points = [
            carla.Location(x=loc.x, y=loc.y, z=loc.z + 1.0)
            for loc in self._route_points
        ]
        for draw_loc in draw_points:
            self.world.debug.draw_point(
                draw_loc,
                size=0.22,
                color=color,
                life_time=0.0,
                persistent_lines=True,
            )
        for start, end in zip(draw_points[:-1], draw_points[1:]):
            if start.distance(end) <= 8.0:
                self.world.debug.draw_line(
                    start,
                    end,
                    thickness=0.08,
                    color=color,
                    life_time=0.0,
                    persistent_lines=True,
                )
        self._route_debug_drawn = True

    # Conflict geometry helpers.

    def _conflict_state(self, ego_v2v, other_v2v, path_points):
        """
        Compute the projected crossing state between the ego path and another vehicle motion segment.
        Inputs: Ego V2V record, another vehicle V2V record, and ego path points.
        Outputs: Returns a conflict-state dictionary or None if there is no path intersection.
        """
        if len(path_points) < 2:
            return None
        motion_state = self._other_motion_state(ego_v2v, other_v2v)
        if motion_state is None:
            return None

        hit = self._path_first_intersection(
            path_points,
            motion_state["other_loc"],
            motion_state["other_end"],
        )
        if hit is None:
            return None

        ego_dist, other_dist, hit_loc = hit
        ego_heading = ego_v2v.get("heading", 0.0)
        ego_loc = self.vehicle.get_location()
        ego_long, _ = self._geom_project_to_heading(
            ego_heading,
            hit_loc.x - ego_loc.x,
            hit_loc.y - ego_loc.y,
        )
        other_long, _ = self._geom_project_to_heading(
            motion_state["other_heading"],
            hit_loc.x - motion_state["other_loc"].x,
            hit_loc.y - motion_state["other_loc"].y,
        )
        return {
            "other_v2v": other_v2v,
            "other_loc": motion_state["other_loc"],
            "other_speed": motion_state["other_speed"],
            "other_heading": motion_state["other_heading"],
            "other_end": motion_state["other_end"],
            "projection_dist": motion_state["projection_dist"],
            "ego_dist": ego_dist,
            "other_dist": other_dist,
            "hit_loc": hit_loc,
            "ego_long": ego_long,
            "other_long": other_long,
        }

    def _other_motion_state(self, ego_v2v, other_v2v):
        """
        Build a short projected motion segment for another vehicle from its current V2V state.
        Inputs: Ego V2V record and another vehicle V2V record.
        Outputs: Returns projected motion information or None if the position is invalid.
        """
        other_loc = self._v2v_to_carla_location(ego_v2v, other_v2v)
        if other_loc is None:
            return None
        #other_speed = self._v2v_effective_speed(other_v2v)
        other_speed = max(0.0, other_v2v.get("velocity", 0.0))
        other_heading = other_v2v.get("heading", 0.0)
        projection_dist = max(
            other_speed * self.conflict_horizon_s,
            self.conflict_min_projection_dist,
        )
        other_end = self._geom_project_forward(other_loc, other_heading, projection_dist)
        return {
            "other_loc": other_loc,
            "other_speed": other_speed,
            "other_heading": other_heading,
            "projection_dist": projection_dist,
            "other_end": other_end,
        }

    def _path_first_intersection(self, path_points, seg_start, seg_end):
        """
        Find the first geometric intersection between the ego path polyline and another projected segment.
        Inputs: Ego path points, segment start, and segment end.
        Outputs: Returns ego distance, other distance, and intersection location, or None.
        """
        total = 0.0
        for p0, p1 in zip(path_points[:-1], path_points[1:]):
            hit = self._geom_segment_intersection(p0, p1, seg_start, seg_end)
            seg_len = p0.distance(p1)
            if hit is not None:
                ego_dist = total + p0.distance(hit)
                other_dist = seg_start.distance(hit)
                return ego_dist, other_dist, hit
            total += seg_len
        return None

    def _path_nearest_projection(self, path_points, loc):
        """
        Project a location onto the ego path polyline and return along-path and lateral distances.
        Inputs: CARLA path points and a CARLA location.
        Outputs: Returns a distance dictionary or None for a degenerate path.
        """
        total = 0.0
        best = None
        best_lateral = float("inf")
        for p0, p1 in zip(path_points[:-1], path_points[1:]):
            dx = p1.x - p0.x
            dy = p1.y - p0.y
            seg_len_sq = dx * dx + dy * dy
            seg_len = math.sqrt(seg_len_sq)
            if seg_len <= 1e-6:
                continue

            t = ((loc.x - p0.x) * dx + (loc.y - p0.y) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = p0.x + t * dx
            proj_y = p0.y + t * dy
            lateral = math.hypot(loc.x - proj_x, loc.y - proj_y)
            distance = total + seg_len * t
            if lateral < best_lateral:
                best_lateral = lateral
                best = {
                    "distance": distance,
                    "lateral": lateral,
                    "point": carla.Location(x=proj_x, y=proj_y, z=loc.z),
                }
            total += seg_len
        return best

    def _geom_segment_intersection(self, p0, p1, p2, p3):
        """
        Compute the intersection point of two 2D line segments if it exists.
        Inputs: Four CARLA Locations defining two segments.
        Outputs: Returns the intersection CARLA Location or None.
        """
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

    def _geom_project_to_heading(self, heading_deg, dx, dy):
        """
        Project a relative 2D offset into longitudinal and lateral components under a given heading.
        Inputs: Heading in degrees and a relative dx, dy vector.
        Outputs: Returns longitudinal and lateral distances in the heading-aligned frame.
        """
        heading = math.radians(heading_deg)
        # Verified with current METS-R/BSM headings:
        # heading 0 deg -> CARLA -y, 90 deg -> CARLA +x,
        # 180 deg -> CARLA +y, 270 deg -> CARLA -x.
        forward_x = math.sin(heading)
        forward_y = -math.cos(heading)
        right_x = math.cos(heading)
        right_y = math.sin(heading)
        longitudinal = dx * forward_x + dy * forward_y
        lateral = dx * right_x + dy * right_y
        return longitudinal, lateral

    def _geom_project_forward(self, origin, heading_deg, distance):
        """
        Project a CARLA location forward along a heading by a given distance.
        Inputs: Origin location, heading in degrees, and travel distance.
        Outputs: Returns the forward-projected CARLA Location.
        """
        heading = math.radians(heading_deg)
        # Same verified heading convention as _geom_project_to_heading():
        # 0 deg -> -y, 90 deg -> +x, 180 deg -> +y, 270 deg -> -x.
        dx = math.sin(heading) * distance
        dy = -math.cos(heading) * distance
        return carla.Location(x=origin.x + dx, y=origin.y + dy, z=origin.z)

    # V2V coordinate helpers.

    def _ego_state_record(self, data_stream=None):
        """
        Build the ego record from the CARLA actor; V2V stream self-records are only compatibility data.
        Inputs: Optional current V2V stream.
        Outputs: Returns a controller-ready ego state dictionary.
        """
        record = self._v2v_ego_record(data_stream or []) or {}
        loc = self.vehicle.get_location()
        yaw = self.vehicle.get_transform().rotation.yaw
        heading = (yaw + 90.0) % 360.0
        speed = max(0.0, get_speed(self.vehicle) / 3.6)
        record = dict(record)
        record.update(
            {
                "vid": self.ego_vid,
                "vehicle_id": self.ego_vid,
                "sender_id": self.ego_vid,
                "x": loc.x,
                "y": -loc.y,
                "true_x": loc.x,
                "true_y": -loc.y,
                "velocity": speed,
                "speed": speed,
                "speed_mps": speed,
                "heading": heading,
                "heading_deg": heading,
            }
        )
        return record

    def _v2v_ego_record(self, data_stream):
        """
        Find the optional ego compatibility record in the current controller stream.
        Inputs: A list of V2V message dictionaries.
        Outputs: Returns the ego V2V dictionary or None if it is missing.
        """
        for v2v_record in data_stream:
            if v2v_record.get("vid") == self.ego_vid:
                return v2v_record
        return None

    def _v2v_other_records(self, data_stream):
        """
        Iterate over all non-ego V2V records in the current stream.
        Inputs: A list of V2V message dictionaries.
        Outputs: Yields one non-ego V2V record at a time.
        """
        for v2v_record in data_stream:
            if v2v_record.get("vid") != self.ego_vid:
                yield v2v_record

    def _v2v_relative_xy(self, ego_v2v, other_v2v):
        """
        Compute the relative 2D position of another vehicle with respect to the ego from V2V data.
        Inputs: Ego V2V record and another vehicle's V2V record.
        Outputs: Returns relative dx and dy in the configured V2V position mode.
        """
        if self.v2v_position_mode == "local":
            ex = ego_v2v.get(self.v2v_x_key)
            ey = ego_v2v.get(self.v2v_y_key)
            ox = other_v2v.get(self.v2v_x_key)
            oy = other_v2v.get(self.v2v_y_key)
            if None in (ex, ey, ox, oy):
                return 0.0, 0.0
            return ox - ex, -(oy - ey)
        return self._v2v_latlon_delta(ego_v2v, other_v2v)

    def _v2v_latlon_delta(self, ego_v2v, other_v2v):
        """
        Convert two geodetic V2V positions into an approximate local Cartesian offset.
        Inputs: Ego V2V record and another vehicle's V2V record with latitude/longitude.
        Outputs: Returns relative dx and dy in meters.
        """
        lat1 = ego_v2v.get(self.v2v_lat_key)
        lon1 = ego_v2v.get(self.v2v_lon_key)
        lat2 = other_v2v.get(self.v2v_lat_key)
        lon2 = other_v2v.get(self.v2v_lon_key)
        if None in (lat1, lon1, lat2, lon2):
            return 0.0, 0.0
        r = 6371000.0
        east = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2.0)) * r
        north = math.radians(lat2 - lat1) * r
        # Verified against CARLA ground truth in the current Town05 co-sim setup:
        # northward motion in BSM lat/lon aligns with CARLA +x,
        # eastward motion in BSM lat/lon aligns with CARLA -y.
        dx = north
        dy = -east
        return dx, dy

    def _v2v_to_carla_location(self, ego_v2v, other_v2v):
        """
        Convert another vehicle's V2V-reported position into a CARLA location in the current world frame.
        Inputs: Ego V2V record and another vehicle's V2V record.
        Outputs: Returns a CARLA Location or None if the V2V position is invalid.
        """
        if self.v2v_position_mode == "local":
            x = other_v2v.get(self.v2v_x_key)
            y = other_v2v.get(self.v2v_y_key)
            if x is None or y is None:
                return None
            return carla.Location(x=x, y=-y, z=0.0)
        dx, dy = self._v2v_relative_xy(ego_v2v, other_v2v)
        ego_loc = self.vehicle.get_location()
        return carla.Location(x=ego_loc.x + dx, y=ego_loc.y + dy, z=0.0)

    def _v2v_effective_speed(self, other_v2v):
        """
        Build a non-zero speed estimate for projected V2V motion even when reported velocity is missing.
        Inputs: Another vehicle's V2V record.
        Outputs: Returns the effective speed used by conflict and junction checks.
        """
        reported_speed = max(0.0, other_v2v.get("velocity", 0.0))
        min_speed = self.conflict_min_projection_dist / max(0.1, self.conflict_horizon_s)
        return max(reported_speed, min_speed)

    @staticmethod
    def _v2v_record_speed(record):
        for key in ("velocity", "speed_mps", "speed"):
            value = record.get(key)
            if value is not None:
                return max(0.0, float(value))
        return 0.0

    @staticmethod
    def _vehicle_priority_key(value):
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, str(value))

    # Small utilities.

    def _metsr_to_carla_location(self, x, y):
        """
        Convert METS-R XY coordinates into a CARLA location.
        Inputs: METS-R x and y coordinates.
        Outputs: Returns the corresponding CARLA Location.
        """
        return carla.Location(x=x, y=-y, z=0.0)

    def _to_kmh(self, speed_mps):
        """
        Convert speed from meters per second to kilometers per hour.
        Inputs: Speed in m/s.
        Outputs: Returns speed in km/h.
        """
        return speed_mps * 3.6
