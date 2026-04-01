import math

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
        min_gap=2.5,
        lane_half_width=2.0,
        conflict_horizon_s=4.0,
        conflict_time_gap=2.5,
        conflict_time_safe=3.0,
        conflict_ignore_dist=2.0,
        conflict_max_dist=40.0,
        conflict_min_projection_dist=4.0,
        junction_yield_radius=12.0,
        junction_stop_buffer=3.0,
        lane_change_lookahead_s=4.0,
        enable_overtake_lane_change=False,
        enable_debug_draw=False,
        v2v_position_mode="geodetic",
        v2v_lat_key="latitude",
        v2v_lon_key="longitude",
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
        self.lane_half_width = lane_half_width
        self.conflict_horizon_s = conflict_horizon_s
        self.conflict_time_gap = conflict_time_gap
        self.conflict_time_safe = conflict_time_safe
        self.conflict_ignore_dist = conflict_ignore_dist
        self.conflict_max_dist = conflict_max_dist
        self.conflict_min_projection_dist = conflict_min_projection_dist
        self.junction_yield_radius = junction_yield_radius
        self.junction_stop_buffer = junction_stop_buffer
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
        local_planner = self.agent.get_local_planner()
        vehicle_controller = getattr(local_planner, "_vehicle_controller", None)
        if vehicle_controller is not None:
            vehicle_controller.past_steering = 0.0
        self._route_points = []
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
        lane_points = self.path_planner.build_lane_points(route_ids, start_point_carla=start_point_carla)
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

        # Gather the ego V2V state and the short CARLA path segment used for all downstream checks.
        ego_v2v = self._v2v_ego_record(data_stream)
        path_points = self._path_points()

        # If a turn is coming up, try to move into the appropriate turn lane before the junction.
        self._lane_ensure_turn_alignment(ego_speed, ego_v2v, data_stream)

        # Check whether there is a same-lane vehicle ahead that should trigger car-following behavior.
        lead = self._decision_lead_vehicle(ego_v2v, data_stream)
        if lead is not None:
            # A lead vehicle means the ego should honor longitudinal headway instead of free cruising.
            desired_speed = min(desired_speed, self._decision_speed_from_gap(ego_speed, lead["distance"]))

        # Check for the most relevant crossing vehicle and reduce speed if a time-critical conflict exists.
        conflict = self._decision_conflict_vehicle(ego_v2v, data_stream, path_points, ego_speed)
        if conflict is not None:
            # A crossing conflict means we keep moving only at the yield factor selected by the conflict logic.
            desired_speed = min(desired_speed, ego_speed * conflict["speed_factor"])

        # Before entering a junction, stop or slow if another truly intersecting flow is already occupying it.
        junction_blocked = self._decision_junction_blocked(ego_v2v, data_stream, path_points)
        junction_entry_dist = self._path_distance_to_junction_entry(path_points) if junction_blocked else None
        if junction_blocked:
            if junction_entry_dist is None:
                # Fallback: if we cannot estimate a stop line distance, use a full stop.
                desired_speed = 0.0
            else:
                # Otherwise taper speed toward the junction entry so the ego stops near the stop line.
                desired_speed = min(
                    desired_speed,
                    self._decision_speed_to_stop_line(junction_entry_dist),
                )

        # Optionally try an overtaking lane change if a slow lead vehicle is blocking progress.
        if self.enable_overtake_lane_change:
            # Only consider overtaking when this optional behavior is enabled by the scenario.
            self._lane_try_overtake(lead, ego_v2v, data_stream)

        # Hand the final speed target to BasicAgent and let it generate the low-level CARLA control.
        self.agent.set_target_speed(self._to_kmh(desired_speed))
        control = self.agent.run_step()
        local_planner = self.agent.get_local_planner()
        target_wp = getattr(local_planner, "target_waypoint", None)
        if target_wp is not None:
            target_loc = target_wp.transform.location
            print(
                f"[target-wp] veh={self.ego_vid} "
                f"loc=({target_loc.x:.2f},{target_loc.y:.2f},{target_loc.z:.2f}) "
                f"road={target_wp.road_id} lane={target_wp.lane_id}"
            )
        else:
            print(f"[target-wp] veh={self.ego_vid} None")

        # Cache a compact summary so the outer script can print one-line debug state per vehicle.
        self._last_debug_state = {
            "ego_vid": self.ego_vid,
            "ego_speed": ego_speed,
            "desired_speed": desired_speed,
            "lead_vid": lead["vehicle"].get("vid") if lead is not None else None,
            "lead_distance": lead["distance"] if lead is not None else None,
            "conflict_vid": conflict["vehicle"].get("vid") if conflict is not None else None,
            "conflict_distance": conflict["distance"] if conflict is not None else None,
            "conflict_speed_factor": conflict["speed_factor"] if conflict is not None else None,
            "junction_blocked": junction_blocked,
            "junction_entry_distance": junction_entry_dist,
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
        return self.agent.done()

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
        Find the closest vehicle ahead of the ego within the current lane-width envelope.
        Inputs: The ego V2V record and the full V2V data stream.
        Outputs: Returns a lead-vehicle dictionary with distance information or None.
        """
        if ego_v2v is None:
            return None
        ego_heading = ego_v2v.get("heading", 0.0)
        best = None
        best_dist = float("inf")
        for other_v2v in self._v2v_other_records(data_stream):
            dx, dy = self._v2v_relative_xy(ego_v2v, other_v2v)
            longitudinal, lateral = self._geom_project_to_heading(ego_heading, dx, dy)
            if longitudinal <= 0.0 or abs(lateral) > self.lane_half_width:
                continue
            if longitudinal < best_dist:
                best_dist = longitudinal
                best = other_v2v
        if best is None:
            return None
        return {"vehicle": best, "distance": best_dist}

    def _decision_conflict_vehicle(self, ego_v2v, data_stream, path_points, ego_speed):
        """
        Detect the most relevant crossing-conflict vehicle based on predicted intersection timing along the ego path.
        Inputs: Ego V2V record, full V2V data stream, ego path points, and ego speed.
        Outputs: Returns a conflict dictionary with speed_factor information or None.
        """
        # No conflict search is possible without ego V2V data or a usable path polyline.
        if ego_v2v is None or len(path_points) < 2:
            return None

        best = None
        best_gap = float("inf")
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
            # Drop cases where the other vehicle is clearly going to clear the conflict long before ego arrives.
            if other_time < ego_time - self.conflict_time_safe:
                continue
            # Drop cases where the other vehicle is clearly far enough behind that it is not an immediate conflict.
            if other_time > ego_time + self.conflict_time_safe:
                continue

            gap = abs(ego_time - other_time)
            # Keep only the closest-in-time conflict that falls inside the active conflict window.
            if gap < self.conflict_time_gap and gap < best_gap:
                best_gap = gap
                # If ego arrives first, only soften speed; otherwise yield more aggressively.
                speed_factor = 0.8 if ego_time < other_time else 0.3
                best = {
                    "vehicle": other_v2v,
                    "distance": ego_dist,
                    "speed_factor": speed_factor,
                }
        return best

    def _decision_junction_blocked(self, ego_v2v, data_stream, path_points):
        """
        Check whether a true crossing vehicle is already occupying or about to occupy the ego junction path.
        Inputs: Ego V2V record, full V2V data stream, and ego path points.
        Outputs: Returns True if the junction should be treated as blocked, otherwise False.
        """
        if ego_v2v is None or len(path_points) < 2:
            return False

        ego_wp = self.map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_wp is None or ego_wp.is_junction:
            return False

        junction_points = self._path_junction_points(path_points)
        if len(junction_points) < 2:
            return False

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
                return True
        return False

    def _decision_speed_from_gap(self, ego_speed, distance):
        """
        Convert a front-vehicle gap into a reduced target speed using headway and minimum-gap rules.
        Inputs: Ego speed and longitudinal distance to the lead vehicle.
        Outputs: Returns a target speed in m/s.
        """
        gap = max(0.1, distance - self.min_gap)
        return min(self.target_speed_mps, gap / max(0.1, self.time_headway))

    def _decision_speed_to_stop_line(self, distance_to_stop_line):
        """
        Convert remaining distance to a junction stop line into a smooth target speed cap.
        Inputs: Distance from the ego to the junction entry along the planned path.
        Outputs: Returns a target speed in m/s that aims to stop shortly before the junction.
        """
        remaining = max(0.0, distance_to_stop_line - self.junction_stop_buffer)
        return min(self.target_speed_mps, remaining / max(0.1, self.time_headway))

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

        plan = self._lane_change_plan(target_lane, direction)
        if plan:
            self.agent.set_global_plan(plan, clean_queue=True)
            self._lane_change_cooldown = 3.0

    def _lane_try_overtake(self, lead, ego_v2v, data_stream):
        """
        Attempt an overtaking lane change when enabled and the current lead vehicle is too close.
        Inputs: Lead-vehicle info, ego V2V record, and full V2V data stream.
        Outputs: May replace the current agent plan and update the lane-change cooldown.
        """
        if self._lane_change_cooldown > 0.0:
            return
        if lead is None or lead["distance"] > 15.0:
            return

        ego_wp = self.map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_wp is None:
            return

        for direction in ("left", "right"):
            target_lane = ego_wp.get_left_lane() if direction == "left" else ego_wp.get_right_lane()
            if target_lane is None or target_lane.lane_type != carla.LaneType.Driving:
                continue
            if not self._lane_is_clear(target_lane, ego_v2v, data_stream):
                continue
            plan = self._lane_change_plan(target_lane, direction)
            if plan:
                self.agent.set_global_plan(plan, clean_queue=True)
                self._lane_change_cooldown = 3.0
            break

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

    def _lane_change_plan(self, target_lane, direction, steps=25, step_dist=2.0):
        """
        Build a short CARLA waypoint plan that performs a lane change and then follows the new lane.
        Inputs: Target lane waypoint, lane-change direction, number of follow-up steps, and step distance.
        Outputs: Returns a waypoint plan list for BasicAgent.set_global_plan().
        """
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

    def _route_draw_points(self):
        """
        Draw the cached global route points for debugging in the CARLA world.
        Inputs: No additional inputs.
        Outputs: Renders the cached route points in the CARLA debug layer.
        """
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

    def _v2v_ego_record(self, data_stream):
        """
        Find the ego vehicle record in the current V2V data stream.
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
            return ox - ex, oy - ey
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
