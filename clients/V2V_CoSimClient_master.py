"""CARLA + METS-R + Simu5G/VEINS co-simulation client."""

from pathlib import Path

from clients.CoSimClient import CoSimClient
from cosim_utils.C_V2X_manager import CV2XManager, normalize_vehicle
from cosim_utils.Sensor_manager import SensorManager
from cosim_utils.cosim_pathplanner import CosimPathPlanner
from cosim_utils.v2v_controller_carla import V2VControllerCarla
from utils.carla_util import release_ready_cosim_vehicles_from_queue


class V2VCoSimClientMaster(CoSimClient):
    """Coordinate CARLA physics, METS-R traffic, and Simu5G C-V2X delivery."""

    def __init__(
        self,
        config,
        carla_client,
        tm_client,
        controller_vids=None,
        controller_cls=V2VControllerCarla,
        controller_kwargs=None,
        cv2x_manager=None,
        enable_v2x=True,
        require_simu5g_uu=True,
        camera_layout=None,
    ):
        super().__init__(config, carla_client, tm_client)
        self.dt = float(getattr(config, "sim_step_size", 0.1))
        self.step_index = 0
        self.controller_cls = controller_cls
        self.controller_kwargs = dict(controller_kwargs or {})
        self.controller_vids = set(controller_vids or getattr(config, "controller_vids", []))
        self.controllers = {}
        self.route_synced = {}
        self.last_v2x_stream = []
        self.last_v2x_streams = {}
        self.last_v2x_rows = []
        self.last_v2x_result = {}
        self.last_controls = {}
        self.enable_v2x = enable_v2x
        self.ghost_attacks = []
        self.handoff_path_planner = None
        self.carla_tick_timeout = float(getattr(config, "carla_tick_timeout", 5.0))
        self.metsr_tick_timeout = float(getattr(config, "metsr_tick_timeout", 5.0))
        self.release_queued_cosim_vehicles = bool(
            getattr(config, "release_queued_cosim_vehicles", True)
        )

        self.sensor_manager = SensorManager(
            self.carla,
            self._vehicle_for_sensor,
            output_path=getattr(config, "sensor_output_path", "_out"),
            camera_layout=camera_layout or getattr(config, "camera_layout", "front_rear"),
            camera_interval_ticks=getattr(config, "camera_interval_ticks", 5),
            lidar_interval_ticks=getattr(config, "lidar_interval_ticks", 10),
        )
        self.cv2x = cv2x_manager or CV2XManager(
            config=config,
            require_simu5g_uu=require_simu5g_uu,
            duration_s=self.dt,
        )

    def set_controller_vehicles(self, vids):
        self.controller_vids = set(vids or [])

    def add_controller_vehicle(self, vid):
        self.controller_vids.add(vid)

    def register_ghost_attack(self, ghost_vehicle, start_tick=0, end_tick=None):
        """Inject one falsified BSM sender through Simu5G during the active tick window."""
        self.ghost_attacks.append(
            {
                "vehicle": dict(ghost_vehicle),
                "start_tick": int(start_tick),
                "end_tick": None if end_tick is None else int(end_tick),
            }
        )

    def step(self, extra_v2x_messages=None, phase="run"):
        """Advance one co-simulation step and return compact V2X/debug data."""
        self._ensure_controllers()
        self._sync_controller_routes()
        self._apply_controller_controls()

        self._base_step()

        self._ensure_controllers()
        self._sync_controller_routes()
        done_vids = self._finish_completed_routes()
        v2x_payload = self._sync_v2x(extra_v2x_messages=extra_v2x_messages, phase=phase)
        self.step_index += 1
        return {
            "tick": self.current_tick,
            "done_vids": done_vids,
            "v2x": v2x_payload,
            "controllers": self.controllers,
            "controls": dict(self.last_controls),
        }

    def run(self, max_steps=None):
        max_steps = max_steps or int(self.config.sim_minutes * 60 / self.config.sim_step_size)
        try:
            for _ in range(int(max_steps)):
                self.step()
        except KeyboardInterrupt:
            print("simulation interrupted by user")
        finally:
            self.close()

    @property
    def current_tick(self):
        tick = getattr(self.metsr, "current_tick", None)
        return self.step_index if tick is None else int(tick)

    def close(self):
        try:
            if hasattr(self, "sensor_manager"):
                for vid in list(self.sensor_manager.sensors):
                    self.sensor_manager.destroy_vehicle_sensors(vid)
            self.cv2x.close()
        finally:
            self.metsr.terminate()

    def enable_vehicle_sensor(self, vid):
        self.carla_veh_dataCollect.add(vid)
        self.sensor_manager.enable_vehicle(vid)

    def disable_vehicle_sensor(self, vid):
        self.carla_veh_dataCollect.discard(vid)
        self.sensor_manager.disable_vehicle(vid)

    def deploy_vehicle_sensors(self, vid):
        if hasattr(self, "sensor_manager"):
            self.sensor_manager.deploy_vehicle_sensors(vid)

    def destroy_vehicle_sensors(self, vid):
        if hasattr(self, "sensor_manager"):
            self.sensor_manager.destroy_vehicle_sensors(vid)

    def collect_sensor_data(self, output_path=None, tick=None):
        self.sensor_manager.collect_sensor_data(output_path=output_path, tick=tick)

    def get_collision_events(self):
        return self.sensor_manager.get_collision_events()

    def save_sensor_data(self, vid, output_path=None, tick=None):
        self.sensor_manager.save_sensor_data(vid, output_path=output_path, tick=tick)

    def _carla_tick(self):
        try:
            return self.carla.tick(seconds=self.carla_tick_timeout)
        except TypeError:
            return self.carla.tick()

    def _base_step(self):
        """CoSimClient.step() logic with queued co-sim vehicle release."""
        self._carla_tick()
        self.metsr.tick(1, max_wait_seconds=self.metsr_tick_timeout, poll_timeout=1)

        if self.release_queued_cosim_vehicles:
            release_ready_cosim_vehicles_from_queue(self.metsr)

        cosim_vehs = self.metsr.query_coSimVehicle()["DATA"]
        cosim_ids = [vehicle["ID"] for vehicle in cosim_vehs]
        cosim_private_flags = [vehicle["v_type"] for vehicle in cosim_vehs]

        cosim_info_map = {}
        cosim_meta_map = {}
        if cosim_ids:
            all_data = self.metsr.query_vehicle(
                cosim_ids,
                cosim_private_flags,
                transform_coords=True,
            )["DATA"]
            for cosim_id, cosim_veh, private_flag, veh_info in zip(
                cosim_ids,
                cosim_vehs,
                cosim_private_flags,
                all_data,
            ):
                cosim_meta_map[cosim_id] = cosim_veh
                cosim_info_map[cosim_id] = (private_flag, veh_info)

        current_cosim_ids = set(cosim_ids)
        managed_ids = set(self.carla_vehs.keys())

        for vid in managed_ids - current_cosim_ids:
            print(f"Vehicle {vid} left the co-sim ownership set and is no longer CARLA-managed.")
            self.handoff_carla_vehicle(vid)

        for cosim_id in cosim_ids:
            private_flag, veh_info = cosim_info_map[cosim_id]
            self.carla_private_flags[cosim_id] = private_flag
            if cosim_id not in self.carla_vehs and veh_info["state"] > 0:
                if cosim_id in self.displayOnly_vehs:
                    print(f"Vehicle {cosim_id} switched from display-only to CARLA-managed.")
                    self.destroy_carla_vehicle(cosim_id)
                self.carla_coordMaps[cosim_id] = cosim_meta_map[cosim_id].get("coord_map", [])
                self.carla_route[cosim_id] = cosim_meta_map[cosim_id].get("route", [])
                route = self.carla_route[cosim_id]
                self.carla_destRoad[cosim_id] = route[-1] if route else None
                handoff_loc = self.get_carla_location(veh_info["x"], veh_info["y"])
                _, handoff_yaw = self.get_carla_rotation(veh_info)
                route_handoff = self._route_handoff_pose(route, handoff_loc, cosim_id)
                if route_handoff is not None:
                    handoff_loc, handoff_yaw, lane_path = route_handoff
                    if getattr(self.config, "verbose", False):
                        print(
                            f"[handoff-route] veh={cosim_id} route={route} "
                            f"lane_path={lane_path} "
                            f"spawn=({handoff_loc.x:.2f},{handoff_loc.y:.2f}) yaw={handoff_yaw:.2f}"
                        )
                self.carla_handoff_locs[cosim_id] = handoff_loc
                self.carla_handoff_yaws[cosim_id] = handoff_yaw
                blocking_vid = self._handoff_spawn_blocker(cosim_id, handoff_loc)
                if blocking_vid is not None:
                    if cosim_id not in self.carla_spawn_pending:
                        print(
                            f"Vehicle {cosim_id} handoff delayed because vehicle {blocking_vid} "
                            f"is still within {self.handoff_spawn_clearance_m:.1f} m of "
                            f"({handoff_loc.x:.2f},{handoff_loc.y:.2f})."
                        )
                    self.carla_spawn_pending.add(cosim_id)
                    continue
                spawned_actor = self.spawn_carla_vehicle(cosim_id, private_flag, veh_info, display_only=False)
                if spawned_actor is None:
                    if cosim_id not in self.carla_spawn_pending:
                        handoff_loc = self.carla_handoff_locs[cosim_id]
                        print(
                            f"Vehicle {cosim_id} handoff delayed because CARLA could not spawn it "
                            f"at ({handoff_loc.x:.2f},{handoff_loc.y:.2f})."
                        )
                    self.carla_spawn_pending.add(cosim_id)
                    continue
                self.carla_spawn_pending.discard(cosim_id)
                self.carla_entered[cosim_id] = True
                print(f"Vehicle {cosim_id} entered the co-sim ownership set and is now CARLA-managed.")

        for cosim_id in cosim_ids:
            if cosim_id in self.carla_vehs:
                private_flag, veh_info = cosim_info_map[cosim_id]
                self.sync_carla_vehicle(cosim_id, private_flag, veh_info)

        if self.display_all:
            self.sync_display_only_vehicles(current_cosim_ids)

    def _route_handoff_pose(self, route_ids, reference_loc, vid=None):
        if not route_ids:
            return None
        net_path = getattr(self.config, "network_file", None)
        if not net_path:
            return None
        if self.handoff_path_planner is None:
            self.handoff_path_planner = CosimPathPlanner(self.carla, str(Path(net_path).resolve()))
        return self.handoff_path_planner.route_handoff_pose(
            route_ids,
            reference_carla_location=reference_loc,
            preferred_lane_ids=self._preferred_lane_ids(vid),
        )

    def _preferred_lane_ids(self, vid):
        lane_paths = getattr(self.config, "controller_lane_paths", {}) or {}
        if vid in lane_paths:
            return list(lane_paths[vid])
        if str(vid) in lane_paths:
            return list(lane_paths[str(vid)])
        return []

    def _vehicle_for_sensor(self, vid):
        return self.carla_vehs.get(vid) or self.displayOnly_vehs.get(vid)

    def _ensure_controllers(self):
        for vid in list(self.controller_vids):
            if vid in self.controllers:
                continue
            carla_vehicle = self.carla_vehs.get(vid)
            if carla_vehicle is None:
                continue
            carla_vehicle.set_autopilot(False)
            kwargs = self._controller_kwargs(vid)
            self.controllers[vid] = self.controller_cls(
                vehicle=carla_vehicle,
                ego_vid=vid,
                **kwargs,
            )
            self.route_synced[vid] = False

    def _controller_kwargs(self, vid):
        kwargs = {
            "net_path": str(Path(getattr(self.config, "network_file")).resolve())
            if getattr(self.config, "network_file", None)
            else None,
            "target_speed_mps": float(getattr(self.config, "v2v_target_speed_mps", 10.0)),
            "control_dt": self.dt,
            "enable_debug_draw": bool(getattr(self.config, "enable_debug_draw", False)),
            "v2v_position_mode": getattr(self.config, "v2v_position_mode", "local"),
            "preferred_lane_ids": self._preferred_lane_ids(vid),
        }
        kwargs.update(self.controller_kwargs)
        kwargs.update(getattr(self.config, "controller_kwargs", {}) or {})
        return {key: value for key, value in kwargs.items() if value is not None}

    def _sync_controller_routes(self):
        for vid, controller in list(self.controllers.items()):
            if self.route_synced.get(vid, False):
                continue
            if not self.carla_entered.get(vid, False):
                continue
            carla_vehicle = self.carla_vehs.get(vid)
            if carla_vehicle is None:
                continue
            route_ids = self.carla_route.get(vid, [])
            start_loc = self.carla_handoff_locs.get(vid, carla_vehicle.get_location())
            start_yaw = self.carla_handoff_yaws.get(vid, carla_vehicle.get_transform().rotation.yaw)
            if route_ids and controller.set_route_from_metsr_route(
                route_ids,
                stop_waypoint_creation=True,
                draw_plan=bool(getattr(self.config, "draw_route_plan", False)),
                start_point_carla=start_loc,
                start_yaw_carla=start_yaw,
            ):
                self.route_synced[vid] = True
                if getattr(self.config, "enable_debug_draw", False) and controller.path_planner is not None:
                    controller.path_planner.draw_lane_points()
                if getattr(self.config, "verbose", False):
                    points = list(getattr(controller, "_route_points", []) or [])
                    if points:
                        xs = [point.x for point in points]
                        ys = [point.y for point in points]
                        print(
                            f"[route-plan] veh={vid} points={len(points)} "
                            f"first=({points[0].x:.2f},{points[0].y:.2f},{points[0].z:.2f}) "
                            f"last=({points[-1].x:.2f},{points[-1].y:.2f},{points[-1].z:.2f}) "
                            f"bbox=({min(xs):.2f},{min(ys):.2f})-({max(xs):.2f},{max(ys):.2f})"
                        )
            elif route_ids and getattr(self.config, "verbose", False):
                print(
                    f"WARNING: failed to sync CARLA route for veh={vid}; "
                    f"route={route_ids}, start=({start_loc.x:.2f},{start_loc.y:.2f}), yaw={start_yaw:.2f}"
                )

    def _apply_controller_controls(self):
        self.last_controls = {}
        for vid, controller in list(self.controllers.items()):
            carla_vehicle = self.carla_vehs.get(vid)
            if carla_vehicle is None or not self.carla_entered.get(vid, False):
                continue
            if not self.route_synced.get(vid, False):
                continue
            control = controller.run_step(self.last_v2x_streams.get(vid, []), dt=self.dt)
            carla_vehicle.apply_control(control)
            self.last_controls[vid] = control

    def _finish_completed_routes(self):
        done_vids = []
        for vid, controller in list(self.controllers.items()):
            if not self.carla_entered.get(vid, False):
                continue
            if not self.route_synced.get(vid, False):
                continue
            if not controller.is_route_complete():
                continue
            private_flag = self.carla_private_flags.get(vid, False)
            dest_road = self.carla_destRoad.get(vid)
            if dest_road in getattr(self.config, "metsr_road", []):
                res = self.metsr.reach_dest(vid, private_flag)
                if self._status_ok(res):
                    self.destroy_carla_vehicle(vid)
                    done_vids.append(vid)
            else:
                carla_vehicle = self.carla_vehs.get(vid)
                if carla_vehicle is None:
                    continue
                loc = carla_vehicle.get_location()
                res = self._exit_cosim_region(vid, loc.x, -loc.y, private_flag, True)
                if self._status_ok(res):
                    self.destroy_carla_vehicle(vid)
                    done_vids.append(vid)

        for vid in done_vids:
            self.controllers.pop(vid, None)
            self.route_synced.pop(vid, None)
            self.controller_vids.discard(vid)
        return done_vids

    @staticmethod
    def _status_ok(response):
        try:
            return response["DATA"][0]["STATUS"] == "OK"
        except Exception:
            return False

    def _exit_cosim_region(self, vid, x, y, private_veh=False, transform_coord=True):
        if hasattr(self.metsr, "exit_cosim_region"):
            return self.metsr.exit_cosim_region(vid, x, y, private_veh, transform_coord)
        msg = {
            "TYPE": "CTRL_exitCoSimRegion",
            "DATA": [
                {
                    "vehID": vid,
                    "vehType": private_veh,
                    "transformCoord": transform_coord,
                    "x": x,
                    "y": y,
                }
            ],
        }
        res = self.metsr.send_receive_msg(msg, ignore_heartbeats=True)
        assert res["TYPE"] == "CTRL_exitCoSimRegion", res["TYPE"]
        assert res["CODE"] == "OK", res["CODE"]
        return res

    def _sync_v2x(self, extra_v2x_messages=None, phase="run"):
        if not self.enable_v2x:
            return {}
        vehicles = self._current_v2x_vehicles()
        if not vehicles:
            self.last_v2x_stream = []
            self.last_v2x_streams = {}
            self.last_v2x_rows = []
            self.last_v2x_result = {}
            return {}

        extra_injection = self._resolve_extra_v2x(extra_v2x_messages, vehicles)
        extra_messages = list(extra_injection.get("messages", []))
        extra_vehicles = list(extra_injection.get("vehicles", []))
        mobility_vehicles = list(vehicles)
        for vehicle in extra_vehicles:
            mobility_vehicles.append(
                normalize_vehicle(
                    vehicle,
                    private_veh=True,
                    role=vehicle.get("role", "v2x_attacker"),
                    map_name=getattr(self.config, "carla_map", None),
                )
            )
        for attack in self.ghost_attacks:
            if not self._attack_active(attack, self.current_tick):
                continue
            ghost = normalize_vehicle(
                attack["vehicle"],
                private_veh=True,
                role=attack["vehicle"].get("role", "ghost_attacker"),
                map_name=getattr(self.config, "carla_map", None),
            )
            mobility_vehicles.append(ghost)
            extra_messages.extend(self.cv2x.make_ghost_bsm_messages(self.current_tick, ghost, vehicles))

        result, rows, stream = self.cv2x.sync_tick(
            tick=self.current_tick,
            vehicles=mobility_vehicles,
            messages=self.cv2x.make_pairwise_bsm_messages(
                self.current_tick,
                vehicles,
                extra_messages=extra_messages,
            ),
            phase=phase,
        )
        self.last_v2x_result = result
        self.last_v2x_rows = rows
        self.last_v2x_stream = stream
        self.last_v2x_streams = dict(getattr(self.cv2x, "last_streams_by_receiver", {}) or {})
        return {
            "result": result,
            "rows": rows,
            "stream": stream,
            "streams_by_receiver": self.last_v2x_streams,
        }

    def _resolve_extra_v2x(self, extra_v2x_messages, vehicles):
        if extra_v2x_messages is None:
            return {"messages": [], "vehicles": [], "events": []}
        if callable(extra_v2x_messages):
            extra_v2x_messages = extra_v2x_messages(
                self,
                tick=self.current_tick,
                vehicles=vehicles,
            )
        if isinstance(extra_v2x_messages, dict):
            return {
                "messages": list(extra_v2x_messages.get("messages", extra_v2x_messages.get("v2x_messages", [])) or []),
                "vehicles": list(extra_v2x_messages.get("vehicles", extra_v2x_messages.get("mobility_vehicles", [])) or []),
                "events": list(extra_v2x_messages.get("events", [])) or [],
            }
        return {"messages": list(extra_v2x_messages or []), "vehicles": [], "events": []}

    @staticmethod
    def _attack_active(attack, tick):
        if tick < attack["start_tick"]:
            return False
        end_tick = attack.get("end_tick")
        return end_tick is None or tick <= end_tick

    def _current_v2x_vehicles(self):
        vehicles = []
        for vid, carla_vehicle in self.carla_vehs.items():
            loc = carla_vehicle.get_location()
            vel = carla_vehicle.get_velocity()
            speed = (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z) ** 0.5
            heading = self.get_metsr_rotation(carla_vehicle.get_transform().rotation.yaw)
            vehicles.append(
                {
                    "ID": vid,
                    "role": "controlled",
                    "x": loc.x,
                    "y": -loc.y,
                    "z": loc.z,
                    "speed": speed,
                    "bearing": heading,
                    "private_veh": self.carla_private_flags.get(vid, False),
                    "sensor_type": "cv2x",
                    "map_name": getattr(self.config, "carla_map", None),
                }
            )
        return vehicles
