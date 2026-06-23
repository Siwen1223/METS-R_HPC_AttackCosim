"""CARLA + METS-R + Simu5G/VEINS co-simulation client."""

from pathlib import Path

from clients.CoSimClient import CoSimClient
from cosim_utils.C_V2X_manager import CV2XManager, normalize_vehicle
from cosim_utils.Sensor_manager import SensorManager
from cosim_utils.v2v_controller_carla import V2VControllerCarla


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
        self.last_v2x_rows = []
        self.last_v2x_result = {}
        self.enable_v2x = enable_v2x
        self.ghost_attacks = []

        self.sensor_manager = SensorManager(
            self.carla,
            self._vehicle_for_sensor,
            output_path=getattr(config, "sensor_output_path", "_out"),
            camera_layout=camera_layout or getattr(config, "camera_layout", "front"),
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

        super().step()

        self._ensure_controllers()
        self._sync_controller_routes()
        done_vids = self._finish_completed_routes()
        v2x_payload = self._sync_v2x(extra_v2x_messages=extra_v2x_messages, phase=phase)
        self.sensor_manager.collect_sensor_data(
            output_path=getattr(self.config, "sensor_output_path", None)
        )
        self.step_index += 1
        return {
            "tick": self.current_tick,
            "done_vids": done_vids,
            "v2x": v2x_payload,
            "controllers": self.controllers,
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

    def collect_sensor_data(self, output_path=None):
        self.sensor_manager.collect_sensor_data(output_path=output_path)

    def save_sensor_data(self, vid, output_path=None):
        self.sensor_manager.save_sensor_data(vid, output_path=output_path)

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
            "enable_debug_draw": bool(getattr(self.config, "enable_debug_draw", False)),
            "v2v_position_mode": getattr(self.config, "v2v_position_mode", "local"),
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
            route_ids = self.carla_route.get(vid, [])
            if route_ids and controller.set_route_from_metsr_route(
                route_ids,
                stop_waypoint_creation=True,
            ):
                self.route_synced[vid] = True

    def _apply_controller_controls(self):
        for vid, controller in list(self.controllers.items()):
            carla_vehicle = self.carla_vehs.get(vid)
            if carla_vehicle is None or not self.carla_entered.get(vid, False):
                continue
            control = controller.run_step(self.last_v2x_stream, dt=self.dt)
            carla_vehicle.apply_control(control)

    def _finish_completed_routes(self):
        done_vids = []
        for vid, controller in list(self.controllers.items()):
            if not self.carla_entered.get(vid, False):
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
            self.last_v2x_rows = []
            self.last_v2x_result = {}
            return {}

        extra_messages = list(extra_v2x_messages or [])
        mobility_vehicles = list(vehicles)
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
        return {"result": result, "rows": rows, "stream": stream}

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
