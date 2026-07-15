"""V2X attack helpers that emit Simu5G-compatible BSM messages."""

import csv
import json
import math
from pathlib import Path

try:
    import carla
except ImportError:
    carla = None

from cosim_utils.C_V2X_manager import normalize_vehicle, vehicle_id


def _to_float(value, default=0.0):
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _same_id(left, right):
    return str(left) == str(right)


def _state_by_id(vehicles, vid):
    for vehicle in vehicles or []:
        if _same_id(vehicle_id(vehicle), vid):
            return vehicle
    return None


def next_available_vehicle_id(existing_ids, count=1):
    numeric_ids = []
    for vid in existing_ids or []:
        try:
            numeric_ids.append(int(vid))
        except (TypeError, ValueError):
            continue
    start = max(numeric_ids, default=0) + 1
    if int(count) <= 1:
        return start
    return list(range(start, start + int(count)))


def vehicle_ids_from_states(vehicles):
    return [vehicle_id(vehicle) for vehicle in vehicles or [] if vehicle_id(vehicle) is not None]


def _heading_forward_xy(heading_deg):
    heading = math.radians(_to_float(heading_deg))
    return math.sin(heading), math.cos(heading)


def _heading_right_xy(heading_deg):
    heading = math.radians(_to_float(heading_deg))
    return math.cos(heading), -math.sin(heading)


def _heading_from_points(first, second, coordinate_frame="carla"):
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    if coordinate_frame == "carla":
        return (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def _location_tuple(point):
    if point is None:
        return None
    if hasattr(point, "x") and hasattr(point, "y"):
        return (float(point.x), float(point.y), _to_float(getattr(point, "z", 0.0)))
    if isinstance(point, dict):
        return (
            _to_float(point.get("x")),
            _to_float(point.get("y")),
            _to_float(point.get("z", 0.0)),
        )
    if len(point) >= 2:
        return (_to_float(point[0]), _to_float(point[1]), _to_float(point[2]) if len(point) > 2 else 0.0)
    return None


def _carla_to_local_state(point, heading_deg, speed_mps, vid, role, map_name):
    x, y, z = _location_tuple(point)
    return {
        "ID": vid,
        "role": role,
        "x": x,
        "y": -y,
        "z": z,
        "speed": max(0.0, _to_float(speed_mps)),
        "bearing": heading_deg % 360.0,
        "private_veh": True,
        "sensor_type": "cv2x",
        "map_name": map_name,
    }


def _local_state(x, y, z, heading_deg, speed_mps, vid, role, map_name):
    return {
        "ID": vid,
        "role": role,
        "x": x,
        "y": y,
        "z": z,
        "speed": max(0.0, _to_float(speed_mps)),
        "bearing": heading_deg % 360.0,
        "private_veh": True,
        "sensor_type": "cv2x",
        "map_name": map_name,
    }


def _current_vehicle_states(cosim_client):
    if cosim_client is None:
        return []
    if hasattr(cosim_client, "_current_v2x_vehicles"):
        return list(cosim_client._current_v2x_vehicles())
    return []


def _controller_route_points(cosim_client, vid):
    controller = getattr(cosim_client, "controllers", {}).get(vid)
    if controller is None:
        return []
    return list(getattr(controller, "_route_points", []) or [])


def _world_map(cosim_client):
    world = getattr(cosim_client, "carla", None)
    if world is None:
        return None
    try:
        return world.get_map()
    except Exception:
        return None


def _filter_receivers(receivers, sender_id):
    return [vehicle for vehicle in receivers or [] if not _same_id(vehicle_id(vehicle), sender_id)]


def _attack_messages(cv2x, tick, sender, receivers, attack_id, attack_type, content=None):
    sender = normalize_vehicle(sender, private_veh=True, role=sender.get("role", "attacker"))
    receivers = _filter_receivers(receivers, vehicle_id(sender))
    messages = cv2x.make_ghost_bsm_messages(
        tick,
        sender,
        receivers,
        attack_id=attack_id,
        attack_type=attack_type,
    )
    if content is not None:
        for message in messages:
            operational = message.get("operational_data") or message.get("operationalData") or {}
            operational["content"] = content
    return messages


class KinematicLimiter:
    """Clamp emitted fake states to simple speed and acceleration bounds."""

    def __init__(self, min_accel_mps2=-5.5, max_accel_mps2=3.5, min_speed_mps=0.0, max_speed_mps=20.0):
        self.min_accel_mps2 = float(min_accel_mps2)
        self.max_accel_mps2 = float(max_accel_mps2)
        self.min_speed_mps = float(min_speed_mps)
        self.max_speed_mps = float(max_speed_mps)
        self._previous = {}

    def apply(self, key, state, dt):
        state = dict(state)
        speed = max(self.min_speed_mps, min(self.max_speed_mps, _to_float(state.get("speed"))))
        previous = self._previous.get(key)
        if previous is not None and dt > 0.0:
            prev_speed = _to_float(previous.get("speed"))
            speed = max(
                prev_speed + self.min_accel_mps2 * dt,
                min(prev_speed + self.max_accel_mps2 * dt, speed),
            )
            max_distance = max(prev_speed, speed, 0.1) * dt + 0.5 * max(abs(self.min_accel_mps2), abs(self.max_accel_mps2)) * dt * dt
            dx = _to_float(state.get("x")) - _to_float(previous.get("x"))
            dy = _to_float(state.get("y")) - _to_float(previous.get("y"))
            distance = math.hypot(dx, dy)
            if distance > max_distance and distance > 1e-6:
                scale = max_distance / distance
                state["x"] = _to_float(previous.get("x")) + dx * scale
                state["y"] = _to_float(previous.get("y")) + dy * scale
        state["speed"] = speed
        if previous is not None and dt > 0.0:
            state["acc"] = (speed - _to_float(previous.get("speed"))) / dt
        self._previous[key] = dict(state)
        return state


class V2XAttack:
    attack_type = "v2x_attack"

    def __init__(self, attack_id=None, start_tick=0, end_tick=None, enabled=True):
        self.attack_id = attack_id or self.attack_type
        self.start_tick = int(start_tick)
        self.end_tick = None if end_tick is None else int(end_tick)
        self.enabled = bool(enabled)

    def active(self, tick):
        if not self.enabled:
            return False
        tick = int(tick)
        return tick >= self.start_tick and (self.end_tick is None or tick <= self.end_tick)

    def generate(self, *, cosim_client, cv2x, tick, vehicles, dt):
        return {"messages": [], "vehicles": [], "events": []}


class StaticGhostVehicleAttack(V2XAttack):
    attack_type = "static_ghost_vehicle"

    def __init__(
        self,
        target_vehicle_id,
        ghost_id=None,
        scenario_type="intersection",
        route_points=None,
        coordinate_frame="carla",
        straight_ahead_distance_m=50.0,
        speed_mps=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_vehicle_id = target_vehicle_id
        self.ghost_id = ghost_id
        self.scenario_type = scenario_type
        self.route_points = list(route_points or [])
        self.coordinate_frame = coordinate_frame
        self.straight_ahead_distance_m = float(straight_ahead_distance_m)
        self.speed_mps = float(speed_mps)
        self._ghost_state = None

    def generate(self, *, cosim_client, cv2x, tick, vehicles, dt):
        if self.ghost_id is None:
            self.ghost_id = next_available_vehicle_id(vehicle_ids_from_states(vehicles))
        if self._ghost_state is None:
            route_points = self.route_points or _controller_route_points(cosim_client, self.target_vehicle_id)
            target = _state_by_id(vehicles, self.target_vehicle_id)
            self._ghost_state = self._build_ghost_state(cosim_client, route_points, target_state=target)
        if self._ghost_state is None:
            return {"messages": [], "vehicles": [], "events": []}
        messages = _attack_messages(
            cv2x,
            tick,
            self._ghost_state,
            vehicles,
            attack_id=self.attack_id,
            attack_type=self.attack_type,
            content="static ghost BSM",
        )
        return {"messages": messages, "vehicles": [self._ghost_state], "events": []}

    def _build_ghost_state(self, cosim_client, route_points, target_state=None):
        points = [_location_tuple(point) for point in route_points]
        points = [point for point in points if point is not None]
        if len(points) < 2:
            return None
        index = self._select_point_index(cosim_client, points, target_state=target_state)
        index = max(0, min(index, len(points) - 2))
        heading = _heading_from_points(points[index], points[index + 1], coordinate_frame=self.coordinate_frame)
        map_name = getattr(getattr(cosim_client, "config", None), "carla_map", None)
        if self.coordinate_frame == "local":
            x, y, z = points[index]
            return _local_state(x, y, z, heading, self.speed_mps, self.ghost_id, "static_ghost_attacker", map_name)
        return _carla_to_local_state(points[index], heading, self.speed_mps, self.ghost_id, "static_ghost_attacker", map_name)

    def _select_point_index(self, cosim_client, points, target_state=None):
        scenario_type = str(self.scenario_type or "").lower()
        if scenario_type == "intersection":
            return self._select_intersection_point(cosim_client, points)
        if scenario_type in {"merging", "merge", "roundabout"}:
            return self._select_placeholder_point(points)
        if scenario_type in {"straight", "road"}:
            return self._select_straight_ahead_point(points, target_state)
        return self._select_middle_point(points)

    def _select_intersection_point(self, cosim_client, points):
        carla_map = _world_map(cosim_client)
        if carla_map is not None and carla is not None and self.coordinate_frame == "carla":
            for index, point in enumerate(points[:-1]):
                loc = carla.Location(x=point[0], y=point[1], z=point[2])
                waypoint = carla_map.get_waypoint(loc, project_to_road=True)
                if waypoint is not None and waypoint.is_junction:
                    return index
        return self._select_middle_point(points)

    def _select_placeholder_point(self, points):
        return self._select_middle_point(points)

    def _select_straight_ahead_point(self, points, target_state):
        if not target_state:
            return self._select_middle_point(points)

        if self.coordinate_frame == "carla":
            target = (
                _to_float(target_state.get("x")),
                -_to_float(target_state.get("y")),
                _to_float(target_state.get("z", 0.0)),
            )
        else:
            target = (
                _to_float(target_state.get("x")),
                _to_float(target_state.get("y")),
                _to_float(target_state.get("z", 0.0)),
            )

        closest_index = min(
            range(len(points)),
            key=lambda index: math.hypot(points[index][0] - target[0], points[index][1] - target[1]),
        )
        distance = 0.0
        desired_distance = max(0.0, self.straight_ahead_distance_m)
        for index in range(closest_index, len(points) - 1):
            current = points[index]
            nxt = points[index + 1]
            distance += math.hypot(nxt[0] - current[0], nxt[1] - current[1])
            if distance >= desired_distance:
                return index + 1
        return max(0, len(points) - 2)

    def _select_middle_point(self, points):
        start = int(0.3 * (len(points) - 1))
        end = max(start, int(0.7 * (len(points) - 1)))
        return (start + end) // 2


class ObstacleGhostVehicleAttack(V2XAttack):
    attack_type = "obstacle_ghost_vehicle"

    def __init__(self, target_vehicle_id, ghost_id=None, lead_time_s=0.0, base_distance_m=13.0, ghost_speed_mps=2.0, **kwargs):
        super().__init__(**kwargs)
        self.target_vehicle_id = target_vehicle_id
        self.ghost_id = ghost_id
        self.lead_time_s = float(lead_time_s)
        self.base_distance_m = float(base_distance_m)
        self.ghost_speed_mps = float(ghost_speed_mps)

    def generate(self, *, cosim_client, cv2x, tick, vehicles, dt):
        target = _state_by_id(vehicles, self.target_vehicle_id)
        if target is None:
            return {"messages": [], "vehicles": [], "events": []}
        if self.ghost_id is None:
            self.ghost_id = next_available_vehicle_id(vehicle_ids_from_states(vehicles))
        heading = _to_float(target.get("bearing", target.get("heading_deg")))
        ghost = self._ghost_state_from_target(target, heading)
        messages = _attack_messages(
            cv2x,
            tick,
            ghost,
            vehicles,
            attack_id=self.attack_id,
            attack_type=self.attack_type,
            content="moving obstacle ghost BSM",
        )
        return {"messages": messages, "vehicles": [ghost], "events": []}

    def _ghost_state_from_target(self, target, heading):
        dx, dy = _heading_forward_xy(heading)
        distance = self.base_distance_m
        return _local_state(
            _to_float(target.get("x")) + dx * distance,
            _to_float(target.get("y")) + dy * distance,
            _to_float(target.get("z", 0.0)),
            heading,
            self.ghost_speed_mps,
            self.ghost_id,
            "obstacle_ghost_attacker",
            target.get("map_name"),
        )


class FalseInformationInjectionAttack(V2XAttack):
    attack_type = "false_information_injection"

    def __init__(
        self,
        attacker_vehicle_id,
        fixed_offset_xy=(0.0, 0.0),
        forward_offset_m=0.0,
        lateral_offset_m=0.0,
        drift_rate_xy_mps=(0.0, 0.0),
        speed_bias_mps=0.0,
        limiter=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attacker_vehicle_id = attacker_vehicle_id
        self.fixed_offset_xy = tuple(fixed_offset_xy)
        self.forward_offset_m = float(forward_offset_m)
        self.lateral_offset_m = float(lateral_offset_m)
        self.drift_rate_xy_mps = tuple(drift_rate_xy_mps)
        self.speed_bias_mps = float(speed_bias_mps)
        self.limiter = limiter or KinematicLimiter()

    def generate(self, *, cosim_client, cv2x, tick, vehicles, dt):
        attacker = _state_by_id(vehicles, self.attacker_vehicle_id)
        if attacker is None:
            return {"messages": [], "vehicles": [], "events": []}
        heading = _to_float(attacker.get("bearing", attacker.get("heading_deg")))
        fx, fy = _heading_forward_xy(heading)
        rx, ry = _heading_right_xy(heading)
        elapsed_s = max(0.0, (int(tick) - self.start_tick) * float(dt))
        desired = dict(attacker)
        desired["role"] = "false_info_attacker"
        desired["x"] = (
            _to_float(attacker.get("x"))
            + _to_float(self.fixed_offset_xy[0])
            + fx * self.forward_offset_m
            + rx * self.lateral_offset_m
            + _to_float(self.drift_rate_xy_mps[0]) * elapsed_s
        )
        desired["y"] = (
            _to_float(attacker.get("y"))
            + _to_float(self.fixed_offset_xy[1])
            + fy * self.forward_offset_m
            + ry * self.lateral_offset_m
            + _to_float(self.drift_rate_xy_mps[1]) * elapsed_s
        )
        desired["speed"] = _to_float(attacker.get("speed", attacker.get("speed_mps"))) + self.speed_bias_mps
        desired = self.limiter.apply(f"false_info:{self.attacker_vehicle_id}", desired, float(dt))
        messages = _attack_messages(
            cv2x,
            tick,
            desired,
            vehicles,
            attack_id=self.attack_id,
            attack_type=self.attack_type,
            content="falsified self BSM",
        )
        return {"messages": messages, "vehicles": [], "events": []}


class SybilAttack(V2XAttack):
    attack_type = "sybil"

    def __init__(
        self,
        attacker_vehicle_id,
        sybil_count=5,
        offsets_xy=None,
        base_sybil_id=None,
        limiter=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attacker_vehicle_id = attacker_vehicle_id
        self.sybil_count = int(sybil_count)
        self.offsets_xy = list(offsets_xy or [(10.0, 0.0), (20.0, 0.0), (-10.0, 0.0), (0.0, -5.0), (10.0, -5.0)])
        self.base_sybil_id = None if base_sybil_id is None else int(base_sybil_id)
        self.limiter = limiter or KinematicLimiter()

    def generate(self, *, cosim_client, cv2x, tick, vehicles, dt):
        attacker = _state_by_id(vehicles, self.attacker_vehicle_id)
        if attacker is None:
            return {"messages": [], "vehicles": [], "events": []}
        if self.base_sybil_id is None:
            sybil_ids = next_available_vehicle_id(
                vehicle_ids_from_states(vehicles),
                count=self.sybil_count,
            )
            self.base_sybil_id = sybil_ids[0] if isinstance(sybil_ids, list) else sybil_ids
        receivers = [vehicle for vehicle in vehicles if not _same_id(vehicle_id(vehicle), self.attacker_vehicle_id)]
        messages = []
        fake_vehicles = []
        for index, offset in enumerate(self.offsets_xy[: self.sybil_count], start=1):
            fake_id = self.base_sybil_id + index - 1
            fake = dict(attacker)
            fake["ID"] = fake_id
            fake["role"] = "sybil_attacker"
            fake["x"] = _to_float(attacker.get("x")) + _to_float(offset[0])
            fake["y"] = _to_float(attacker.get("y")) + _to_float(offset[1])
            fake = self.limiter.apply(f"sybil:{fake_id}", fake, float(dt))
            fake_vehicles.append(fake)
            messages.extend(
                _attack_messages(
                    cv2x,
                    tick,
                    fake,
                    receivers,
                    attack_id=self.attack_id,
                    attack_type=self.attack_type,
                    content=f"sybil BSM {index}",
                )
            )
        return {"messages": messages, "vehicles": fake_vehicles, "events": []}


class ReplayAttack(V2XAttack):
    attack_type = "replay"

    def __init__(
        self,
        replay_name=None,
        replay_path=None,
        records=None,
        replay_sender_id=None,
        loop=True,
        use_limiter=True,
        limiter=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.replay_name = replay_name
        self.replay_path = replay_path
        self.records = list(records or self._load_records(replay_path))
        self.replay_sender_id = replay_sender_id
        self.loop = bool(loop)
        self.limiter = (limiter or KinematicLimiter()) if use_limiter else None

    def generate(self, *, cosim_client, cv2x, tick, vehicles, dt):
        record = self._record_for_tick(tick)
        if record is None:
            return {"messages": [], "vehicles": [], "events": []}
        if self.replay_sender_id is None:
            self.replay_sender_id = next_available_vehicle_id(vehicle_ids_from_states(vehicles))
        sender = self._record_to_vehicle(record, cosim_client)
        if self.limiter is not None:
            sender = self.limiter.apply(f"replay:{self.replay_sender_id}", sender, float(dt))
        messages = _attack_messages(
            cv2x,
            tick,
            sender,
            vehicles,
            attack_id=self.attack_id,
            attack_type=self.attack_type,
            content=f"replayed BSM {self.replay_name or ''}".strip(),
        )
        return {"messages": messages, "vehicles": [sender], "events": []}

    def _record_for_tick(self, tick):
        if not self.records:
            return None
        index = int(tick) - self.start_tick
        if index < 0:
            return None
        if self.loop:
            return self.records[index % len(self.records)]
        if index >= len(self.records):
            return None
        return self.records[index]

    def _record_to_vehicle(self, record, cosim_client):
        map_name = getattr(getattr(cosim_client, "config", None), "carla_map", None)
        operational = record.get("operational_data") or record.get("operationalData") or {}
        local_position = operational.get("localPositionM") or {}
        recorded_id = record.get("ID", record.get("vehicle_id", record.get("vid")))
        sender_id = self.replay_sender_id if self.replay_sender_id is not None else recorded_id
        sender = {
            "ID": sender_id,
            "role": "replay_attacker",
            "x": record.get("x", record.get("true_x", local_position.get("x"))),
            "y": record.get("y", record.get("true_y", local_position.get("y"))),
            "z": record.get("z", local_position.get("z", 0.0)),
            "speed": record.get(
                "speed",
                record.get("speed_mps", record.get("velocity", operational.get("speedMps", 0.0))),
            ),
            "bearing": record.get(
                "bearing",
                record.get("heading_deg", record.get("heading", operational.get("headingDeg", 0.0))),
            ),
            "latitude": record.get("latitude", record.get("bsm_latitude_deg")),
            "longitude": record.get("longitude", record.get("bsm_longitude_deg")),
            "private_veh": True,
            "sensor_type": "cv2x",
            "map_name": record.get("map_name", map_name),
        }
        return sender

    def _load_records(self, replay_path):
        if replay_path is None:
            return []
        path = Path(replay_path)
        if not path.exists():
            return []
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else data.get("records", [])
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as file_obj:
                return list(csv.DictReader(file_obj))
        return []


def assign_attack_vehicle_ids(attack, existing_vehicle_ids):
    """Assign compact fake sender IDs for one attack or a list of attacks."""
    if isinstance(attack, (list, tuple)):
        used_ids = list(existing_vehicle_ids)
        for item in attack:
            _assign_one_attack_vehicle_id(item, used_ids)
        return attack
    return _assign_one_attack_vehicle_id(attack, list(existing_vehicle_ids))


def _assign_one_attack_vehicle_id(attack, used_ids):
    if isinstance(attack, FalseInformationInjectionAttack):
        return attack
    if isinstance(attack, (StaticGhostVehicleAttack, ObstacleGhostVehicleAttack)):
        if attack.ghost_id is None:
            attack.ghost_id = next_available_vehicle_id(used_ids)
            used_ids.append(attack.ghost_id)
        return attack
    if isinstance(attack, ReplayAttack):
        if attack.replay_sender_id is None:
            attack.replay_sender_id = next_available_vehicle_id(used_ids)
            used_ids.append(attack.replay_sender_id)
        return attack
    if isinstance(attack, SybilAttack):
        if attack.base_sybil_id is None:
            ids = next_available_vehicle_id(used_ids, count=attack.sybil_count)
            ids = ids if isinstance(ids, list) else [ids]
            attack.base_sybil_id = ids[0]
            used_ids.extend(ids)
        return attack
    return attack


class V2XAttackManager:
    def __init__(self, attacks=None):
        self.attacks = list(attacks or [])
        self.last_injection = {"messages": [], "vehicles": [], "events": []}

    def add(self, attack):
        self.attacks.append(attack)
        return attack

    def __call__(self, cosim_client, tick=None, vehicles=None):
        return self.injection_for_tick(cosim_client, tick=tick, vehicles=vehicles)

    def injection_for_tick(self, cosim_client, tick=None, vehicles=None):
        tick = int(getattr(cosim_client, "current_tick", 0) if tick is None else tick)
        vehicles = list(_current_vehicle_states(cosim_client) if vehicles is None else vehicles)
        cv2x = getattr(cosim_client, "cv2x")
        dt = float(getattr(cosim_client, "dt", getattr(getattr(cosim_client, "config", None), "sim_step_size", 0.1)))
        injection = {"messages": [], "vehicles": [], "events": []}
        for attack in self.attacks:
            if not attack.active(tick):
                continue
            payload = attack.generate(cosim_client=cosim_client, cv2x=cv2x, tick=tick, vehicles=vehicles, dt=dt)
            injection["messages"].extend(payload.get("messages", []))
            injection["vehicles"].extend(payload.get("vehicles", []))
            injection["events"].extend(payload.get("events", []))
        self.last_injection = injection
        return injection

    def messages_for_tick(self, cosim_client, tick=None, vehicles=None):
        return self.injection_for_tick(cosim_client, tick=tick, vehicles=vehicles)["messages"]

    def step(self, cosim_client, phase="attack"):
        return cosim_client.step(extra_v2x_messages=self, phase=phase)
