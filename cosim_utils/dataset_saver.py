import csv
import json
import math
from pathlib import Path


class DatasetSaver:
    """Save one co-simulation run into the V2V attack dataset layout."""

    def __init__(self, dataset_root, meta, attack, run_id=None, sensor_every_n=None):
        self.dataset_root = Path(dataset_root)
        self.runs_dir = self.dataset_root / "runs"
        self.sensor_every_n = sensor_every_n
        self.run_id = run_id or self._next_run_id()
        self.run_dir = self.runs_dir / self.run_id
        self.bsm_dir = self.run_dir / "bsm"
        self.attack_dir = self.run_dir / "attack"
        self.sensors_dir = self.run_dir / "sensors"
        self.vehicle_state_dir = self.run_dir / "vehicle_state"
        self.events_path = self.run_dir / "events.log"
        self.meta_path = self.run_dir / "meta.json"
        self.attack_path = self.run_dir / "attack.json"

        self._ensure_dirs()
        self.meta = dict(meta)
        self.meta.setdefault("run_id", self.run_id)
        self._normalize_meta_paths()
        self.attack = dict(attack) if attack is not None else {"attack_type": "none"}
        self._attack_tick_rows = {}
        self._attack_tick_vehicle_ids = set()
        self._high_impact_vehicle_ids = set()
        self._init_vehicle_impact_fields()
        self._write_json(self.meta_path, self.meta)
        self._write_json(self.attack_path, self.attack)

        self._events_fp = self.events_path.open("a", encoding="utf-8")
        self._bsm_summary_fp = (self.bsm_dir / "summary.jsonl").open("a", encoding="utf-8")
        self._bsm_vehicle_fps = {}
        self._trajectory_fps = {}
        self._trajectory_writers = {}
        self._control_fps = {}
        self._control_writers = {}
        self._valid_vehicle_state_seen = set()
        self._seen_collision_events = 0

    def log_event(self, sim_time, message):
        self._events_fp.write(f"[{float(sim_time):.2f}s] {message}\n")
        self._events_fp.flush()

    def record_step(self, tick, sim_time, cosim_client, step_result=None, vehicle_ids=None):
        step_result = step_result or {}
        vehicle_ids = list(vehicle_ids or cosim_client.carla_vehs.keys())
        if self._ensure_vehicle_impact_ids(vehicle_ids):
            self._write_json(self.attack_path, self.attack)
        self.record_vehicle_state(tick, sim_time, cosim_client, vehicle_ids)
        self.record_controls(tick, sim_time, step_result.get("controls", {}))
        bsm_rows = (step_result.get("v2x") or {}).get("rows", [])
        self.record_bsm_rows(tick, sim_time, bsm_rows)
        self.record_attack_impact(tick, sim_time, cosim_client, step_result, vehicle_ids, bsm_rows)
        self.save_sensors(tick, cosim_client)

    def record_vehicle_state(self, tick, sim_time, cosim_client, vehicle_ids):
        for vid in vehicle_ids:
            vehicle = cosim_client.carla_vehs.get(vid)
            if vehicle is None or not cosim_client.carla_entered.get(vid, False):
                continue
            loc = vehicle.get_location()
            yaw = vehicle.get_transform().rotation.yaw
            vel = vehicle.get_velocity()
            speed = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
            if self._is_unsynced_initial_vehicle_state(vid, loc, yaw, speed):
                continue
            self._valid_vehicle_state_seen.add(vid)
            writer = self._trajectory_writer(vid)
            writer.writerow([tick, sim_time, vid, loc.x, loc.y, loc.z, yaw, speed])

    def record_controls(self, tick, sim_time, controls):
        for vid, control in (controls or {}).items():
            if control is None:
                continue
            if vid not in self._valid_vehicle_state_seen:
                continue
            writer = self._control_writer(vid)
            writer.writerow([
                tick,
                sim_time,
                vid,
                control.throttle,
                control.brake,
                control.steer,
                int(control.hand_brake),
                int(control.reverse),
            ])

    def record_bsm_rows(self, tick, sim_time, rows):
        for row in rows or []:
            core_data = row.get("bsm_core_data") or {}
            summary = {
                "tick": tick,
                "source_tick": row.get("tick"),
                "sim_time": sim_time,
                "message_id": row.get("message_id"),
                "sender_id": row.get("sender_id"),
                "receiver_id": row.get("receiver_id"),
                "message_count": row.get("message_count"),
                "bsm_core_data": core_data,
                "transport_envelope": row.get("transport_envelope") or {},
                "link_result": row.get("link_result") or {},
                "attacked": row.get("attacked", False),
                "attack_id": row.get("attack_id", ""),
                "attack_type": row.get("attack_type", ""),
                "phase": row.get("phase", ""),
            }
            self._write_jsonl(self._bsm_summary_fp, summary)

            if not (row.get("link_result") or {}).get("delivered", row.get("delivered")):
                continue
            receiver_id = row.get("receiver_id")
            if receiver_id is None:
                continue
            vehicle_record = {
                "tick": tick,
                "source_tick": row.get("tick"),
                "sim_time": sim_time,
                "sender_id": row.get("sender_id"),
                "receiver_id": receiver_id,
                "bsm_core_data": core_data,
            }
            self._write_jsonl(self._bsm_vehicle_fp(receiver_id), vehicle_record)

    def record_bsm(self, tick, sim_time, data_stream):
        if isinstance(data_stream, list):
            self.record_bsm_rows(tick, sim_time, data_stream)

    def record_attack_impact(self, tick, sim_time, cosim_client, step_result, vehicle_ids, bsm_rows=None):
        tick_impacts = {}
        attack_type = self.attack.get("attack_type", "none")
        if attack_type in {"none", "no_attack"}:
            self._record_attack_tick_rows(tick, sim_time, vehicle_ids, bsm_rows, tick_impacts)
            return

        for vid, controller in (step_result.get("controllers") or {}).items():
            debug_state = controller.get_last_debug_state() if hasattr(controller, "get_last_debug_state") else {}
            if debug_state.get("attack_influenced_speed"):
                vehicle_key = self._vehicle_key(vid)
                tick_impacts[vehicle_key] = "medium"
                self._set_vehicle_impact_level(
                    vid,
                    "medium",
                    tick,
                    sim_time,
                    f"Vehicle {vid} reduced target speed due to attacked BSM ({debug_state.get('attack_influence_source', '')})",
                )

        for collision in self._new_collision_events(cosim_client):
            if not self._attack_started(sim_time):
                continue
            other_actor_type = collision.get("other_actor_type", "")
            if not str(other_actor_type).startswith("vehicle."):
                continue
            for vid in self._collision_vehicle_ids(cosim_client, collision):
                self._high_impact_vehicle_ids.add(self._vehicle_key(vid))
                self._set_vehicle_impact_level(
                    vid,
                    "high",
                    tick,
                    sim_time,
                    "Vehicle-to-vehicle collision detected by CARLA collision sensor "
                    f"for vehicle {vid} with {other_actor_type}",
                )
        self._record_attack_tick_rows(tick, sim_time, vehicle_ids, bsm_rows, tick_impacts)

    def save_sensors(self, tick, cosim_client=None):
        if cosim_client is None:
            cosim_client = tick
            tick = None
        cosim_client.collect_sensor_data(output_path=str(self.sensors_dir), tick=tick)

    def finalize(self, duration_sec=None):
        if duration_sec is not None:
            self.meta["duration_sec"] = duration_sec
            self._write_json(self.meta_path, self.meta)
        self._write_json(self.attack_path, self.attack)
        self._write_attack_tick_csvs()
        for fp in (
            [self._events_fp, self._bsm_summary_fp]
            + list(self._bsm_vehicle_fps.values())
            + list(self._trajectory_fps.values())
            + list(self._control_fps.values())
        ):
            try:
                fp.flush()
                fp.close()
            except Exception:
                pass

    def _trajectory_writer(self, vid):
        if vid not in self._trajectory_writers:
            folder = self.vehicle_state_dir / f"vehicle_{vid}"
            folder.mkdir(parents=True, exist_ok=True)
            fp = (folder / "trajectory.csv").open("w", newline="", encoding="utf-8")
            writer = csv.writer(fp)
            writer.writerow(["tick", "sim_time", "vid", "x", "y", "z", "yaw", "speed_mps"])
            self._trajectory_fps[vid] = fp
            self._trajectory_writers[vid] = writer
        return self._trajectory_writers[vid]

    def _control_writer(self, vid):
        if vid not in self._control_writers:
            folder = self.vehicle_state_dir / f"vehicle_{vid}"
            folder.mkdir(parents=True, exist_ok=True)
            fp = (folder / "control.csv").open("w", newline="", encoding="utf-8")
            writer = csv.writer(fp)
            writer.writerow(["tick", "sim_time", "vid", "throttle", "brake", "steer", "hand_brake", "reverse"])
            self._control_fps[vid] = fp
            self._control_writers[vid] = writer
        return self._control_writers[vid]

    def _is_unsynced_initial_vehicle_state(self, vid, loc, yaw, speed):
        if vid in self._valid_vehicle_state_seen:
            return False
        eps = 1e-6
        return (
            abs(float(loc.x)) <= eps
            and abs(float(loc.y)) <= eps
            and abs(float(loc.z)) <= eps
            and abs(float(yaw)) <= eps
            and abs(float(speed)) <= eps
        )

    def _bsm_vehicle_fp(self, vid):
        if vid not in self._bsm_vehicle_fps:
            folder = self.bsm_dir / f"vehicle_{vid}"
            folder.mkdir(parents=True, exist_ok=True)
            self._bsm_vehicle_fps[vid] = (folder / "bsm.jsonl").open("a", encoding="utf-8")
        return self._bsm_vehicle_fps[vid]

    def _ensure_dirs(self):
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        for path in (
            self.runs_dir,
            self.run_dir,
            self.bsm_dir,
            self.attack_dir,
            self.sensors_dir,
            self.vehicle_state_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _next_run_id(self):
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        run_numbers = []
        for item in self.runs_dir.iterdir():
            if item.is_dir() and item.name.startswith("run_"):
                try:
                    run_numbers.append(int(item.name.split("_", 1)[1]))
                except ValueError:
                    continue
        return f"run_{max(run_numbers, default=0) + 1:06d}"

    def _normalize_meta_paths(self):
        scenario = self.meta.get("scenario")
        if not scenario:
            return
        scenario_path = Path(str(scenario))
        try:
            relative = scenario_path.resolve().relative_to(self.dataset_root.resolve())
        except (OSError, ValueError):
            return
        self.meta["scenario"] = relative.as_posix()

    def _set_vehicle_impact_level(self, vid, level, tick, sim_time, reason):
        vehicle_key = self._vehicle_key(vid)
        self._ensure_vehicle_impact_ids([vehicle_key])
        if level == "high":
            self._high_impact_vehicle_ids.add(vehicle_key)
        current = self.attack["impact_level"].get(vehicle_key, "N/A")
        if current != "N/A" and self._impact_rank(level) <= self._impact_rank(current):
            return
        evidence = {
            "vehicle_id": self._vehicle_id_value(vehicle_key),
            "tick": int(tick),
            "sim_time": float(sim_time),
            "level": level,
            "reason": reason,
        }
        self.attack["impact_level"][vehicle_key] = level
        self.attack["impact_evidence"].append(evidence)
        self.log_event(sim_time, f"vehicle {vehicle_key} attack impact upgraded to {level}: {reason}")
        self._write_json(self.attack_path, self.attack)

    def _record_attack_tick_rows(self, tick, sim_time, vehicle_ids, bsm_rows, tick_impacts):
        attacked_senders_by_receiver = self._attacked_senders_by_receiver(bsm_rows)
        self._ensure_vehicle_impact_ids(vehicle_ids)
        for vid in vehicle_ids or []:
            vehicle_key = self._vehicle_key(vid)
            received_senders = attacked_senders_by_receiver.get(vehicle_key, [])
            if not received_senders:
                impact_level = "N/A"
            elif vehicle_key in self._high_impact_vehicle_ids:
                self._set_vehicle_min_impact_level(vehicle_key, "low")
                impact_level = "high"
            else:
                self._set_vehicle_min_impact_level(vehicle_key, "low")
                impact_level = tick_impacts.get(vehicle_key, "low")
            row = {
                "tick": int(tick),
                "sim_time": float(sim_time),
                "vehicle_id": self._vehicle_id_value(vehicle_key),
                "received_attack_message": "yes" if received_senders else "no",
                "attack_sender_id": ";".join(str(sender_id) for sender_id in received_senders) if received_senders else "none",
                "impact_level": impact_level,
            }
            self._attack_tick_rows.setdefault(vehicle_key, []).append(row)

    def _attacked_senders_by_receiver(self, bsm_rows):
        senders_by_receiver = {}
        for row in bsm_rows or []:
            if not row.get("attacked", False):
                continue
            if not (row.get("link_result") or {}).get("delivered", row.get("delivered")):
                continue
            receiver_id = row.get("receiver_id")
            if receiver_id is None:
                continue
            receiver_key = self._vehicle_key(receiver_id)
            sender_id = row.get("sender_id")
            sender_values = senders_by_receiver.setdefault(receiver_key, [])
            if sender_id is not None and sender_id not in sender_values:
                sender_values.append(sender_id)
        return senders_by_receiver

    def _write_attack_tick_csvs(self):
        for vehicle_key in sorted(self._attack_tick_vehicle_ids, key=self._vehicle_sort_key):
            folder = self.attack_dir / f"vehicle_{vehicle_key}"
            folder.mkdir(parents=True, exist_ok=True)
            csv_path = folder / "attack_impact.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as fp:
                fieldnames = [
                    "tick",
                    "sim_time",
                    "vehicle_id",
                    "received_attack_message",
                    "attack_sender_id",
                    "impact_level",
                ]
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                for row in self._attack_tick_rows.get(vehicle_key, []):
                    output_row = dict(row)
                    if output_row.get("received_attack_message") == "no":
                        output_row["impact_level"] = "N/A"
                    elif vehicle_key in self._high_impact_vehicle_ids:
                        output_row["impact_level"] = "high"
                    writer.writerow(output_row)

    def _init_vehicle_impact_fields(self):
        existing_levels = self.attack.get("impact_level", "N/A")
        if isinstance(existing_levels, dict):
            impact_levels = {self._vehicle_key(vid): level for vid, level in existing_levels.items()}
        else:
            impact_levels = {
                self._vehicle_key(vid): existing_levels
                for vid in self._initial_vehicle_ids()
            }
        self.attack["impact_level"] = impact_levels
        self._ensure_vehicle_impact_ids(self._initial_vehicle_ids())

        evidence = self.attack.get("impact_evidence", [])
        if not isinstance(evidence, list):
            evidence = []
        self.attack["impact_evidence"] = [
            dict(item) for item in evidence if isinstance(item, dict)
        ]

    def _set_vehicle_min_impact_level(self, vid, level):
        vehicle_key = self._vehicle_key(vid)
        self._ensure_vehicle_impact_ids([vehicle_key])
        current = self.attack["impact_level"].get(vehicle_key, "N/A")
        if current == "N/A" or self._impact_rank(level) > self._impact_rank(current):
            self.attack["impact_level"][vehicle_key] = level

    def _initial_vehicle_ids(self):
        vehicle_ids = []
        vehicle_route = self.meta.get("vehicle_route") or {}
        if isinstance(vehicle_route, dict):
            vehicle_ids.extend(vehicle_route.keys())
        elif isinstance(vehicle_route, (list, tuple, set)):
            vehicle_ids.extend(vehicle_route)

        attack_receiver = self.attack.get("attack_receiver")
        if isinstance(attack_receiver, (list, tuple, set)):
            vehicle_ids.extend(attack_receiver)

        attack_target = self.attack.get("attack_target")
        if attack_target not in (None, "all", "intersection"):
            vehicle_ids.append(attack_target)

        return list(dict.fromkeys(self._vehicle_key(vid) for vid in vehicle_ids))

    def _ensure_vehicle_impact_ids(self, vehicle_ids):
        impact_levels = self.attack.setdefault("impact_level", {})
        if not isinstance(impact_levels, dict):
            impact_levels = {}
            self.attack["impact_level"] = impact_levels
        changed = False
        for vid in vehicle_ids or []:
            vehicle_key = self._vehicle_key(vid)
            self._attack_tick_vehicle_ids.add(vehicle_key)
            if vehicle_key not in impact_levels:
                impact_levels[vehicle_key] = "N/A"
                changed = True
        return changed

    def _collision_vehicle_ids(self, cosim_client, collision):
        vehicle_ids = []
        primary_vid = collision.get("vid")
        if primary_vid is not None:
            vehicle_ids.append(primary_vid)

        other_actor_id = collision.get("other_actor_id")
        if other_actor_id is not None:
            for vid, actor in getattr(cosim_client, "carla_vehs", {}).items():
                if getattr(actor, "id", None) == other_actor_id:
                    vehicle_ids.append(vid)
                    break

        return list(dict.fromkeys(self._vehicle_key(vid) for vid in vehicle_ids))

    @staticmethod
    def _vehicle_key(vid):
        return str(vid)

    @staticmethod
    def _vehicle_id_value(vehicle_key):
        try:
            return int(vehicle_key)
        except (TypeError, ValueError):
            return vehicle_key

    @staticmethod
    def _vehicle_sort_key(vehicle_key):
        try:
            return (0, int(vehicle_key))
        except (TypeError, ValueError):
            return (1, str(vehicle_key))

    @staticmethod
    def _impact_rank(level):
        return {"low": 0, "medium": 1, "high": 2}.get(level, 0)

    def _new_collision_events(self, cosim_client):
        if not hasattr(cosim_client, "get_collision_events"):
            return []
        events = cosim_client.get_collision_events()
        new_events = events[self._seen_collision_events:]
        self._seen_collision_events = len(events)
        return new_events

    def _attack_started(self, sim_time):
        start_time = self.attack.get("start_time")
        if start_time is None:
            return True
        return float(sim_time) >= float(start_time)

    @staticmethod
    def _write_json(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _write_jsonl(fp, data):
        fp.write(json.dumps(data, separators=(",", ":")) + "\n")
        fp.flush()
