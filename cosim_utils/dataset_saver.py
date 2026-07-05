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
        self.sensors_dir = self.run_dir / "sensors"
        self.vehicle_state_dir = self.run_dir / "vehicle_state"
        self.events_path = self.run_dir / "events.log"
        self.meta_path = self.run_dir / "meta.json"
        self.attack_path = self.run_dir / "attack.json"

        self._ensure_dirs()
        self.meta = dict(meta)
        self.meta.setdefault("run_id", self.run_id)
        self.attack = dict(attack) if attack is not None else {"attack_type": "none"}
        self.attack.setdefault("impact_level", "low")
        self.attack.setdefault("impact_evidence", [])
        self._write_json(self.meta_path, self.meta)
        self._write_json(self.attack_path, self.attack)

        self._events_fp = self.events_path.open("a", encoding="utf-8")
        self._bsm_summary_fp = (self.bsm_dir / "summary.jsonl").open("a", encoding="utf-8")
        self._bsm_vehicle_fps = {}
        self._trajectory_fps = {}
        self._trajectory_writers = {}
        self._control_fps = {}
        self._control_writers = {}
        self._seen_collision_events = 0

    def log_event(self, sim_time, message):
        self._events_fp.write(f"[{float(sim_time):.2f}s] {message}\n")
        self._events_fp.flush()

    def record_step(self, tick, sim_time, cosim_client, step_result=None, vehicle_ids=None):
        step_result = step_result or {}
        vehicle_ids = list(vehicle_ids or cosim_client.carla_vehs.keys())
        self.record_vehicle_state(tick, sim_time, cosim_client, vehicle_ids)
        self.record_controls(tick, sim_time, step_result.get("controls", {}))
        self.record_bsm_rows(tick, sim_time, (step_result.get("v2x") or {}).get("rows", []))
        self.record_attack_impact(tick, sim_time, cosim_client, step_result, vehicle_ids)
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
            writer = self._trajectory_writer(vid)
            writer.writerow([tick, sim_time, vid, loc.x, loc.y, loc.z, yaw, speed])

    def record_controls(self, tick, sim_time, controls):
        for vid, control in (controls or {}).items():
            if control is None:
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

    def record_attack_impact(self, tick, sim_time, cosim_client, step_result, vehicle_ids):
        if self.attack.get("attack_type", "none") in {"none", "no_attack"}:
            return

        for vid, controller in (step_result.get("controllers") or {}).items():
            debug_state = controller.get_last_debug_state() if hasattr(controller, "get_last_debug_state") else {}
            if debug_state.get("attack_influenced_speed"):
                self._set_impact_level(
                    "medium",
                    tick,
                    sim_time,
                    f"Vehicle {vid} reduced target speed due to attacked BSM ({debug_state.get('attack_influence_source', '')})",
                )
                break

        for collision in self._new_collision_events(cosim_client):
            if not self._attack_started(sim_time):
                continue
            other_actor_type = collision.get("other_actor_type", "")
            if not str(other_actor_type).startswith("vehicle."):
                continue
            self._set_impact_level(
                "high",
                tick,
                sim_time,
                "Vehicle-to-vehicle collision detected by CARLA collision sensor "
                f"for vehicle {collision.get('vid')} with {other_actor_type}",
            )
            break

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

    def _set_impact_level(self, level, tick, sim_time, reason):
        current = self.attack.get("impact_level", "low")
        if self._impact_rank(level) <= self._impact_rank(current):
            return
        evidence = {
            "tick": int(tick),
            "sim_time": float(sim_time),
            "level": level,
            "reason": reason,
        }
        self.attack["impact_level"] = level
        self.attack["impact_evidence"].append(evidence)
        self.log_event(sim_time, f"attack impact upgraded to {level}: {reason}")
        self._write_json(self.attack_path, self.attack)

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
