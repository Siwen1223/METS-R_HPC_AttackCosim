import csv
import json
import math
import os
from pathlib import Path


class RunDataSaver:
    """
    Save one simulation run into a structured dataset folder with metadata, BSM logs, sensor data, and vehicle states.
    Inputs: Dataset root path, run-level metadata, attack metadata, and optional run configuration.
    Outputs: Creates run directories/files and stores subsequent simulation records into them.
    """
    def __init__(
        self,
        dataset_root,
        meta,
        attack,
        run_id=None,
        sensor_every_n=5,
    ):
        """
        Initialize the run saver and create the output files for the current run.
        Inputs: Dataset root path, meta dictionary, attack dictionary, optional run_id, and sensor save interval.
        Outputs: Creates the run folder structure, writes meta/attack JSON files, and opens log files.
        """
        self.dataset_root = Path(dataset_root)
        self.runs_dir = self.dataset_root / "runs"
        self.sensor_every_n = sensor_every_n
        self._ensure_dataset_dirs()

        self.run_id = run_id or self._next_run_id()
        self.run_dir = self.runs_dir / self.run_id
        self.bsm_dir = self.run_dir / "bsm"
        self.sensors_dir = self.run_dir / "sensors"
        self.vehicle_state_dir = self.run_dir / "vehicle_state"
        self.events_path = self.run_dir / "events.log"
        self.meta_path = self.run_dir / "meta.json"
        self.attack_path = self.run_dir / "attack.json"
        self._init_run_dirs()

        self.meta = dict(meta)
        self.meta.setdefault("run_id", self.run_id)
        self.attack = dict(attack) if attack is not None else {"attack_type": "none"}

        self._write_json(self.meta_path, self.meta)
        self._write_json(self.attack_path, self.attack)

        self._events_fp = open(self.events_path, "a", encoding="utf-8")
        self._trajectory_fp = open(self.vehicle_state_dir / "trajectory.csv", "w", newline="", encoding="utf-8")
        self._control_fp = open(self.vehicle_state_dir / "control.csv", "w", newline="", encoding="utf-8")
        self._trajectory_writer = csv.writer(self._trajectory_fp)
        self._control_writer = csv.writer(self._control_fp)
        self._trajectory_writer.writerow(["tick", "sim_time", "vid", "x", "y", "yaw", "speed_mps"])
        self._control_writer.writerow(["tick", "sim_time", "vid", "throttle", "brake", "steer", "hand_brake", "reverse"])

        self._bsm_fp = open(self.bsm_dir / "bsm.jsonl", "a", encoding="utf-8")

    def log_event(self, sim_time, message):
        """
        Append one time-stamped event message to the run event log.
        Inputs: Simulation time in seconds and the event message string.
        Outputs: Writes one line into events.log.
        """
        self._events_fp.write(f"[{sim_time:.2f}s] {message}\n")
        self._events_fp.flush()

    def record_bsm(self, tick, sim_time, data_stream):
        """
        Record one BSM data snapshot into the run BSM log file.
        Inputs: Simulation tick, simulation time, and the BSM data stream object.
        Outputs: Appends one JSONL record into bsm/bsm.jsonl.
        """
        if data_stream is None:
            return
        payload = {"tick": tick, "sim_time": sim_time, "bsm": data_stream}
        self._bsm_fp.write(json.dumps(payload) + "\n")

    def record_vehicle_state(self, tick, sim_time, cosim_client, vids=None):
        """
        Record the current CARLA vehicle states for selected co-sim vehicles.
        Inputs: Simulation tick, simulation time, CoSimClient instance, and optional vehicle IDs.
        Outputs: Appends rows into vehicle_state/trajectory.csv.
        """
        if vids is None:
            vids = list(cosim_client.carla_vehs.keys())
        for vid in vids:
            if not cosim_client.carla_entered.get(vid, False):
                continue
            vehicle = cosim_client.carla_vehs.get(vid)
            if vehicle is None:
                continue
            loc = vehicle.get_location()
            yaw = vehicle.get_transform().rotation.yaw
            vel = vehicle.get_velocity()
            speed = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
            self._trajectory_writer.writerow([tick, sim_time, vid, loc.x, loc.y, yaw, speed])

    def record_control(self, tick, sim_time, vid, control):
        """
        Record one control command applied to a vehicle in the co-sim area.
        Inputs: Simulation tick, simulation time, vehicle ID, and a CARLA VehicleControl object.
        Outputs: Appends one row into vehicle_state/control.csv.
        """
        if control is None:
            return
        self._control_writer.writerow([
            tick,
            sim_time,
            vid,
            control.throttle,
            control.brake,
            control.steer,
            int(control.hand_brake),
            int(control.reverse),
        ])

    def save_sensors(self, cosim_client):
        """
        Trigger sensor data saving for the current run through the CoSim client.
        Inputs: A CoSimClient instance with deployed sensors.
        Outputs: Saves the current sensor outputs under the run sensors directory.
        """
        cosim_client.collect_sensor_data(str(self.sensors_dir))

    def finalize(self, duration_sec=None):
        """
        Finalize the run output files and optionally update the run duration in metadata.
        Inputs: Optional simulation duration in seconds.
        Outputs: Updates meta.json and closes all opened log files.
        """
        if duration_sec is not None:
            self.meta["duration_sec"] = duration_sec
            self._write_json(self.meta_path, self.meta)
        for fp in (self._events_fp, self._trajectory_fp, self._control_fp, self._bsm_fp):
            try:
                fp.flush()
                fp.close()
            except Exception:
                pass

    def _write_json(self, path, data):
        """
        Write a Python object as a formatted JSON file.
        Inputs: Target file path and serializable data object.
        Outputs: Creates or overwrites the JSON file at the given path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)

    def _ensure_dataset_dirs(self):
        """
        Ensure that the top-level dataset directory structure exists.
        Inputs: No external inputs.
        Outputs: Creates dataset-level folders such as runs, metadata, annotations, splits, and scenarios.
        """
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        for name in ("runs", "metadata", "annotations", "splits", "scenarios"):
            (self.dataset_root / name).mkdir(parents=True, exist_ok=True)

    def _init_run_dirs(self):
        """
        Ensure that the directory structure for the current run exists.
        Inputs: No external inputs.
        Outputs: Creates the run, bsm, sensors, and vehicle_state directories.
        """
        for p in (self.run_dir, self.bsm_dir, self.sensors_dir, self.vehicle_state_dir):
            p.mkdir(parents=True, exist_ok=True)

    def _next_run_id(self):
        """
        Generate the next sequential run identifier under the dataset runs directory.
        Inputs: No external inputs beyond the existing runs directory contents.
        Outputs: Returns a new run ID string such as run_000001.
        """
        if not self.runs_dir.exists():
            return "run_000001"
        run_ids = []
        for item in self.runs_dir.iterdir():
            if item.is_dir() and item.name.startswith("run_"):
                try:
                    run_ids.append(int(item.name.split("_")[1]))
                except (IndexError, ValueError):
                    continue
        next_id = max(run_ids, default=0) + 1
        return f"run_{next_id:06d}"
