"""Shared helpers for co-simulation scripts."""

import os
import random
import socket

import yaml


def is_port_open(port, host="127.0.0.1", timeout=0.5):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, int(port))) == 0


def set_random_seed(seed, config=None, traffic_manager=None):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    '''try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass'''

    if config is not None:
        num_simulations = int(getattr(config, "num_simulations", 1) or 1)
        config.random_seeds = [seed for _ in range(num_simulations)]

    if traffic_manager is not None and hasattr(traffic_manager, "set_random_device_seed"):
        traffic_manager.set_random_device_seed(seed)

    return seed


def load_scenario(path):
    with open(path, "r", encoding="utf-8") as scenario_file:
        scenario = yaml.safe_load(scenario_file)
    if not isinstance(scenario, dict):
        raise ValueError(f"Scenario file did not contain a YAML mapping: {path}")
    return scenario


def scenario_trip_specs(scenario, movement_names):
    movements = scenario.get("movements") or {}
    trips = []
    for vid, movement_name in enumerate(movement_names, start=1):
        movement = movements.get(movement_name)
        if not isinstance(movement, dict):
            raise ValueError(f"Scenario movement is missing or invalid: {movement_name}")
        try:
            origin = movement["origin"]
            destination = movement["destination"]
        except KeyError as exc:
            raise ValueError(f"Scenario movement {movement_name} is missing {exc.args[0]}") from exc
        trips.append((vid, origin, destination, movement_name))
    return trips


def scenario_lane_paths_for_trips(scenario, trip_specs):
    lane_paths_by_movement = scenario.get("lane_paths") or {}
    lane_paths_by_vehicle = {}
    for vid, _origin, _destination, movement_name in trip_specs:
        lane_path = lane_paths_by_movement.get(movement_name)
        if lane_path:
            lane_paths_by_vehicle[vid] = list(lane_path)
    return lane_paths_by_vehicle
