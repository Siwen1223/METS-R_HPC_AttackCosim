"""Shared helpers for co-simulation scripts."""

import socket

import yaml


def is_port_open(port, host="127.0.0.1", timeout=0.5):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, int(port))) == 0


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
