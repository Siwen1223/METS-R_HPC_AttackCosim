"""Reusable C-V2X helpers for METS-R/CARLA/Simu5G co-simulation."""

import math
import shlex
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

try:
    from pyproj import CRS, Transformer
except ImportError:
    CRS = None
    Transformer = None

from clients.KafkaDataProcessor import (
    bsm_core_heading_degrees,
    bsm_core_latitude_degrees,
    bsm_core_longitude_degrees,
    bsm_core_speed_mps,
    normalize_sensor_record,
)
from clients.VeinsClient import VeinsClient, build_mobility_records


BSM_MESSAGE_NAME = "BasicSafetyMessage"
BSM_MESSAGE_ID = "basicSafetyMessage"
BSM_STANDARD = "SAE J2735"
CV2X_RADIO_MODE = "cv2x"
EARTH_RADIUS_M = 6378137.0


def _float_or_none(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def bridge_field(payload, key, default=None):
    if not isinstance(payload, Mapping):
        return default
    data = payload.get("data")
    if isinstance(data, Mapping) and key in data:
        return data[key]
    return payload.get(key, default)


def vehicle_id(record):
    if not isinstance(record, Mapping):
        return record
    return record.get("ID", record.get("vehicle_id", record.get("vid")))


def _config_value(config, names, default=None):
    for name in names:
        if isinstance(config, Mapping) and name in config:
            return config[name]
        if config is not None and hasattr(config, name):
            return getattr(config, name)
    return default


def _same_vehicle_id(left, right):
    return str(left) == str(right)


def _xy_distance_m(first, second):
    try:
        dx = float(first.get("x")) - float(second.get("x"))
        dy = float(first.get("y")) - float(second.get("y"))
    except (TypeError, ValueError):
        return None
    return (dx * dx + dy * dy) ** 0.5


def _parse_proj_parameters(proj_parameter):
    params = {}
    for token in shlex.split(proj_parameter or ""):
        if not token.startswith("+"):
            continue
        key, _, value = token[1:].partition("=")
        params[key] = True if not value else value
    return params


class SumoNetGeoConverter:
    """Convert CARLA/METS-R local metric coordinates to WGS84 degrees."""

    def __init__(self, net_path):
        self.net_path = Path(net_path)
        self.net_offset = (0.0, 0.0)
        self.proj_parameter = ""
        self.proj_params = {}
        self.transformer = None
        self._load_location()
        self._load_transformer()

    def _load_location(self):
        root = ET.parse(self.net_path).getroot()
        location = root.find("location")
        if location is None:
            return
        offset_parts = location.get("netOffset", "0,0").split(",")
        if len(offset_parts) >= 2:
            self.net_offset = (float(offset_parts[0]), float(offset_parts[1]))
        self.proj_parameter = location.get("projParameter", "")
        self.proj_params = _parse_proj_parameters(self.proj_parameter)

    def _load_transformer(self):
        if not self.proj_parameter or self.proj_parameter == "!" or Transformer is None:
            return
        try:
            self.transformer = Transformer.from_crs(
                CRS.from_proj4(self.proj_parameter),
                "EPSG:4326",
                always_xy=True,
            )
        except Exception:
            self.transformer = None

    def to_lat_lon(self, x, y):
        projected_x = _float_or_none(x)
        projected_y = _float_or_none(y)
        if projected_x is None or projected_y is None:
            return None, None
        if self.transformer is not None:
            lon, lat = self.transformer.transform(projected_x, projected_y)
            return lat, lon
        return self._local_projection_to_lat_lon(projected_x, projected_y)

    def sumo_to_lat_lon(self, x, y):
        x = _float_or_none(x)
        y = _float_or_none(y)
        if x is None or y is None:
            return None, None
        projected_x = x - self.net_offset[0]
        projected_y = y - self.net_offset[1]
        return self.to_lat_lon(projected_x, projected_y)

    def _local_projection_to_lat_lon(self, projected_x, projected_y):
        lon_0 = float(self.proj_params.get("lon_0", 0.0) or 0.0)
        lat_0 = float(self.proj_params.get("lat_0", 0.0) or 0.0)
        scale = float(self.proj_params.get("k", self.proj_params.get("k_0", 1.0)) or 1.0)
        false_easting = float(self.proj_params.get("x_0", 0.0) or 0.0)
        false_northing = float(self.proj_params.get("y_0", 0.0) or 0.0)
        dx = projected_x - false_easting
        dy = projected_y - false_northing
        lat = lat_0 + math.degrees(dy / (EARTH_RADIUS_M * scale))
        cos_lat = math.cos(math.radians(lat_0))
        if abs(cos_lat) < 1e-9:
            return lat, lon_0
        lon = lon_0 + math.degrees(dx / (EARTH_RADIUS_M * scale * cos_lat))
        return lat, lon


def make_geo_converter(config):
    network_file = _config_value(config, ("network_file", "net_file", "sumo_net_file"))
    if not network_file:
        return None
    net_path = Path(network_file)
    if not net_path.is_absolute():
        net_path = Path.cwd() / net_path
    if not net_path.exists():
        return None
    try:
        return SumoNetGeoConverter(net_path)
    except Exception:
        return None


def normalize_vehicle(record, private_veh=False, role="vehicle", map_name=None):
    return {
        "ID": vehicle_id(record),
        "role": record.get("role", role),
        "x": record.get("x"),
        "y": record.get("y"),
        "z": record.get("z", 0.0),
        "latitude": record.get("latitude", record.get("lat")),
        "longitude": record.get("longitude", record.get("lon")),
        "speed": record.get("speed", record.get("speed_mps", 0.0)),
        "bearing": record.get("bearing", record.get("heading_deg", 0.0)),
        "acc": record.get("acc", record.get("acceleration_mps2")),
        "road": record.get("road", record.get("road_id")),
        "lane": record.get("lane", record.get("lane_id")),
        "state": record.get("state"),
        "v_type": record.get("v_type", record.get("vehicle_type")),
        "private_veh": bool(record.get("private_veh", private_veh)),
        "sensor_type": record.get("sensor_type", "cv2x"),
        "map_name": record.get("map_name", map_name),
    }


def make_basic_safety_message(sender, tick, sequence, message_count=None, sec_mark_ms=None):
    sensor_record = {
        "vehicle_id": vehicle_id(sender),
        "tick": tick,
        "message_count": (
            int(message_count) % 128
            if message_count is not None
            else (int(tick) * 16 + int(sequence)) % 128
        ),
        "secMark": sec_mark_ms,
        "speed_mps": float(sender.get("speed", sender.get("speed_mps", 0.0)) or 0.0),
        "heading_deg": float(sender.get("bearing", sender.get("heading_deg", 0.0)) or 0.0),
        "acceleration_mps2": sender.get("acc", sender.get("acceleration_mps2")),
        "x": sender.get("x"),
        "y": sender.get("y"),
        "z": sender.get("z", 0.0),
    }
    if sender.get("latitude") is not None:
        sensor_record["latitude"] = sender.get("latitude")
    if sender.get("longitude") is not None:
        sensor_record["longitude"] = sender.get("longitude")
    return {"coreData": normalize_sensor_record(sensor_record, topic="bsm")["coreData"]}


def basic_safety_message(message):
    if not isinstance(message, Mapping):
        return {}
    payload = message.get("BasicSafetyMessage")
    if isinstance(payload, Mapping):
        return payload
    frame = message.get("messageFrame")
    if isinstance(frame, Mapping):
        value = frame.get("value")
        if isinstance(value, Mapping) and isinstance(value.get("BasicSafetyMessage"), Mapping):
            return value["BasicSafetyMessage"]
    if isinstance(message.get("coreData"), Mapping):
        return {"coreData": message["coreData"]}
    return {}


def bsm_core_data(message):
    return dict(basic_safety_message(message).get("coreData", {}))


def make_message_frame(core_data):
    return {
        "messageId": BSM_MESSAGE_ID,
        "value": {BSM_MESSAGE_NAME: {"coreData": dict(core_data)}},
    }


def make_transport_envelope(
    *,
    tick,
    sender_id,
    receiver_id,
    message_id,
    payload_bytes=300,
    tx_time_s=None,
    radio_mode=CV2X_RADIO_MODE,
):
    return {
        "messageId": message_id,
        "tick": int(tick),
        "senderId": sender_id,
        "receiverId": receiver_id,
        "payloadBytes": payload_bytes,
        "txTimeS": tx_time_s,
        "radioMode": radio_mode,
    }


def make_operational_data(
    sender,
    receiver=None,
    content=None,
    attacked=False,
    attack_id="",
    attack_type="",
):
    speed_mps = float(sender.get("speed", sender.get("speed_mps", 0.0)) or 0.0)
    heading_deg = float(sender.get("bearing", sender.get("heading_deg", 0.0)) or 0.0)
    return {
        "content": content,
        "mapName": sender.get("map_name"),
        "senderRole": sender.get("role"),
        "receiverRole": receiver.get("role") if isinstance(receiver, Mapping) else None,
        "privateVehicle": bool(sender.get("private_veh", False)),
        "localPositionM": {
            "x": sender.get("x"),
            "y": sender.get("y"),
            "z": sender.get("z", 0.0),
        },
        "speedMps": speed_mps,
        "headingDeg": heading_deg,
        "accelerationMps2": sender.get("acc", sender.get("acceleration_mps2")),
        "truth": {
            "x": sender.get("truth_x", sender.get("x")),
            "y": sender.get("truth_y", sender.get("y")),
            "speedMps": sender.get("truth_speed_mps", speed_mps),
            "headingDeg": sender.get("truth_heading_deg", heading_deg),
        },
        "attack": {
            "attacked": bool(attacked),
            "attackId": attack_id,
            "attackType": attack_type,
        },
        "custom": {},
    }


def transport_envelope_from_message(message):
    transport = dict(message.get("transport_envelope") or message.get("transport") or {})
    if "messageId" not in transport and message.get("message_id") is not None:
        transport["messageId"] = message.get("message_id")
    if "tick" not in transport and message.get("tick") is not None:
        transport["tick"] = message.get("tick")
    if "senderId" not in transport:
        transport["senderId"] = message.get("sender_id", message.get("vehicle_id"))
    if "receiverId" not in transport:
        transport["receiverId"] = message.get("receiver_id", message.get("target_vehicle_id"))
    if "payloadBytes" not in transport and message.get("payload_bytes") is not None:
        transport["payloadBytes"] = message.get("payload_bytes")
    if "txTimeS" not in transport and message.get("tx_time_s") is not None:
        transport["txTimeS"] = message.get("tx_time_s")
    if "radioMode" not in transport:
        transport["radioMode"] = message.get("radio_mode", CV2X_RADIO_MODE)
    return transport


def operational_data_from_message(message):
    return dict(message.get("operational_data") or message.get("operationalData") or {})


def make_bsm_message(
    *,
    tick,
    sequence,
    sender,
    receiver,
    message_id,
    payload_bytes=300,
    tx_time_s=None,
    message_count=None,
    sec_mark_ms=None,
    content=None,
    attacked=False,
    attack_id="",
    attack_type="",
):
    sender_id = vehicle_id(sender)
    receiver_id = vehicle_id(receiver)
    basic_message = make_basic_safety_message(
        sender,
        tick,
        sequence,
        message_count=message_count,
        sec_mark_ms=sec_mark_ms,
    )
    core_data = bsm_core_data(basic_message)
    message_frame = make_message_frame(core_data)
    transport = make_transport_envelope(
        tick=tick,
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_id=message_id,
        payload_bytes=payload_bytes,
        tx_time_s=tx_time_s,
    )
    operational_data = make_operational_data(
        sender,
        receiver=receiver,
        content=content,
        attacked=attacked,
        attack_id=attack_id,
        attack_type=attack_type,
    )
    return {
        "messageFrame": message_frame,
        "bsm_core_data": core_data,
        "transport": transport,
        "transport_envelope": transport,
        "operationalData": operational_data,
        "operational_data": operational_data,
    }


def bridge_message_from_bsm(message):
    transport = transport_envelope_from_message(message)
    operational_data = operational_data_from_message(message)
    core_data = bsm_core_data(message)
    speed_mps = bsm_core_speed_mps({"coreData": core_data})
    heading_deg = bsm_core_heading_degrees({"coreData": core_data})
    latitude_deg = bsm_core_latitude_degrees({"coreData": core_data})
    longitude_deg = bsm_core_longitude_degrees({"coreData": core_data})
    return {
        "message_id": transport.get("messageId"),
        "tick": transport.get("tick"),
        "vehicle_id": transport.get("senderId"),
        "sender_id": transport.get("senderId"),
        "receiver_id": transport.get("receiverId"),
        "target_vehicle_id": transport.get("receiverId"),
        "message_name": BSM_MESSAGE_NAME,
        "message_standard": BSM_STANDARD,
        "message_count": core_data.get("msgCnt"),
        "bsm_core_data": core_data,
        "payload_bytes": transport.get("payloadBytes", 300),
        "tx_time_s": transport.get("txTimeS"),
        "radio_mode": transport.get("radioMode", CV2X_RADIO_MODE),
        "transport_envelope": transport,
        "operational_data": operational_data,
        "messageFrame": message.get("messageFrame"),
        "operationalData": operational_data,
        "bsm_latitude_deg": latitude_deg,
        "bsm_longitude_deg": longitude_deg,
        "bsm_speed_mps": speed_mps,
        "bsm_heading_deg": heading_deg,
        "attacked": bool(operational_data.get("attack", {}).get("attacked", False)),
        "attack_id": operational_data.get("attack", {}).get("attackId", ""),
        "attack_type": operational_data.get("attack", {}).get("attackType", ""),
    }


def communication_records_from_result(result, vehicles, messages):
    vehicles_by_id = {vehicle["ID"]: vehicle for vehicle in vehicles if "ID" in vehicle}
    messages_by_link = {}
    for message in messages:
        key = (
            message.get("sender_id", message.get("vehicle_id")),
            message.get("receiver_id", message.get("target_vehicle_id")),
            message.get("message_count"),
        )
        messages_by_link[key] = message
        messages_by_link.setdefault((key[0], key[1], None), message)

    records = []
    for metric in result.get("link_metrics", []):
        sender_id = metric.get("sender_id")
        receiver_id = metric.get("receiver_id", metric.get("target_vehicle_id"))
        message_count = metric.get("message_count")
        message = messages_by_link.get(
            (sender_id, receiver_id, message_count),
            messages_by_link.get((sender_id, receiver_id, None), {}),
        )
        sender = vehicles_by_id.get(sender_id, {})
        receiver = vehicles_by_id.get(receiver_id, {})
        transport = dict(message.get("transport_envelope") or {})
        operational_data = dict(message.get("operational_data") or {})
        core_data = dict(message.get("bsm_core_data", message.get("coreData", {})) or {})
        bsm_payload = {"messageFrame": message.get("messageFrame")}
        link_result = {
            "delivered": metric.get("delivered"),
            "drop_reason": metric.get("drop_reason", ""),
            "latency_ms": None if metric.get("latency_ms") is None else metric.get("latency_ms"),
            "distance_m": metric.get("distance_m"),
            "radio_mode": metric.get("radio_mode", message.get("radio_mode")),
            "backend_implementation": metric.get("backend_implementation"),
        }
        records.append(
            {
                "tick": metric.get("tick"),
                "message_id": metric.get("message_id", message.get("message_id")),
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "origin_vehicle_id": sender_id,
                "target_vehicle_id": receiver_id,
                "message_name": metric.get("message_name", message.get("message_name")),
                "message_count": message_count,
                "message_standard": message.get("message_standard", BSM_STANDARD),
                "bsm_payload": bsm_payload,
                "bsm_core_data": core_data,
                "transport_envelope": transport,
                "operational_data": operational_data,
                "link_result": link_result,
                "bsm_latitude_deg": message.get("bsm_latitude_deg"),
                "bsm_longitude_deg": message.get("bsm_longitude_deg"),
                "bsm_speed_mps": message.get("bsm_speed_mps"),
                "bsm_heading_deg": message.get("bsm_heading_deg"),
                "delivered": link_result["delivered"],
                "drop_reason": link_result["drop_reason"],
                "latency_ms": "" if link_result["latency_ms"] is None else link_result["latency_ms"],
                "distance_m": link_result["distance_m"],
                "radio_mode": link_result["radio_mode"],
                "backend_implementation": link_result["backend_implementation"],
                "attacked": message.get("attacked", False),
                "attack_id": message.get("attack_id", ""),
                "attack_type": message.get("attack_type", ""),
                "_sender_state": {
                    "x": (operational_data.get("localPositionM") or {}).get("x", sender.get("x")),
                    "y": (operational_data.get("localPositionM") or {}).get("y", sender.get("y")),
                    "z": (operational_data.get("localPositionM") or {}).get("z", sender.get("z")),
                    "latitude": message.get("bsm_latitude_deg", sender.get("latitude")),
                    "longitude": message.get("bsm_longitude_deg", sender.get("longitude")),
                    "speed_mps": operational_data.get("speedMps", sender.get("speed", sender.get("speed_mps", 0.0))),
                    "heading_deg": operational_data.get("headingDeg", sender.get("bearing", sender.get("heading_deg", 0.0))),
                },
            }
        )
    return records


class CV2XManager:
    """Thin Simu5G/VEINS facade that returns controller-ready V2V streams."""

    def __init__(
        self,
        config=None,
        veins_client=None,
        host=None,
        port=None,
        require_simu5g_uu=True,
        duration_s=None,
        payload_bytes=300,
        communication_range_m=None,
        geo_converter=None,
    ):
        self.config = config
        self.veins = veins_client or VeinsClient(config=config, host=host, port=port)
        self.require_simu5g_uu = require_simu5g_uu
        self.duration_s = duration_s if duration_s is not None else getattr(config, "sim_step_size", 0.1)
        self.payload_bytes = payload_bytes
        self.geo_converter = geo_converter if geo_converter is not None else make_geo_converter(config)
        self.communication_range_m = float(
            communication_range_m
            if communication_range_m is not None
            else _config_value(
                config,
                ("cv2x_communication_range_m", "cv2x_range_m", "communication_range_m"),
                500.0,
            )
        )
        self.last_result = {}
        self.last_rows = []
        self.last_stream = []
        self.last_streams_by_receiver = {}
        self._bsm_message_counts = {}

    def close(self):
        if self.veins is not None:
            self.veins.close()

    def make_pairwise_bsm_messages(self, tick, vehicles, extra_messages=None):
        vehicles = [
            self._with_bsm_geography(
                normalize_vehicle(vehicle, map_name=getattr(self.config, "carla_map", None))
            )
            for vehicle in vehicles
        ]
        messages = []
        sequence = 0
        tx_time_s = int(tick) * float(self.duration_s)
        sec_mark_ms = int(round(tx_time_s * 1000.0)) % 60000
        for sender in vehicles:
            sender_id = vehicle_id(sender)
            if sender_id is None:
                continue
            sender_message_count = self._next_message_count(sender_id)
            for receiver in vehicles:
                receiver_id = vehicle_id(receiver)
                if receiver_id is None or receiver_id == sender_id:
                    continue
                if not self._within_communication_range(sender, receiver):
                    continue
                sequence += 1
                messages.append(
                    make_bsm_message(
                        tick=tick,
                        sequence=sequence,
                        sender=sender,
                        receiver=receiver,
                        message_id=f"cv2x:{tick}:{sender_id}>{receiver_id}:{sequence}",
                        payload_bytes=self.payload_bytes,
                        tx_time_s=tx_time_s,
                        message_count=sender_message_count,
                        sec_mark_ms=sec_mark_ms,
                    )
                )
        messages.extend(extra_messages or [])
        return messages

    def _next_message_count(self, sender_id):
        key = str(sender_id)
        value = self._bsm_message_counts.get(key, 0) % 128
        self._bsm_message_counts[key] = (value + 1) % 128
        return value

    def make_ghost_bsm_messages(
        self,
        tick,
        ghost_vehicle,
        receivers,
        attack_id="simu5g_bsm_position_falsification",
        attack_type="position_falsification",
    ):
        messages = []
        ghost_vehicle = self._with_bsm_geography(normalize_vehicle(ghost_vehicle))
        receivers = [self._with_bsm_geography(normalize_vehicle(receiver)) for receiver in receivers]
        ghost_id = vehicle_id(ghost_vehicle)
        tx_time_s = int(tick) * float(self.duration_s)
        sec_mark_ms = int(round(tx_time_s * 1000.0)) % 60000
        ghost_message_count = self._next_message_count(ghost_id) if ghost_id is not None else None
        for sequence, receiver in enumerate(receivers, start=1):
            if not self._within_communication_range(ghost_vehicle, receiver):
                continue
            messages.append(
                make_bsm_message(
                    tick=tick,
                    sequence=sequence,
                    sender=ghost_vehicle,
                    receiver=receiver,
                    message_id=f"attack:{tick}:{vehicle_id(ghost_vehicle)}>{vehicle_id(receiver)}:{sequence}",
                    payload_bytes=self.payload_bytes,
                    tx_time_s=tx_time_s,
                    message_count=ghost_message_count,
                    sec_mark_ms=sec_mark_ms,
                    content="falsified BSM position",
                    attacked=True,
                    attack_id=attack_id,
                    attack_type=attack_type,
                )
            )
        return messages

    def sync_tick(self, tick, vehicles, messages=None, attacks=None, phase="run"):
        vehicles = [
            self._with_bsm_geography(
                normalize_vehicle(vehicle, map_name=getattr(self.config, "carla_map", None))
            )
            for vehicle in vehicles
        ]
        messages = self.make_pairwise_bsm_messages(tick, vehicles) if messages is None else list(messages)
        bridge_messages = [bridge_message_from_bsm(message) for message in messages]
        result = self.veins.sync_tick(
            tick=tick,
            vehicles=build_mobility_records(vehicles),
            bsm_messages=bridge_messages,
            attacks=attacks or [],
            duration_s=self.duration_s,
        )
        if self.require_simu5g_uu:
            implementation = bridge_field(result, "backend_implementation")
            if implementation != "simu5g_cellular_uu":
                raise RuntimeError(
                    "The bridge did not report backend_implementation=simu5g_cellular_uu. "
                    f"Reported value: {implementation!r}. Start veins_bridge/omnetpp/run_sim5g_uu.sh."
                )
        rows = communication_records_from_result(result, vehicles, bridge_messages)
        for row in rows:
            row["phase"] = phase
        self.last_result = result
        self.last_rows = rows
        self.last_streams_by_receiver = self.controller_streams_by_receiver(vehicles, rows)
        self.last_stream = self.controller_stream(vehicles, rows)
        return result, rows, self.last_stream

    def controller_stream(self, vehicles, rows=None):
        stream = [self._vehicle_to_controller_record(vehicle) for vehicle in vehicles]
        for row in rows or []:
            if not row.get("delivered"):
                continue
            stream.append(self._row_to_controller_record(row))
        return [record for record in stream if record.get("vid") is not None]

    def controller_streams_by_receiver(self, vehicles, rows=None):
        streams = {}
        for vehicle in vehicles:
            vid = vehicle_id(vehicle)
            if vid is None:
                continue
            record = self._vehicle_to_controller_record(vehicle)
            if record.get("vid") is not None:
                streams[vid] = [record]

        for row in rows or []:
            if not row.get("delivered"):
                continue
            receiver_id = row.get("target_vehicle_id")
            if receiver_id is None:
                continue
            record = self._row_to_controller_record(row)
            if record.get("vid") is None:
                continue
            for stream_vid in list(streams):
                if _same_vehicle_id(stream_vid, receiver_id):
                    streams[stream_vid].append(record)
                    break
        return streams

    def _within_communication_range(self, sender, receiver):
        distance = _xy_distance_m(sender, receiver)
        if distance is None:
            return False
        return distance <= self.communication_range_m

    def _with_bsm_geography(self, vehicle):
        if vehicle.get("latitude") is not None and vehicle.get("longitude") is not None:
            return vehicle
        if self.geo_converter is None:
            return vehicle
        latitude, longitude = self.geo_converter.to_lat_lon(vehicle.get("x"), vehicle.get("y"))
        if latitude is not None and longitude is not None:
            vehicle = dict(vehicle)
            vehicle["latitude"] = latitude
            vehicle["longitude"] = longitude
        return vehicle

    @staticmethod
    def _vehicle_to_controller_record(vehicle):
        vid = vehicle_id(vehicle)
        speed = vehicle.get("speed", vehicle.get("speed_mps", 0.0))
        heading = vehicle.get("bearing", vehicle.get("heading_deg", 0.0))
        return {
            "vid": vid,
            "vehicle_id": vid,
            "sender_id": vid,
            "x": vehicle.get("x"),
            "y": vehicle.get("y"),
            "true_x": vehicle.get("x"),
            "true_y": vehicle.get("y"),
            "latitude": vehicle.get("latitude"),
            "longitude": vehicle.get("longitude"),
            "velocity": speed,
            "speed": speed,
            "speed_mps": speed,
            "heading": heading,
            "heading_deg": heading,
        }

    @staticmethod
    def _row_to_controller_record(row):
        vid = row.get("origin_vehicle_id")
        sender_state = row.get("_sender_state", {})
        x = sender_state.get("x")
        y = sender_state.get("y")
        speed = row.get("bsm_speed_mps")
        heading = row.get("bsm_heading_deg")
        if speed is None:
            speed = sender_state.get("speed_mps", 0.0)
        if heading is None:
            heading = sender_state.get("heading_deg", 0.0)
        return {
            "vid": vid,
            "vehicle_id": vid,
            "sender_id": vid,
            "receiver_id": row.get("target_vehicle_id"),
            "x": x,
            "y": y,
            "true_x": x,
            "true_y": y,
            "latitude": row.get("bsm_latitude_deg"),
            "longitude": row.get("bsm_longitude_deg"),
            "velocity": speed,
            "speed": speed,
            "speed_mps": speed,
            "heading": heading,
            "heading_deg": heading,
            "attacked": row.get("attacked", False),
            "attack_id": row.get("attack_id", ""),
            "attack_type": row.get("attack_type", ""),
        }
