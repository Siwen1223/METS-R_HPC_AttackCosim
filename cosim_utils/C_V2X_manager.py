"""Reusable C-V2X helpers for METS-R/CARLA/Simu5G co-simulation."""

from collections.abc import Mapping

from clients.KafkaDataProcessor import (
    bsm_core_heading_degrees,
    bsm_core_speed_mps,
    normalize_sensor_record,
)
from clients.VeinsClient import VeinsClient, build_mobility_records


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


def normalize_vehicle(record, private_veh=False, role="vehicle", map_name=None):
    return {
        "ID": vehicle_id(record),
        "role": record.get("role", role),
        "x": record.get("x"),
        "y": record.get("y"),
        "z": record.get("z", 0.0),
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


def make_basic_safety_message(sender, tick, sequence):
    sensor_record = {
        "vehicle_id": vehicle_id(sender),
        "tick": tick,
        "message_count": (int(tick) * 16 + int(sequence)) % 128,
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


def make_bsm_message(
    *,
    tick,
    sequence,
    sender,
    receiver,
    message_id,
    payload_bytes=300,
    tx_time_s=None,
    content=None,
    attacked=False,
    attack_id="",
    attack_type="",
):
    sender_id = vehicle_id(sender)
    receiver_id = vehicle_id(receiver)
    speed_mps = float(sender.get("speed", sender.get("speed_mps", 0.0)) or 0.0)
    heading_deg = float(sender.get("bearing", sender.get("heading_deg", 0.0)) or 0.0)
    basic_message = make_basic_safety_message(sender, tick, sequence)
    return {
        "messageFrame": {
            "messageId": "basicSafetyMessage",
            "value": {"BasicSafetyMessage": basic_message},
        },
        "transport": {
            "messageId": message_id,
            "tick": int(tick),
            "senderId": sender_id,
            "receiverId": receiver_id,
            "payloadBytes": payload_bytes,
            "txTimeS": tx_time_s,
            "radioMode": "cv2x",
        },
        "operationalData": {
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
        },
    }


def bridge_message_from_bsm(message):
    transport = message.get("transport", {})
    operational = message.get("operationalData", {})
    position = operational.get("localPositionM", {})
    truth = operational.get("truth", {})
    attack = operational.get("attack", {})
    core_data = bsm_core_data(message)
    speed_mps = bsm_core_speed_mps({"coreData": core_data})
    heading_deg = bsm_core_heading_degrees({"coreData": core_data})
    return {
        "message_id": transport.get("messageId"),
        "tick": transport.get("tick"),
        "vehicle_id": transport.get("senderId"),
        "sender_id": transport.get("senderId"),
        "receiver_id": transport.get("receiverId"),
        "target_vehicle_id": transport.get("receiverId"),
        "message_name": "BasicSafetyMessage",
        "message_standard": "SAE J2735",
        "message_count": core_data.get("msgCnt"),
        "payload_bytes": transport.get("payloadBytes", 300),
        "tx_time_s": transport.get("txTimeS"),
        "radio_mode": transport.get("radioMode", "cv2x"),
        "content": operational.get("content"),
        "map_name": operational.get("mapName"),
        "sender_role": operational.get("senderRole"),
        "receiver_role": operational.get("receiverRole"),
        "messageFrame": message.get("messageFrame"),
        "BasicSafetyMessage": basic_safety_message(message),
        "coreData": core_data,
        "operationalData": operational,
        "x": position.get("x"),
        "y": position.get("y"),
        "z": position.get("z", 0.0),
        "speed_mps": speed_mps,
        "heading_deg": heading_deg,
        "acceleration_mps2": operational.get("accelerationMps2"),
        "truth_x": truth.get("x", position.get("x")),
        "truth_y": truth.get("y", position.get("y")),
        "truth_speed_mps": truth.get("speedMps", speed_mps),
        "truth_heading_deg": truth.get("headingDeg", heading_deg),
        "attacked": bool(attack.get("attacked", False)),
        "attack_id": attack.get("attackId", ""),
        "attack_type": attack.get("attackType", ""),
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
        records.append(
            {
                "tick": metric.get("tick"),
                "origin_vehicle_id": sender_id,
                "target_vehicle_id": receiver_id,
                "origin_role": sender.get("role", message.get("sender_role", "")),
                "target_role": receiver.get("role", message.get("receiver_role", "")),
                "origin_x": sender.get("x"),
                "origin_y": sender.get("y"),
                "origin_z": sender.get("z"),
                "target_x": receiver.get("x"),
                "target_y": receiver.get("y"),
                "target_z": receiver.get("z"),
                "distance_m": metric.get("distance_m"),
                "message_name": metric.get("message_name", message.get("message_name")),
                "message_id": metric.get("message_id", message.get("message_id")),
                "message_count": message_count,
                "message_content": message.get("content", metric.get("message_content", "")),
                "attacked": metric.get("attacked", message.get("attacked", False)),
                "attack_id": metric.get("attack_id", message.get("attack_id", "")),
                "attack_type": metric.get("attack_type", message.get("attack_type", "")),
                "delivered": metric.get("delivered"),
                "drop_reason": metric.get("drop_reason", ""),
                "latency_ms": "" if metric.get("latency_ms") is None else metric.get("latency_ms"),
                "packet_error_rate": metric.get("packet_error_rate"),
                "delivery_probability": metric.get("delivery_probability"),
                "radio_mode": metric.get("radio_mode", message.get("radio_mode")),
                "bridge_backend": metric.get("bridge_backend"),
                "backend_implementation": metric.get("backend_implementation"),
                "radio_access": metric.get("radio_access"),
                "bridge_model": metric.get("bridge_model"),
                "network_model": metric.get("network_model"),
                "tx_x": message.get("x"),
                "tx_y": message.get("y"),
                "tx_speed_mps": message.get("speed_mps"),
                "tx_heading_deg": message.get("heading_deg"),
                "truth_x": message.get("truth_x"),
                "truth_y": message.get("truth_y"),
                "truth_speed_mps": message.get("truth_speed_mps"),
                "truth_heading_deg": message.get("truth_heading_deg"),
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
    ):
        self.config = config
        self.veins = veins_client or VeinsClient(config=config, host=host, port=port)
        self.require_simu5g_uu = require_simu5g_uu
        self.duration_s = duration_s if duration_s is not None else getattr(config, "sim_step_size", 0.1)
        self.payload_bytes = payload_bytes
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

    def close(self):
        if self.veins is not None:
            self.veins.close()

    def make_pairwise_bsm_messages(self, tick, vehicles, extra_messages=None):
        messages = []
        sequence = 0
        tx_time_s = int(tick) * float(self.duration_s)
        for sender in vehicles:
            sender_id = vehicle_id(sender)
            if sender_id is None:
                continue
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
                        content=f"BSM tick={tick} veh={sender_id}",
                    )
                )
        messages.extend(extra_messages or [])
        return messages

    def make_ghost_bsm_messages(
        self,
        tick,
        ghost_vehicle,
        receivers,
        attack_id="simu5g_bsm_position_falsification",
        attack_type="position_falsification",
    ):
        messages = []
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
                    tx_time_s=int(tick) * float(self.duration_s),
                    content="falsified BSM position",
                    attacked=True,
                    attack_id=attack_id,
                    attack_type=attack_type,
                )
            )
        return messages

    def sync_tick(self, tick, vehicles, messages=None, attacks=None, phase="run"):
        vehicles = [normalize_vehicle(vehicle, map_name=getattr(self.config, "carla_map", None)) for vehicle in vehicles]
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
            "velocity": speed,
            "speed": speed,
            "speed_mps": speed,
            "heading": heading,
            "heading_deg": heading,
        }

    @staticmethod
    def _row_to_controller_record(row):
        vid = row.get("origin_vehicle_id")
        x = row.get("tx_x", row.get("truth_x"))
        y = row.get("tx_y", row.get("truth_y"))
        speed = row.get("tx_speed_mps", row.get("truth_speed_mps", 0.0))
        heading = row.get("tx_heading_deg", row.get("truth_heading_deg", 0.0))
        return {
            "vid": vid,
            "vehicle_id": vid,
            "sender_id": vid,
            "receiver_id": row.get("target_vehicle_id"),
            "x": x,
            "y": y,
            "true_x": row.get("truth_x", x),
            "true_y": row.get("truth_y", y),
            "velocity": speed,
            "speed": speed,
            "speed_mps": speed,
            "heading": heading,
            "heading_deg": heading,
            "attacked": row.get("attacked", False),
            "attack_type": row.get("attack_type"),
        }
