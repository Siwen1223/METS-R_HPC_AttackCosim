import argparse
import math
import time

import carla


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal CARLA autopilot + V2V influence example.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--attack", action="store_true", help="Inject a fake V2V vehicle ahead.")
    parser.add_argument("--attack-distance", type=float, default=8.0)
    return parser.parse_args()


def get_or_spawn_ego(world):
    vehicles = world.get_actors().filter("vehicle.*")
    if vehicles:
        return vehicles[0]
    blueprint = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points in map.")
    return world.spawn_actor(blueprint, spawn_points[0])


def ego_frame_delta(ego_transform, other_location):
    ex = ego_transform.location.x
    ey = ego_transform.location.y
    heading = math.radians(ego_transform.rotation.yaw)
    dx = other_location.x - ex
    dy = other_location.y - ey
    forward_x = math.cos(heading)
    forward_y = math.sin(heading)
    right_x = math.cos(heading + math.pi / 2.0)
    right_y = math.sin(heading + math.pi / 2.0)
    longitudinal = dx * forward_x + dy * forward_y
    lateral = dx * right_x + dy * right_y
    return longitudinal, lateral


def build_v2v_stream_from_carla(ego, vehicles):
    ego_tf = ego.get_transform()
    data = []
    for v in vehicles:
        if v.id == ego.id:
            continue
        dx, dy = ego_frame_delta(ego_tf, v.get_location())
        data.append({"vid": v.id, "dx": dx, "dy": dy})
    return data


def find_lead_vehicle(v2v_stream, lane_half_width=2.0):
    lead = None
    lead_dist = float("inf")
    for v in v2v_stream:
        if v["dx"] <= 0:
            continue
        if abs(v["dy"]) > lane_half_width:
            continue
        if v["dx"] < lead_dist:
            lead_dist = v["dx"]
            lead = v
    return lead, lead_dist


def main():
    args = parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    tm = client.get_trafficmanager(args.tm_port)

    settings = world.get_settings()
    if settings.synchronous_mode:
        tm.set_synchronous_mode(True)

    ego = get_or_spawn_ego(world)
    ego.set_autopilot(True, tm.get_port())

    tm.distance_to_leading_vehicle(ego, 3.0)
    tm.vehicle_percentage_speed_difference(ego, 0.0)

    print(f"Ego vehicle id: {ego.id}")
    print("Running V2V-influenced autopilot loop...")

    try:
        while True:
            if settings.synchronous_mode:
                world.tick()
            else:
                world.wait_for_tick()

            vehicles = world.get_actors().filter("vehicle.*")
            v2v_stream = build_v2v_stream_from_carla(ego, vehicles)

            if args.attack:
                v2v_stream.append({"vid": -1, "dx": args.attack_distance, "dy": 0.0})

            lead, lead_dist = find_lead_vehicle(v2v_stream)
            if lead is not None and lead_dist < 12.0:
                tm.distance_to_leading_vehicle(ego, 8.0)
                tm.vehicle_percentage_speed_difference(ego, 60.0)
            else:
                tm.distance_to_leading_vehicle(ego, 3.0)
                tm.vehicle_percentage_speed_difference(ego, 0.0)

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
