import math


class V2VController:
    """
    Simple V2V-aware longitudinal controller with decision/planning/control steps.
    Assumes BSM-like data stream with fields: vid, latitude, longitude, heading, velocity.
    """

    def __init__(
        self,
        ego_vid,
        desired_speed=10.0,
        max_accel=2.0,
        max_decel=-4.0,
        time_headway=1.2,
        min_gap=2.0,
        lane_half_width=2.0,
        max_heading_diff_deg=50.0,
        accel_exp=4.0,
        max_jerk=1.5,
        smoothing=0.6,
    ):
        self.ego_vid = ego_vid
        self.desired_speed = desired_speed
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.time_headway = time_headway
        self.min_gap = min_gap
        self.lane_half_width = lane_half_width
        self.max_heading_diff_deg = max_heading_diff_deg
        self.accel_exp = accel_exp
        self.max_jerk = max_jerk
        self.smoothing = smoothing
        self.prev_accel = 0.0

    def compute_acceleration(self, data_stream, dt=0.1):
        ego = self._get_ego(data_stream)
        if ego is None:
            return 0.0
        lead = self._decide_lead_vehicle(ego, data_stream)
        target_accel = self._plan_acceleration(ego, lead)
        return self._apply_limits(target_accel, dt)

    def _get_ego(self, data_stream):
        for v in data_stream:
            if v.get("vid") == self.ego_vid:
                return v
        return None

    def _decide_lead_vehicle(self, ego, data_stream):
        ego_lat = ego["latitude"]
        ego_lon = ego["longitude"]
        ego_heading = ego["heading"]

        best = None
        best_dist = float("inf")

        for v in data_stream:
            if v.get("vid") == self.ego_vid:
                continue
            if not self._heading_compatible(ego_heading, v.get("heading")):
                continue
            dx, dy = self._enu_delta(ego_lat, ego_lon, v["latitude"], v["longitude"])
            longitudinal, lateral = self._project_to_ego(ego_heading, dx, dy)
            if longitudinal <= 0:
                continue
            if abs(lateral) > self.lane_half_width:
                continue
            if longitudinal < best_dist:
                best = v
                best_dist = longitudinal

        if best is None:
            return None

        return {
            "vehicle": best,
            "distance": best_dist,
        }

    def _plan_acceleration(self, ego, lead):
        v = max(0.0, ego["velocity"])
        a = self.max_accel
        b = max(0.1, -self.max_decel)
        v0 = max(0.1, self.desired_speed)

        if lead is None:
            accel = a * (1.0 - (v / v0) ** self.accel_exp)
            return accel

        lead_vehicle = lead["vehicle"]
        s = max(0.1, lead["distance"])
        dv = v - max(0.0, lead_vehicle["velocity"])

        s_star = self.min_gap + v * self.time_headway + (v * dv) / (2.0 * math.sqrt(a * b))
        accel = a * (1.0 - (v / v0) ** self.accel_exp - (s_star / s) ** 2)
        return accel

    def _apply_limits(self, accel, dt):
        accel = max(self.max_decel, min(self.max_accel, accel))
        max_delta = self.max_jerk * max(0.01, dt)
        accel = max(self.prev_accel - max_delta, min(self.prev_accel + max_delta, accel))
        accel = self.smoothing * accel + (1.0 - self.smoothing) * self.prev_accel
        self.prev_accel = accel
        return accel

    def _heading_compatible(self, ego_heading, other_heading):
        if other_heading is None:
            return True
        diff = abs(((other_heading - ego_heading + 180.0) % 360.0) - 180.0)
        return diff <= self.max_heading_diff_deg

    def _enu_delta(self, lat1, lon1, lat2, lon2):
        r = 6371000.0
        x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2.0)) * r
        y = math.radians(lat2 - lat1) * r
        return x, y

    def _project_to_ego(self, heading_deg, dx, dy):
        heading = math.radians(heading_deg)
        forward_x = math.sin(heading)
        forward_y = math.cos(heading)
        right_x = math.cos(heading)
        right_y = -math.sin(heading)
        longitudinal = dx * forward_x + dy * forward_y
        lateral = dx * right_x + dy * right_y
        return longitudinal, lateral
