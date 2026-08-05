"""Vendored motion patterns for test_integration_synthetic.py.

The integration suite imported these from a `motion_patterns` module that
shipped with a long-gone `synthetic-adsb` project — the import failed at
collection, so the entire 1,100-line suite (supersonic, instant-turn,
spoofing, missed-detection scenarios) had been silently skipped in CI for
its whole life.  This is a minimal reimplementation of the six generators
against the API the suite actually uses: each pattern exposes
``get_position(t) -> (lat, lon)`` for wall-clock ``t``.

Kinematics are deliberately simple (flat-earth degrees at the test site's
latitude); the suite asserts tracker *behaviour* — anomaly flags, velocity
classes — not geodesy.
"""

import math

_KM_PER_DEG_LAT = 111.1949
_KNOTS_TO_MS = 0.514444
_MACH_1_MS = 343.0


class _VelocityFromPosition:
    """get_velocity (knots) and get_heading (deg) by numeric differentiation
    of get_position — correct for every pattern within a motion segment."""

    _DT = 0.1

    def get_velocity(self, t: float) -> float:
        lat1, lon1 = self.get_position(t)
        lat2, lon2 = self.get_position(t + self._DT)
        north_m = (lat2 - lat1) * _KM_PER_DEG_LAT * 1000.0
        east_m = (lon2 - lon1) * _KM_PER_DEG_LAT * 1000.0 * max(math.cos(math.radians(lat1)), 1e-6)
        return math.hypot(north_m, east_m) / self._DT / _KNOTS_TO_MS

    def get_heading(self, t: float) -> float:
        lat1, lon1 = self.get_position(t)
        lat2, lon2 = self.get_position(t + self._DT)
        north = lat2 - lat1
        east = (lon2 - lon1) * max(math.cos(math.radians(lat1)), 1e-6)
        return math.degrees(math.atan2(east, north)) % 360.0


def _deg_offsets(distance_m: float, direction_deg: float, lat: float):
    """(dlat, dlon) for a ground displacement along a compass bearing."""
    north_m = distance_m * math.cos(math.radians(direction_deg))
    east_m = distance_m * math.sin(math.radians(direction_deg))
    dlat = north_m / (_KM_PER_DEG_LAT * 1000.0)
    dlon = east_m / (_KM_PER_DEG_LAT * 1000.0 * max(math.cos(math.radians(lat)), 1e-6))
    return dlat, dlon


class CircularMotion(_VelocityFromPosition):
    """Constant-rate orbit around a centre point."""

    def __init__(self, center_lat, center_lon, radius_deg, angular_speed, start_time: float = 0.0):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius_deg = radius_deg
        self.angular_speed = angular_speed  # rad/s
        self.start_time = start_time

    def get_position(self, t: float):
        theta = self.angular_speed * (t - self.start_time)
        return (
            self.center_lat + self.radius_deg * math.sin(theta),
            self.center_lon + self.radius_deg * math.cos(theta),
        )


class _LinearMotion(_VelocityFromPosition):
    def __init__(self, start_lat, start_lon, speed_ms, direction_deg, start_time):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.speed_ms = speed_ms
        self.direction_deg = direction_deg
        self.start_time = start_time

    def get_position(self, t: float):
        dist = self.speed_ms * max(t - self.start_time, 0.0)
        dlat, dlon = _deg_offsets(dist, self.direction_deg, self.start_lat)
        return self.start_lat + dlat, self.start_lon + dlon


class SupersonicLinearMotion(_LinearMotion):
    """Straight-line flight at a fixed Mach number."""

    def __init__(self, start_lat, start_lon, mach_number, direction_deg, start_time=0.0):
        super().__init__(start_lat, start_lon, mach_number * _MACH_1_MS, direction_deg, start_time)
        self.mach_number = mach_number


class _SegmentedHeadingMotion(_VelocityFromPosition):
    """Constant speed; heading jumps by 90° every change_interval_sec."""

    def __init__(self, start_lat, start_lon, speed_ms, initial_direction_deg, change_interval_sec, start_time):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.speed_ms = speed_ms
        self.initial_direction_deg = initial_direction_deg
        self.change_interval_sec = change_interval_sec
        self.start_time = start_time

    def get_position(self, t: float):
        elapsed = max(t - self.start_time, 0.0)
        lat, lon = self.start_lat, self.start_lon
        seg = 0
        while elapsed > 0:
            step = min(elapsed, self.change_interval_sec)
            heading = (self.initial_direction_deg + 90.0 * seg) % 360.0
            dlat, dlon = _deg_offsets(self.speed_ms * step, heading, lat)
            lat += dlat
            lon += dlon
            elapsed -= step
            seg += 1
        return lat, lon


class InstantDirectionChangeMotion(_SegmentedHeadingMotion):
    def __init__(
        self, start_lat, start_lon, velocity_knots, initial_direction_deg, change_interval_sec, start_time=0.0
    ):
        super().__init__(
            start_lat, start_lon, velocity_knots * _KNOTS_TO_MS, initial_direction_deg, change_interval_sec, start_time
        )


class SupersonicDirectionChangeMotion(_SegmentedHeadingMotion):
    def __init__(self, start_lat, start_lon, mach_number, initial_direction_deg, change_interval_sec, start_time=0.0):
        super().__init__(
            start_lat, start_lon, mach_number * _MACH_1_MS, initial_direction_deg, change_interval_sec, start_time
        )


class _ProfileSpeedMotion(_VelocityFromPosition):
    """Fixed heading; speed follows [(duration_s, speed), ...] segments,
    holding the last speed after the profile is exhausted."""

    def __init__(self, start_lat, start_lon, direction_deg, profile_ms, start_time):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.direction_deg = direction_deg
        self.profile_ms = profile_ms
        self.start_time = start_time

    def get_position(self, t: float):
        elapsed = max(t - self.start_time, 0.0)
        dist = 0.0
        last_speed = self.profile_ms[-1][1] if self.profile_ms else 0.0
        for duration, speed in self.profile_ms:
            step = min(elapsed, duration)
            dist += speed * step
            elapsed -= step
            if elapsed <= 0:
                break
        if elapsed > 0:
            dist += last_speed * elapsed
        dlat, dlon = _deg_offsets(dist, self.direction_deg, self.start_lat)
        return self.start_lat + dlat, self.start_lon + dlon


class InstantAccelerationMotion(_ProfileSpeedMotion):
    def __init__(self, start_lat, start_lon, direction_deg, speed_profile, start_time=0.0):
        # profile entries are (duration_s, speed_knots)
        super().__init__(
            start_lat, start_lon, direction_deg, [(d, s * _KNOTS_TO_MS) for d, s in speed_profile], start_time
        )


class SupersonicAccelerationMotion(_ProfileSpeedMotion):
    def __init__(self, start_lat, start_lon, direction_deg, speed_profile_mach, start_time=0.0):
        # profile entries are (duration_s, mach)
        super().__init__(
            start_lat, start_lon, direction_deg, [(d, m * _MACH_1_MS) for d, m in speed_profile_mach], start_time
        )
