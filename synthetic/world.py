"""Bistatic site geometry and the aircraft that fly through it.

The delay and Doppler formulae here are a direct port of blah2-api's
`api/lib/bistatic.js`, including its sign conventions, so that a detection this
package emits and the `expected_delay` / `expected_doppler` it carries are
consistent with what a real node would have computed for the same aircraft.

Site defaults describe the Atlanta reference node. Its receiver position is the
one committed in `retina_tracker/config.yaml`; its transmitter and centre
frequency are NOT exposed by any node API, so the values here are a documented
stand-in placed in the VHF-high broadcast band, chosen so the Doppler spread of
the generated stream matches the measured reference. Point `Site` at real
values whenever they are known: nothing else in this package assumes Atlanta.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from retina_tracker.geometry import SPEED_OF_LIGHT, enu2lla, ft2m, lla2ecef

from . import profile

KNOTS_TO_MS = 0.514444
M_TO_FT = 1.0 / 0.3048


@dataclass(frozen=True)
class Site:
    rx_lat: float = 33.939182
    rx_lon: float = -84.651910
    rx_alt: float = 290.0
    tx_lat: float = 33.789000
    tx_lon: float = -84.335000
    tx_alt: float = 300.0
    fc_hz: float = 177_000_000.0

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT / self.fc_hz

    @property
    def rx_ecef(self) -> np.ndarray:
        return np.array(lla2ecef(self.rx_lat, self.rx_lon, self.rx_alt))

    @property
    def tx_ecef(self) -> np.ndarray:
        return np.array(lla2ecef(self.tx_lat, self.tx_lon, self.tx_alt))

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.rx_ecef - self.tx_ecef))


def bistatic(site: Site, pos_ecef: np.ndarray, vel_ecef: np.ndarray) -> tuple[float, float]:
    """Bistatic range (km) and Doppler (Hz) for one target state."""
    to_rx = site.rx_ecef - pos_ecef
    to_tx = site.tx_ecef - pos_ecef
    d_rx = float(np.linalg.norm(to_rx))
    d_tx = float(np.linalg.norm(to_tx))
    delay_km = (d_rx + d_tx - site.baseline_m) / 1000.0
    doppler_hz = float(vel_ecef @ (to_rx / d_rx) + vel_ecef @ (to_tx / d_tx)) / site.wavelength_m
    return delay_km, doppler_hz


def enu_to_ecef_velocity(vel_east: float, vel_north: float, vel_up: float, lat: float, lon: float) -> np.ndarray:
    lat_rad, lon_rad = math.radians(lat), math.radians(lon)
    sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
    sin_lon, cos_lon = math.sin(lon_rad), math.cos(lon_rad)
    return np.array(
        [
            -sin_lon * vel_east - sin_lat * cos_lon * vel_north + cos_lat * cos_lon * vel_up,
            cos_lon * vel_east - sin_lat * sin_lon * vel_north + cos_lat * sin_lon * vel_up,
            cos_lat * vel_north + sin_lat * vel_up,
        ]
    )


@dataclass
class Aircraft:
    """One target on a constant-speed track with an optional steady turn and climb."""

    hex: str
    east_m: float
    north_m: float
    alt_m: float
    speed_ms: float
    heading_deg: float
    turn_rate_deg_s: float = 0.0
    vertical_rate_ms: float = 0.0
    enter_s: float = 0.0
    exit_s: float = math.inf
    adsb_equipped: bool = True
    radar_visible: bool = True
    strength_db: float = 0.0

    def alive(self, t: float) -> bool:
        return self.enter_s <= t <= self.exit_s

    def enu_state(self, t: float) -> tuple[float, float, float, float, float, float]:
        dt = t - self.enter_s
        h0 = math.radians(self.heading_deg)
        omega = math.radians(self.turn_rate_deg_s)
        if abs(omega) < 1e-9:
            east = self.east_m + self.speed_ms * math.sin(h0) * dt
            north = self.north_m + self.speed_ms * math.cos(h0) * dt
            heading = h0
        else:
            radius = self.speed_ms / omega
            east = self.east_m + radius * (math.cos(h0) - math.cos(h0 + omega * dt))
            north = self.north_m + radius * (math.sin(h0 + omega * dt) - math.sin(h0))
            heading = h0 + omega * dt
        up = self.alt_m + self.vertical_rate_ms * dt
        return (
            east,
            north,
            up,
            self.speed_ms * math.sin(heading),
            self.speed_ms * math.cos(heading),
            self.vertical_rate_ms,
        )

    def observe(self, site: Site, t: float) -> dict:
        east, north, up, vel_e, vel_n, vel_u = self.enu_state(t)
        lat, lon, alt = enu2lla(east, north, up, site.rx_lat, site.rx_lon, site.rx_alt)
        pos_ecef = np.array(lla2ecef(lat, lon, alt))
        vel_ecef = enu_to_ecef_velocity(vel_e, vel_n, vel_u, lat, lon)
        delay_km, doppler_hz = bistatic(site, pos_ecef, vel_ecef)
        return {
            "hex": self.hex,
            "lat": lat,
            "lon": lon,
            "alt_ft": alt * M_TO_FT,
            "gs_kt": math.hypot(vel_e, vel_n) / KNOTS_TO_MS,
            "track_deg": math.degrees(math.atan2(vel_e, vel_n)) % 360.0,
            "delay_km": delay_km,
            "doppler_hz": doppler_hz,
            "strength_db": self.strength_db,
            "adsb_equipped": self.adsb_equipped,
        }


CLASSES = (
    {"name": "cruise", "weight": 0.30, "alt_ft": (28000, 43000), "speed_kt": (400, 490), "climb_fpm": (-200, 200)},
    {"name": "climb", "weight": 0.22, "alt_ft": (6000, 24000), "speed_kt": (260, 400), "climb_fpm": (900, 2600)},
    {"name": "descend", "weight": 0.22, "alt_ft": (5000, 22000), "speed_kt": (240, 380), "climb_fpm": (-2400, -800)},
    {"name": "approach", "weight": 0.18, "alt_ft": (1600, 8000), "speed_kt": (130, 250), "climb_fpm": (-900, -200)},
    {"name": "ga", "weight": 0.08, "alt_ft": (2000, 6500), "speed_kt": (85, 155), "climb_fpm": (-400, 400)},
)


@dataclass
class Fleet:
    site: Site
    aircraft: list[Aircraft] = field(default_factory=list)

    def alive(self, t: float) -> list[Aircraft]:
        return [a for a in self.aircraft if a.alive(t)]


def build_fleet(
    site: Site,
    duration_s: float,
    rng: np.random.Generator,
    concurrent: int | None = None,
    spawn_radius_km: float | None = None,
    adsb_equipped_rate: float | None = None,
) -> Fleet:
    """A stream of aircraft crossing the coverage area.

    Aircraft appear on the boundary of a disc around the site and live exactly
    as long as their straight-line chord keeps them inside it, so how long a
    target is available is set by geometry rather than by an independent dwell
    draw. That matters twice over: the population is stationary without any
    hand-placed initial cohort, and aircraft clipping the edge of the disc are
    naturally short-lived, which is the skew real traffic shows.

    Arrivals run from well before the window so the population is already at
    equilibrium by t=0. The chord distribution is heavy-tailed enough that a
    short run-up leaves detections per frame visibly climbing through the run.
    Arrivals that never overlap the window are discarded rather than carried.

    `concurrent` is the mean number airborne inside the disc, and is the lever
    that sets detections per frame: each one contributes about
    `profile.CONTINUITY` detections while it is also inside the delay and
    Doppler window.
    """
    concurrent = profile.CONCURRENT_AIRCRAFT if concurrent is None else concurrent
    equipped = profile.ADSB_EQUIPPED_RATE if adsb_equipped_rate is None else adsb_equipped_rate
    radius_m = (profile.SPAWN_RADIUS_KM if spawn_radius_km is None else spawn_radius_km) * 1000.0
    weights = np.array([c["weight"] for c in CLASSES], dtype=float)
    weights /= weights.sum()

    def draw(radius: float) -> tuple[dict, float, float, float, float, float]:
        cls = CLASSES[rng.choice(len(CLASSES), p=weights)]
        speed_ms = rng.uniform(*cls["speed_kt"]) * KNOTS_TO_MS
        bearing = rng.uniform(0, 2 * math.pi)
        offset = rng.uniform(-0.92, 0.92) * radius
        east = radius * math.sin(bearing) + offset * math.cos(bearing)
        north = radius * math.cos(bearing) - offset * math.sin(bearing)
        norm = math.hypot(east, north) or 1.0
        east, north = east * radius / norm, north * radius / norm
        heading = math.radians(
            (
                math.degrees(math.atan2(-east, -north))
                + rng.uniform(-profile.HEADING_SPREAD_DEG, profile.HEADING_SPREAD_DEG)
            )
            % 360.0
        )
        vel_e, vel_n = speed_ms * math.sin(heading), speed_ms * math.cos(heading)
        chord_s = -2.0 * (east * vel_e + north * vel_n) / (speed_ms * speed_ms)
        return cls, east, north, speed_ms, math.degrees(heading) % 360.0, max(chord_s, 1.0)

    def spawn(enter_s: float, radius: float, visible: bool = True) -> Aircraft:
        cls, east, north, speed_ms, heading, chord_s = draw(radius)
        return Aircraft(
            hex=f"{0xA00000 + int(rng.integers(0, 0x0FFFFF)):06x}",
            east_m=east,
            north_m=north,
            alt_m=ft2m(rng.uniform(*cls["alt_ft"])),
            speed_ms=speed_ms,
            heading_deg=heading,
            turn_rate_deg_s=float(rng.normal(0.0, 0.12)) if rng.random() < 0.35 else 0.0,
            vertical_rate_ms=rng.uniform(*cls["climb_fpm"]) * 0.00508,
            enter_s=enter_s,
            exit_s=enter_s + chord_s,
            adsb_equipped=bool(rng.random() < equipped),
            radar_visible=visible,
            strength_db=float(rng.normal(0.0, profile.TARGET_STRENGTH_SIGMA_DB)),
        )

    def populate(radius: float, mean_alive: float, visible: bool) -> list[Aircraft]:
        out: list[Aircraft] = []
        expected = float(np.mean([draw(radius)[5] for _ in range(1500)]))
        burn_in = expected * 20.0
        n = int(round(mean_alive / expected * (burn_in + duration_s)))
        for _ in range(n):
            enter_s = float(rng.uniform(-burn_in, duration_s))
            candidate = spawn(enter_s, radius, visible=visible)
            if candidate.exit_s >= 0.0 and candidate.enter_s <= duration_s:
                out.append(candidate)
        return out

    aircraft = populate(radius_m, concurrent, True)
    aircraft += populate(profile.ADSB_FEED_RADIUS_KM * 1000.0, profile.ADSB_FEED_AIRCRAFT, False)
    return Fleet(site=site, aircraft=aircraft)
