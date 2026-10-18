"""The node's ADS-B-proven coverage wedge, and delay arcs cut to it.

Positions come only from ADS-B fixes blah2 matched to a detection, never from
the tracker's own estimate. See docs/coverage-wedge.md.
"""

import json
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .config import (
    COVERAGE_BIN_DEG,
    COVERAGE_HOLD_S,
    COVERAGE_MAX_POSITION_AGE_S,
    COVERAGE_MIN_PASS_DETECTIONS,
)

EARTH_RADIUS_KM = 6371.0
SCHEMA = 1
DEFAULT_BIN_DEG = 5.0
MIN_RANGE_KM = 0.5
ARC_MIN_DIFFERENTIAL_KM = 3.0
ARC_QUANTUM_KM = 0.03
ARC_CACHE_SIZE = 4096
LOBE_STEP_DEG = 1.0
HELD_REFRESH_MS = 10_000
SPAN_LIMITED_FRACTION = 0.85
HIGH_WATER_MEMORY = 512
REPLAY_MAX_ARCS = 240
SAVE_INTERVAL_S = 60
DP = 5


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.subtract(lon2, lon1))
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(x, y)) % 360.0


def ground_range_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.subtract(lon2, lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def destination(lat, lon, bearing, range_km):
    p1, l1 = np.radians(lat), np.radians(lon)
    th = np.radians(bearing)
    d = np.asarray(range_km, dtype=float) / EARTH_RADIUS_KM
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(th))
    l2 = l1 + np.arctan2(np.sin(th) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2), (np.degrees(l2) + 540.0) % 360.0 - 180.0


def ellipse_range_km(differential_km, baseline_km, psi_deg):
    """Ground range from the RX focus to the constant-differential ellipse, psi off the RX->TX bearing."""
    d, lb = differential_km, baseline_km
    return d * (d + 2 * lb) / (2 * ((lb + d) - lb * np.cos(np.radians(psi_deg))))


def verified_position(detection, track_hex, max_age_s=None):
    """The ADS-B fix a detection was matched to, if it can stand as this track's position.

    See docs/coverage-wedge.md#what-counts-as-a-verified-position.
    """
    adsb = detection.get("adsb")
    if not track_hex or not isinstance(adsb, dict) or adsb.get("hex") != track_hex:
        return None
    limit = COVERAGE_MAX_POSITION_AGE_S() if max_age_s is None else max_age_s
    age = adsb.get("position_age")
    if not _finite(age) or abs(age) > limit:
        return None
    lat, lon = adsb.get("lat"), adsb.get("lon")
    if not (_finite(lat) and _finite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return float(lat), float(lon)


@dataclass(frozen=True)
class Geometry:
    rx_lat: float
    rx_lon: float
    tx_lat: float
    tx_lon: float

    @classmethod
    def from_location(cls, location):
        """From blah2's `location` block, or None when either end is not placed."""
        try:
            rx, tx = location["rx"], location["tx"]
            values = [rx["latitude"], rx["longitude"], tx["latitude"], tx["longitude"]]
        except (KeyError, TypeError):
            return None
        if not all(_finite(v) for v in values):
            return None
        rx_lat, rx_lon, tx_lat, tx_lon = (float(v) for v in values)
        if not (-90 <= rx_lat <= 90 and -90 <= tx_lat <= 90 and -180 <= rx_lon <= 180 and -180 <= tx_lon <= 180):
            return None
        if (rx_lat, rx_lon) == (0.0, 0.0) or (tx_lat, tx_lon) == (0.0, 0.0):
            return None
        return cls(rx_lat, rx_lon, tx_lat, tx_lon)

    @property
    def baseline_km(self):
        return float(ground_range_km(self.rx_lat, self.rx_lon, self.tx_lat, self.tx_lon))

    @property
    def tx_bearing_deg(self):
        return float(bearing_deg(self.rx_lat, self.rx_lon, self.tx_lat, self.tx_lon))

    def key(self):
        return [round(v, DP) for v in (self.rx_lat, self.rx_lon, self.tx_lat, self.tx_lon)]

    def to_dict(self):
        return {
            "rx": {"lat": self.rx_lat, "lon": self.rx_lon},
            "tx": {"lat": self.tx_lat, "lon": self.tx_lon},
            "baseline_km": round(self.baseline_km, 3),
        }


@dataclass
class Pass:
    """One aircraft's verified detections in one bearing bin, under one track."""

    hex: str
    bin: int
    n: int
    first_ms: int
    last_ms: int
    range_km: float

    def to_list(self):
        return [self.hex, self.bin, self.n, self.first_ms, self.last_ms, round(self.range_km, 3)]

    @classmethod
    def from_list(cls, row):
        hex_, bin_, n, first_ms, last_ms, range_km = row
        return cls(str(hex_), int(bin_), int(n), int(first_ms), int(last_ms), float(range_km))


def _point(lat, lon):
    return [round(float(lat), DP), round(float(lon), DP)]


def _now_ms():
    return int(time.time() * 1000)


class Coverage:
    """Records ADS-B-verified passes and serves the sliding max-hold wedge.

    Written from the frame thread through write_event, read from HTTP threads,
    so every method that touches passes takes the lock.
    """

    def __init__(
        self,
        geometry,
        bin_deg=None,
        hold_s=None,
        min_pass_detections=None,
        state_path=None,
        now_ms=None,
        max_differential_km=None,
    ):
        self._lock = threading.Lock()
        self.geometry = geometry
        bin_deg = float(bin_deg or COVERAGE_BIN_DEG())
        if bin_deg <= 0 or abs(360.0 / bin_deg - round(360.0 / bin_deg)) > 1e-9:
            print(f"Coverage bin of {bin_deg} deg does not divide 360; using {DEFAULT_BIN_DEG}", file=sys.stderr)
            bin_deg = DEFAULT_BIN_DEG
        self.bin_deg = bin_deg
        self.n_bins = int(round(360.0 / bin_deg))
        self.hold_s = float(hold_s or COVERAGE_HOLD_S())
        self.min_pass_detections = int(min_pass_detections or COVERAGE_MIN_PASS_DETECTIONS())
        self.state_path = state_path
        self.max_differential_km = max_differential_km if _finite(max_differential_km) else None
        self._live = {}
        self._closed = []
        self._through = OrderedDict()
        self.started_ms = _now_ms() if now_ms is None else now_ms
        self.reset_ms = None
        self.reset_reason = None
        self._held = None
        self._arcs = OrderedDict()

    # ── Recording ──────────────────────────────────────────────

    def write_event(self, track_id, timestamp, length, detections, adsb_hex=None, **_unused):
        """The tracker's event-writer duck type. Each event repeats a window, so a high-water mark dedupes it."""
        if self.geometry is None or not adsb_hex:
            return
        with self._lock:
            through = self._through.get(track_id)
            newest = through
            for det in detections:
                ts = det["timestamp"]
                if through is not None and ts <= through:
                    continue
                newest = ts if newest is None else max(newest, ts)
                position = verified_position(det, adsb_hex)
                if position is not None:
                    self._add(track_id, adsb_hex, int(ts), *position)
            if newest is not None:
                self._through[track_id] = newest
                self._through.move_to_end(track_id)
                while len(self._through) > HIGH_WATER_MEMORY:
                    self._through.popitem(last=False)

    def _add(self, track_id, hex_, ts, lat, lon):
        g = self.geometry
        range_km = float(ground_range_km(g.rx_lat, g.rx_lon, lat, lon))
        if range_km < MIN_RANGE_KM:
            return
        bin_ = int(float(bearing_deg(g.rx_lat, g.rx_lon, lat, lon)) // self.bin_deg) % self.n_bins
        key = (track_id, hex_, bin_)
        found = self._live.get(key)
        if found is None:
            self._live[key] = Pass(hex_, bin_, 1, ts, ts, range_km)
        else:
            found.n += 1
            found.first_ms = min(found.first_ms, ts)
            found.last_ms = max(found.last_ms, ts)
            found.range_km = max(found.range_km, range_km)

    def reset(self, reason="manual", now_ms=None):
        now = _now_ms() if now_ms is None else now_ms
        with self._lock:
            self._clear(reason, now)
        self.save()

    def _clear(self, reason, now):
        self._live = {}
        self._closed = []
        self._through = OrderedDict()
        self.started_ms = now
        self.reset_ms = now
        self.reset_reason = reason
        self._held = None
        self._arcs = OrderedDict()

    # ── The wedge ──────────────────────────────────────────────

    def _prune(self, now):
        cutoff = now - self.hold_s * 1000
        self._live = {k: p for k, p in self._live.items() if p.last_ms >= cutoff}
        self._closed = [p for p in self._closed if p.last_ms >= cutoff]

    def _passes(self):
        return [p for p in (*self._live.values(), *self._closed) if p.n >= self.min_pass_detections]

    def _held_bins(self, now):
        """Caller holds the lock. Per bin, the second-best range among distinct aircraft."""
        cached = self._held
        if cached is not None and 0 <= now - cached[0] < HELD_REFRESH_MS:
            return cached[1], cached[2]
        self._prune(now)
        best = {}
        for p in self._passes():
            key = (p.bin, p.hex)
            best[key] = max(best.get(key, 0.0), p.range_km)
        per_bin = [[] for _ in range(self.n_bins)]
        for (bin_, _), range_km in best.items():
            per_bin[bin_].append(range_km)
        held = tuple(sorted(r)[-2] if len(r) >= 2 else 0.0 for r in per_bin)
        aircraft = len({hex_ for _, hex_ in best})
        self._held = (now, held, aircraft)
        return held, aircraft

    def held(self, now_ms=None):
        now = _now_ms() if now_ms is None else now_ms
        with self._lock:
            return self._held_bins(now)[0]

    def lobes(self, held):
        """Rings of [lat, lon], one per run of open bins, each bin drawn edge to edge."""
        g = self.geometry
        if g is None or not any(held):
            return []
        apex = _point(g.rx_lat, g.rx_lon)
        n = self.n_bins
        if all(held):
            return [self._bin_edges(range(n), held)]
        start = next(i for i in range(n) if not held[i])
        rings, run = [], []
        for k in range(1, n + 1):
            i = (start + k) % n
            if held[i] and k < n:
                run.append(i)
            elif run:
                rings.append([apex, *self._bin_edges(run, held), apex])
                run = []
        return rings

    def _bin_edges(self, bins, held):
        g = self.geometry
        steps = max(1, int(round(self.bin_deg / LOBE_STEP_DEG)))
        points = []
        for i in bins:
            bearings = i * self.bin_deg + np.linspace(0.0, self.bin_deg, steps + 1)
            lat, lon = destination(g.rx_lat, g.rx_lon, bearings, np.full(bearings.shape, held[i]))
            points.extend(_point(a, b) for a, b in zip(lat, lon))
        return points

    def span_km(self):
        """Per bin, the ground range at the bin centre of the furthest differential blah2 processes, or None."""
        g, d = self.geometry, self.max_differential_km
        if g is None or d is None or d <= 0:
            return None
        centres = (np.arange(self.n_bins) + 0.5) * self.bin_deg
        return [float(r) for r in ellipse_range_km(d, g.baseline_km, centres - g.tx_bearing_deg)]

    def span_ring(self):
        """The processed-range limit as a closed [lat, lon] ring.

        See docs/coverage-wedge.md#the-processed-range-limit.
        """
        g, d = self.geometry, self.max_differential_km
        if g is None or d is None or d <= 0:
            return []
        bearings = np.arange(0.0, 360.0 + LOBE_STEP_DEG / 2, LOBE_STEP_DEG)
        lat, lon = destination(
            g.rx_lat, g.rx_lon, bearings, ellipse_range_km(d, g.baseline_km, bearings - g.tx_bearing_deg)
        )
        return [_point(a, b) for a, b in zip(lat, lon)]

    def state(self, now_ms=None):
        """Everything GET /coverage serves."""
        now = _now_ms() if now_ms is None else now_ms
        with self._lock:
            held, aircraft = self._held_bins(now)
            started, reset_ms, reason = self.started_ms, self.reset_ms, self.reset_reason
        span = self.span_km()
        limited = [bool(span and h > 0 and h >= SPAN_LIMITED_FRACTION * s) for h, s in zip(held, span or held)]
        return {
            "enabled": self.geometry is not None,
            "now_ms": now,
            "geometry": None if self.geometry is None else self.geometry.to_dict(),
            "bin_deg": self.bin_deg,
            "hold_s": self.hold_s,
            "min_pass_detections": self.min_pass_detections,
            "since_ms": int(max(now - self.hold_s * 1000, started)),
            "reset_ms": reset_ms,
            "reset_reason": reason,
            "held_km": [round(h, 2) for h in held],
            "proven_bins": sum(1 for h in held if h > 0),
            "aircraft": aircraft,
            "lobes": self.lobes(held),
            "max_differential_km": self.max_differential_km,
            "span_ring": self.span_ring(),
            "limited": limited,
            "limited_bins": sum(limited),
        }

    # ── Arcs ───────────────────────────────────────────────────

    def arc(self, differential_km, now_ms=None):
        """The delay ellipse as {"solid": [...], "faint": [...]} polylines, or None.

        Solid where it lies inside the held wedge, faint elsewhere. Flat, so a
        target at altitude sits off its own arc. See docs/coverage-wedge.md#arcs.
        """
        if self.geometry is None or not _finite(differential_km) or differential_km < ARC_MIN_DIFFERENTIAL_KM:
            return None
        quantum = round(differential_km / ARC_QUANTUM_KM)
        held = self.held(now_ms)
        key = (quantum, held)
        with self._lock:
            cached = self._arcs.get(key)
            if cached is not None:
                self._arcs.move_to_end(key)
                return cached
        built = self._build_arc(quantum * ARC_QUANTUM_KM, held)
        with self._lock:
            self._arcs[key] = built
            while len(self._arcs) > ARC_CACHE_SIZE:
                self._arcs.popitem(last=False)
        return built

    def _build_arc(self, differential_km, held):
        g = self.geometry
        step = self.bin_deg / 2
        n = int(round(360.0 / step))
        bearings = np.arange(n + 1) * step
        ranges = ellipse_range_km(differential_km, g.baseline_km, bearings - g.tx_bearing_deg)
        lat, lon = destination(g.rx_lat, g.rx_lon, bearings, ranges)
        points = [_point(a, b) for a, b in zip(lat, lon)]
        out = {"solid": [], "faint": []}
        state = {"kind": None, "line": None}

        def emit(kind, a, b):
            if kind != state["kind"]:
                state["line"] = [a]
                out[kind].append(state["line"])
                state["kind"] = kind
            state["line"].append(b)

        for k in range(n):
            limit = held[(k // 2) % self.n_bins]
            g0, g1 = ranges[k] - limit, ranges[k + 1] - limit
            kind0 = "solid" if g0 <= 0 else "faint"
            kind1 = "solid" if g1 <= 0 else "faint"
            if kind0 == kind1:
                emit(kind0, points[k], points[k + 1])
                continue
            f = g0 / (g0 - g1)
            cross_lat, cross_lon = destination(
                g.rx_lat, g.rx_lon, bearings[k] + f * step, ranges[k] + f * (ranges[k + 1] - ranges[k])
            )
            cross = _point(cross_lat, cross_lon)
            emit(kind0, points[k], cross)
            emit(kind1, cross, points[k + 1])
        return out

    # ── Persistence ────────────────────────────────────────────

    def to_dict(self):
        with self._lock:
            return {
                "schema": SCHEMA,
                "geometry": None if self.geometry is None else self.geometry.key(),
                "bin_deg": self.bin_deg,
                "started_ms": self.started_ms,
                "reset_ms": self.reset_ms,
                "reset_reason": self.reset_reason,
                "passes": [p.to_list() for p in (*self._live.values(), *self._closed)],
            }

    def save(self):
        if not self.state_path:
            return
        tmp = self.state_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self.to_dict(), f, separators=(",", ":"))
            os.replace(tmp, self.state_path)
        except OSError as e:
            print(f"Could not save coverage to {self.state_path}: {e}", file=sys.stderr)

    def load(self, now_ms=None):
        """Restore saved passes, or start again when they were held under another geometry."""
        now = _now_ms() if now_ms is None else now_ms
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as f:
                saved = json.load(f)
            passes = [Pass.from_list(row) for row in saved.get("passes", [])]
        except (OSError, ValueError, TypeError) as e:
            print(f"Ignoring unreadable coverage state {self.state_path}: {e}", file=sys.stderr)
            return
        with self._lock:
            if saved.get("schema") != SCHEMA or saved.get("bin_deg") != self.bin_deg:
                print("Coverage state is from another schema or bin width; starting again", file=sys.stderr)
                self._clear("format", now)
                return
            if self.geometry is None or saved.get("geometry") != self.geometry.key():
                print("Receiver or transmitter moved since the coverage was saved; starting again", file=sys.stderr)
                self._clear("geometry", now)
                return
            self._closed = [p for p in passes if 0 <= p.bin < self.n_bins]
            self.started_ms = int(saved.get("started_ms") or now)
            self.reset_ms = saved.get("reset_ms")
            self.reset_reason = saved.get("reset_reason")
            self._prune(now)


def replay_arcs(coverage, track, now_ms=None):
    """At most REPLAY_MAX_ARCS arcs spread evenly over a track's points, for the page's replay."""
    times, delays = track["t"], track["delay"]
    n = len(times)
    if n == 0:
        return []
    picks = sorted({int(round(i)) for i in np.linspace(0, n - 1, min(n, REPLAY_MAX_ARCS))})
    return [{"t": times[i], "delay": delays[i], "arc": coverage.arc(delays[i], now_ms)} for i in picks]


def start_saver(coverage, interval_s=SAVE_INTERVAL_S, stop_event=None):
    """Save on a cadence, so a restart loses at most one interval of passes."""

    def loop():
        while True:
            if stop_event is not None:
                if stop_event.wait(interval_s):
                    return
            else:
                time.sleep(interval_s)
            coverage.save()

    thread = threading.Thread(target=loop, daemon=True, name="coverage-saver")
    thread.start()
    return thread
