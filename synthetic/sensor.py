"""The detection process: what a CFAR chain and blah2-api make of a fleet.

Turns true target states into the frame a node would actually emit. The steps
that matter for realism, in the order they are applied:

1. Coverage. A target is observable only inside the delay window and outside
   the +/-15 Hz Doppler blanking, so targets crossing the bistatic ellipse
   tangentially drop out for a while whatever their signal strength.
2. Detection. A two-state Markov chain per target, not an independent draw,
   because real dropouts arrive in bursts (see `profile`).
3. SNR and measurement noise, the noise scaled by SNR the way a real
   estimator's variance is.
4. Clutter, as independent one-shot false alarms.
5. ADS-B association, a port of the matcher in blah2-api's `server.js`, run
   against the whole feed rather than assigned from truth. Targets can miss
   their own match and clutter can steal someone else's, which is what makes
   the dataset useful for testing association.
6. Wire shaping: Doppler-ascending order, 2 dp rounding, and the `alt` key
   that live data actually carries.
"""

from __future__ import annotations

import math

import numpy as np

from . import profile
from .world import KNOTS_TO_MS, Aircraft, Fleet, Site


def _lognormal_params(floor: float, median: float, p95: float) -> tuple[float, float]:
    mu = math.log(median - floor)
    sigma = math.log((p95 - floor) / (median - floor)) / 1.6448536269514722
    return mu, sigma


TARGET_SNR_MU, TARGET_SNR_SIGMA = _lognormal_params(
    profile.SNR_FLOOR_DB, profile.TARGET_SNR_MEDIAN_DB, profile.TARGET_SNR_P95_DB
)
CLUTTER_SNR_MU, CLUTTER_SNR_SIGMA = _lognormal_params(
    profile.SNR_FLOOR_DB, profile.CLUTTER_SNR_MEDIAN_DB, profile.CLUTTER_SNR_P95_DB
)

P_STAY_DETECTED = 1.0 - 1.0 / profile.DETECTED_RUN_MEAN
P_ENTER_DETECTED = profile.CHAIN_CONTINUITY * (1.0 - P_STAY_DETECTED) / (1.0 - profile.CHAIN_CONTINUITY)

NOISE_REFERENCE_SNR_DB = profile.TARGET_SNR_MEDIAN_DB


def in_coverage(delay_km: float, doppler_hz: float) -> bool:
    return (
        profile.DELAY_FLOOR_KM <= delay_km <= profile.DELAY_MAX_KM
        and profile.DOPPLER_BLANK_HZ <= abs(doppler_hz) <= profile.DOPPLER_MAX_HZ
    )


class Sensor:
    def __init__(self, site: Site, rng: np.random.Generator, clutter_per_frame: float | None = None):
        self.site = site
        self.rng = rng
        self.clutter_per_frame = profile.CLUTTER_PER_FRAME if clutter_per_frame is None else clutter_per_frame
        self._detected: dict[str, bool] = {}
        self._adsb_cache: dict[str, tuple[float, dict]] = {}
        self._adsb_slow: dict[str, bool] = {}

    def _step_detection(self, aircraft: Aircraft, covered: bool) -> bool:
        if not covered:
            self._detected[aircraft.hex] = False
            return False
        stay = P_STAY_DETECTED + 0.010 * aircraft.strength_db
        enter = P_ENTER_DETECTED * math.exp(0.055 * aircraft.strength_db)
        was = self._detected.get(aircraft.hex, False)
        threshold = min(max(stay, 0.05), 0.98) if was else min(max(enter, 0.005), 0.9)
        now = bool(self.rng.random() < threshold)
        self._detected[aircraft.hex] = now
        return now

    def _draw_snr(self, aircraft: Aircraft, delay_km: float) -> float:
        tilt = -profile.RANGE_TILT_DB_PER_DECADE * math.log10(max(delay_km, 1.0) / profile.RANGE_TILT_REFERENCE_KM)
        snr = profile.SNR_FLOOR_DB + math.exp(self.rng.normal(TARGET_SNR_MU, TARGET_SNR_SIGMA))
        return max(snr + aircraft.strength_db + tilt + profile.SNR_SELECTION_OFFSET_DB, profile.SNR_FLOOR_DB)

    def _noise_scale(self, snr_db: float) -> float:
        ratio = 10 ** ((NOISE_REFERENCE_SNR_DB - snr_db) / 20.0)
        return float(np.clip(ratio, *profile.NOISE_SCALE_CLIP))

    def _measure(self, delay_km: float, doppler_hz: float, snr_db: float) -> tuple[float, float]:
        scale = self._noise_scale(snr_db)
        delay = delay_km + profile.DELAY_NOISE_SCALE_KM * scale * self.rng.standard_t(profile.DELAY_NOISE_DF)
        doppler = doppler_hz + profile.DOPPLER_NOISE_SCALE_HZ * scale * self.rng.standard_t(profile.DOPPLER_NOISE_DF)
        return delay, doppler

    def _clutter(self) -> list[dict]:
        out = []
        decades = np.array(profile.CLUTTER_DELAY_DECADE_WEIGHTS, dtype=float)
        bands = np.array(profile.CLUTTER_DOPPLER_BAND_WEIGHTS, dtype=float)
        for _ in range(int(self.rng.poisson(self.clutter_per_frame))):
            decade = int(self.rng.choice(len(decades), p=decades / decades.sum()))
            delay = float(
                np.clip(
                    self.rng.uniform(decade * 10.0, decade * 10.0 + 10.0), profile.DELAY_FLOOR_KM, profile.DELAY_MAX_KM
                )
            )
            for _ in range(24):
                band = int(self.rng.choice(len(bands), p=bands / bands.sum()))
                low = -profile.DOPPLER_MAX_HZ + band * profile.CLUTTER_DOPPLER_BAND_HZ
                doppler = float(self.rng.uniform(low, low + profile.CLUTTER_DOPPLER_BAND_HZ))
                if abs(doppler) >= profile.DOPPLER_BLANK_HZ:
                    break
            else:
                continue
            snr = profile.SNR_FLOOR_DB + math.exp(self.rng.normal(CLUTTER_SNR_MU, CLUTTER_SNR_SIGMA))
            out.append({"delay": delay, "doppler": doppler, "snr": snr, "source": None})
        return out

    def _adsb_feed(self, fleet: Fleet, t: float) -> list[dict]:
        """Feed entries a detection could conceivably match.

        A real node scores every aircraft tar1090 knows about, but anything
        whose bistatic delay lies outside the map plus the gate can never win,
        so dropping those early is behaviour-preserving and keeps generation
        fast with a realistically large feed.
        """
        low = profile.DELAY_FLOOR_KM - profile.ADSB_DELAY_TOLERANCE_KM
        high = profile.DELAY_MAX_KM + profile.ADSB_DELAY_TOLERANCE_KM
        feed = []
        for aircraft in fleet.alive(t):
            if not aircraft.adsb_equipped:
                continue
            cached = self._adsb_cache.get(aircraft.hex)
            if cached is None or t >= cached[0]:
                state = aircraft.observe(self.site, t)
                if aircraft.hex not in self._adsb_slow:
                    self._adsb_slow[aircraft.hex] = bool(self.rng.random() < profile.ADSB_STALE_FRACTION)
                window = profile.ADSB_REFRESH_SLOW_S if self._adsb_slow[aircraft.hex] else profile.ADSB_REFRESH_FAST_S
                expiry = t + float(self.rng.uniform(*window))
                self._adsb_cache[aircraft.hex] = (expiry, state)
            else:
                state = cached[1]
            if low <= state["delay_km"] <= high:
                feed.append(state)
        return feed

    def _match(self, delay: float, doppler: float, feed: list[dict]) -> dict | None:
        best, best_score = None, math.inf
        for ac in feed:
            delay_err = abs(delay - ac["delay_km"])
            doppler_err = abs(doppler - ac["doppler_hz"])
            if delay_err >= profile.ADSB_DELAY_TOLERANCE_KM or doppler_err >= profile.ADSB_DOPPLER_TOLERANCE_HZ:
                continue
            score = delay_err / profile.ADSB_DELAY_TOLERANCE_KM + doppler_err / profile.ADSB_DOPPLER_TOLERANCE_HZ
            if score < best_score:
                best_score = score
                best = {
                    "hex": ac["hex"],
                    "lat": round(ac["lat"], 6),
                    "lon": round(ac["lon"], 6),
                    "alt": round(ac["alt_ft"] / 25.0) * 25,
                    "gs": round(ac["gs_kt"], 1),
                    "track": round(ac["track_deg"], 2),
                    "expected_delay": round(ac["delay_km"], 2),
                    "expected_doppler": round(ac["doppler_hz"], 2),
                    "delay_residual": round(delay - ac["delay_km"], 2),
                    "doppler_residual": round(doppler - ac["doppler_hz"], 2),
                }
        return best

    def frame(self, fleet: Fleet, t: float, timestamp_ms: int) -> tuple[dict, dict]:
        raw: list[dict] = []
        targets: dict[str, dict] = {}

        for aircraft in fleet.alive(t):
            if not aircraft.radar_visible:
                continue
            state = aircraft.observe(self.site, t)
            covered = in_coverage(state["delay_km"], state["doppler_hz"])
            detected = self._step_detection(aircraft, covered)
            targets[aircraft.hex] = {
                "delay": round(state["delay_km"], 3),
                "doppler": round(state["doppler_hz"], 3),
                "in_coverage": covered,
                "detected": detected,
                "adsb_equipped": aircraft.adsb_equipped,
            }
            if not detected:
                continue
            snr = self._draw_snr(aircraft, state["delay_km"])
            delay, doppler = self._measure(state["delay_km"], state["doppler_hz"], snr)
            if not in_coverage(delay, doppler):
                targets[aircraft.hex]["detected"] = False
                self._detected[aircraft.hex] = False
                continue
            raw.append({"delay": delay, "doppler": doppler, "snr": snr, "source": aircraft.hex})

        raw.extend(self._clutter())
        raw.sort(key=lambda d: d["doppler"])
        if len(raw) > 1 and self.rng.random() < profile.DOPPLER_SORT_INVERSION_RATE:
            i = int(self.rng.integers(0, len(raw) - 1))
            raw[i], raw[i + 1] = raw[i + 1], raw[i]

        for d in raw:
            d["delay"] = round(d["delay"], 2)
            d["doppler"] = round(d["doppler"], 2)
            d["snr"] = round(d["snr"], 2)

        feed = self._adsb_feed(fleet, t)
        adsb = [self._match(d["delay"], d["doppler"], feed) for d in raw]

        frame = {
            "timestamp": timestamp_ms,
            "delay": [d["delay"] for d in raw],
            "doppler": [d["doppler"] for d in raw],
            "snr": [d["snr"] for d in raw],
            "adsb": adsb,
        }
        truth = {
            "timestamp": timestamp_ms,
            "t": round(t, 3),
            "sources": [d["source"] for d in raw],
            "matched": [a["hex"] if a else None for a in adsb],
            "targets": targets,
        }
        return frame, truth


def knots(speed_ms: float) -> float:
    return speed_ms / KNOTS_TO_MS
