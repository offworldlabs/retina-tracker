"""Track class and state management for radar tracking."""

from collections import deque
from datetime import datetime
from enum import Enum

import numpy as np

from .config import (
    DOPPLER_BIN_HZ,
    DOPPLER_BLANK_HZ,
    GATE_THRESHOLD,
    INITIAL_RANGE_ACCEL_VARIANCE,
    M_THRESHOLD,
    MEASUREMENT_NOISE_DELAY_KM2,
    MEASUREMENT_NOISE_DOPPLER_HZ2,
    N_COAST,
    N_DELETE,
    N_WINDOW,
    PROCESS_NOISE_ADAPTIVE,
    PROCESS_NOISE_MAX_SCALE,
    PROCESS_NOISE_NIS_MEMORY,
    SHADOW_MIN_FRACTION,
    WAVELENGTH_KM,
)
from .kalman import MEASUREMENT_DIM, doppler_to_range_rate, range_rate_to_doppler

TRACK_HISTORY_MAX = 600
MAX_OCCLUDED_S = 120.0


class TrackState(Enum):
    """Track states following blah2 architecture."""

    TENTATIVE = 0
    ASSOCIATED = 1
    ACTIVE = 2
    COASTING = 3


class Track:
    """Represents a single radar track with state history."""

    _daily_counter = 0
    _last_date = None

    @property
    def state_status(self):
        return self._state_status

    @state_status.setter
    def state_status(self, value):
        """Latches ever_confirmed on the way through ACTIVE.

        A track reaches ACTIVE by three routes: promotion on M-of-N, a
        coasting track re-associating, and tracklet initiation. Latching at
        the assignment catches all of them, and any fourth added later,
        which matters because the flag is load-bearing for detection
        classification and a missed route reads as "this detection was never
        in a confirmed track".

        Its current status cannot stand in for the flag. A track that was
        ACTIVE and has since coasted or been deleted reads the same as one
        that never got there.
        """
        self._state_status = value
        if value is TrackState.ACTIVE:
            self.ever_confirmed = True

    def __init__(self, detection, timestamp, kf, frame=0, config=None):
        self.id = None
        # Before state_status: its setter latches ever_confirmed.
        self.ever_confirmed = False
        self.retired = False
        self.state_status = TrackState.TENTATIVE
        self.kf = kf
        self.adsb_hex = None
        self.adsb_initialized = False

        adsb_config = config.get("adsb", {}) if config else {}
        if detection.get("adsb") and adsb_config.get("enabled") and adsb_config.get("reference_location"):
            self._init_from_adsb(detection, adsb_config)
        else:
            self._init_from_delay_doppler(detection)

        # Capture ADS-B hex even when not using ADS-B for state initialization
        if self.adsb_hex is None and detection.get("adsb"):
            _det_hex = detection["adsb"].get("hex")
            if _det_hex and isinstance(_det_hex, str):
                self.adsb_hex = _det_hex

        _hist_max = (
            int(config.get("tracker", {}).get("track_history_max", TRACK_HISTORY_MAX)) if config else TRACK_HISTORY_MAX
        )
        self.history = {
            "timestamps": deque([timestamp], maxlen=_hist_max),
            "frames": deque([frame], maxlen=_hist_max),
            "states": deque([self.state.copy()], maxlen=_hist_max),
            "measurements": deque([detection], maxlen=_hist_max),
            "state_status": deque([self.state_status.name], maxlen=_hist_max),
        }

        self.n_frames = 1
        self.n_associated = 0
        self.n_missed = 0
        self.occluded_s = 0.0
        self.blanked = False
        self.last_doppler = detection["doppler"]
        self.last_update_ms = timestamp
        self.n_observations = 1
        self.nis_ema = float(MEASUREMENT_DIM)
        self.last_dt = None
        self.last_residual = None
        self.last_q_scale = 1.0
        self.last_n_missed = 0
        self.n_shadowed = 1 if detection.get("shadowed") else 0
        self.n_interfering = 1 if detection.get("interfering") else 0

        self.total_snr = detection["snr"]
        self.birth_timestamp = timestamp
        self.death_timestamp = timestamp

        self.max_velocity_ms = 0.0
        self._observe_speed_floor(detection)

    @staticmethod
    def _validate_adsb_data(adsb):
        if not isinstance(adsb, dict):
            return False

        if "lat" in adsb:
            lat = adsb["lat"]
            if not isinstance(lat, (int, float)) or np.isnan(lat) or not (-90 <= lat <= 90):
                return False

        if "lon" in adsb:
            lon = adsb["lon"]
            if not isinstance(lon, (int, float)) or np.isnan(lon) or not (-180 <= lon <= 180):
                return False

        if "alt_baro" in adsb:
            alt = adsb["alt_baro"]
            if not isinstance(alt, (int, float)) or np.isnan(alt) or not (-1000 <= alt <= 60000):
                return False

        if "gs" in adsb:
            gs = adsb["gs"]
            if not isinstance(gs, (int, float)) or np.isnan(gs) or gs < 0:
                return False

        if "track" in adsb:
            track = adsb["track"]
            if not isinstance(track, (int, float)) or np.isnan(track) or not (0 <= track < 360):
                return False

        return True

    def _init_from_adsb(self, detection, adsb_config):
        adsb = detection["adsb"]

        if not self._validate_adsb_data(adsb):
            self._init_from_delay_doppler(detection)
            return

        self.adsb_hex = adsb.get("hex")
        self.adsb_initialized = True

        self._init_from_delay_doppler(detection)
        self.covariance[0, 0] = (adsb_config["initial_covariance"]["position"] / 1000.0) ** 2

    def _init_from_delay_doppler(self, detection):
        wavelength = WAVELENGTH_KM()
        self.state = np.array([detection["delay"], doppler_to_range_rate(detection["doppler"]), 0.0])
        self.covariance = np.diag(
            [
                MEASUREMENT_NOISE_DELAY_KM2(),
                wavelength * wavelength * MEASUREMENT_NOISE_DOPPLER_HZ2(),
                INITIAL_RANGE_ACCEL_VARIANCE,
            ]
        )

    @classmethod
    def _generate_id(cls, timestamp_ms):
        """A fresh identity per track, never one derived from the aircraft.

        An id built from adsb_hex gives every pass of an aircraft the same
        identity, which costs twice over: fragmentation becomes unmeasurable,
        because live_score counts distinct ids per hex and would only ever
        find one, and two tracks of the same aircraft share OutputWriter's
        per-track high-water mark, so whichever emits first silently suppresses
        the other's earlier detections. The label already travels in the
        event's own adsb_hex field, so nothing downstream needs it here.
        """
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0)
        date_str = dt.strftime("%y%m%d")

        if cls._last_date != date_str:
            cls._daily_counter = 0
            cls._last_date = date_str

        track_id = f"{date_str}-{cls._daily_counter:06X}"
        cls._daily_counter += 1
        return track_id

    def process_noise_scale(self):
        """How far the filter's own residuals say its motion model is wrong.

        A matched filter produces a normalised innovation squared averaging
        the measurement dimension, so the ratio to that is how badly the
        constant-acceleration assumption is failing right now. A banking
        aircraft drives it up and widens the gate for exactly as long as it
        banks; a cruising one leaves it at the floor. Floored at 1 rather
        than allowed to shrink, so a well-modelled target behaves exactly as
        it did before this existed.
        """
        if not PROCESS_NOISE_ADAPTIVE():
            return 1.0
        return min(max(self.nis_ema / MEASUREMENT_DIM, 1.0), PROCESS_NOISE_MAX_SCALE())

    def predict(self, dt):
        self.kf.dt = dt
        self.last_dt = dt
        state_pred, cov_pred = self.kf.predict(self.state, self.covariance, self.process_noise_scale())
        self.state = state_pred
        # Freeze covariance growth once coasting exceeds N_COAST so the
        # association gate stops inflating toward unrelated detections while
        # the track dead-reckons on toward its N_DELETE deletion point.
        if self.n_missed <= N_COAST():
            self.covariance = cov_pred

    def update(self, detection, timestamp, frame=0):
        measurement = np.array([detection["delay"], doppler_to_range_rate(detection["doppler"])])
        q_scale = self.process_noise_scale()
        self.state, self.covariance, residual = self.kf.update(
            self.state, self.covariance, measurement, detection.get("snr")
        )
        memory = PROCESS_NOISE_NIS_MEMORY()
        self.nis_ema = (1.0 - memory) * self.nis_ema + memory * residual.nis
        self.last_residual = residual
        self.last_q_scale = q_scale
        # Before the reset below, which is what makes this worth latching: how
        # long the track had been coasting is a property of the prediction this
        # innovation was measured against, and by the end of this method it is
        # gone.
        self.last_n_missed = self.n_missed

        # Capture ADS-B hex on first association that carries one.  Tracks
        # initialised from clutter miss this in __init__; this back-fills it.
        _det_adsb = detection.get("adsb")
        if self.adsb_hex is None and _det_adsb:
            adsb = _det_adsb
            if self._validate_adsb_data(adsb) and adsb.get("hex"):
                self.adsb_hex = adsb["hex"]
                self.adsb_initialized = True

        self.n_observations += 1
        if detection.get("shadowed"):
            self.n_shadowed += 1
        if detection.get("interfering"):
            self.n_interfering += 1

        self.history["timestamps"].append(timestamp)
        self.history["frames"].append(frame)
        self.history["states"].append(self.state.copy())
        self.history["measurements"].append(detection)
        self.history["state_status"].append(self.state_status.name)

        self.n_associated += 1
        self.n_missed = 0
        self.occluded_s = 0.0
        self.blanked = False
        self.last_doppler = detection["doppler"]
        self.last_update_ms = timestamp
        self.n_frames += 1

        self.total_snr += detection.get("snr", 0)
        self.death_timestamp = timestamp

        self._observe_speed_floor(detection)

    def _observe_speed_floor(self, detection):
        """Bistatic range rate never exceeds twice the target's speed, so |range rate| / 2 floors it in any geometry."""
        floor_ms = abs(doppler_to_range_rate(detection["doppler"])) * 1000.0 / 2.0
        self.max_velocity_ms = max(self.max_velocity_ms, floor_ms)

    def mark_missed(self, timestamp, frame=0):
        """Mark track as not associated this frame.

        A miss the zero-Doppler blank explains is not counted towards deletion.
        """
        if self.is_occluded():
            self.blanked = self.blanked or self._predicted_through_blank()
            self.occluded_s += self.last_dt or 0.0
        else:
            self.n_missed += 1
        self.n_frames += 1

        self.history["timestamps"].append(timestamp)
        self.history["frames"].append(frame)
        self.history["states"].append(self.state.copy())
        self.history["measurements"].append(None)
        self.history["state_status"].append(self.state_status.name)

        self.death_timestamp = timestamp

    def is_occluded(self):
        """Whether blah2's zero-Doppler blank can explain a confirmed track going undetected.

        Once the prediction has entered or crossed the blank this latches until the next
        association: when the target leaves the blank depends on an acceleration too poorly
        estimated going in to trust. See docs/zero-doppler-blank.md.
        """
        if DOPPLER_BLANK_HZ() is None or self.state_status == TrackState.TENTATIVE:
            return False
        if self.blanked or self._predicted_through_blank():
            return True
        reach = np.sqrt(GATE_THRESHOLD() * self.covariance[1, 1]) / WAVELENGTH_KM()
        return abs(range_rate_to_doppler(self.state[1])) - reach < self._blank_edge_hz()

    def max_occluded_s(self):
        """A track is held no longer than it was observed: extrapolating further is a guess."""
        return min(MAX_OCCLUDED_S, (self.last_update_ms - self.birth_timestamp) / 1000.0)

    def _blank_edge_hz(self):
        return DOPPLER_BLANK_HZ() + DOPPLER_BIN_HZ()

    def _predicted_through_blank(self):
        doppler = range_rate_to_doppler(self.state[1])
        return abs(doppler) < self._blank_edge_hz() or (doppler > 0) != (self.last_doppler > 0)

    def get_predicted_measurement(self):
        return self.kf.H @ self.state

    def get_innovation_covariance(self, snr=None):
        return self.kf.get_innovation_covariance(self.covariance, snr)

    def get_innovation_base(self):
        return self.kf.get_innovation_base(self.covariance)

    def is_shadowed(self):
        """Whether this track lives in the delay shadow of brighter returns.

        A strong aircraft casts multipath and sidelobe replicas at longer
        bistatic range and lower SNR. Measured on a live node over 93
        same-frame pairs: 89% of the spurious detections around an aircraft
        sat at longer delay than it and 99% were weaker, median 6.9 dB down.
        They are kinematically consistent, so the filter cannot reject them;
        only their position relative to a brighter return gives them away.
        """
        if self.n_observations < 3:
            return False
        return self.shadow_fraction() >= SHADOW_MIN_FRACTION()

    def shadow_fraction(self):
        """Share of this track's detections that sat behind a brighter return.

        Reported so a site where the shadow thresholds do not fit is visible
        without anyone analysing anything: on the node these were measured
        against, every ADS-B-matched aircraft stayed at 0.00 while replica
        tracks ran a median of 0.64. An identified aircraft drifting up from
        zero means the thresholds are wrong here.
        """
        return self.n_shadowed / max(self.n_observations, 1)

    def interference_fraction(self):
        """Share of this track's detections that sat in an interfering bin.

        This is how the occupancy map is held to account. A track carrying an
        ADS-B hex is an aircraft whatever the map thinks, so a labelled track
        reading above zero is the map reaching for something it must not have,
        and the fraction says how close it came before anything was suppressed.
        Reported whether or not suppression is on, so the question can be
        settled from a recording made before it is turned on anywhere.
        """
        return self.n_interfering / max(self.n_observations, 1)

    def promote_if_ready(self):
        if self.state_status == TrackState.TENTATIVE:
            if self.n_frames >= N_WINDOW():
                if self.n_associated >= M_THRESHOLD() and not self.is_shadowed():
                    self.state_status = TrackState.ACTIVE
                    return True
        return False

    def get_quality_score(self):
        if self.n_frames == 0:
            return 0.0

        continuity = (self.n_associated / self.n_frames) * 40.0
        avg_snr = self.total_snr / max(self.n_associated, 1)
        snr_score = min((avg_snr / 15.0) * 30.0, 30.0)
        duration_sec = (self.death_timestamp - self.birth_timestamp) / 1000.0
        duration_score = min((duration_sec / 60.0) * 20.0, 20.0)
        assoc_score = min((self.n_associated / 50.0) * 10.0, 10.0)

        return continuity + snr_score + duration_score + assoc_score

    def is_high_quality(self):
        if self.n_associated < 3:
            return False

        continuity = self.n_associated / max(self.n_frames, 1)
        if continuity < 0.4:
            return False

        avg_snr = self.total_snr / max(self.n_associated, 1)
        if avg_snr < 8.0:
            return False

        duration_sec = (self.death_timestamp - self.birth_timestamp) / 1000.0
        return duration_sec >= 5.0

    def should_delete(self):
        if self.state_status == TrackState.TENTATIVE:
            return self.n_frames > N_WINDOW()
        else:
            return self.n_missed > N_DELETE() or self.occluded_s > self.max_occluded_s()

    def get_length_bucket(self):
        if self.n_associated < 10:
            return "short"
        elif self.n_associated < 50:
            return "medium"
        else:
            return "long"

    def get_recent_detections(self, n=20):
        """Return the last *n* non-None detections (reverse scan, early exit)."""
        # Called off the frame worker (analytics, feed builders) while that worker
        # may be appending, and a deque raises RuntimeError if it is mutated during
        # Python-level iteration.  list(deque) is a single C call, so under the GIL
        # it is an atomic snapshot; take both buffers before iterating.  The zip is
        # unchanged: a reader that lands mid-append can see one more timestamp than
        # measurement (the mutator appends timestamps first), and zip truncating to
        # the shorter buffer is the same behaviour these buffers had as lists.
        meas = list(self.history["measurements"])
        ts_hist = list(self.history["timestamps"])
        result = []
        for m, ts in zip(reversed(meas), reversed(ts_hist)):
            if m is not None:
                result.append(
                    {
                        "timestamp": ts,
                        "delay": m["delay"],
                        "doppler": m["doppler"],
                        "snr": m["snr"],
                        "adsb": m.get("adsb"),
                    }
                )
                if len(result) >= n:
                    break
        result.reverse()
        return result

    def to_dict(self):
        # Served from admin/state-snapshot threads while the frame worker appends.
        # list(deque) is atomic under the GIL; the comprehensions below are not, so
        # snapshot every buffer first.  The snapshots can differ in length by one
        # (the mutator appends timestamps -> frames -> states -> measurements), so
        # truncate to the common length to keep the emitted arrays aligned.
        hist_timestamps = list(self.history["timestamps"])
        hist_states = list(self.history["states"])
        hist_measurements = list(self.history["measurements"])
        hist_state_status = list(self.history["state_status"])
        n_hist = min(len(hist_timestamps), len(hist_states), len(hist_measurements), len(hist_state_status))
        hist_timestamps = hist_timestamps[:n_hist]
        hist_states = hist_states[:n_hist]
        hist_measurements = hist_measurements[:n_hist]
        hist_state_status = hist_state_status[:n_hist]

        duration_sec = (self.death_timestamp - self.birth_timestamp) / 1000.0
        avg_snr = self.total_snr / max(self.n_associated, 1)
        continuity = self.n_associated / max(self.n_frames, 1)

        return {
            "id": self.id,
            "adsb_hex": self.adsb_hex,
            "adsb_initialized": self.adsb_initialized,
            "state_status": self.state_status.name,
            "n_frames": self.n_frames,
            "n_associated": self.n_associated,
            "length_bucket": self.get_length_bucket(),
            "quality_score": self.get_quality_score(),
            "avg_snr": avg_snr,
            "duration_sec": duration_sec,
            "continuity": continuity,
            "shadow_fraction": self.shadow_fraction(),
            "interference_fraction": self.interference_fraction(),
            "birth_timestamp": self.birth_timestamp,
            "death_timestamp": self.death_timestamp,
            "max_velocity_ms": self.max_velocity_ms,
            "history": {
                "timestamps": hist_timestamps,
                "states": [s.tolist() for s in hist_states],
                "delays": [m["delay"] if m else None for m in hist_measurements],
                "dopplers": [m["doppler"] if m else None for m in hist_measurements],
                "snrs": [m["snr"] if m else None for m in hist_measurements],
                "state_status": hist_state_status,
            },
        }
