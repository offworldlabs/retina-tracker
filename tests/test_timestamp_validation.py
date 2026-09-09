"""A frame's timestamp is read by far more than the dt clamp.

Past the clamp the raw value reaches datetime.fromtimestamp() when a track
promotes and takes an ID, death_timestamp on every track the frame missed, and
the merge-window cutoff. A non-finite stamp defeats all three: the ID
generation raises out of process_frame, the cutoff comparison is false for
every track so the whole merge working set empties in one frame, and a NaN
death_timestamp makes the quality score NaN. Guarding only the two derived
values (dt, and the clock mark) leaves each of those open, so the frame is
rejected at the entry instead.

The clock mark has the mirror problem. It only ever advances, so one finite but
implausible future timestamp pins it: every later frame is then backwards,
clamps to dt = 0, and the filter stops predicting for good. It resyncs after a
run of frames that never passes it. A run rather than a single frame, because
two node clocks interleaved into one stream alternate above and below the mark
indefinitely without either being wrong.
"""

import numpy as np
import pytest

from retina_tracker.config import set_config
from retina_tracker.kalman import range_rate_to_doppler
from retina_tracker.tracker import BACKWARDS_RUN_BEFORE_RESYNC, Tracker

NON_FINITE = [float("nan"), float("inf"), float("-inf")]
NON_NUMERIC = [None, "1700000000000", [1], {}]

N_WINDOW = 20
CADENCE_OUTSPANNING_TRACKLETS_MS = 2000


def build_config(**overrides):
    config = {
        "tracker": {
            "m_threshold": 4,
            "n_window": N_WINDOW,
            "n_delete": 20,
            "n_coast": 3,
            "min_snr": 7.0,
            "gate_threshold": 9.0,
            "detection_window": 20,
        },
        "process_noise": {"range_jerk": 1e-7},
        "tracklet": {"max_delay_residual": 2.0, "max_doppler_residual": 10.0, "max_time_span": 3.0},
        "adsb": {
            "enabled": False,
            "priority": True,
            "reference_location": None,
            "initial_covariance": {"position": 100.0, "velocity": 5.0},
        },
        "radar": {"center_frequency": 200000000},
    }
    config["tracker"].update(overrides)
    return config


def _det(delay, doppler, snr=15.0):
    return {"delay": delay, "doppler": doppler, "snr": snr}


def _tracker():
    config = build_config()
    set_config(config)
    return Tracker(config=config)


def _drift_doppler(cadence_ms):
    """The Doppler that matches a 0.2 km delay step at this cadence."""
    return range_rate_to_doppler(0.2 / (cadence_ms / 1000.0))


def _settled_tracker(n_frames=40, cadence_ms=1000):
    tracker = _tracker()
    doppler = _drift_doppler(cadence_ms)
    for i in range(n_frames):
        tracker.process_frame([_det(10.0 + 0.2 * i, doppler)], i * cadence_ms)
    return tracker


def _tentative_tracker(n_frames):
    """A track held TENTATIVE until promote_if_ready() takes it at N_WINDOW.

    The cadence puts any three detections 4 s apart, past TRACKLET_MAX_TIME_SPAN,
    so tracklet initiation never fires and never promotes the track early.
    """
    tracker = _tracker()
    for i in range(n_frames):
        tracker.process_frame(
            [_det(10.0 + 0.2 * i, _drift_doppler(CADENCE_OUTSPANNING_TRACKLETS_MS))],
            i * CADENCE_OUTSPANNING_TRACKLETS_MS,
        )
    return tracker


class TestNonFiniteFrameCarryingDetections:
    """Every existing non-finite test passes an empty detection list, which is
    the one shape of frame that never reaches the timestamp's other readers."""

    @pytest.mark.parametrize("timestamp", NON_FINITE)
    def test_a_non_finite_frame_carrying_detections_does_not_raise(self, timestamp):
        tracker = _settled_tracker()

        tracker.process_frame([_det(18.0, 320.0), _det(52.0, -140.0)], timestamp)

        assert tracker.tracks

    @pytest.mark.parametrize("timestamp", NON_FINITE)
    def test_a_non_finite_frame_on_the_promotion_frame_does_not_raise(self, timestamp):
        """Track._generate_id calls datetime.fromtimestamp(t / 1000.0), which
        rejects NaN and overflows on an infinity."""
        tracker = _tentative_tracker(N_WINDOW - 1)
        track = tracker.tracks[0]
        assert track.id is None and track.n_frames == N_WINDOW - 1

        tracker.process_frame([_det(10.0 + 0.2 * 19, _drift_doppler(1000))], timestamp)

        assert track.n_frames == N_WINDOW - 1, "the rejected frame must not advance the track"

    @pytest.mark.parametrize("timestamp", NON_FINITE)
    def test_the_promotion_still_happens_on_the_next_good_frame(self, timestamp):
        """Rejecting the frame defers the promotion rather than losing it."""
        tracker = _tentative_tracker(N_WINDOW - 1)
        track = tracker.tracks[0]

        tracker.process_frame([_det(10.0 + 0.2 * 19, _drift_doppler(1000))], timestamp)
        tracker.process_frame([_det(10.0 + 0.2 * 19, _drift_doppler(1000))], 19 * CADENCE_OUTSPANNING_TRACKLETS_MS)

        assert track.id is not None
        assert "NAN" not in track.id.upper()


class TestNonFiniteFrameLeavesStateIntact:
    def test_a_nan_frame_does_not_drain_the_merge_working_set(self):
        """cutoff = timestamp - MERGE_WINDOW_MS is NaN, so `death_timestamp >=
        cutoff` is false for every entry and all_tracks empties in one frame."""
        tracker = _settled_tracker()
        for i in range(40, 65):
            tracker.process_frame([], i * 1000)
        assert tracker.all_tracks and not tracker.completed_tracks

        before = list(tracker.all_tracks)
        tracker.process_frame([], float("nan"))

        assert tracker.all_tracks == before
        assert not tracker.completed_tracks

    @pytest.mark.parametrize("timestamp", NON_FINITE)
    def test_a_non_finite_frame_does_not_poison_the_quality_score(self, timestamp):
        """mark_missed() sets death_timestamp on every track the frame missed,
        and get_quality_score() divides by it."""
        tracker = _settled_tracker()
        track = tracker.tracks[0]
        death, quality = track.death_timestamp, track.get_quality_score()
        assert np.isfinite(death) and np.isfinite(quality)

        tracker.process_frame([], timestamp)

        assert track.death_timestamp == death
        assert track.get_quality_score() == quality

    def test_a_rejected_frame_is_still_counted_as_a_frame(self):
        tracker = _settled_tracker()
        before = tracker.frame_count

        tracker.process_frame([], float("nan"))

        assert tracker.frame_count == before + 1

    @pytest.mark.parametrize("timestamp", NON_NUMERIC)
    def test_a_non_numeric_timestamp_drops_the_frame(self, timestamp):
        """math.isfinite raises rather than returning False off a real number,
        and server.process_streaming_frame reads the stamp straight out of an
        external frame without checking its type."""
        tracker = _settled_tracker()
        clock = tracker.last_timestamp
        rejected = tracker.n_frames_rejected

        tracker.process_frame([_det(18.0, 330.0)], timestamp)

        assert tracker.n_frames_rejected == rejected + 1
        assert tracker.last_timestamp == clock

    def test_a_rejected_frame_is_counted_apart_from_a_clamped_dt(self):
        """An unusable frame and an out-of-order but usable one are different
        failures, and one counter for both cannot tell an operator which."""
        tracker = _settled_tracker()

        tracker.process_frame([], tracker.last_timestamp - 6000)
        assert (tracker.n_dt_clamped, tracker.n_frames_rejected) == (1, 0)

        tracker.process_frame([], float("nan"))
        assert (tracker.n_dt_clamped, tracker.n_frames_rejected) == (1, 1)


class TestClockResync:
    """The high-water mark stops a clamped frame rewinding the clock, but on
    its own it is a one-way ratchet with no way back down."""

    def test_a_far_future_frame_does_not_pin_the_clock_for_good(self):
        tracker = _settled_tracker()
        base = tracker.last_timestamp
        tracker.process_frame([], base + 86_400_000)  # a day ahead of the stream
        clamped = tracker.n_dt_clamped

        for i in range(1, 11):
            tracker.process_frame([_det(18.0 + 0.2 * i, _drift_doppler(1000))], base + i * 1000)

        assert tracker.last_timestamp == base + 10_000
        assert tracker.n_clock_resyncs == 1
        assert tracker.n_dt_clamped - clamped == BACKWARDS_RUN_BEFORE_RESYNC, (
            "recovery costs the run and nothing after it"
        )

    def test_the_filter_predicts_again_after_a_resync(self):
        """While the mark is pinned every dt is 0, so predict is F = I with
        Q = 0 and the covariance never moves."""
        tracker = _settled_tracker()
        base = tracker.last_timestamp
        tracker.process_frame([], base + 86_400_000)
        for i in range(1, 11):
            tracker.process_frame([_det(18.0 + 0.2 * i, _drift_doppler(1000))], base + i * 1000)

        track = next(t for t in tracker.tracks if t.n_missed == 0)
        assert track.n_associated >= 5, "the pinned track loses the target and the stream respawns it"
        before = np.diag(track.covariance).copy()
        next_frame_on_the_streams_own_clock = base + 11_000
        tracker.process_frame([], next_frame_on_the_streams_own_clock)

        assert np.all(np.diag(track.covariance) > before)

    def test_interleaved_node_clocks_do_not_resync(self):
        """Two nodes 30 s apart in one stream alternate either side of the mark
        forever. Alternating is not a run, and neither clock is the outlier."""
        tracker = _settled_tracker()
        base = tracker.last_timestamp

        for i in range(1, 31):
            tracker.process_frame([_det(18.0 + 0.2 * i, _drift_doppler(1000))], base + i * 1000)
            assert tracker.n_backwards < BACKWARDS_RUN_BEFORE_RESYNC
            tracker.process_frame([_det(60.0 + 0.2 * i, _drift_doppler(1000))], base + 30_000 + i * 1000)
            assert tracker.n_backwards < BACKWARDS_RUN_BEFORE_RESYNC

        assert tracker.n_clock_resyncs == 0

    def test_a_duplicate_timestamp_counts_toward_the_run(self):
        """A stream stuck on one stamp advances no more than a backwards one
        does, and clamping never fires on it because dt is already 0."""
        tracker = _settled_tracker()
        stuck = tracker.last_timestamp
        clamped = tracker.n_dt_clamped

        for _ in range(BACKWARDS_RUN_BEFORE_RESYNC):
            tracker.process_frame([], stuck)

        assert tracker.n_backwards == 0
        assert tracker.n_clock_resyncs == 1
        assert tracker.n_dt_clamped == clamped

    def test_a_non_finite_frame_does_not_count_toward_the_run(self):
        """It is dropped, not late, so it says nothing about the mark."""
        tracker = _settled_tracker()
        stuck = tracker.last_timestamp
        tracker.process_frame([], stuck)
        tracker.process_frame([], stuck)
        assert tracker.n_backwards == BACKWARDS_RUN_BEFORE_RESYNC - 1

        tracker.process_frame([], float("nan"))

        assert tracker.n_backwards == BACKWARDS_RUN_BEFORE_RESYNC - 1
        assert tracker.n_clock_resyncs == 0

    def test_the_resync_counters_reset_with_the_tracker(self):
        tracker = _settled_tracker()
        stuck = tracker.last_timestamp
        for _ in range(BACKWARDS_RUN_BEFORE_RESYNC - 1):
            tracker.process_frame([], stuck)
        assert tracker.n_backwards > 0

        tracker.process_frame([], float("nan"))
        assert tracker.n_frames_rejected > 0

        tracker.reset()

        assert tracker.n_backwards == 0
        assert tracker.n_clock_resyncs == 0
        assert tracker.n_frames_rejected == 0
