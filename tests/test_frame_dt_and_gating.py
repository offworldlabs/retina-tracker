"""An out-of-order frame must not let one track out-compete every other.

The chain: dt is a bare subtraction of two timestamps, so a frame that
arrives late makes it negative; a negative dt makes F P Fᵀ + Q
non-positive-definite; the gate check tested only that S was invertible, so
such a track kept gating; and its Mahalanobis distances came out negative,
which wins a minimisation. The corrupted track then took detections from
healthy tracks with no exception, no log and no counter.

A large positive dt is the mirror image: Q grows as dt³, so after a node
outage the gate is wider than the delay axis and everything falls inside it.
Tracks are deleted by missed-frame count rather than by elapsed time, so
nothing ages out across the gap to stop it.

Clamping alone does not close either: the clock has to stop rewinding too, or
the frame after a clamped one computes the whole excursion as its dt, and a
non-finite timestamp passes straight through min and max untouched.
"""

import numpy as np
import pytest

from retina_tracker.config import set_config
from retina_tracker.kalman import range_rate_to_doppler
from retina_tracker.tracker import MAX_FRAME_DT_S, Tracker


def build_config():
    return {
        "tracker": {
            "m_threshold": 4,
            "n_window": 20,
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


def _det(delay, doppler, snr=15.0):
    return {"delay": delay, "doppler": doppler, "snr": snr}


def _settled_tracker():
    """A single track carried far enough that its covariance has converged."""
    config = build_config()
    set_config(config)
    tracker = Tracker(config=config)
    for i in range(40):
        tracker.process_frame([_det(10.0 + 0.2 * i, -70.0 + 10.0 * i)], i * 1000)
    return tracker


class TestDtClamp:
    def test_backwards_frame_is_clamped_and_counted(self):
        tracker = _settled_tracker()
        before = tracker.n_dt_clamped
        tracker.process_frame([_det(18.0, 320.0)], tracker.last_timestamp - 6000)
        assert tracker.n_dt_clamped == before + 1

    def test_long_gap_is_clamped_and_counted(self):
        tracker = _settled_tracker()
        before = tracker.n_dt_clamped
        tracker.process_frame([_det(18.0, 320.0)], tracker.last_timestamp + 300_000)
        assert tracker.n_dt_clamped == before + 1

    def test_ordinary_cadence_is_not_clamped(self):
        tracker = _settled_tracker()
        before = tracker.n_dt_clamped
        tracker.process_frame([_det(18.0, 320.0)], tracker.last_timestamp + 1000)
        assert tracker.n_dt_clamped == before

    def test_covariance_stays_positive_definite_after_a_backwards_frame(self):
        """The whole failure chain starts here: without the clamp this goes negative."""
        tracker = _settled_tracker()
        tracker.process_frame([_det(18.0, 320.0)], tracker.last_timestamp - 6000)
        assert tracker.tracks
        for track in tracker.tracks:
            assert np.all(np.linalg.eigvalsh(track.covariance) > 0)

    def test_gap_recovery_survives_a_drought_shorter_than_the_clamp(self):
        """Empty frames are dropped upstream, so multi-second droughts are normal."""
        tracker = _settled_tracker()
        ts = tracker.last_timestamp + int(MAX_FRAME_DT_S * 1000) - 1000
        tracker.process_frame([_det(18.0, 320.0)], ts)
        assert tracker.tracks, "the track must survive a gap inside the clamp window"

    def test_a_long_gap_predicts_exactly_the_clamped_interval(self):
        """Pins the bound itself, not just that some bound applies.

        Process noise grows as dt cubed, so the covariance after the gap is what
        distinguishes a clamp at 60 s from one at 10 s or 600 s.
        """
        clamped = _settled_tracker()
        clamped.process_frame([], clamped.last_timestamp + 3_600_000)

        reference = _settled_tracker()
        reference.process_frame([], reference.last_timestamp + 60_000)

        assert MAX_FRAME_DT_S == 60.0
        assert clamped.tracks[0].covariance == pytest.approx(reference.tracks[0].covariance)

    def test_frame_after_a_clamped_one_gets_the_ordinary_dt(self):
        """Clamping the late frame is wasted if it still rewinds the clock.

        A frame 6 s late is clamped to 0 and counted, but leaves the clock at
        T-6 s, so the next legitimate frame computes 7 s: inside the bounds, so
        neither clamped nor counted, and 343x the process noise.

        Both sides coast the same number of frames on purpose: covariance
        growth freezes above N_COAST misses, so an asymmetric coast would
        diverge for a reason that has nothing to do with the clock.
        """
        late = _settled_tracker()
        t = late.last_timestamp
        late.process_frame([], t - 6000)
        late.process_frame([], t + 1000)

        reference = _settled_tracker()
        t_ref = reference.last_timestamp
        reference.process_frame([], t_ref)  # dt = 0 without needing the clamp
        reference.process_frame([], t_ref + 1000)

        assert late.tracks and reference.tracks
        assert late.tracks[0].n_missed == reference.tracks[0].n_missed
        assert late.tracks[0].covariance == pytest.approx(reference.tracks[0].covariance), (
            "dt = 0 predicts nothing, so the clamped frame must leave the covariance alone"
        )

    def test_counter_resets_with_the_tracker(self):
        tracker = _settled_tracker()
        tracker.process_frame([_det(18.0, 320.0)], tracker.last_timestamp - 6000)
        assert tracker.n_dt_clamped > 0
        tracker.reset()
        assert tracker.n_dt_clamped == 0


class TestNonFiniteTimestamp:
    """A NaN defeats min/max entirely, so the clamp has to reject it by name.

    `min(max(nan, 0.0), 60.0)` is nan: Python's min and max return the first
    operand whenever the comparison is false, which every NaN comparison is.
    The nan then reaches F and Q and destroys every track on the node, while
    `dt != raw_dt` reads true forever and the counter reports clamping working.
    """

    def test_nan_timestamp_leaves_the_filter_finite(self):
        tracker = _settled_tracker()
        before = tracker.tracks[0].covariance.copy()

        tracker.process_frame([], float("nan"))

        assert tracker.tracks
        assert tracker.tracks[0].covariance == pytest.approx(before)

    def test_infinite_timestamp_leaves_the_filter_finite(self):
        tracker = _settled_tracker()
        before = tracker.tracks[0].covariance.copy()

        tracker.process_frame([], float("inf"))

        assert tracker.tracks
        assert tracker.tracks[0].covariance == pytest.approx(before)

    def test_non_finite_timestamp_does_not_become_the_clock(self):
        """Otherwise no later frame can ever pass the mark, and tracking stops."""
        tracker = _settled_tracker()
        t = tracker.last_timestamp

        tracker.process_frame([], float("nan"))
        tracker.process_frame([], float("inf"))

        assert tracker.last_timestamp == t

    def test_the_counter_stops_once_the_finite_frames_resume(self):
        """The counter must report the bad frames, not every frame after them."""
        tracker = _settled_tracker()
        tracker.process_frame([], float("nan"))
        assert tracker.n_frames_rejected == 1

        for i in range(1, 4):
            tracker.process_frame([_det(18.0 + i, 320.0 + 10.0 * i)], tracker.last_timestamp + 1000)

        assert tracker.n_frames_rejected == 1

    def test_tracking_recovers_after_a_non_finite_frame(self):
        tracker = _settled_tracker()
        t = tracker.last_timestamp
        tracker.process_frame([], float("nan"))
        tracker.process_frame([], t + 1000)

        reference = _settled_tracker()
        reference.process_frame([], reference.last_timestamp + 1000)

        assert tracker.tracks and reference.tracks
        assert tracker.tracks[0].covariance == pytest.approx(reference.tracks[0].covariance)


def _make_negative_definite(track, snr=15.0):
    """Drive S negative definite, as an out-of-order frame does.

    S = H P Hᵀ + R·noise_scale, so pushing the two variances below their
    measurement-noise terms flips both diagonal entries negative while leaving
    det_S = a*d positive (the state a determinant-sign test cannot see).
    """
    noise_scale = 1.0 / max(10 ** (snr / 10) / 10, 0.1)
    covariance = track.covariance.copy()
    covariance[0, 0] = -track.kf.R[0, 0] * noise_scale - 1.0
    covariance[1, 1] = -track.kf.R[1, 1] * noise_scale - 1.0
    track.covariance = covariance

    base = track.get_innovation_base()
    a = base[0, 0] + track.kf.R[0, 0] * noise_scale
    d = base[1, 1] + track.kf.R[1, 1] * noise_scale
    assert a < 0 and d < 0 and a * d - base[0, 1] * base[1, 0] > 1e-15, (
        "fixture guard: holds only while the calling test's detections all share this snr"
    )
    return track


class TestCorruptTrackTakesNothing:
    """The property the gate and the cost lower bound exist to protect.

    They are layered, and the first two tests here show it: a corrupt track on
    its own is rejected by either change alone, so that pair only both-fails
    when both are reverted. Only the last test separates them: with the
    determinant test in place of the definiteness one, the corrupt track still
    takes a healthy track's detection.
    """

    def test_negative_definite_covariance_wins_nothing(self):
        tracker = _settled_tracker()
        _make_negative_definite(tracker.tracks[0])

        associations = tracker._associate([_det(22.8, 380.0), _det(48.0, -155.0)])

        assert associations == []

    def test_healthy_track_still_associates(self):
        """The added a > 0 term must not reject legitimate detections."""
        tracker = _settled_tracker()
        z = tracker.tracks[0].kf.H @ tracker.tracks[0].state

        associations = tracker._associate([_det(float(z[0]), range_rate_to_doppler(float(z[1])))])

        assert associations == [(0, 0)]

    def test_corrupt_track_does_not_steal_a_healthy_track_detection(self):
        config = build_config()
        set_config(config)
        tracker = Tracker(config=config)

        for i in range(40):
            tracker.process_frame(
                [_det(10.0 + 0.2 * i, -70.0 + 10.0 * i), _det(60.0 - 0.2 * i, 400.0 - 5.0 * i)],
                i * 1000,
            )
        assert len(tracker.tracks) >= 2

        healthy = tracker.tracks[0]
        _make_negative_definite(tracker.tracks[1])
        z = healthy.kf.H @ healthy.state
        wanted = _det(float(z[0]), range_rate_to_doppler(float(z[1])))

        associations = tracker._associate([wanted])

        claimed = [track_i for track_i, det_i in associations if det_i == 0]
        assert claimed == [tracker.tracks.index(healthy)]


class TestGateStructuralInvariant:
    """The gate tests `a > 0` and `det_S > 0`, which is the whole of Sylvester's
    criterion for a symmetric 2x2 and so admits an off-diagonal S. Coupling
    range to its rate makes S genuinely non-diagonal — a constant-acceleration
    F correlates position with velocity — so symmetry, not diagonality, is what
    the gate now depends on and what is pinned here."""

    def test_the_covariance_stays_symmetric(self):
        tracker = _settled_tracker()
        for track in tracker.tracks:
            assert np.allclose(track.covariance, track.covariance.T, atol=0.0, rtol=0.0)

    def test_the_innovation_base_is_symmetric_and_positive_definite(self):
        tracker = _settled_tracker()
        for track in tracker.tracks:
            base = track.get_innovation_base()
            assert base[0, 1] == base[1, 0]
            assert base[0, 0] > 0.0
            assert base[0, 0] * base[1, 1] - base[0, 1] * base[1, 0] > 0.0

    def test_the_position_velocity_correlation_is_actually_present(self):
        tracker = _settled_tracker()
        assert any(track.covariance[0, 1] != 0.0 for track in tracker.tracks)
