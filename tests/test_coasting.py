"""Tests for coasting covariance growth.

A track that misses several detections in a row keeps predicting forward
(dead reckoning) so it can still be deleted cleanly at n_delete, but its
association gate must not keep inflating the whole time — otherwise a
long-coasting track eventually gates onto an unrelated stray detection far
from its true predicted position, producing a visible jump in the track's
path that then jumps back once it reacquires the real target.
"""

from retina_tracker.config import set_config
from retina_tracker.tracker import Tracker


def build_config(**overrides):
    config = {
        "tracker": {
            "m_threshold": 4,
            "n_window": 20,
            "n_delete": 20,
            "n_coast": 3,
            "min_snr": 7.0,
            "gate_threshold": 9.0,
            "detection_window": 20,
        },
        "process_noise": {"delay": 0.1, "doppler": 0.5},
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


def make_coasting_track(n_coast):
    config = build_config(n_coast=n_coast)
    set_config(config)
    tracker = Tracker(config=config)
    tracker.process_frame([{"delay": 10.0, "doppler": -70.0, "snr": 15.0}], 0)
    for i in range(1, 9):
        tracker.process_frame([], i * 500)
    return tracker


def test_covariance_growth_freezes_after_n_coast():
    delay_variances = []
    doppler_variances = []
    config = build_config(n_coast=3)
    set_config(config)
    tracker = Tracker(config=config)
    tracker.process_frame([{"delay": 10.0, "doppler": -70.0, "snr": 15.0}], 0)
    for i in range(1, 9):
        tracker.process_frame([], i * 500)
        track = tracker.tracks[0]
        delay_variances.append(track.covariance[0, 0])
        doppler_variances.append(track.covariance[2, 2])

    # Grows for the first N_COAST + 1 missed frames...
    assert delay_variances[0] < delay_variances[1] < delay_variances[2] < delay_variances[3]
    assert doppler_variances[0] < doppler_variances[1] < doppler_variances[2] < doppler_variances[3]
    # ...then freezes for every frame after that.
    assert all(v == delay_variances[3] for v in delay_variances[3:])
    assert all(v == doppler_variances[3] for v in doppler_variances[3:])


def test_long_coasting_track_rejects_stray_detection():
    """A detection far from the predicted position must not be swallowed
    just because the track has been coasting for a while."""
    tracker = make_coasting_track(n_coast=3)
    stray = {"delay": 30.0, "doppler": -70.0, "snr": 15.0}

    assert tracker._associate([stray]) == []


def test_long_coasting_track_still_accepts_a_true_reacquisition():
    """The cap must not make coasting tracks impossible to reacquire —
    a detection close to the predicted position still associates."""
    tracker = make_coasting_track(n_coast=3)
    nearby = {"delay": 10.5, "doppler": -69.0, "snr": 15.0}

    assert tracker._associate([nearby]) == [(0, 0)]
