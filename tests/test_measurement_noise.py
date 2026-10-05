"""Tests for measurement noise sized from the resolution cell.

R was a fixed 1 km and 2.24 Hz of standard deviation, about 7 delay cells, and
the SNR scaling took it past 2 km at the detection floor. Measured scatter is a
small fraction of a cell, so every gate was kilometres wide and a confirmed
track would take a stray detection kilometres off its path and come straight
back. See docs/measurement-noise.md.
"""

import numpy as np
import pytest

from retina_tracker.config import (
    MEASUREMENT_NOISE_DELAY_KM2,
    MEASUREMENT_NOISE_DOPPLER_HZ2,
    WAVELENGTH_KM,
    load_config,
    set_config,
)
from retina_tracker.kalman import KalmanFilter, doppler_to_range_rate
from retina_tracker.track import Track, TrackState
from retina_tracker.tracker import Tracker

FRAME_MS = 500


def use(**radar):
    config = load_config(None)
    config["radar"].update(radar)
    set_config(config)
    return config


def test_noise_is_one_resolution_cell_of_the_nodes_own_capture():
    use(sample_rate=2000000, cpi=0.5)
    assert MEASUREMENT_NOISE_DELAY_KM2() == pytest.approx(0.149896**2, rel=1e-4)
    assert MEASUREMENT_NOISE_DOPPLER_HZ2() == pytest.approx(2.0**2)

    use(sample_rate=4000000, cpi=1.0)
    assert MEASUREMENT_NOISE_DELAY_KM2() == pytest.approx(0.074948**2, rel=1e-4)
    assert MEASUREMENT_NOISE_DOPPLER_HZ2() == pytest.approx(1.0**2)


def test_doppler_noise_converts_to_range_rate_at_the_nodes_carrier():
    use(center_frequency=545000000)
    r = KalmanFilter().R
    assert r[1, 1] == pytest.approx((WAVELENGTH_KM() * 2.0) ** 2)


def test_process_noise_carries_a_range_rate_random_walk():
    config = use()
    config["process_noise"]["range_jerk"] = 0.0
    config["process_noise"]["range_rate"] = 1e-5
    kf = KalmanFilter()
    kf.dt = 2.0
    _, covariance = kf.predict(np.zeros(3), np.zeros((3, 3)))
    assert covariance[1, 1] == pytest.approx(1e-5 * 2.0)
    assert covariance[0, 0] == pytest.approx(1e-5 * 8.0 / 3.0)


def confirmed(tracker, doppler=-80.0, frames=10):
    step = doppler_to_range_rate(doppler) * FRAME_MS / 1000.0
    delay = 20.0
    for frame in range(frames):
        tracker.process_frame([{"delay": delay, "doppler": doppler, "snr": 12.0}], frame * FRAME_MS)
        delay += step
    track = tracker.tracks[0]
    assert track.state_status == TrackState.ACTIVE
    return track, delay, step, frames


def test_a_confirmed_track_refuses_a_stray_detection_kilometres_off_its_path():
    use()
    tracker = Tracker()
    track, delay, step, frame = confirmed(tracker)
    stray = {"delay": delay + 3.0, "doppler": -80.0, "snr": 5.0}
    tracker.process_frame([stray], frame * FRAME_MS)
    assert track.n_missed == 1
    tracker.process_frame([{"delay": delay + step, "doppler": -80.0, "snr": 12.0}], (frame + 1) * FRAME_MS)
    assert track.n_missed == 0


def test_an_adsb_start_is_tighter_than_a_radar_only_one():
    config = use()
    config["adsb"].update({"enabled": True, "reference_location": {"latitude": 0.0, "longitude": 0.0, "altitude": 0}})
    adsb = {"hex": "abc123", "lat": 37.0, "lon": -122.0, "alt_baro": 9000, "gs": 250, "track": 90}
    radar_only = Track({"delay": 20.0, "doppler": -80.0, "snr": 12.0}, 0, KalmanFilter(), config=config)
    labelled = Track({"delay": 20.0, "doppler": -80.0, "snr": 12.0, "adsb": adsb}, 0, KalmanFilter(), config=config)
    assert labelled.adsb_initialized
    assert labelled.covariance[0, 0] == pytest.approx(0.1**2)
    assert labelled.covariance[0, 0] < radar_only.covariance[0, 0]
