"""Tests for measurement noise as measured on three nodes.

R was a fixed 1 km and 2.24 Hz of standard deviation, about 7 delay cells, and
the SNR scaling took it past 2 km at the detection floor, so every gate was
kilometres wide. Measured from smoother residuals on jn1, fairforest B and jn2,
delay noise is 0.12 cells at 10 dB falling as SNR^-0.4 in variance, and Doppler
noise is 0.4 m/s of range rate on both bands. See docs/measurement-noise.md.
"""

import numpy as np
import pytest

from retina_tracker.config import (
    MEASUREMENT_NOISE_DELAY_KM2,
    MEASUREMENT_NOISE_RANGE_RATE_KM2,
    load_config,
    set_config,
)
from retina_tracker.kalman import KalmanFilter, delay_noise_scale, doppler_to_range_rate
from retina_tracker.track import Track, TrackState
from retina_tracker.tracker import Tracker

FRAME_MS = 500


def use(**radar):
    config = load_config(None)
    config["radar"].update(radar)
    set_config(config)
    return config


def test_delay_noise_is_a_fixed_share_of_the_nodes_own_delay_cell():
    use(sample_rate=2000000)
    assert MEASUREMENT_NOISE_DELAY_KM2() == pytest.approx((0.12 * 0.149896) ** 2, rel=1e-4)
    use(sample_rate=4000000)
    assert MEASUREMENT_NOISE_DELAY_KM2() == pytest.approx((0.12 * 0.074948) ** 2, rel=1e-4)


def test_doppler_noise_is_the_same_range_rate_on_every_band_and_cpi():
    for fc, cpi in ((213000000, 0.5), (545000000, 0.5), (213000000, 1.0)):
        use(center_frequency=fc, cpi=cpi)
        assert KalmanFilter().R[1, 1] == pytest.approx(MEASUREMENT_NOISE_RANGE_RATE_KM2())
        assert MEASUREMENT_NOISE_RANGE_RATE_KM2() == pytest.approx(0.0004**2)


def test_only_delay_noise_falls_with_snr():
    use()
    kf = KalmanFilter()
    at_10, at_20 = kf.measurement_noise(10.0), kf.measurement_noise(20.0)
    assert at_20[0, 0] / at_10[0, 0] == pytest.approx(10.0**-0.4)
    assert at_20[1, 1] == at_10[1, 1]
    assert kf.measurement_noise(None)[0, 0] == pytest.approx(at_10[0, 0])


def test_delay_noise_stops_growing_below_0_db():
    assert delay_noise_scale(-10.0) == pytest.approx(delay_noise_scale(0.0))
    assert np.allclose(delay_noise_scale(np.array([10.0, 20.0])), [1.0, 10.0**-0.4])


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


def test_an_adsb_label_does_not_change_where_a_track_starts():
    config = use()
    config["adsb"].update({"enabled": True, "reference_location": {"latitude": 0.0, "longitude": 0.0, "altitude": 0}})
    adsb = {"hex": "abc123", "lat": 37.0, "lon": -122.0, "alt_baro": 9000, "gs": 250, "track": 90}
    radar_only = Track({"delay": 20.0, "doppler": -80.0, "snr": 12.0}, 0, KalmanFilter(), config=config)
    labelled = Track({"delay": 20.0, "doppler": -80.0, "snr": 12.0, "adsb": adsb}, 0, KalmanFilter(), config=config)
    assert labelled.adsb_initialized and labelled.adsb_hex == "abc123"
    assert np.array_equal(labelled.covariance, radar_only.covariance)
