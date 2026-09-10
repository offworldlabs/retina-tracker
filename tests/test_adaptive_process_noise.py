"""Widening the motion model on the filter's own residuals.

Bistatic range acceleration is mostly geometry, not aircraft: it depends on how
close a target passes, so it differs between sites by more than between flight
phases. Two nodes measured 2.9x apart, from traffic at 29.8 km against 5.8 km.
No single jerk constant is right for a fleet, so range_jerk is a floor and the
filter widens from it by however far its residuals say its model is failing.
"""

import numpy as np
import pytest

from retina_tracker.config import get_config, load_config, set_config
from retina_tracker.kalman import MEASUREMENT_DIM, KalmanFilter
from retina_tracker.track import Track


def build_config(**process_noise):
    config = load_config("nonexistent.yaml")
    config["adsb"]["enabled"] = False
    config["process_noise"].update(process_noise)
    return config


def det(delay=10.0, doppler=-50.0, snr=15.0):
    return {"delay": delay, "doppler": doppler, "snr": snr}


@pytest.fixture(autouse=True)
def _config():
    set_config(build_config())
    yield


def a_track():
    return Track(det(), 0, KalmanFilter(), config=get_config())


class TestTheScaleFollowsTheResiduals:
    def test_a_matched_filter_sits_at_the_floor(self):
        """A track is born assuming its model fits, so it starts unwidened."""
        assert a_track().process_noise_scale() == pytest.approx(1.0)

    def test_residuals_smaller_than_expected_do_not_narrow_it(self):
        """Floored at 1 so a well-modelled target behaves exactly as before."""
        track = a_track()
        track.nis_ema = 0.1
        assert track.process_noise_scale() == pytest.approx(1.0)

    def test_residuals_larger_than_expected_widen_it(self):
        track = a_track()
        track.nis_ema = 8.0
        assert track.process_noise_scale() == pytest.approx(4.0)

    def test_the_ceiling_holds(self):
        """Without it a single wild innovation would open the gate indefinitely."""
        set_config(build_config(max_scale=3.0))
        track = a_track()
        track.nis_ema = 1000.0
        assert track.process_noise_scale() == pytest.approx(3.0)

    def test_disabling_it_pins_the_scale(self):
        set_config(build_config(adaptive=False))
        track = a_track()
        track.nis_ema = 1000.0
        assert track.process_noise_scale() == pytest.approx(1.0)


class TestTheFilterReportsAndUsesIt:
    def test_update_returns_the_normalised_innovation(self):
        kf = KalmanFilter()
        state = np.array([10.0, 0.0, 0.0])
        cov = np.diag([1.0, 1e-5, 1e-4])
        _, _, nis = kf.update(state, cov, np.array([10.0, 0.0]), snr=15.0)
        assert nis == pytest.approx(0.0, abs=1e-9)

    def test_a_large_innovation_gives_a_large_nis(self):
        kf = KalmanFilter()
        state = np.array([10.0, 0.0, 0.0])
        cov = np.diag([1.0, 1e-5, 1e-4])
        _, _, near = kf.update(state, cov, np.array([10.1, 0.0]), snr=15.0)
        _, _, far = kf.update(state, cov, np.array([14.0, 0.0]), snr=15.0)
        assert far > near

    def test_a_wider_scale_grows_the_covariance_faster(self):
        kf = KalmanFilter(dt=1.0)
        state = np.array([10.0, 0.05, 0.0])
        cov = np.diag([1.0, 1e-5, 1e-4])
        _, narrow = kf.predict(state, cov, q_scale=1.0)
        _, wide = kf.predict(state, cov, q_scale=20.0)
        assert wide[2, 2] > narrow[2, 2]
        assert wide[0, 0] > narrow[0, 0]

    def test_the_running_average_moves_toward_what_is_observed(self):
        track = a_track()
        assert track.nis_ema == pytest.approx(float(MEASUREMENT_DIM))
        for i in range(1, 12):
            track.update(det(delay=10.0 + 3.0 * i), i * 1000, frame=i)
        assert track.nis_ema > MEASUREMENT_DIM

    def test_a_target_matching_its_model_stays_at_the_floor(self):
        """The whole point: only a track whose model is failing pays for it."""
        track = a_track()
        rate = track.state[1]
        for i in range(1, 12):
            track.update(det(delay=10.0 + rate * i), i * 1000, frame=i)
        assert track.process_noise_scale() == pytest.approx(1.0)
