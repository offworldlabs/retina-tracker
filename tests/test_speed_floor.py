"""Tests for max_velocity_ms as the radar's speed floor, and the neutral anomaly fields.

Anomaly detection was removed to be rebuilt (docs/anomaly-detection.md). The wire
still carries its fields because retina-telemetry requires them.
"""

import pytest

from retina_tracker.config import WAVELENGTH_KM, load_config, set_config
from retina_tracker.kalman import KalmanFilter
from retina_tracker.track import Track
from retina_tracker.tracker import Tracker, _frame_track


@pytest.fixture(autouse=True)
def _config():
    set_config(load_config(None))


def test_max_velocity_is_the_largest_range_rate_floor_seen():
    track = Track({"delay": 20.0, "doppler": -80.0, "snr": 15.0}, 0, KalmanFilter())
    track.update({"delay": 20.05, "doppler": 120.0, "snr": 15.0}, 500)
    track.update({"delay": 20.10, "doppler": 40.0, "snr": 15.0}, 1000)
    assert track.max_velocity_ms == pytest.approx(WAVELENGTH_KM() * 1000.0 * 120.0 / 2.0)


def test_adsb_ground_speed_no_longer_sets_max_velocity():
    adsb = {"hex": "abc123", "lat": 37.0, "lon": -122.0, "alt_baro": 30000, "gs": 900, "track": 90}
    track = Track({"delay": 20.0, "doppler": 10.0, "snr": 15.0, "adsb": adsb}, 0, KalmanFilter())
    assert track.max_velocity_ms == pytest.approx(WAVELENGTH_KM() * 1000.0 * 10.0 / 2.0)


def test_frame_record_carries_neutral_anomaly_fields():
    tracker = Tracker()
    track = Track({"delay": 20.0, "doppler": -80.0, "snr": 15.0}, 0, tracker.kf)
    record = _frame_track(track, "active", 0)
    assert record["is_anomalous"] is False
    assert record["anomaly_types"] == []
    assert record["max_velocity_ms"] == pytest.approx(track.max_velocity_ms)
