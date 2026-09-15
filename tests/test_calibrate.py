"""Fitting R and Q back out of recorded innovations.

The two cancel: R is around 45x too large in delay while the model error it
stands in for grows as the square of the prediction interval, not the fifth
power white jerk implies, so the sum looks plausible and NIS alone cannot tell
which is wrong. What separates them is that only one of the two grows with how
long the filter had been predicting.

These tests synthesise records from a known R and a known velocity error and
assert the fit recovers both, which is the only way to know the tool is
measuring rather than asserting before any of it is pointed at a node.
"""

import json
import random

import pytest

from retina_tracker.calibrate import (
    calibrate,
    fit_noise_model,
    group_by_interval,
    load_blah2_capture,
    prediction_interval,
    usable,
    wavelength_km,
)
from retina_tracker.output import InnovationWriter
from retina_tracker.tracker import Tracker

FC = 177000000.0
FS = 2000000.0
CPI = 0.5

TRUE_DELAY_SIGMA_KM = 0.022
TRUE_DOPPLER_SIGMA_HZ = 0.356
TRUE_VELOCITY_ERROR = 0.015

INTERVALS = (0.5, 1.0, 1.5, 2.0, 3.0)
PER_INTERVAL = 1200


def record(interval, delay_innovation, rate_innovation, doppler=-120.0, birth=0, **overrides):
    record = {
        "track_id": "250915-000001",
        "birth": birth,
        "timestamp": 1718747745000,
        "dt": 0.5,
        "n_missed": round(interval / 0.5) - 1,
        "snr": 16.0,
        "delay": 20.0,
        "doppler": doppler,
        "interfering": False,
        "q_scale": 1.0,
        "innovation": [delay_innovation, rate_innovation],
        "s_diag": [1.0, 5e-6],
        "nis": 1.0,
    }
    record.update(overrides)
    return record


def synthesise(seed=1, delay_sigma=TRUE_DELAY_SIGMA_KM, velocity_error=TRUE_VELOCITY_ERROR, exponent=2.0):
    """Innovations from a filter whose R is `delay_sigma` and whose model error
    is a velocity held for the whole prediction interval."""
    rng = random.Random(seed)
    rate_sigma = TRUE_DOPPLER_SIGMA_HZ * wavelength_km(FC)
    records = []
    for interval in INTERVALS:
        growth = velocity_error * interval ** (exponent / 2.0)
        delay_spread = (delay_sigma**2 + growth**2) ** 0.5
        rate_spread = (rate_sigma**2 + (growth / 50.0) ** 2) ** 0.5
        for i in range(PER_INTERVAL):
            records.append(
                record(
                    interval,
                    rng.gauss(0.0, delay_spread),
                    rng.gauss(0.0, rate_spread),
                    doppler=-120.0 + i % 40,
                )
            )
    return records


class TestTheFitRecoversWhatWasPutIn:
    def test_r_comes_back_from_the_intercept(self):
        report = calibrate(synthesise(), FC, FS, CPI)
        assert report["axes"]["delay"]["sigma"] == pytest.approx(TRUE_DELAY_SIGMA_KM, rel=0.2)

    def test_r_comes_back_in_resolution_cells(self):
        """The number that has to agree across sites, and the one a shipped
        constant is expressed in."""
        report = calibrate(synthesise(), FC, FS, CPI)
        assert report["axes"]["delay"]["cells"] == pytest.approx(0.147, rel=0.25)

    def test_the_doppler_axis_comes_back_in_hertz(self):
        """The innovation is a range rate. Reported in Hz it is comparable with
        the configured value and with the other axis in cells."""
        report = calibrate(synthesise(), FC, FS, CPI)
        assert report["axes"]["doppler"]["sigma"] == pytest.approx(TRUE_DOPPLER_SIGMA_HZ, rel=0.25)

    def test_the_shape_of_the_model_error_comes_back(self):
        """Two, not five. An exponent near five would mean white jerk really is
        what the filter is missing and range_jerk merely needs raising."""
        report = calibrate(synthesise(), FC, FS, CPI)
        assert report["axes"]["delay"]["exponent"] == pytest.approx(2.0, abs=0.4)

    def test_the_size_of_the_model_error_comes_back(self):
        report = calibrate(synthesise(), FC, FS, CPI)
        assert report["axes"]["delay"]["error_rate"] == pytest.approx(TRUE_VELOCITY_ERROR, rel=0.3)

    def test_a_larger_r_is_reported_larger(self):
        """The oversized R the tracker ships with today, against the measured
        one, on data whose model error is identical."""
        loose = calibrate(synthesise(seed=2, delay_sigma=0.2), FC, FS, CPI)
        tight = calibrate(synthesise(seed=2), FC, FS, CPI)
        assert loose["axes"]["delay"]["sigma"] > 5 * tight["axes"]["delay"]["sigma"]

    def test_white_jerk_is_told_apart_from_a_velocity_error(self):
        report = calibrate(synthesise(seed=3, exponent=5.0), FC, FS, CPI)
        assert report["axes"]["delay"]["exponent"] > 3.0


class TestSeparatingTheTwo:
    def test_a_constant_scatter_reads_as_all_r_and_no_q(self):
        report = calibrate(synthesise(seed=4, velocity_error=0.0), FC, FS, CPI)
        assert report["axes"]["delay"]["sigma"] == pytest.approx(TRUE_DELAY_SIGMA_KM, rel=0.15)
        assert report["axes"]["delay"]["error_rate"] < TRUE_DELAY_SIGMA_KM

    def test_the_interval_is_what_carries_the_separation(self):
        """Without a spread of prediction intervals there is one equation and
        two unknowns, and the honest answer is to decline."""
        records = [r for r in synthesise(seed=5) if prediction_interval(r) == 0.5]
        assert "sigma" not in calibrate(records, FC, FS, CPI)["axes"]["delay"]

    def test_a_coasted_update_is_a_longer_prediction(self):
        assert prediction_interval({"dt": 0.5, "n_missed": 3}) == pytest.approx(2.0)

    def test_scatter_is_reported_against_what_the_filter_claimed(self):
        """Per axis, because NIS sums the two and that is how an oversized R
        and an undersized Q have been hiding inside it."""
        report = calibrate(synthesise(seed=6), FC, FS, CPI)
        # Far below one is the signature being hunted: the filter is carrying
        # an R so large that its innovations never come close to filling it.
        assert report["axes"]["delay"]["scatter_vs_claimed"] < 0.1


class TestWhatIsExcluded:
    def test_interference_is_dropped_on_the_records_own_verdict(self):
        records = synthesise(seed=7)[:200]
        for r in records[:120]:
            r["interfering"] = True
        assert len(usable(records)) == 80

    def test_a_degenerate_update_is_not_a_measurement(self):
        """A singular covariance skipped the measurement, so its innovation is
        NaN and there is nothing in it to fit."""
        records = [record(0.5, float("nan"), float("nan")), record(0.5, 0.01, 1e-5)]
        assert len(usable(records)) == 1

    def test_older_records_fall_back_to_a_tracks_doppler_holding_still(self):
        """Written before the tracker carried the occupancy map's verdict."""
        tone = [record(0.5, 0.01, 1e-5, doppler=27.9, birth=1) for _ in range(12)]
        aircraft = [record(0.5, 0.01, 1e-5, doppler=-120.0 + 4 * i, birth=2) for i in range(12)]
        for r in tone + aircraft:
            del r["interfering"]
        kept = usable(tone + aircraft)
        assert {r["birth"] for r in kept} == {2}

    def test_the_fallback_is_not_used_when_a_verdict_is_present(self):
        """A real target can hold one Doppler bin. Where the map has spoken,
        the heuristic that would convict it must not run."""
        holding = [record(0.5, 0.01, 1e-5, doppler=27.9, birth=1) for _ in range(12)]
        assert len(usable(holding)) == 12

    def test_a_thin_interval_is_not_fitted(self):
        assert fit_noise_model([(0.5, 40, 1e-4), (1.0, 40, 2e-4)]) is None

    def test_a_sparse_interval_is_not_a_measurement(self):
        records = [record(0.5, 0.01, 1e-5) for _ in range(5)]
        assert group_by_interval(records, 0) == []


class TestTheCaptureConfig:
    def test_all_three_parameters_come_out_of_blah2s_own_file(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text(
            "capture:\n  fs: 2000000\n  fc: 177000000\n  device:\n    type: 'RspDuo'\n"
            "process:\n  data:\n    cpi: 0.5\n    buffer: 1.5\n"
        )
        assert load_blah2_capture(str(path)) == {"fs": 2000000.0, "fc": 177000000.0, "cpi": 0.5}


class TestItReadsWhatTheTrackerWrites:
    """The contract between the two is a set of field names in a file. A test
    that synthesises its own records would never notice one being renamed."""

    def test_a_real_recording_calibrates(self, tmp_path):
        path = tmp_path / "innovations.jsonl"
        writer = InnovationWriter(str(path), max_bytes=0)
        tracker = Tracker(innovation_writer=writer)
        for i in range(60):
            tracker.process_frame([{"delay": 20.0 - 0.05 * i, "doppler": -120.0, "snr": 16.0}], 1718747745000 + i * 500)
        writer.close()

        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        report = calibrate(records, FC, FS, CPI)

        assert report["n_records"] == len(records)
        assert report["n_used"] > 0
        assert report["nis_median"] is not None
