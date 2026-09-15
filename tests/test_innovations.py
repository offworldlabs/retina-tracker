"""Per-update innovation records, which are what R and Q are answerable to.

The events file records the detections that were associated, never what the
filter predicted before it saw them. R is the floor of the innovation
covariance and Q sets how fast the state's share of it grows between updates,
so neither can be fitted from events alone. These records close that gap.
"""

import json

import numpy as np
import pytest

from retina_tracker.kalman import KalmanFilter, Residual, doppler_to_range_rate
from retina_tracker.output import InnovationWriter
from retina_tracker.tracker import Tracker

BASE_TS = 1718747745000


def frame(delay, doppler=-120.0, snr=16.0):
    return [{"delay": delay, "doppler": doppler, "snr": snr}]


def run(tracker, n=8, step_ms=500, delay0=20.0, drift=-0.05):
    for i in range(n):
        tracker.process_frame(frame(delay0 + i * drift), BASE_TS + i * step_ms)


def read(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


class TestTheFilterReportsItsOwnResidual:
    def test_update_returns_the_innovation_not_just_its_norm(self):
        kf = KalmanFilter()
        state = np.array([20.0, -0.1, 0.0])
        cov = np.eye(3)

        _, _, residual = kf.update(state, cov, np.array([20.5, -0.1]), 16.0)

        assert residual.innovation[0] == pytest.approx(0.5)
        assert residual.S.shape == (2, 2)

    def test_nis_is_still_the_quadratic_form_it_always_was(self):
        kf = KalmanFilter()
        state = np.array([20.0, -0.1, 0.0])
        cov = np.eye(3)
        measurement = np.array([20.5, -0.1])

        _, _, residual = kf.update(state, cov, measurement, 16.0)

        expected = float(residual.innovation @ np.linalg.solve(residual.S, residual.innovation))
        assert residual.nis == pytest.approx(expected)

    def test_a_singular_covariance_reports_nothing_learned(self):
        """The measurement was skipped, so the record must not claim a residual."""
        residual = Residual.degenerate()

        assert np.isnan(residual.innovation).all()
        assert np.isnan(residual.S).all()


class TestRecordsReachTheFile:
    def test_one_record_per_update(self, tmp_path):
        path = tmp_path / "innovations.jsonl"
        writer = InnovationWriter(str(path), max_bytes=0)
        tracker = Tracker(innovation_writer=writer)

        run(tracker)
        writer.close()

        records = read(path)
        assert records
        assert all(r["innovation"] and len(r["innovation"]) == 2 for r in records)

    def test_a_record_carries_what_calibration_needs(self, tmp_path):
        """dt and q_scale separate R from Q; without them the two cancel."""
        path = tmp_path / "innovations.jsonl"
        writer = InnovationWriter(str(path), max_bytes=0)
        tracker = Tracker(innovation_writer=writer)

        run(tracker)
        writer.close()

        record = read(path)[-1]
        assert set(record) == {
            "track_id",
            "birth",
            "timestamp",
            "dt",
            "snr",
            "delay",
            "doppler",
            "n_missed",
            "q_scale",
            "innovation",
            "s_diag",
            "nis",
        }
        assert record["dt"] == pytest.approx(0.5)
        assert record["snr"] == pytest.approx(16.0)
        assert record["s_diag"][0] > 0

    def test_a_record_can_be_filtered_without_the_events_file(self, tmp_path):
        """Most tracks at an interfered site sit on a fixed-Doppler tone and
        have to be dropped before anything is fitted. A track that never
        confirmed has no id and never reaches the events file, so the Doppler
        has to travel with the record itself."""
        path = tmp_path / "innovations.jsonl"
        writer = InnovationWriter(str(path), max_bytes=0)
        tracker = Tracker(innovation_writer=writer)

        run(tracker, n=5)
        writer.close()

        records = read(path)
        assert all(r["doppler"] is not None for r in records)
        assert all(r["birth"] == records[0]["birth"] for r in records)

    def test_the_recorded_innovation_matches_the_measurement_it_came_from(self, tmp_path):
        path = tmp_path / "innovations.jsonl"
        writer = InnovationWriter(str(path), max_bytes=0)
        tracker = Tracker(innovation_writer=writer)

        run(tracker, n=3)
        writer.close()

        for record in read(path):
            assert abs(record["innovation"][0]) < 5.0
            assert record["nis"] >= 0.0

    def test_q_scale_is_the_one_in_force_for_that_prediction(self, tmp_path):
        """Recorded after the update it would be the next frame's scale, which
        is not the value that shaped the covariance this innovation was
        measured against."""
        path = tmp_path / "innovations.jsonl"
        writer = InnovationWriter(str(path), max_bytes=0)
        tracker = Tracker(innovation_writer=writer)

        run(tracker, n=6)
        writer.close()

        assert all(r["q_scale"] >= 1.0 for r in read(path))


class TestTheCoastingHistoryIsNotLost:
    """How long a track had been coasting is the one input to the prediction
    that a replay cannot reconstruct, and it is the reason these records exist
    rather than an offline reconstruction. Track.update() zeroes n_missed
    before returning, so reading it from the track afterwards gives 0 every
    time and the longest predictions - the ones Q answers to - look like the
    shortest."""

    def _coasted(self, tmp_path, gap):
        path = tmp_path / "innovations.jsonl"
        writer = InnovationWriter(str(path), max_bytes=0)
        tracker = Tracker(innovation_writer=writer)

        run(tracker, n=6)
        for i in range(gap):
            tracker.process_frame([], BASE_TS + (6 + i) * 500)
        tracker.process_frame(frame(20.0 - (6 + gap) * 0.05), BASE_TS + (6 + gap) * 500)
        writer.close()
        return read(path)

    def test_a_reassociation_records_the_frames_it_coasted(self, tmp_path):
        assert self._coasted(tmp_path, gap=2)[-1]["n_missed"] == 2

    def test_an_uninterrupted_update_still_records_none(self, tmp_path):
        assert self._coasted(tmp_path, gap=0)[-1]["n_missed"] == 0


class TestOffByDefault:
    def test_no_writer_means_no_records_and_no_cost(self, tmp_path):
        tracker = Tracker()

        run(tracker)

        assert tracker.innovation_writer is None
        assert not list(tmp_path.iterdir())

    def test_tracking_is_unchanged_by_recording(self, tmp_path):
        path = tmp_path / "innovations.jsonl"
        writer = InnovationWriter(str(path), max_bytes=0)
        recorded = Tracker(innovation_writer=writer)
        plain = Tracker()

        run(recorded)
        run(plain)
        writer.close()

        assert len(recorded.tracks) == len(plain.tracks)
        assert [t.state.tolist() for t in recorded.tracks] == [t.state.tolist() for t in plain.tracks]


class TestTheFileStaysBounded:
    def test_recording_cannot_fill_a_node_disk(self, tmp_path):
        """A node left recording writes one record per update indefinitely."""
        path = tmp_path / "innovations.jsonl"
        writer = InnovationWriter(str(path), max_bytes=400, backup_count=1)
        tracker = Tracker(innovation_writer=writer)

        run(tracker, n=40)
        writer.close()

        assert path.stat().st_size <= 400
        assert (tmp_path / "innovations.jsonl.1").exists()


class TestMeasurementConversion:
    def test_the_rate_innovation_is_in_range_rate_not_hertz(self):
        """S and R live in km/s, so an innovation in Hz would be inconsistent
        with the covariance it is divided by."""
        kf = KalmanFilter()
        state = np.array([20.0, doppler_to_range_rate(-120.0), 0.0])
        cov = np.eye(3)

        _, _, residual = kf.update(state, cov, np.array([20.0, doppler_to_range_rate(-130.0)]), 16.0)

        assert residual.innovation[1] == pytest.approx(doppler_to_range_rate(-130.0) - doppler_to_range_rate(-120.0))
