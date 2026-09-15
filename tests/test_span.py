"""Working at whatever Doppler span the node is set to.

blah2 computes its ambiguity map over a delay and Doppler span it is told, and
that span is a deployment choice: nodes have run +/-200, +/-300 and +/-1000 Hz.
Nothing in the tracker may assume one of them. What the tracker may do is read
the bounds, because they are the only honest answer to two questions it cannot
otherwise answer: whether an arriving detection could have come from the map at
all, and what range an axis drawn over it should cover.

The bounds default to None, and None must mean "nobody has said" rather than a
plausible number. A guessed bound discards real detections and looks exactly
like a quiet sky.
"""

import random

import pytest

from retina_tracker.config import (
    DELAY_CELL_KM,
    DELAY_MAX_KM,
    DELAY_MIN_KM,
    DOPPLER_BIN_HZ,
    DOPPLER_MAX_HZ,
    DOPPLER_MIN_HZ,
    load_blah2_config,
    load_config,
    set_config,
)
from retina_tracker.interference import SPREAD_REFRESH_FRAMES, DopplerOccupancy
from retina_tracker.kalman import doppler_to_range_rate
from retina_tracker.tracker import Tracker

FRAME_MS = 500


def build_config(**radar):
    config = load_config("nonexistent.yaml")
    config["adsb"]["enabled"] = False
    config["radar"].update(radar)
    return config


def span_config(span=300, **radar):
    """A node that has said what it can see: +/-span Hz, 0 to 60 km."""
    return build_config(doppler_min=-span, doppler_max=span, delay_min_bins=-10, delay_max_bins=400, **radar)


@pytest.fixture(autouse=True)
def _config():
    set_config(build_config())
    yield


def det(delay, doppler, snr=15.0):
    return {"delay": delay, "doppler": doppler, "snr": snr}


def run(tracker, detections, n=1):
    for i in range(n):
        tracker.process_frame(list(detections), i * FRAME_MS)
    return tracker


class TestTheBoundsComeFromBlah2:
    def test_the_ambiguity_block_is_read(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text(
            "capture:\n  fs: 2000000\n  fc: 213000000\n"
            "process:\n  data:\n    cpi: 0.5\n"
            "  ambiguity:\n    delayMin: -10\n    delayMax: 400\n"
            "    dopplerMin: -300\n    dopplerMax: 300\n"
        )
        assert load_blah2_config(str(path)) == {
            "fc": 213000000,
            "fs": 2000000,
            "cpi": 0.5,
            "dopplerMin": -300,
            "dopplerMax": 300,
            "delayMin": -10,
            "delayMax": 400,
        }

    def test_a_config_without_an_ambiguity_block_still_loads(self, tmp_path):
        """An older node config, or one being written. The capture parameters
        still arrive; the bounds simply are not known."""
        path = tmp_path / "config.yml"
        path.write_text("capture:\n  fs: 2000000\n  fc: 213000000\nprocess:\n  data:\n    cpi: 0.5\n")
        assert "dopplerMin" not in load_blah2_config(str(path))

    def test_delay_bounds_become_kilometres_via_the_sample_rate(self):
        """blah2 states them in delay bins, which are only kilometres once the
        sample rate says how wide a bin is."""
        set_config(span_config())
        assert DELAY_MAX_KM() == pytest.approx(400 * DELAY_CELL_KM())
        assert DELAY_MAX_KM() == pytest.approx(59.96, abs=0.01)
        assert DELAY_MIN_KM() == pytest.approx(-1.5, abs=0.01)

    def test_unknown_bounds_read_as_unknown_not_as_zero(self):
        assert DOPPLER_MIN_HZ() is None
        assert DOPPLER_MAX_HZ() is None
        assert DELAY_MIN_KM() is None
        assert DELAY_MAX_KM() is None


class TestRejectingWhatTheNodeCannotHaveSeen:
    def test_a_detection_beyond_the_span_is_rejected(self):
        set_config(span_config(span=300))
        tracker = run(Tracker(config=None), [det(20.0, 500.0)])
        assert tracker.n_detections_rejected == 1
        assert not tracker.tracks

    def test_a_detection_inside_the_span_is_kept(self):
        set_config(span_config(span=300))
        tracker = run(Tracker(config=None), [det(20.0, 290.0)])
        assert tracker.n_detections_rejected == 0
        assert tracker.tracks

    def test_one_resolution_cell_of_slack_at_the_edge(self):
        """The centroider interpolates between bin centres, so a real peak in
        the outermost bin can land just outside it."""
        set_config(span_config(span=300))
        tracker = run(Tracker(config=None), [det(20.0, 300.0 + DOPPLER_BIN_HZ() * 0.5)])
        assert tracker.n_detections_rejected == 0

    def test_a_delay_beyond_the_map_is_rejected(self):
        set_config(span_config())
        tracker = run(Tracker(config=None), [det(500.0, 20.0)])
        assert tracker.n_detections_rejected == 1

    def test_nothing_is_rejected_when_nobody_has_said_what_the_span_is(self):
        """The safe direction. An invented bound silently discards real
        detections and is indistinguishable from an empty sky."""
        tracker = run(Tracker(config=None), [det(20.0, 5000.0), det(900.0, 1.0)])
        assert tracker.n_detections_rejected == 0

    def test_a_non_finite_measurement_never_reaches_the_filter(self):
        set_config(span_config())
        tracker = run(Tracker(config=None), [det(float("nan"), 20.0), det(20.0, float("inf"))])
        assert tracker.n_detections_rejected == 2
        assert not tracker.tracks

    def test_a_nan_snr_is_still_recorded_rather_than_rejected(self):
        """It has a real delay and Doppler. The SNR partition routes it to
        below_snr on purpose, which keeps it in the record."""

        class Sink:
            def __init__(self):
                self.below = 0

            def write_detections(self, timestamp, associated, unassociated, below_snr):
                self.below += len(below_snr)

        set_config(span_config())
        sink = Sink()
        tracker = run(Tracker(config=None, detection_sink=sink), [det(20.0, 30.0, snr=float("nan"))], n=3)
        assert tracker.n_detections_rejected == 0
        assert sink.below >= 1

    def test_a_measurement_that_is_not_a_number_is_rejected(self):
        set_config(span_config())
        tracker = run(Tracker(config=None), [det(20.0, 30.0, snr="loud"), {"delay": 20.0, "doppler": 30.0}])
        assert tracker.n_detections_rejected == 2

    def test_the_count_survives_nothing_being_wrong(self):
        set_config(span_config())
        tracker = run(Tracker(config=None), [det(20.0, 30.0)], n=4)
        assert tracker.n_detections_rejected == 0


class TestABoundThatIsNotABound:
    """The failure this whole mechanism exists to avoid is a node that looks
    dead rather than misconfigured, so a nonsense bound must not be honoured."""

    def test_a_transposed_pair_rejects_nothing(self):
        """dopplerMin above dopplerMax is a typo. Honouring it would reject
        every detection at the node and read as an empty sky."""
        set_config(build_config(doppler_min=300, doppler_max=-300))
        tracker = run(Tracker(config=None), [det(20.0, d) for d in (-400.0, 0.0, 400.0)])
        assert tracker.n_detections_rejected == 0

    def test_a_transposed_pair_is_dropped_where_it_is_read(self, tmp_path):
        """Twice over: at the config, so the node says so once at startup, and
        at the bound, so a hand-edited tracker config cannot do it either."""
        path = tmp_path / "config.yml"
        path.write_text(
            "capture:\n  fs: 2000000\n  fc: 213000000\n"
            "process:\n  data:\n    cpi: 0.5\n"
            "  ambiguity:\n    dopplerMin: 300\n    dopplerMax: -300\n"
        )
        found = load_blah2_config(str(path))
        assert "dopplerMin" not in found and "dopplerMax" not in found
        assert found["fc"] == 213000000

    def test_half_a_pair_is_still_a_statement(self):
        """Unlike an axis, which needs both ends, a single stated bound is a
        real limit and is worth holding a detection to."""
        set_config(build_config(doppler_min=-300))
        tracker = run(Tracker(config=None), [det(20.0, -400.0), det(20.0, 400.0)])
        assert tracker.n_detections_rejected == 1

    def test_a_zero_width_span_admits_only_its_own_bin(self):
        """Degenerate but not nonsense: a node computing a single Doppler bin.
        One cell of slack either side, as everywhere else."""
        set_config(build_config(doppler_min=0, doppler_max=0))
        tracker = run(Tracker(config=None), [det(20.0, 0.0), det(20.0, 50.0)])
        assert tracker.n_detections_rejected == 1

    @pytest.mark.parametrize("lo,hi", [(0, 0), (-1, 1), (-15, 15), (-200, 200), (-300, 300), (-1000, 1000)])
    def test_every_span_across_the_range_holds_its_own_line(self, lo, hi):
        """0 to 1000 Hz, symmetric. What is inside is kept and what is outside
        is rejected, with no case left to the reader's imagination."""
        set_config(build_config(doppler_min=lo, doppler_max=hi))
        slack = DOPPLER_BIN_HZ()
        candidates = [float(d) for d in range(-1200, 1201, 25)]
        expected = sum(1 for d in candidates if not (lo - slack <= d <= hi + slack))
        tracker = run(Tracker(config=None), [det(20.0, d) for d in candidates])
        assert tracker.n_detections_rejected == expected

    @pytest.mark.parametrize("lo,hi", [(-100, 300), (0, 300), (-300, 0)])
    def test_an_asymmetric_span_is_not_assumed_symmetric(self, lo, hi):
        set_config(build_config(doppler_min=lo, doppler_max=hi))
        slack = DOPPLER_BIN_HZ()
        candidates = [float(d) for d in range(-1200, 1201, 25)]
        expected = sum(1 for d in candidates if not (lo - slack <= d <= hi + slack))
        tracker = run(Tracker(config=None), [det(20.0, d) for d in candidates])
        assert tracker.n_detections_rejected == expected


class TestTheMapWidensWithTheSpan:
    """The same tone, at three spans, with nothing changed but the span."""

    def _tone_at(self, doppler, window_s=5.0, seed=1):
        config = span_config(span=1000)
        config["interference"]["window_s"] = window_s
        set_config(config)
        occupancy = DopplerOccupancy.from_config()
        rng = random.Random(seed)
        for i in range(int(window_s / 0.5)):
            frame = [det(rng.uniform(5.0, 60.0), doppler + rng.uniform(-0.4, 0.4)) for _ in range(rng.randint(1, 3))]
            occupancy.observe(frame, i * FRAME_MS)
        return occupancy

    @pytest.mark.parametrize("tone", [27.9, -60.0, 180.0, -290.0, 850.0])
    def test_a_tone_anywhere_in_the_span_is_learned(self, tone):
        assert self._tone_at(tone).is_interfering(tone)

    def test_the_bin_width_is_the_cpi_not_the_span(self):
        """Widening what the node sees does not coarsen how finely it sees it,
        so a threshold in cells means the same thing at every span."""
        set_config(span_config(span=1000, cpi=0.5))
        assert DOPPLER_BIN_HZ() == pytest.approx(2.0)
        set_config(span_config(span=200, cpi=0.5))
        assert DOPPLER_BIN_HZ() == pytest.approx(2.0)


class TestTheCostDoesNotGrowWithTheSpan:
    """Occupancy is a counter and is exact every frame. The delay spread is a
    percentile over the whole window, and recomputing every bin every frame is
    the one cost that would scale with how much sky the node can see."""

    def _busy(self, span, frames=40, window_s=5.0):
        config = span_config(span=span)
        config["interference"]["window_s"] = window_s
        set_config(config)
        occupancy = DopplerOccupancy.from_config()
        rng = random.Random(2)
        bins = list(range(-span, span + 1, 2))
        for i in range(frames):
            occupancy.observe([det(rng.uniform(5.0, 60.0), d) for d in bins], i * FRAME_MS)
        return occupancy

    def test_a_bin_is_measured_the_first_time_it_is_asked_about(self):
        """A new interferer must not be judged on a verdict it never had."""
        occupancy = self._busy(span=200)
        assert occupancy.interfering_bins()

    def test_a_stale_verdict_is_not_kept_forever(self):
        """A bin convicted while it was scattered has to be acquitted once it
        is not. Caching the verdict and never revisiting it would leave the
        map refusing a bin on evidence it no longer has."""
        config = span_config(span=200)
        config["interference"]["window_s"] = 10.0
        set_config(config)
        occupancy = DopplerOccupancy.from_config()
        rng = random.Random(3)

        frames = int(10.0 / 0.5)
        for i in range(frames * 2):
            occupancy.observe([det(rng.uniform(5.0, 60.0), 40.0) for _ in range(2)], i * FRAME_MS)
        assert occupancy.is_interfering(40.0)

        # The same bin, now holding a target rather than a ridge: one detection
        # a frame, marching in delay at the rate its own Doppler mandates,
        # which is the whole difference between the two.
        base = frames * 2
        per_frame = doppler_to_range_rate(40.0) * FRAME_MS / 1000.0
        for i in range(frames * 2 + SPREAD_REFRESH_FRAMES):
            t = (base + i) * FRAME_MS
            occupancy.observe([det(30.0 + per_frame * i, 40.0)], t)
        assert not occupancy.is_interfering(40.0)

    def test_a_bin_that_leaves_the_window_leaves_the_cache(self):
        """Otherwise the cache is a slow leak at a node whose interference
        wanders across the span."""
        occupancy = self._busy(span=200, frames=12)
        assert occupancy._spread_exceeded
        for i in range(30):
            occupancy.observe([det(20.0, 0.0)], (200 + i) * FRAME_MS)
        assert set(occupancy._spread_exceeded) <= set(occupancy._frames_occupied)
