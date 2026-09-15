"""Refusing to build tracks on a node's own interference.

Measured on three live nodes on 2026-09-14, each node's own CW tracks excluded:

| Node          | Tone      | % of detections | % of frames | Delay cells |
| ------------- | --------- | --------------- | ----------- | ----------- |
| fairforest B  | +27.9 Hz  | 98%             | 92.5%       | 12 of 12    |
| jn1           | -60 Hz    | 74%             | 38.9%       | 7 of 12     |
| owl-ded9      | none      | 7%              | 18.3%       | 2 of 12     |

Two nodes, two different frequencies, which is the whole reason this is learned
rather than shipped: a notch tuned on either one misses the other. owl-ded9 is
what a clean node looks like, and is the reference for suppressing nothing.

The tracks built on those tones are kinematically flawless - spans of 26, 20 and
18 km with delay rate and Doppler rate agreeing to 1% - because a tone puts a
detection at every delay in every frame, so the filter always finds one exactly
where it predicted and the track manufactures its own confirming evidence.
Nothing downstream of initiation has grounds to reject them, which is why this
acts at initiation and nowhere else.
"""

import json
import random

import pytest

from retina_tracker.config import (
    DELAY_CELL_KM,
    DOPPLER_BIN_HZ,
    load_blah2_config,
    load_config,
    set_config,
)
from retina_tracker.interference import DopplerOccupancy
from retina_tracker.kalman import doppler_to_range_rate
from retina_tracker.track import Track
from retina_tracker.tracker import BACKWARDS_RUN_BEFORE_RESYNC, Tracker

FRAME_MS = 500
WINDOW_S = 5.0
WINDOW_FRAMES = int(WINDOW_S / 0.5)


def build_config(**interference):
    config = load_config("nonexistent.yaml")
    config["adsb"]["enabled"] = False
    config["interference"].update({"window_s": WINDOW_S}, **interference)
    return config


@pytest.fixture(autouse=True)
def _config():
    set_config(build_config())
    yield


def det(delay, doppler, snr=15.0):
    return {"delay": delay, "doppler": doppler, "snr": snr}


def tone_frame(doppler, rng, delay_min=5.0, delay_max=65.0):
    """One frame of a CW ridge, as the detector actually reports it.

    The ridge is continuous in delay, but CFAR thins it to one to three peaks,
    landing wherever it happened to be brightest. That is why a single frame is
    weak evidence and persistence is the signal.
    """
    return [det(rng.uniform(delay_min, delay_max), doppler + rng.uniform(-0.4, 0.4)) for _ in range(rng.randint(1, 3))]


def aircraft(delay_0, doppler, t_ms, jitter=0.0, rng=None):
    """A target whose delay marches at the rate its Doppler mandates.

    Bistatic range rate is the Doppler, so this is not one choice of trajectory
    among many: any real target sitting in a Doppler bin moves in delay like
    this, and that is what makes drift-removed scatter separate the two.
    """
    delay = delay_0 + doppler_to_range_rate(doppler) * t_ms / 1000.0
    if jitter:
        delay += rng.uniform(-jitter, jitter)
    return det(delay, doppler)


def feed(occupancy, frames, start_ms=0):
    for i, detections in enumerate(frames):
        occupancy.observe(detections, start_ms + i * FRAME_MS)
    return occupancy


def make_map(**overrides):
    set_config(build_config(**overrides))
    return DopplerOccupancy.from_config()


class TestLearningWhatTheNodeSees:
    def test_a_persistent_tone_is_learned(self):
        rng = random.Random(1)
        occupancy = feed(make_map(), [tone_frame(27.9, rng) for _ in range(WINDOW_FRAMES)])
        assert occupancy.is_interfering(27.9)

    def test_the_same_constants_catch_a_tone_at_either_frequency(self):
        """fairforest B sits at +27.9 Hz and jn1 at -60 Hz. One threshold set,
        no per-site key, both caught. A fixed notch could only ever do one."""
        for tone in (27.9, -60.0):
            rng = random.Random(2)
            occupancy = feed(make_map(), [tone_frame(tone, rng) for _ in range(WINDOW_FRAMES)])
            assert occupancy.is_interfering(tone), tone

    def test_a_clean_node_learns_nothing(self):
        """owl-ded9: ordinary traffic, no bin above 18.3% of frames."""
        rng = random.Random(3)
        frames = []
        for i in range(WINDOW_FRAMES * 3):
            t = i * FRAME_MS
            frames.append(
                [
                    aircraft(20.0, 80.0 - 4.0 * i, t, jitter=0.02, rng=rng),
                    aircraft(45.0, -30.0 + 3.0 * i, t, jitter=0.02, rng=rng),
                ]
            )
        occupancy = feed(make_map(), frames)
        assert occupancy.interfering_bins() == frozenset()

    def test_an_aircraft_crossing_the_tone_does_not_clear_the_bin(self):
        """The aircraft's few frames in the bin are swamped by the tone's own,
        so a real target flying through an interferer does not unlearn it."""
        rng = random.Random(4)
        frames = []
        for i in range(WINDOW_FRAMES):
            frame = tone_frame(27.9, rng)
            if i in (4, 5):
                frame.append(aircraft(30.0, 27.9, i * FRAME_MS))
            frames.append(frame)
        assert feed(make_map(), frames).is_interfering(27.9)


class TestProtectingARealTarget:
    """The delay-spread condition is the only thing standing between the map
    and an aircraft that holds near-constant bistatic Doppler. Occupancy alone
    would convict one, because it is in the bin in every frame.

    At the shipped window rather than the short one the rest of these tests
    use: the longer the window, the further such a target has moved in delay,
    and it is over a whole shipped window that it has to survive.
    """

    FRAMES = 120

    def _holding(self, seed=5, jitter=0.022):
        rng = random.Random(seed)
        return [[aircraft(40.0, 30.0, i * FRAME_MS, jitter=jitter, rng=rng)] for i in range(self.FRAMES * 2)]

    def test_a_target_holding_one_doppler_bin_is_not_interference(self):
        occupancy = feed(make_map(window_s=60.0), self._holding())
        assert occupancy.frame_fraction(occupancy._bin(30.0)) == 1.0
        assert not occupancy.is_interfering(30.0)

    def test_it_is_the_drift_removal_that_saves_it(self):
        """Its raw delay extent is many cells wide: 45 m/s for a minute. It is
        only narrow once the drift its own Doppler mandates is taken out."""
        occupancy = feed(make_map(window_s=60.0), self._holding())
        delays = [d for _, d in occupancy._samples[occupancy._bin(30.0)]]
        raw_extent_cells = (max(delays) - min(delays)) / DELAY_CELL_KM()
        assert raw_extent_cells > 5 * occupancy.max_delay_spread_cells
        assert occupancy.delay_spread_cells(occupancy._bin(30.0)) < 1.0

    def test_a_target_drifting_across_the_bin_stays_inside_the_threshold(self):
        """Its range rate is not quite constant, so the drift removal leaves a
        curvature term. Over a whole window that is a fraction of a cell."""
        rng = random.Random(6)
        centre = DOPPLER_BIN_HZ() * round(30.0 / DOPPLER_BIN_HZ())
        frames = []
        delay = 40.0
        for i in range(self.FRAMES * 2):
            doppler = centre - DOPPLER_BIN_HZ() / 2 + DOPPLER_BIN_HZ() * i / (self.FRAMES * 2)
            delay += doppler_to_range_rate(doppler) * FRAME_MS / 1000.0
            frames.append([det(delay + rng.uniform(-0.022, 0.022), doppler)])
        occupancy = feed(make_map(window_s=60.0), frames)
        assert not occupancy.is_interfering(centre)

    def test_a_single_stray_detection_does_not_convict_the_bin(self):
        """A percentile range rather than the full extent, because one stray
        in a bin an aircraft otherwise owns would push the extent over."""
        frames = self._holding(seed=7)
        frames[3].append(det(150.0, 30.0))
        assert not feed(make_map(window_s=60.0), frames).is_interfering(30.0)


class TestAgainstTheMeasuredProfiles:
    """The three nodes as they were measured, at the shipped thresholds.

    One set of constants has to convict both interfered nodes at their two
    different frequencies and acquit the clean one. These reproduce each node's
    own numbers - how often the bin was busy, and how far its detections spread
    across the delay axis - rather than the numbers the thresholds were picked
    from, so a later change to either threshold has to answer to all three.
    """

    FRAMES = 240
    AXIS_KM = (5.0, 65.0)

    def _node(self, tones, frame_fraction, axis_share, seed):
        """Frames of interference occupying `axis_share` of the delay axis."""
        rng = random.Random(seed)
        low, high = self.AXIS_KM
        span = (high - low) * axis_share
        frames = []
        for _ in range(self.FRAMES):
            frame = []
            for tone in tones:
                if rng.random() < frame_fraction:
                    for _ in range(rng.randint(1, 3)):
                        frame.append(det(low + rng.uniform(0, span), tone + rng.uniform(-0.4, 0.4)))
            frames.append(frame)
        return frames

    def test_fairforest_b_is_convicted(self):
        """+27.9 Hz, 92.5% of frames, spread over all 12 delay cells."""
        occupancy = feed(make_map(window_s=60.0), self._node([27.9], 0.925, 1.0, seed=20))
        assert occupancy.is_interfering(27.9)

    def test_jonathan_node_1_is_convicted(self):
        """A pair at about plus and minus 60 Hz, 38.9% of frames, 7 of 12
        delay cells. The thinnest margin of the three, and the reason the
        occupancy threshold sits at 0.3 rather than higher."""
        occupancy = feed(make_map(window_s=60.0), self._node([-60.0, 60.0], 0.389, 7 / 12, seed=21))
        assert occupancy.is_interfering(-60.0)
        assert occupancy.is_interfering(60.0)

    def test_owl_ded9_is_acquitted(self):
        """Nothing above 18.3% of frames or 2 of 12 delay cells. A node with
        no interference has to learn an empty map, or the whole design is a
        per-site threshold in disguise."""
        occupancy = feed(make_map(window_s=60.0), self._node([12.0, -85.0], 0.183, 2 / 12, seed=22))
        assert occupancy.interfering_bins() == frozenset()

    def test_a_busy_sky_of_real_traffic_is_acquitted(self):
        """Twelve aircraft crossing the same bins all window. Every one of them
        drifts in delay at the rate its own Doppler mandates, which is what
        separates them from a ridge that occupies the same bins as often."""
        rng = random.Random(23)
        starts = [(10.0 + 4 * k, 90.0 - 15 * k) for k in range(12)]
        frames = []
        for i in range(self.FRAMES):
            t = i * FRAME_MS
            frames.append([aircraft(d0, f0 + 0.05 * i, t, jitter=0.022, rng=rng) for d0, f0 in starts])
        assert feed(make_map(window_s=60.0), frames).interfering_bins() == frozenset()


class TestWhenItIsAllowedToJudge:
    def test_nothing_is_judged_until_the_window_is_full(self):
        rng = random.Random(8)
        occupancy = feed(make_map(), [tone_frame(27.9, rng) for _ in range(WINDOW_FRAMES - 1)])
        assert occupancy.interfering_bins() == frozenset()

    def test_the_verdict_is_dropped_once_the_tone_stops(self):
        """A site's interference is not permanent, and a map that only ever
        accumulates would go on refusing a bin long after it went quiet."""
        rng = random.Random(9)
        occupancy = feed(make_map(), [tone_frame(27.9, rng) for _ in range(WINDOW_FRAMES)])
        assert occupancy.is_interfering(27.9)
        feed(occupancy, [[det(30.0, -120.0)] for _ in range(WINDOW_FRAMES)], start_ms=WINDOW_FRAMES * FRAME_MS)
        assert not occupancy.is_interfering(27.9)

    def test_the_window_holds_only_its_own_frames(self):
        rng = random.Random(10)
        occupancy = feed(make_map(), [tone_frame(27.9, rng) for _ in range(WINDOW_FRAMES * 3)])
        assert len(occupancy._frames) == WINDOW_FRAMES

    def test_a_clock_resync_starts_the_window_again(self):
        """The window would otherwise straddle two time bases, and the drift
        the map removes is a rate times an elapsed time that no longer means
        anything across the jump."""
        set_config(build_config())
        tracker = Tracker(config=None)
        rng = random.Random(14)
        for i in range(WINDOW_FRAMES * 2):
            tracker.process_frame(tone_frame(27.9, rng), i * FRAME_MS)
        assert tracker.occupancy.interfering_bins()

        for i in range(BACKWARDS_RUN_BEFORE_RESYNC):
            tracker.process_frame(tone_frame(27.9, rng), i * FRAME_MS)

        assert tracker.n_clock_resyncs == 1
        assert tracker.occupancy.interfering_bins() == frozenset()

    def test_a_malformed_detection_is_skipped_rather_than_raising(self):
        occupancy = make_map()
        occupancy.observe([{"snr": 12.0}, det(float("nan"), 30.0), det(10.0, None)], 0)
        assert occupancy.is_interfering(float("nan")) is False


class TestSuppressionAtInitiation:
    def _tone_tracker(self, frames, **interference):
        set_config(build_config(**interference))
        tracker = Tracker(config=None)
        for i, detections in enumerate(frames):
            tracker.process_frame(detections, i * FRAME_MS)
        return tracker

    def _tone_frames(self, n, seed=11):
        rng = random.Random(seed)
        return [tone_frame(27.9, rng) for _ in range(n)]

    def test_the_tone_stops_starting_tracks_once_it_is_learned(self):
        tracker = self._tone_tracker(self._tone_frames(WINDOW_FRAMES * 3), suppress=True)
        assert tracker.n_initiations_suppressed > 0

    def test_it_is_off_unless_asked_for(self):
        tracker = self._tone_tracker(self._tone_frames(WINDOW_FRAMES * 3))
        assert tracker.n_initiations_suppressed == 0

    def test_marking_happens_even_when_suppression_does_not(self):
        """So a node can record what suppression would have refused, next to
        the ADS-B labels that say whether refusing it would have cost an
        aircraft, without suppressing anything anywhere first."""
        frames = self._tone_frames(WINDOW_FRAMES * 3)
        assert self._tone_tracker(frames).n_initiations_suppressed == 0
        assert any(d["interfering"] for d in frames[-1])

    def test_an_established_track_still_associates_inside_the_tone(self):
        """Exactly as one does crossing blah2's own notch. Suppression is
        refusing the interferer its own tracks, not refusing its detections."""
        rng = random.Random(12)
        n_frames = WINDOW_FRAMES * 3
        frames = []
        for i in range(n_frames):
            frame = tone_frame(27.9, rng)
            frame.append(aircraft(30.0, 27.9, i * FRAME_MS, jitter=0.02, rng=rng))
            frames.append(frame)
        set_config(build_config(suppress=True))
        tracker = Tracker(config=None)
        for i, detections in enumerate(frames):
            tracker.process_frame(detections, i * FRAME_MS)

        final = aircraft(30.0, 27.9, (n_frames - 1) * FRAME_MS)["delay"]
        followed = [
            t
            for t in tracker.get_confirmed_tracks()
            if t.n_associated > WINDOW_FRAMES and abs(t.history["measurements"][-1]["delay"] - final) < 0.2
        ]
        assert followed, "the aircraft in the tone's own bin lost its track"

    def test_a_suppressed_detection_is_still_accounted_for(self):
        """The record says what became of every detection. One refused a track
        is unassociated, which is what it is, not absent."""

        class Sink:
            def __init__(self):
                self.unassociated = 0

            def write_detections(self, timestamp, associated, unassociated, below_snr):
                self.unassociated += len(unassociated)

        sink = Sink()
        set_config(build_config(suppress=True))
        tracker = Tracker(config=None, detection_sink=sink)
        rng = random.Random(13)
        for i in range(WINDOW_FRAMES * 4):
            tracker.process_frame(tone_frame(27.9, rng), i * FRAME_MS)
        assert tracker.n_initiations_suppressed > 0
        assert sink.unassociated >= tracker.n_initiations_suppressed

    def test_a_reset_forgets_the_map(self):
        tracker = self._tone_tracker(self._tone_frames(WINDOW_FRAMES * 2), suppress=True)
        tracker.reset()
        assert tracker.occupancy.interfering_bins() == frozenset()


class TestTheGuardrailIsReported:
    """A track carrying an ADS-B hex is an aircraft whatever the map thinks, so
    a labelled track reading above zero is the map reaching for something it
    must not have. Reported whether or not suppression is on, so the question
    can be settled from a recording made before it is turned on anywhere."""

    def _track(self, flags):
        config = load_config("nonexistent.yaml")
        tracker = Tracker(config=config)
        first = det(11.0, 27.9)
        first["interfering"] = flags[0]
        track = Track(first, 0, tracker.kf, config=config)
        for i, flag in enumerate(flags[1:], start=1):
            d = det(11.0 + 0.02 * i, 27.9)
            d["interfering"] = flag
            track.update(d, i * FRAME_MS, frame=i)
        return track

    def test_the_fraction_is_exposed_on_the_track(self):
        assert self._track([True, True, False, False]).interference_fraction() == pytest.approx(0.5)

    def test_a_clean_track_reads_zero(self):
        assert self._track([False] * 5).interference_fraction() == 0.0

    def test_it_reaches_the_serialised_track(self):
        assert self._track([True, False, False, False]).to_dict()["interference_fraction"] == pytest.approx(0.25)

    def test_it_reaches_the_event_stream(self, tmp_path):
        from retina_tracker.output import TrackEventWriter

        path = tmp_path / "events.jsonl"
        writer = TrackEventWriter(str(path))
        writer.write_event("t1", 0, 3, [], interference_fraction=0.76)
        writer.close()
        assert json.loads(path.read_text())["interference_fraction"] == pytest.approx(0.76)


class TestTheCellsComeFromTheCapture:
    """Rule one of the design: a number the tracker depends on is derived from
    blah2's capture config, not configured a second time next to it."""

    def test_the_delay_cell_is_the_sample_rate(self):
        set_config(build_config())
        assert DELAY_CELL_KM() == pytest.approx(0.14990, abs=1e-5)

    def test_the_doppler_bin_is_the_cpi(self):
        set_config(build_config())
        assert DOPPLER_BIN_HZ() == pytest.approx(2.0)

    def test_a_differently_captured_node_gets_different_cells(self):
        config = build_config()
        config["radar"].update({"sample_rate": 4000000, "cpi": 1.0})
        set_config(config)
        assert DELAY_CELL_KM() == pytest.approx(0.07495, abs=1e-5)
        assert DOPPLER_BIN_HZ() == pytest.approx(1.0)

    def test_they_are_read_from_blah2s_own_config(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text("capture:\n  fc: 177000000\n  fs: 2000000\nprocess:\n  data:\n    cpi: 0.5\n")
        assert load_blah2_config(str(path)) == {"fc": 177000000, "fs": 2000000, "cpi": 0.5}

    def test_a_missing_blah2_config_is_not_fatal(self, tmp_path):
        assert load_blah2_config(str(tmp_path / "absent.yml")) == {}
