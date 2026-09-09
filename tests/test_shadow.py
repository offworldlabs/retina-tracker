"""Suppressing multipath and sidelobe replicas of strong targets.

A large aircraft casts weaker copies of itself at longer bistatic range. They
move with it, so they obey range_rate = -wavelength * doppler exactly and the
filter cannot tell them from real targets. What gives them away is a brighter
detection sitting a short way in front of them in the same frame.

Measured on a live node over 93 same-frame pairs: 89% of the spurious detections
around an aircraft sat at longer delay than it, 99% were weaker, median 6.9 dB
down. Across whole tracks the separation is wider still - ADS-B-matched aircraft
never exceeded a shadowed fraction of 0.10, while replica tracks ran a median of
0.64 - which is what the 0.5 promotion threshold sits between.

Doppler is deliberately not compared. The replica travels a different path, so
its Doppler differs: a median of 35 Hz even for copies within 1.5 km.
"""

import json

import pytest

from retina_tracker.config import get_config, load_config, set_config
from retina_tracker.track import Track
from retina_tracker.tracker import Tracker


def build_config(**shadow):
    config = load_config("nonexistent.yaml")
    config["adsb"]["enabled"] = False
    config["shadow"].update(shadow)
    return config


def det(delay, doppler, snr):
    return {"delay": delay, "doppler": doppler, "snr": snr}


@pytest.fixture(autouse=True)
def _config():
    set_config(build_config())
    yield


def marked(detections):
    Tracker._mark_shadows(detections)
    return [d["shadowed"] for d in detections]


class TestShadowMarking:
    def test_a_weaker_return_behind_a_brighter_one_is_shadowed(self):
        bright = det(10.0, -50.0, 18.0)
        replica = det(11.0, -20.0, 11.0)
        assert marked([bright, replica]) == [False, True]

    def test_the_brighter_return_is_never_shadowed_by_its_own_replica(self):
        assert marked([det(10.0, -50.0, 18.0), det(11.0, -20.0, 11.0)])[0] is False

    def test_a_return_in_front_of_the_bright_one_is_not_shadowed(self):
        """Replicas are delayed. Something at shorter range did not come from it."""
        assert marked([det(10.0, -50.0, 18.0), det(9.0, -20.0, 11.0)]) == [False, False]

    def test_beyond_the_delay_window_nothing_is_shadowed(self):
        set_config(build_config(delay_km=1.0))
        assert marked([det(10.0, -50.0, 18.0), det(11.5, -20.0, 11.0)]) == [False, False]

    def test_a_comparably_bright_return_does_not_shadow(self):
        assert marked([det(10.0, -50.0, 12.0), det(11.0, -20.0, 11.0)]) == [False, False]

    def test_doppler_is_not_compared(self):
        """The replica takes a different path, so its Doppler genuinely differs."""
        assert marked([det(10.0, 150.0, 18.0), det(11.0, -150.0, 11.0)])[1] is True

    def test_disabling_it_marks_everything_unshadowed(self):
        set_config(build_config(enabled=False))
        assert marked([det(10.0, -50.0, 18.0), det(11.0, -20.0, 11.0)]) == [False, False]


class TestShadowedTracksDoNotPromote:
    def _track(self, shadowed_flags):
        config = get_config()
        tracker = Tracker(config=config)
        first = det(11.0, -20.0, 11.0)
        first["shadowed"] = shadowed_flags[0]
        track = Track(first, 0, tracker.kf, config=config)
        for i, flag in enumerate(shadowed_flags[1:], start=1):
            d = det(11.0 + 0.02 * i, -20.0, 11.0)
            d["shadowed"] = flag
            track.update(d, i * 1000, frame=i)
        return track

    def test_a_consistently_shadowed_track_is_shadowed(self):
        assert self._track([True] * 5).is_shadowed()

    def test_a_never_shadowed_track_is_not(self):
        assert not self._track([False] * 5).is_shadowed()

    def test_below_the_fraction_it_is_not_blocked(self):
        assert not self._track([True, False, False, False, False]).is_shadowed()

    def test_too_few_observations_to_judge(self):
        """Two detections is not evidence of a persistent replica."""
        assert not self._track([True, True]).is_shadowed()

    def test_a_shadowed_track_does_not_promote(self):
        track = self._track([True] * 8)
        track.n_frames = 99
        assert track.promote_if_ready() is False

    def test_an_unshadowed_track_with_the_same_history_does_promote(self):
        track = self._track([False] * 8)
        track.n_frames = 99
        assert track.promote_if_ready() is True


class TestTheGuardrailIsReported:
    """The thresholds were fitted at one site. Reporting the fraction is what
    makes a site where they do not fit visible, rather than silently blocking
    real aircraft."""

    def test_the_fraction_is_exposed_on_the_track(self):
        track = TestShadowedTracksDoNotPromote()._track([True, True, False, False])
        assert track.shadow_fraction() == pytest.approx(0.5)

    def test_it_reaches_the_serialised_track(self):
        track = TestShadowedTracksDoNotPromote()._track([True, False, False, False])
        assert track.to_dict()["shadow_fraction"] == pytest.approx(0.25)

    def test_it_reaches_the_event_stream(self, tmp_path):
        from retina_tracker.output import TrackEventWriter

        path = tmp_path / "events.jsonl"
        writer = TrackEventWriter(str(path))
        writer.write_event("t1", 0, 3, [], shadow_fraction=0.64)
        writer.close()
        assert json.loads(path.read_text())["shadow_fraction"] == pytest.approx(0.64)
