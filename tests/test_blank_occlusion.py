"""Tests for holding confirmed tracks through blah2's zero-Doppler blank.

blah2's detector never reports within process.detection.minDoppler of zero, so
a target crossing zero Doppler is invisible for as long as the crossing takes:
seconds near the baseline, a minute or more further out. Those misses say
nothing about whether the target is still there, so they must not count
towards deletion. See docs/zero-doppler-blank.md.
"""

from retina_tracker.config import load_blah2_config, set_config
from retina_tracker.kalman import doppler_to_range_rate
from retina_tracker.track import MAX_OCCLUDED_S, TrackState
from retina_tracker.tracker import Tracker

FRAME_MS = 500
BLANK_HZ = 15.0


def build_config(doppler_blank=BLANK_HZ):
    return {
        "tracker": {
            "m_threshold": 3,
            "n_window": 6,
            "n_delete": 10,
            "n_coast": 3,
            "min_snr": 4.0,
            "gate_threshold": 9.0,
            "detection_window": 20,
        },
        "process_noise": {"range_jerk": 1e-7, "adaptive": True, "max_scale": 25.0, "nis_memory": 0.25},
        "tracklet": {"max_delay_residual": 2.0, "max_doppler_residual": 10.0, "max_time_span": 3.0},
        "shadow": {"enabled": True, "delay_km": 3.0, "snr_margin_db": 3.0, "min_fraction": 0.5},
        "interference": {"enabled": False, "suppress": False},
        "adsb": {"enabled": False, "priority": True, "reference_location": None},
        "radar": {"center_frequency": 200000000, "sample_rate": 2000000, "cpi": 0.5, "doppler_blank": doppler_blank},
    }


def make_tracker(doppler_blank=BLANK_HZ):
    config = build_config(doppler_blank)
    set_config(config)
    return Tracker(config=config)


def crossing(tracker, slew_hz_s=-4.0, start_hz=60.0, duration_s=30.0):
    """Fly a target through zero Doppler, detected only outside the blank.

    Returns the ids of the confirmed tracks holding it before and after the blank.
    """
    delay = 20.0
    before, after = set(), set()
    for frame in range(int(duration_s * 1000 / FRAME_MS)):
        t = frame * FRAME_MS / 1000.0
        doppler = start_hz + slew_hz_s * t
        delay += doppler_to_range_rate(doppler) * FRAME_MS / 1000.0
        visible = abs(doppler) >= BLANK_HZ
        tracker.process_frame([{"delay": delay, "doppler": doppler, "snr": 20.0}] if visible else [], frame * FRAME_MS)
        side = before if doppler > 0 else after
        side.update(track.id for track in tracker.tracks if track.id and track.n_missed == 0 and visible)
    return before, after


def test_track_keeps_its_identity_through_the_blank():
    before, after = crossing(make_tracker())
    assert len(before) == 1
    assert after == before


def test_without_a_known_blank_the_crossing_breaks_the_track():
    before, after = crossing(make_tracker(doppler_blank=None))
    assert len(before) == 1
    assert after and after.isdisjoint(before)


def test_blanked_misses_do_not_count_towards_deletion():
    tracker = make_tracker()
    crossing(tracker, duration_s=17.0)
    track = tracker.tracks[0]
    assert track.n_missed == 0
    assert track.occluded_s > tracker.config["tracker"]["n_delete"] * FRAME_MS / 1000.0


def confirmed_track(tracker, doppler, frames=8):
    step = doppler_to_range_rate(doppler) * FRAME_MS / 1000.0
    for frame in range(frames):
        tracker.process_frame([{"delay": 20.0 + frame * step, "doppler": doppler, "snr": 20.0}], frame * FRAME_MS)
    track = tracker.tracks[0]
    assert track.state_status == TrackState.ACTIVE
    return track, frames


def miss(tracker, start, n):
    for i in range(n):
        tracker.process_frame([], (start + i) * FRAME_MS)
    return start + n


def test_misses_far_from_the_blank_still_delete():
    tracker = make_tracker()
    track, frame = confirmed_track(tracker, doppler=150.0)
    miss(tracker, frame, 11)
    assert track not in tracker.tracks
    assert track.occluded_s == 0.0


def test_a_track_is_held_no_longer_than_it_was_observed():
    """A short tracklet held for long enough wanders onto unrelated detections."""
    tracker = make_tracker()
    track, frame = confirmed_track(tracker, doppler=10.0, frames=20)
    observed_s = track.max_occluded_s()
    assert observed_s == (19 * FRAME_MS) / 1000.0
    frame = miss(tracker, frame, int(observed_s * 1000 / FRAME_MS))
    assert track in tracker.tracks
    miss(tracker, frame, 2)
    assert track not in tracker.tracks


def test_a_long_track_is_held_no_longer_than_the_cap():
    tracker = make_tracker()
    track, frame = confirmed_track(tracker, doppler=10.0, frames=int(2 * MAX_OCCLUDED_S * 1000 / FRAME_MS))
    assert track.max_occluded_s() == MAX_OCCLUDED_S
    frame = miss(tracker, frame, int(MAX_OCCLUDED_S * 1000 / FRAME_MS))
    assert track in tracker.tracks
    miss(tracker, frame, 2)
    assert track not in tracker.tracks


def test_tentative_tracks_are_never_held():
    tracker = make_tracker()
    tracker.process_frame([{"delay": 20.0, "doppler": 16.0, "snr": 20.0}], 0)
    for frame in range(1, 8):
        tracker.process_frame([], frame * FRAME_MS)
    assert tracker.tracks == []


def test_blank_is_read_from_blah2_detection_config(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("capture:\n  fc: 204640000\nprocess:\n  detection:\n    minDoppler: 15\n")
    assert load_blah2_config(str(path))["minDoppler"] == 15
