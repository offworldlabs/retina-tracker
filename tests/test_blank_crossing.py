"""Tests for targets crossing blah2's zero-Doppler blank.

blah2's detector never reports within process.detection.minDoppler of zero, so a
target crossing zero Doppler goes unseen for as long as the crossing takes. The
filter does not try to carry a track across: the crossing ends one track and the
far side starts another, and joining them is stitching's job. See
docs/zero-doppler-blank.md.
"""

from retina_tracker.config import set_config
from retina_tracker.kalman import doppler_to_range_rate
from retina_tracker.track import TrackState
from retina_tracker.tracker import Tracker

FRAME_MS = 500
BLANK_HZ = 15.0


def make_tracker():
    config = {
        "tracker": {
            "m_threshold": 3,
            "n_window": 6,
            "n_delete": 10,
            "n_coast": 3,
            "gate_threshold": 9.0,
            "detection_window": 20,
        },
        "process_noise": {"range_jerk": 1e-6, "adaptive": True, "max_scale": 25.0, "nis_memory": 0.25},
        "tracklet": {"max_delay_residual": 2.0, "max_doppler_residual": 10.0, "max_time_span": 3.0},
        "shadow": {"enabled": True, "delay_km": 3.0, "snr_margin_db": 3.0, "min_fraction": 0.5},
        "interference": {"enabled": False, "suppress": False},
        "adsb": {"enabled": False, "priority": True, "reference_location": None},
        "radar": {"center_frequency": 200000000, "sample_rate": 2000000, "cpi": 0.5},
    }
    set_config(config)
    return Tracker(config=config)


def crossing(tracker, slew_hz_s=-4.0, start_hz=60.0, duration_s=30.0):
    """Fly a target through zero Doppler, detected only outside the blank.

    Returns the ids of the confirmed tracks holding it on each side of the blank.
    """
    delay = 20.0
    before, after = set(), set()
    for frame in range(int(duration_s * 1000 / FRAME_MS)):
        doppler = start_hz + slew_hz_s * frame * FRAME_MS / 1000.0
        delay += doppler_to_range_rate(doppler) * FRAME_MS / 1000.0
        visible = abs(doppler) >= BLANK_HZ
        tracker.process_frame([{"delay": delay, "doppler": doppler, "snr": 20.0}] if visible else [], frame * FRAME_MS)
        side = before if doppler > 0 else after
        side.update(track.id for track in tracker.tracks if track.id and track.n_missed == 0 and visible)
    return before, after


def test_a_crossing_ends_one_track_and_starts_another():
    before, after = crossing(make_tracker())
    assert len(before) == 1 and len(after) == 1
    assert after.isdisjoint(before)


def test_a_track_lost_in_the_blank_is_deleted_like_any_other():
    tracker = make_tracker()
    step = doppler_to_range_rate(20.0) * FRAME_MS / 1000.0
    for frame in range(10):
        tracker.process_frame([{"delay": 20.0 + frame * step, "doppler": 20.0, "snr": 20.0}], frame * FRAME_MS)
    track = tracker.tracks[0]
    assert track.state_status == TrackState.ACTIVE
    for frame in range(10, 21):
        tracker.process_frame([], frame * FRAME_MS)
    assert track not in tracker.tracks
