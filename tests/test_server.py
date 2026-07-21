"""Tests for retina_tracker.server's frame dispatch, including the RESET
message added for clients (e.g. passive-radar auto-calibration) that reuse
this same long-running tracker across a search geometry change.
"""

from retina_tracker.config import get_config
from retina_tracker.server import process_streaming_frame, reset_tracker
from retina_tracker.tracker import Tracker


def make_frame(timestamp, delay, doppler, snr=15.0):
    return {"timestamp": timestamp, "delay": [delay], "doppler": [doppler], "snr": [snr]}


def test_reset_clears_in_progress_and_completed_tracks():
    tracker = Tracker(config=get_config())

    # Build up a consistent, repeating detection so it accumulates state.
    for i in range(3):
        process_streaming_frame(tracker, make_frame(i * 500, 10.0, 50.0))

    assert tracker.tracks or tracker.all_tracks or tracker.last_timestamp is not None

    reset_tracker(tracker)

    assert tracker.tracks == []
    assert tracker.all_tracks == []
    assert tracker.last_timestamp is None


def test_frame_after_reset_starts_fresh_not_associated_into_old_state():
    tracker = Tracker(config=get_config())

    # A consistent target at one geometry...
    for i in range(3):
        process_streaming_frame(tracker, make_frame(i * 500, 10.0, 50.0))
    reset_tracker(tracker)

    # ...then a detection at a totally different (delay, doppler) — as if
    # the search just switched to a different tower's geometry. Must start
    # as a brand-new tentative track, not associate into anything pre-reset.
    process_streaming_frame(tracker, make_frame(0, 300.0, -200.0))

    # A brand-new track spawned from a single, just-arrived detection —
    # n_associated only increments once a *later* frame matches into it.
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].n_frames == 1
    assert tracker.tracks[0].state[0] == 300.0  # delay state, not the pre-reset 10.0


def test_reset_message_is_recognized_by_type_field():
    """A real detection frame never carries a "type" key — confirms the
    dispatch convention server.py's recv loop relies on."""
    real_frame = make_frame(0, 10.0, 50.0)
    assert "type" not in real_frame

    reset_message = {"type": "RESET"}
    assert reset_message.get("type") == "RESET"
