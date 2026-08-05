"""Tests that confirmed tracks stay reportable after they age out of the
merge window.

The merge working set (`all_tracks`) is bounded to MERGE_WINDOW_MS so
`_merge_tracks` stays O(window²), but aged entries are archived into
`completed_tracks` rather than discarded — otherwise batch output
(`to_dict`, `visualize_tracks`) would only ever show the last few seconds
of a capture. The archive itself is bounded (MAX_COMPLETED_TRACKS) so a
long-running tracker cannot grow without limit.
"""

from retina_tracker.config import get_config
from retina_tracker.tracker import MAX_COMPLETED_TRACKS, MERGE_WINDOW_MS, Tracker


def make_detections(delay, doppler, snr=20.0):
    return [{"delay": delay, "doppler": doppler, "snr": snr}]


def run_track(tracker, start_ts, n_frames, delay, doppler, step=0.0):
    ts = start_ts
    for i in range(n_frames):
        tracker.process_frame(make_detections(delay + i * step, doppler), ts)
        ts += 500
    return ts


def idle(tracker, start_ts, duration_ms):
    ts = start_ts
    while ts < start_ts + duration_ms:
        tracker.process_frame([], ts)
        ts += 500
    return ts


def test_confirmed_track_survives_aging_out_of_merge_window():
    tracker = Tracker(config=get_config())

    ts = run_track(tracker, 0, 12, 10.0, 50.0, step=0.2)
    confirmed_while_live = len(tracker.get_confirmed_tracks())
    assert confirmed_while_live > 0

    idle(tracker, ts, MERGE_WINDOW_MS * 4)

    assert tracker.all_tracks == []
    assert len(tracker.completed_tracks) > 0
    assert len(tracker.get_confirmed_tracks()) >= confirmed_while_live


def test_archived_tracks_accumulate_across_many_windows():
    tracker = Tracker(config=get_config())

    ts = 0
    for n in range(3):
        ts = run_track(tracker, ts, 12, 10.0 + n * 40.0, 50.0 + n * 100.0, step=0.2)
        ts = idle(tracker, ts, MERGE_WINDOW_MS * 3)

    assert len(tracker.get_confirmed_tracks()) >= 3


def test_no_double_counting_between_archive_and_merge_window():
    tracker = Tracker(config=get_config())

    ts = run_track(tracker, 0, 12, 10.0, 50.0, step=0.2)
    idle(tracker, ts, MERGE_WINDOW_MS * 4)

    confirmed = tracker.get_confirmed_tracks()
    assert len(confirmed) == len({id(t) for t in confirmed})


def test_archive_is_bounded():
    tracker = Tracker(config=get_config())

    assert tracker.completed_tracks.maxlen == MAX_COMPLETED_TRACKS

    tracker.completed_tracks.extend(object() for _ in range(MAX_COMPLETED_TRACKS + 100))
    assert len(tracker.completed_tracks) == MAX_COMPLETED_TRACKS


def test_reset_clears_the_archive():
    tracker = Tracker(config=get_config())

    ts = run_track(tracker, 0, 12, 10.0, 50.0, step=0.2)
    idle(tracker, ts, MERGE_WINDOW_MS * 4)
    assert len(tracker.completed_tracks) > 0

    tracker.reset()

    assert len(tracker.completed_tracks) == 0
    assert tracker.get_confirmed_tracks() == []
