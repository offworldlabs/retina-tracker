"""Tests that per-track history is a bounded ring buffer.

`Track.history` is appended to once per processed frame on BOTH the update
path and the coast path, so as plain lists it grew for the entire life of a
track — ~1.4 MB per track-hour at 1 Hz, and without limit for a loitering
aircraft or a persistent clutter return that coasts and re-associates
forever. The five buffers are now `deque(maxlen=TRACK_HISTORY_MAX)`.

Every reader wants only the tail (the tracklet fit reads the last 3
associated samples; `get_recent_detections` is called with n <=
`detection_window`), so the cap must not change what those readers see —
that is what the `get_recent_detections` cases below pin down.
"""

from retina_tracker.config import get_config
from retina_tracker.track import TRACK_HISTORY_MAX
from retina_tracker.tracker import MAX_COMPLETED_TRACKS, Tracker

HISTORY_KEYS = ("timestamps", "frames", "states", "measurements", "state_status")


def make_detections(delay, doppler, snr=20.0):
    return [{"delay": delay, "doppler": doppler, "snr": snr}]


def run_frames(tracker, n_frames, start_ts=0, delay=10.0, doppler=50.0, step=0.05, dt_ms=1000):
    """Feed one steadily-drifting target for n_frames, returning the last ts."""
    ts = start_ts
    for i in range(n_frames):
        tracker.process_frame(make_detections(delay + i * step, doppler), ts)
        ts += dt_ms
    return ts - dt_ms


def test_history_stays_bounded_over_5000_frames():
    tracker = Tracker(config=get_config())
    run_frames(tracker, 5000)

    tracks = tracker.get_tracks()
    assert tracks, "the drifting target should still be tracked after 5000 frames"

    for track in tracks:
        assert track.n_frames > TRACK_HISTORY_MAX, "test target must outlive the cap to be meaningful"
        for key in HISTORY_KEYS:
            buf = track.history[key]
            assert buf.maxlen == TRACK_HISTORY_MAX
            assert len(buf) <= TRACK_HISTORY_MAX, key


def test_get_recent_detections_returns_newest_n_in_order():
    tracker = Tracker(config=get_config())
    last_ts = run_frames(tracker, 5000)

    track = max(tracker.get_tracks(), key=lambda t: t.n_associated)
    recent = track.get_recent_detections(n=20)

    assert len(recent) == 20
    # Oldest-first, contiguous, and ending on the frame just processed.
    timestamps = [d["timestamp"] for d in recent]
    assert timestamps == sorted(timestamps)
    assert timestamps[-1] == last_ts
    assert timestamps == list(range(last_ts - 19 * 1000, last_ts + 1000, 1000))
    assert all(d["delay"] is not None and d["snr"] is not None for d in recent)


def test_get_recent_detections_skips_coasted_frames():
    """mark_missed() appends a None measurement; the reverse scan must skip it."""
    tracker = Tracker(config=get_config())
    ts = run_frames(tracker, 30)

    track = max(tracker.get_tracks(), key=lambda t: t.n_associated)
    associated_before = [d["timestamp"] for d in track.get_recent_detections(n=5)]

    for _ in range(3):
        ts += 1000
        track.mark_missed(ts, frame=track.n_frames)

    assert None in list(track.history["measurements"])[-3:]
    assert [d["timestamp"] for d in track.get_recent_detections(n=5)] == associated_before


def test_history_cap_is_configurable():
    config = get_config()
    config["tracker"]["track_history_max"] = 25
    tracker = Tracker(config=config)
    run_frames(tracker, 200, step=0.05)

    for track in tracker.get_tracks():
        for key in HISTORY_KEYS:
            assert track.history[key].maxlen == 25
            assert len(track.history[key]) <= 25


def test_completed_track_archive_ceiling_is_small():
    """The archive holds whole Tracks; one Tracker exists per radar node."""
    assert MAX_COMPLETED_TRACKS <= 50

    tracker = Tracker(config=get_config())
    assert tracker.completed_tracks.maxlen == MAX_COMPLETED_TRACKS

    # ...and a consumer that wants a deeper archive can still ask for one.
    assert Tracker(config=get_config(), max_completed_tracks=500).completed_tracks.maxlen == 500


def test_archived_tracks_carry_bounded_history():
    """A retired track keeps its history, but only TRACK_HISTORY_MAX of it."""
    tracker = Tracker(config=get_config())
    ts = run_frames(tracker, 2000)

    # Idle past N_DELETE and past the merge window so the track retires.
    for _ in range(60):
        ts += 1000
        tracker.process_frame([], ts)

    assert len(tracker.completed_tracks) > 0
    for track in tracker.completed_tracks:
        assert 0 < len(track.history["timestamps"]) <= TRACK_HISTORY_MAX
        assert len(track.history["states"]) <= TRACK_HISTORY_MAX
