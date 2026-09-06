"""Off-thread readers of `Track.history` must not see a mutating deque.

The five history buffers are `deque(maxlen=TRACK_HISTORY_MAX)`.  Only the frame
worker appends to a given node's tracker, but several readers run on other
threads (the aircraft-feed flush executor, the analytics executor, the admin
state-snapshot routes).  A deque raises `RuntimeError: deque mutated during
iteration` if it is appended to while a Python-level loop walks it, which a
plain list tolerated -- so `get_recent_detections` and `to_dict` must snapshot
each buffer with `list()` (one C call, atomic under the GIL) before iterating.
"""

import threading

from retina_tracker.config import get_config
from retina_tracker.tracker import Tracker


def make_detection(i):
    return {"delay": 10.0 + i * 0.01, "doppler": 50.0, "snr": 20.0}


def make_track():
    """A live track with a real Kalman filter, via the normal Tracker path."""
    tracker = Tracker(config=get_config())
    ts = 0
    for i in range(20):
        tracker.process_frame([make_detection(i)], ts)
        ts += 1000
    return max(tracker.get_tracks(), key=lambda t: t.n_associated), ts


def test_readers_survive_concurrent_appends():
    track, start_ts = make_track()

    errors = []
    stop = threading.Event()

    def appender():
        ts = start_ts
        i = track.n_frames
        try:
            while not stop.is_set():
                ts += 1000
                i += 1
                if i % 5 == 0:
                    track.mark_missed(ts, frame=i)
                else:
                    track.update(make_detection(i), ts, frame=i)
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(("appender", exc))

    writer = threading.Thread(target=appender, daemon=True)
    writer.start()
    try:
        deadline = threading.Event()
        timer = threading.Timer(0.5, deadline.set)
        timer.start()
        reads = 0
        try:
            while not deadline.is_set():
                for _ in range(50):
                    recent = track.get_recent_detections(n=5)
                    for det in recent:
                        assert isinstance(det, dict)
                        assert det["timestamp"] is not None
                        assert det["delay"] is not None
                        assert det["doppler"] is not None
                        assert det["snr"] is not None
                    dumped = track.to_dict()
                    history = dumped["history"]
                    n = len(history["timestamps"])
                    assert len(history["states"]) == n
                    assert len(history["delays"]) == n
                    assert len(history["dopplers"]) == n
                    assert len(history["snrs"]) == n
                    assert len(history["state_status"]) == n
                    reads += 1
        finally:
            timer.cancel()
    finally:
        stop.set()
        writer.join(timeout=2.0)

    assert not errors, errors
    assert reads > 100, f"stress loop did too little work to be meaningful ({reads} reads)"
    assert track.n_frames > 100, f"appender did too little work to be meaningful ({track.n_frames} frames)"
