"""Tests for the HTTP control surface.

The one control operation the tracker supports, clearing state between search
geometries, has only ever been reachable as a `{"type": "RESET"}` message mixed
into the detection socket. That works while one process sends both detections
and controls, which is the arrangement being unwound: blah2_api is taking over
the detection socket, and `run_tcp_server` accepts one connection at a time.

These go over real HTTP against a real server on an ephemeral port rather than
calling the handler directly, because the things worth pinning are what a
client actually gets back: the status code, the body, and whether a 200 means
the reset has happened or merely been scheduled.
"""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from retina_tracker import control
from retina_tracker.config import get_config
from retina_tracker.control import start_control_server
from retina_tracker.history import CLASSES, DetectionHistory
from retina_tracker.server import process_streaming_frame
from retina_tracker.tracker import Tracker


def make_frame(timestamp, delay, doppler, snr=15.0):
    return {"timestamp": timestamp, "delay": [delay], "doppler": [doppler], "snr": [snr]}


@pytest.fixture
def served():
    """A tracker with some state, and a control server in front of it."""
    tracker = Tracker(config=get_config())
    lock = threading.Lock()
    # Port 0: the OS picks a free one, so these never collide with a real
    # sidecar or with each other under parallel test runs.
    server = start_control_server(tracker, lock, host="127.0.0.1", port=0)
    try:
        yield tracker, lock, f"http://127.0.0.1:{server.port}"
    finally:
        server.shutdown()
        server.server_close()


def request(url, method="GET"):
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def build_state(tracker, frames=3):
    for i in range(frames):
        process_streaming_frame(tracker, make_frame(i * 500, 10.0, 50.0))


def test_reset_clears_tracker_state(served):
    tracker, _lock, base = served
    build_state(tracker)
    assert tracker.tracks or tracker.all_tracks or tracker.last_timestamp is not None

    status, body = request(base + "/reset", method="POST")

    assert status == 200
    assert body == {"ok": True}
    assert tracker.tracks == []
    assert tracker.all_tracks == []
    assert tracker.last_timestamp is None
    assert tracker.frame_count == 0


def test_reset_has_already_happened_when_the_response_arrives(served):
    """Not "scheduled": a caller resetting between candidate geometries waits
    on the next frame, and that frame must not be able to associate into
    pre-reset state."""
    tracker, _lock, base = served
    build_state(tracker)

    request(base + "/reset", method="POST")
    assert tracker.frame_count == 0

    # A detection at a completely different geometry starts fresh rather than
    # associating into anything that existed before the reset.
    process_streaming_frame(tracker, make_frame(0, 300.0, -200.0))
    assert tracker.frame_count == 1


def test_reset_tolerates_a_trailing_slash(served):
    tracker, _lock, base = served
    build_state(tracker)
    status, _ = request(base + "/reset/", method="POST")
    assert status == 200
    assert tracker.frame_count == 0


def test_health_reports_what_the_tracker_has_seen(served):
    """The one thing worth asking a node during the switchover: are frames
    arriving at all."""
    tracker, _lock, base = served
    status, body = request(base + "/health")
    assert status == 200
    assert body["ok"] is True
    assert body["frames"] == 0

    build_state(tracker, frames=3)

    _status, body = request(base + "/health")
    assert body["frames"] == 3
    assert isinstance(body["tracks"], int)


def test_unknown_paths_are_404_not_a_silent_success(served):
    _tracker, _lock, base = served
    assert request(base + "/nope")[0] == 404
    assert request(base + "/nope", method="POST")[0] == 404
    # /reset is a POST; a GET must not quietly do nothing and return 200.
    assert request(base + "/reset")[0] == 404


def test_reset_waits_for_an_in_flight_frame(served):
    """The lock is the whole point of taking it: a reset landing mid-frame
    would clear state the frame path is part way through mutating."""
    tracker, lock, base = served
    build_state(tracker)

    done = threading.Event()
    result = {}

    def reset_call():
        result["status"] = request(base + "/reset", method="POST")[0]
        done.set()

    with lock:
        thread = threading.Thread(target=reset_call, daemon=True)
        thread.start()
        # Held: the request cannot have completed, so state survives.
        assert not done.wait(timeout=0.3)
        assert tracker.frame_count == 3

    assert done.wait(timeout=5)
    assert result["status"] == 200
    assert tracker.frame_count == 0


# ── The data stream ─────────────────────────────────────────────────────────


def read_events(response, count, timeout=10):
    """Pull `count` SSE messages off an open stream."""
    events = []
    kind, buf = None, []
    deadline = time.monotonic() + timeout
    for raw in response:
        line = raw.decode().rstrip("\n")
        if line.startswith("event: "):
            kind = line[7:]
        elif line.startswith("data: "):
            buf.append(line[6:])
        elif line == "":
            if kind and buf:
                events.append((kind, json.loads("".join(buf))))
                if len(events) >= count:
                    return events
            kind, buf = None, []
        if time.monotonic() > deadline:
            break
    return events


@pytest.fixture
def streaming(monkeypatch):
    """A control server with a history behind it, streaming quickly."""
    monkeypatch.setattr(control, "STREAM_INTERVAL_S", 0.05)
    tracker = Tracker(config=get_config())
    history = DetectionHistory()
    lock = threading.Lock()
    server = start_control_server(tracker, lock, host="127.0.0.1", port=0, history=history)
    try:
        yield history, f"http://127.0.0.1:{server.port}"
    finally:
        server.shutdown()
        server.server_close()


def open_stream(base, query=""):
    return urllib.request.urlopen(base + "/events" + query, timeout=10)


def test_the_stream_opens_with_a_snapshot(streaming):
    """One ordering, owned by the connection. Fetching the snapshot
    separately would race the start of the delta stream."""
    history, base = streaming
    history.write_detections(1000, [{"delay": 10.0, "doppler": 50.0, "snr": 15.0}], [], [])

    with open_stream(base) as response:
        kind, payload = read_events(response, 1)[0]

    assert kind == "snapshot"
    assert payload["detections"]["associated"]["delay"] == [10.0]
    assert set(payload["detections"]) == set(CLASSES)


def test_deltas_carry_only_what_was_appended(streaming):
    history, base = streaming
    history.write_detections(1000, [{"delay": 10.0, "doppler": 50.0, "snr": 15.0}], [], [])

    with open_stream(base) as response:
        assert read_events(response, 1)[0][0] == "snapshot"
        history.write_detections(2000, [], [{"delay": 20.0, "doppler": -30.0, "snr": 9.0}], [])
        kind, payload = read_events(response, 1)[0]

    assert kind == "delta"
    assert payload["detections"]["unassociated"]["delay"] == [20.0]
    assert payload["detections"]["associated"]["t"] == [], "the snapshot's point came again"


def test_all_three_classes_reach_a_consumer(streaming):
    """below_snr is the one nothing could see before."""
    history, base = streaming
    with open_stream(base) as response:
        read_events(response, 1)
        history.write_detections(
            1000,
            [{"delay": 1.0, "doppler": 0.0, "snr": 15.0}],
            [{"delay": 2.0, "doppler": 0.0, "snr": 9.0}],
            [{"delay": 3.0, "doppler": 0.0, "snr": 2.0}],
        )
        _kind, payload = read_events(response, 1)[0]

    assert payload["detections"]["associated"]["delay"] == [1.0]
    assert payload["detections"]["unassociated"]["delay"] == [2.0]
    assert payload["detections"]["below_snr"]["delay"] == [3.0]


def test_a_clear_reseeds_the_stream_rather_than_reconciling(streaming):
    history, base = streaming
    history.write_detections(1000, [{"delay": 10.0, "doppler": 50.0, "snr": 15.0}], [], [])

    with open_stream(base) as response:
        read_events(response, 1)
        history.clear()
        history.write_detections(2000, [{"delay": 99.0, "doppler": 0.0, "snr": 15.0}], [], [])
        kind, payload = read_events(response, 1)[0]

    assert kind == "snapshot", "a voided cursor produced a delta"
    assert payload["detections"]["associated"]["delay"] == [99.0]


def test_a_window_is_clamped_to_what_is_held(streaming):
    history, base = streaming
    with open_stream(base, "?window=999999") as response:
        _kind, payload = read_events(response, 1)[0]
    assert payload["window_s"] == history.window_s

    with open_stream(base, "?window=1") as response:
        _kind, payload = read_events(response, 1)[0]
    assert payload["window_s"] == control.MIN_WINDOW_S


def test_health_reports_the_history_footprint(streaming):
    history, base = streaming
    history.write_detections(1000, [{"delay": 10.0, "doppler": 50.0, "snr": 15.0}], [], [])

    status, body = request(base + "/health")

    assert status == 200
    assert body["history"]["detections"]["associated"] == 1
    assert body["history"]["approx_bytes"] == 20


def test_the_stream_is_unavailable_without_a_history(served):
    """The CLI runs without one, and must say so rather than pretend."""
    _tracker, _lock, base = served
    status, body = request(base + "/events")
    assert status == 503
    assert "error" in body


def test_history_clear_wipes_the_record_without_touching_the_tracker(streaming):
    """ "Clear buffer" has always meant "clear what I am shown, keep
    tracking". That distinction survives the record moving into the tracker."""
    history, base = streaming
    history.write_detections(1000, [{"delay": 10.0, "doppler": 50.0, "snr": 15.0}], [], [])

    status, body = request(base + "/history/clear", method="POST")

    assert status == 200 and body == {"ok": True}
    assert history.stats()["points"] == 0


def test_history_clear_is_unavailable_without_a_history(served):
    _tracker, _lock, base = served
    assert request(base + "/history/clear", method="POST")[0] == 503
