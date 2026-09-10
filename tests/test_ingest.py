"""Tests for the detection feed's connection handling.

Over real sockets rather than a fake, because every case here is about what
the socket layer does: a peer that goes silent without closing, a second peer
arriving while the first still holds the slot, a partial line stranded when
one is replaced. None of that is observable against a stub.

The case this exists for is the handover from retina-gui to blah2_api, which
is precisely a new peer connecting while the old one may still be holding on.
"""

import json
import socket
import threading
import time

import pytest

from retina_tracker.config import get_config
from retina_tracker.server import serve_detections
from retina_tracker.tracker import Tracker


def frame(timestamp, delay=10.0, doppler=50.0, snr=15.0):
    return {"timestamp": timestamp, "delay": [delay], "doppler": [doppler], "snr": [snr]}


def line(obj):
    return (json.dumps(obj) + "\n").encode()


@pytest.fixture
def feed():
    """A listening detection feed on an ephemeral port, served on a thread."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    port = server.getsockname()[1]

    tracker = Tracker(config=get_config())
    lock = threading.Lock()
    stop = threading.Event()
    thread = threading.Thread(
        target=serve_detections, args=(server, tracker, lock, stop), daemon=True)
    thread.start()
    try:
        yield tracker, port
    finally:
        stop.set()
        thread.join(timeout=3)
        server.close()


def connect(port):
    return socket.create_connection(("127.0.0.1", port), timeout=3)


def wait_for_frames(tracker, n, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if tracker.frame_count >= n:
            return True
        time.sleep(0.02)
    return tracker.frame_count >= n


def test_frames_on_a_single_connection_are_processed(feed):
    tracker, port = feed
    conn = connect(port)
    try:
        for i in range(3):
            conn.sendall(line(frame(1000 + i * 500)))
        assert wait_for_frames(tracker, 3)
    finally:
        conn.close()


def test_frames_split_across_packets_are_reassembled(feed):
    """A frame is not a packet. The buffer has to survive a boundary landing
    mid-JSON."""
    tracker, port = feed
    conn = connect(port)
    try:
        payload = line(frame(1000))
        conn.sendall(payload[:9])
        time.sleep(0.15)
        conn.sendall(payload[9:])
        assert wait_for_frames(tracker, 1)
    finally:
        conn.close()


def test_a_new_connection_is_served_while_the_old_one_sits_silent(feed):
    """The regression. A peer that stops sending without closing used to
    leave the loop blocked in recv, with the next peer unserved behind a
    backlog of one. That is the shape of the retina-gui to blah2_api
    handover, so it has to work."""
    tracker, port = feed
    stale = connect(port)
    try:
        # Never sends, never closes: exactly a killed container or a dropped link.
        time.sleep(0.2)

        fresh = connect(port)
        try:
            for i in range(3):
                fresh.sendall(line(frame(5000 + i * 500)))
            assert wait_for_frames(tracker, 3), "new peer was not served"
        finally:
            fresh.close()
    finally:
        stale.close()


def test_the_replaced_connection_stops_being_read(feed):
    """Newest wins. The tracker does not deduplicate, so two live feeders
    would give it every frame twice."""
    tracker, port = feed
    first = connect(port)
    try:
        first.sendall(line(frame(1000)))
        assert wait_for_frames(tracker, 1)

        second = connect(port)
        try:
            time.sleep(0.3)  # let the replacement land
            # The displaced peer keeps writing into a socket nobody reads.
            for i in range(5):
                try:
                    first.sendall(line(frame(2000 + i * 500)))
                except OSError:
                    break  # closed on us, which is the same outcome
            time.sleep(0.4)
            assert tracker.frame_count == 1, "frames from the replaced peer were read"

            second.sendall(line(frame(9000)))
            assert wait_for_frames(tracker, 2)
        finally:
            second.close()
    finally:
        first.close()


def test_a_partial_line_from_a_replaced_peer_does_not_corrupt_the_next(feed):
    """Half a frame left in the buffer would otherwise be prefixed onto the
    new peer's first line and take both of them out."""
    tracker, port = feed
    first = connect(port)
    try:
        first.sendall(b'{"timestamp": 1000, "delay": [10.0], "dop')  # cut mid-key
        time.sleep(0.2)

        second = connect(port)
        try:
            time.sleep(0.3)
            second.sendall(line(frame(7000)))
            assert wait_for_frames(tracker, 1), "the new peer's first frame was lost"
        finally:
            second.close()
    finally:
        first.close()


def test_a_reconnect_after_a_clean_close_is_served(feed):
    tracker, port = feed
    conn = connect(port)
    conn.sendall(line(frame(1000)))
    assert wait_for_frames(tracker, 1)
    conn.close()
    time.sleep(0.3)

    again = connect(port)
    try:
        again.sendall(line(frame(2000)))
        assert wait_for_frames(tracker, 2)
    finally:
        again.close()


@pytest.mark.parametrize("junk", [
    b"not json at all\n",
    b"[1, 2, 3]\n",                      # valid JSON, not an object
    b'{"delay": [1.0]}\n',               # object, but no timestamp
    b'{"timestamp": "nonsense"}\n',      # timestamp of the wrong type
])
def test_a_malformed_frame_does_not_take_the_feed_down(feed, junk):
    """One bad line used to be able to end detection ingest until the
    container was restarted."""
    tracker, port = feed
    conn = connect(port)
    try:
        conn.sendall(junk)
        time.sleep(0.2)
        conn.sendall(line(frame(4000)))
        assert wait_for_frames(tracker, 1), "the feed stopped after a malformed frame"
    finally:
        conn.close()


def test_reset_in_the_stream_still_works(feed):
    """Kept until retina-gui stops feeding this socket."""
    tracker, port = feed
    conn = connect(port)
    try:
        for i in range(3):
            conn.sendall(line(frame(1000 + i * 500)))
        assert wait_for_frames(tracker, 3)

        conn.sendall(line({"type": "RESET"}))
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and tracker.frame_count != 0:
            time.sleep(0.02)
        assert tracker.frame_count == 0
        assert tracker.tracks == []
    finally:
        conn.close()
