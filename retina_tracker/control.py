"""Loopback HTTP control surface for the streaming tracker.

The tracker's only inbound channel has been the detection socket, which means
the one control operation it supports — clearing state between search
geometries — has had to travel as a `{"type": "RESET"}` message mixed into the
detection stream. That works only while the sender of detections and the sender
of controls are the same process. They are about to stop being: blah2_api
forwards detections directly, and the auto-calibration search lives in
retina-gui. `run_tcp_server` accepts one connection at a time, so a second
consumer cannot simply open its own.

So control moves to its own door. Nothing here is specific to a caller: the
auto-calibration search and the Tracker page are equal consumers of a tracker
that does not know which is which.

Bound to loopback by default for the same reason the ingest socket is (see the
sidecar's compose command): the container runs with network_mode host, so
0.0.0.0 would publish this on the LAN.

Stdlib only, deliberately. This is a handful of routes on a link with one or
two clients; a framework would be a dependency and an image layer for nothing.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 30101

# How often a stream looks for new points. The tracker appends about once a
# second, so this is the resolution of the feed rather than a poll of anything
# expensive: since() on an unchanged history is a few length comparisons.
STREAM_INTERVAL_S = 1.0

# A comment line keeps an idle connection open through anything that times out
# silent sockets, and is how a stream notices the client has gone: the write
# fails.
HEARTBEAT_S = 15.0

# Clamped rather than rejected. A window is a display preference, and must
# never let a query string ask for more than is held.
MIN_WINDOW_S = 60


class _Handler(BaseHTTPRequestHandler):
    """Routes are matched on the path with any trailing slash removed, so
    /reset and /reset/ are the same endpoint."""

    protocol_version = "HTTP/1.1"

    # The default handler logs every request to stderr. A health check on a
    # short interval would bury the tracker's own output, which is the only
    # thing anyone reads that stream for.
    def log_message(self, fmt, *args):
        pass

    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self._route() == "/history/clear":
            # Wipes the record without touching the tracker. The page's
            # "Clear buffer" has always meant "clear what I am being shown,
            # keep tracking", and that distinction survives the record moving
            # here from retina-gui. An active track repopulates on its own
            # within a few events.
            if self.server.history is None:
                self._send(503, {"error": "history not enabled"})
                return
            self.server.history.clear()
            self._send(200, {"ok": True})
            return
        if self._route() == "/reset":
            # Held for the duration, so a 200 means the tracker is already
            # clear rather than scheduled to be. A caller that resets between
            # candidate geometries needs that: the next frame it waits on must
            # not be able to associate into pre-reset state.
            with self.server.tracker_lock:
                self.server.tracker.reset()
            self._send(200, {"ok": True})
            return
        self._send(404, {"error": "not found"})

    def do_GET(self):
        route = self._route()
        if route in ("/health", ""):
            with self.server.tracker_lock:
                payload = {
                    "ok": True,
                    "frames": self.server.tracker.frame_count,
                    "tracks": len(self.server.tracker.tracks),
                }
            if self.server.history is not None:
                payload["history"] = self.server.history.stats()
            self._send(200, payload)
            return
        if route == "/events":
            self._stream()
            return
        self._send(404, {"error": "not found"})

    # ── The data stream ────────────────────────────────────────

    def _window(self):
        raw = parse_qs(urlparse(self.path).query).get("window", [None])[0]
        if raw is None:
            return None
        try:
            seconds = int(raw)
        except ValueError:
            return None
        if seconds <= 0:
            return None
        return max(MIN_WINDOW_S, min(seconds, self.server.history.window_s))

    def _stream(self):
        """Server-sent events: a snapshot, then only what has been appended.

        The connection is the session. Its cursor lives in this thread and
        nowhere else, so there is no per-consumer state on the server to
        expire, and no negotiation: a reconnect simply takes a fresh
        snapshot, which is always a valid thing to start from.

        Sending the snapshot down this same stream rather than having the
        consumer fetch it separately is what removes the race between what
        the snapshot contained and where the delta stream began. There is one
        ordering, and this generator owns it.
        """
        if self.server.history is None:
            self._send(503, {"error": "history not enabled"})
            return

        window_s = self._window()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        # Chunked, not "read until the connection closes". No length is
        # knowable and this never ends of its own accord, so a client told
        # only "Connection: close" has to read to EOF to find a message
        # boundary — which for urllib3, and therefore for retina-gui's
        # requests-based proxy, means blocking until the read timeout rather
        # than delivering each event as it arrives. Framing every message as
        # its own chunk is what makes it stream to any client.
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()

        history = self.server.history
        try:
            payload, cursor = history.snapshot(window_s=window_s)
            self._event("snapshot", payload)

            last_sent = time.monotonic()
            while not self.server.stopping.is_set():
                time.sleep(STREAM_INTERVAL_S)

                delta, new_cursor = history.since(cursor)
                if delta is None:
                    # clear() ran, so every outstanding cursor is void.
                    # Re-seed rather than trying to reconcile.
                    payload, cursor = history.snapshot(window_s=window_s)
                    self._event("snapshot", payload)
                    last_sent = time.monotonic()
                    continue

                cursor = new_cursor
                if _has_points(delta):
                    self._event("delta", delta)
                    last_sent = time.monotonic()
                elif time.monotonic() - last_sent >= HEARTBEAT_S:
                    self._chunk(b": keepalive\n\n")
                    last_sent = time.monotonic()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # the consumer went away, which is how a stream ends

    def _event(self, kind, payload):
        body = json.dumps(payload, separators=(",", ":"))
        self._chunk(f"event: {kind}\ndata: {body}\n\n".encode())

    def _chunk(self, data):
        """One HTTP/1.1 chunk, so each message is its own frame on the wire."""
        self.wfile.write(b"%X\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()

    def _route(self):
        return self.path.split("?", 1)[0].rstrip("/")


class ControlServer(ThreadingHTTPServer):
    """HTTP control surface over a running Tracker.

    `tracker_lock` is the same lock the frame path takes, so a reset can never
    land in the middle of one. Frames arrive about once a second and take
    milliseconds, so the contention this introduces is not measurable; the
    alternative, deferring the reset to a flag the frame loop reads, would make
    a 200 mean "queued" and would never apply at all on a node whose detections
    have stopped.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, tracker, tracker_lock, host=DEFAULT_HOST, port=DEFAULT_PORT,
                 history=None):
        super().__init__((host, port), _Handler)
        self.tracker = tracker
        self.tracker_lock = tracker_lock
        self.history = history
        # Lets an open stream wind down on shutdown instead of holding the
        # process up for a full interval.
        self.stopping = threading.Event()

    def shutdown(self):
        self.stopping.set()
        super().shutdown()

    @property
    def port(self):
        """The bound port, which is what was asked for unless 0 was, in which
        case it is whatever the OS chose."""
        return self.server_address[1]


def _has_points(delta):
    if delta["tracks"]:
        return True
    return any(cols["t"] for cols in delta["detections"].values())


def start_control_server(tracker, tracker_lock, host=DEFAULT_HOST, port=DEFAULT_PORT,
                         history=None):
    """Serve the control surface on a daemon thread and return the server.

    The thread is a daemon so it never holds up interpreter shutdown: the
    tracker process is killed by its supervisor, not asked to wind down."""
    server = ControlServer(tracker, tracker_lock, host=host, port=port, history=history)
    thread = threading.Thread(target=server.serve_forever, daemon=True,
                              name="tracker-control")
    thread.start()
    server.thread = thread
    return server
