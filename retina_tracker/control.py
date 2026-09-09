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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 30101


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
        if self._route() in ("/health", ""):
            with self.server.tracker_lock:
                payload = {
                    "ok": True,
                    "frames": self.server.tracker.frame_count,
                    "tracks": len(self.server.tracker.tracks),
                }
            self._send(200, payload)
            return
        self._send(404, {"error": "not found"})

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

    def __init__(self, tracker, tracker_lock, host=DEFAULT_HOST, port=DEFAULT_PORT):
        super().__init__((host, port), _Handler)
        self.tracker = tracker
        self.tracker_lock = tracker_lock

    @property
    def port(self):
        """The bound port, which is what was asked for unless 0 was, in which
        case it is whatever the OS chose."""
        return self.server_address[1]


def start_control_server(tracker, tracker_lock, host=DEFAULT_HOST, port=DEFAULT_PORT):
    """Serve the control surface on a daemon thread and return the server.

    The thread is a daemon so it never holds up interpreter shutdown: the
    tracker process is killed by its supervisor, not asked to wind down."""
    server = ControlServer(tracker, tracker_lock, host=host, port=port)
    thread = threading.Thread(target=server.serve_forever, daemon=True,
                              name="tracker-control")
    thread.start()
    server.thread = thread
    return server
