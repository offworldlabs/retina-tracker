"""TCP server for receiving detection frames from blah2."""

import json
import select
import socket
import sys
import threading

from .config import get_config
from .control import DEFAULT_HOST as CONTROL_HOST
from .control import DEFAULT_PORT as CONTROL_PORT
from .control import start_control_server
from .tracker import Tracker


def process_streaming_frame(tracker, frame):
    """Convert blah2 streaming frame format to detections and process.

    Args:
        tracker: Tracker instance
        frame: Dict with 'timestamp', 'delay', 'doppler', 'snr', 'adsb' arrays
    """
    timestamp = frame["timestamp"]
    delays = frame.get("delay", [])
    dopplers = frame.get("doppler", [])
    snrs = frame.get("snr", [])
    adsb_list = frame.get("adsb", [])

    detections = []
    for idx, (delay, doppler, snr) in enumerate(zip(delays, dopplers, snrs)):
        detection = {
            "delay": delay,
            "doppler": doppler,
            "snr": snr,
        }
        if adsb_list and idx < len(adsb_list) and adsb_list[idx] is not None:
            detection["adsb"] = adsb_list[idx]
        detections.append(detection)

    tracker.process_frame(detections, timestamp)


# A new peer must never queue behind a dead one, so the backlog is bigger than
# the single slot it used to be.
LISTEN_BACKLOG = 8

# Bounds how long the loop sits in select() with nothing happening, so a
# stop_event is noticed promptly. Nothing else depends on it.
SELECT_TIMEOUT_S = 0.5


def _close(sock):
    if sock is None:
        return
    try:
        sock.close()
    except OSError:
        pass


def _handle_line(line, tracker, tracker_lock):
    """One newline-delimited frame from the detection feed.

    Nothing a peer can send may take the feed down. A frame that is not JSON,
    not an object, or missing the fields process_frame needs is logged and
    dropped: the alternative is one malformed line ending detection ingest
    until the container is restarted.
    """
    line = line.strip()
    if not line:
        return
    try:
        frame = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        print(f"JSON parse error: {e}", file=sys.stderr)
        return
    if not isinstance(frame, dict):
        print("Ignoring non-object frame", file=sys.stderr)
        return
    # A real detection frame never carries a "type" key, so this can never
    # misfire on genuine data.
    if frame.get("type") == "RESET":
        # Kept alongside POST /reset while retina-gui is still the process
        # feeding this socket. It goes when that does: a control message
        # riding in the data stream only works while one process sends both,
        # which is the arrangement being unwound.
        with tracker_lock:
            tracker.reset()
        print("Tracker state reset", file=sys.stderr)
        return
    try:
        with tracker_lock:
            process_streaming_frame(tracker, frame)
    except Exception as e:
        print(f"Dropping unusable frame: {e!r}", file=sys.stderr)


def serve_detections(server, tracker, tracker_lock, stop_event=None):
    """Read detection frames from `server`, newest connection winning.

    Exactly one peer feeds the tracker at a time, and it is whichever
    connected most recently. Two things follow, both of which the old
    accept-one-then-read-to-EOF loop got wrong.

    A peer that vanishes without closing — a killed container, a dropped
    link — leaves a half-open socket that never returns from recv and never
    reaches EOF, so the loop blocked there for as long as the kernel took to
    notice while a new peer sat unserved in a backlog of one. Selecting on
    the listener as well means a new connection is heard immediately,
    whatever state the old one is in. That is the case this exists for: the
    handover from retina-gui to blah2_api is exactly a new peer arriving
    while the old one may still be holding the slot.

    And only one peer should ever be feeding. The tracker does not
    deduplicate, so two senders of the same detections would hand it every
    frame twice at dt=0, and every track twice the evidence it earned.
    Replacing rather than multiplexing makes that unrepresentable.
    """
    conn = None
    buffer = b""

    while stop_event is None or not stop_event.is_set():
        watching = [server] if conn is None else [server, conn]
        try:
            ready, _, _ = select.select(watching, [], [], SELECT_TIMEOUT_S)
        except OSError:
            break  # listener closed underneath us: shutting down

        if server in ready:
            try:
                new_conn, addr = server.accept()
            except OSError:
                continue
            if conn is not None:
                print("Detection feed replaced by a newer connection", file=sys.stderr)
                _close(conn)
                # Whatever is half-read belongs to the peer being replaced.
                buffer = b""
            conn = new_conn
            print(f"Detections connected from {addr}", file=sys.stderr)

        if conn is not None and conn in ready:
            try:
                data = conn.recv(4096)
            except OSError:
                data = b""
            if not data:
                print("Detection feed disconnected", file=sys.stderr)
                _close(conn)
                conn = None
                buffer = b""
                continue

            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                _handle_line(line, tracker, tracker_lock)

    _close(conn)


def run_tcp_server(host="0.0.0.0", port=3012, event_writer=None, detection_window=20,
                   config=None, control_host=CONTROL_HOST, control_port=CONTROL_PORT):
    """Run tracker as TCP server receiving detection frames from blah2.

    Args:
        host: Bind address (default: 0.0.0.0)
        port: TCP port to listen on (default: 3012)
        event_writer: TrackEventWriter for streaming output
        detection_window: Number of detections in sliding window
        config: Configuration dict
        control_host: Bind address for the HTTP control surface
        control_port: Port for the HTTP control surface; 0 disables it
    """
    tracker = Tracker(
        event_writer=event_writer,
        detection_window=detection_window,
        config=config or get_config(),
    )

    # Guards every mutation of `tracker`. The frame path has always been
    # single-threaded, so this is uncontended right up until the control
    # surface below can reset from a request thread.
    tracker_lock = threading.Lock()

    if control_port:
        control = start_control_server(tracker, tracker_lock,
                                       host=control_host, port=control_port)
        print(f"Tracker control on {control_host}:{control.port}", file=sys.stderr)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(LISTEN_BACKLOG)

    print(f"Tracker listening on {host}:{port}", file=sys.stderr)

    serve_detections(server, tracker, tracker_lock)
