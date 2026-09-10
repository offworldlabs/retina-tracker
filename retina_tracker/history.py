"""The rolling record of what this node has seen, and the shape it is served in.

The tracker's own Track objects keep a bounded per-track ring and drop
completed tracks after a short merge window, so nothing in it can answer "what
was seen three hours ago". This is that record: every detection the tracker was
given, classified, plus the points and metadata of every track, held for a
rolling window.

In memory, deliberately. It does not need to survive a restart, and the
alternative was writing several hundred megabytes a day to an SD card that the
fleet cannot afford to wear out.

Storage
-------
`array.array` columns rather than lists of tuples or dicts. A CPython tuple of
four floats costs about 192 bytes once the list slot, the tuple header and four
boxed floats are counted; the same point here is 20, so a buffer that would
have been 83 MB is 8.7. Columns are also what goes on the wire, so serving a
snapshot is a slice rather than a transposition.

Bounded twice over. The time window is the point of the thing, but a window
alone makes memory a function of how busy the sky is, which is not something a
node can promise. `max_points` is a hard ceiling underneath it: whichever binds
first, wins. That is what makes the footprint predictable on hardware that has
no headroom to spare.

Deltas
------
Same contract as the page already speaks: a snapshot on connect and only what
has been appended since, per consumer. Columns are append-only between prunes,
so a consumer's position is a count. Because pruning drops from the *front*
that count is monotonic rather than an index — hence the `_dropped` counters,
which record how many points have fallen off ahead of the ones still held.
clear() bumps `_gen`, which voids every outstanding cursor at once and makes
consumers take a fresh snapshot rather than trying to reconcile.
"""

import bisect
import threading
import time
from array import array

WINDOW_S = 4 * 3600

# A hard ceiling on each class, independent of the window. 500,000 points is
# about 10 MB per class at 20 bytes each, and covers four hours at rates well
# above anything measured. A node that starts seeing far more loses the oldest
# points rather than its memory.
MAX_POINTS = 500_000

# Tracks are far fewer than detections and each is small, but a run that
# accumulates thousands still has to stop somewhere.
MAX_TRACKS = 2000

ASSOCIATED = "associated"
UNASSOCIATED = "unassociated"
BELOW_SNR = "below_snr"
CLASSES = (ASSOCIATED, UNASSOCIATED, BELOW_SNR)

# Rounded on the way out, to the precision the measurement actually has: 2 dp
# of bistatic range is 10 m, 2 dp of Doppler is 0.01 Hz, 1 dp of SNR. Also
# undoes float32's repr — 16.1 stored as a float32 reads back as
# 16.100000381469727, which would be 18 bytes on the wire for 4 bytes of
# meaning.
DELAY_DP = 2
DOPPLER_DP = 2
SNR_DP = 1


def _columns():
    """One point-set. 'q' is 8 bytes, 'f' is 4: 20 bytes a point."""
    return {"t": array("q"), "delay": array("f"),
            "doppler": array("f"), "snr": array("f")}


def _append(cols, timestamp, delay, doppler, snr):
    cols["t"].append(int(timestamp))
    cols["delay"].append(delay)
    cols["doppler"].append(doppler)
    cols["snr"].append(snr)


def _drop_front(cols, n):
    if n <= 0:
        return 0
    for key in cols:
        del cols[key][:n]
    return n


def _slice(cols, start):
    """The wire form of cols[start:], rounded."""
    return {
        "t": list(cols["t"][start:]),
        "delay": [round(v, DELAY_DP) for v in cols["delay"][start:]],
        "doppler": [round(v, DOPPLER_DP) for v in cols["doppler"][start:]],
        "snr": [round(v, SNR_DP) for v in cols["snr"][start:]],
    }


def _first_at_or_after(times, cutoff_ms):
    """Points are appended in timestamp order, so the cutoff is a bisect
    rather than a scan. That is what makes a view window a saving."""
    return bisect.bisect_left(times, cutoff_ms)


class DetectionHistory:
    """Everything the node has seen recently, and what became of it.

    Written from the tracker's frame thread and read from HTTP request
    threads, so every method takes the lock. Reads are slices of contiguous
    arrays, so they are short.
    """

    def __init__(self, window_s=WINDOW_S, max_points=MAX_POINTS, max_tracks=MAX_TRACKS):
        self._lock = threading.Lock()
        self.window_s = window_s
        self.max_points = max_points
        self.max_tracks = max_tracks

        # One point-set per class rather than one with a class column: the
        # page asks for these separately, and keeping them apart makes a
        # snapshot a slice instead of a filter over everything held.
        self._det = {name: _columns() for name in CLASSES}
        self._det_dropped = dict.fromkeys(CLASSES, 0)

        self._tracks = {}          # id -> columns
        self._track_meta = {}      # id -> what the tracker thinks of it
        self._track_dropped = {}   # id -> points dropped from the front
        self._track_last_ts = {}   # id -> newest timestamp held

        self._gen = 0

    # ── Writing ────────────────────────────────────────────────

    def write_detections(self, timestamp, associated, unassociated, below_snr):
        """The tracker's detection sink. One call per frame, in frame order,
        with each detection's classification already final."""
        with self._lock:
            for name, dets in ((ASSOCIATED, associated),
                               (UNASSOCIATED, unassociated),
                               (BELOW_SNR, below_snr)):
                cols = self._det[name]
                for det in dets:
                    _append(cols, timestamp, det["delay"], det["doppler"],
                            det.get("snr", 0.0))
                self._enforce_ceiling(name)

    def write_event(self, track_id, timestamp, length, detections,
                    adsb_hex=None, is_anomalous=False, anomaly_types=None,
                    max_velocity_ms=0.0, shadow_fraction=0.0, **_unused):
        """The tracker's event-writer duck type.

        Named keywords rather than **kwargs so it is visible which of the
        sidecar's fields are kept and which are dropped on purpose:
        adsb_initialized is the only deliberate omission. Anything added
        later lands in **_unused rather than raising.

        Detections arrive already deduplicated by TrackEventWriter, but the
        timestamp guard stays: this is also reachable from a consumer that
        replays, and appending an older point would break the ordering
        every read here relies on.
        """
        with self._lock:
            cols = self._tracks.get(track_id)
            if cols is None:
                if len(self._tracks) >= self.max_tracks:
                    self._evict_oldest_track()
                cols = self._tracks[track_id] = _columns()
                self._track_dropped[track_id] = 0

            last = self._track_last_ts.get(track_id)
            for det in detections:
                ts = det["timestamp"]
                if last is not None and ts <= last:
                    continue
                _append(cols, ts, det["delay"], det["doppler"], det.get("snr", 0.0))
                last = ts if last is None or ts > last else last
            if last is not None:
                self._track_last_ts[track_id] = last

            self._track_meta[track_id] = {
                "adsb_hex": adsb_hex,
                "length": length,
                "max_velocity_ms": round(max_velocity_ms or 0.0, 1),
                "is_anomalous": bool(is_anomalous),
                "anomaly_types": sorted(anomaly_types or []),
                "shadow_fraction": round(shadow_fraction or 0.0, 3),
            }

    def prune(self, now_ms):
        """Drop everything older than the window. Runs on its own cadence,
        independent of whether anyone is watching: the ceiling below is a
        backstop, but this is what keeps the window a window."""
        cutoff = now_ms - self.window_s * 1000
        with self._lock:
            for name in CLASSES:
                cols = self._det[name]
                dropped = _drop_front(cols, _first_at_or_after(cols["t"], cutoff))
                self._det_dropped[name] += dropped

            for track_id in list(self._tracks):
                cols = self._tracks[track_id]
                cut = _first_at_or_after(cols["t"], cutoff)
                if cut >= len(cols["t"]):
                    self._forget_track(track_id)
                elif cut:
                    self._track_dropped[track_id] += _drop_front(cols, cut)

    def clear(self):
        """Wipe everything and void every outstanding cursor."""
        with self._lock:
            self._det = {name: _columns() for name in CLASSES}
            self._det_dropped = dict.fromkeys(CLASSES, 0)
            self._tracks = {}
            self._track_meta = {}
            self._track_dropped = {}
            self._track_last_ts = {}
            self._gen += 1

    # ── Bounds ─────────────────────────────────────────────────

    def _enforce_ceiling(self, name):
        """Caller holds the lock. The window is the intent; this is the
        promise about memory when the sky is busier than expected."""
        cols = self._det[name]
        excess = len(cols["t"]) - self.max_points
        if excess > 0:
            self._det_dropped[name] += _drop_front(cols, excess)

    def _evict_oldest_track(self):
        """Caller holds the lock. Whichever track has the oldest last point:
        a live one is still being written to, so this reaches for a dead one
        first without needing to be told which are dead."""
        oldest = min(self._track_last_ts, key=self._track_last_ts.get, default=None)
        if oldest is None:
            oldest = next(iter(self._tracks))
        self._forget_track(oldest)

    def _forget_track(self, track_id):
        """Caller holds the lock."""
        self._tracks.pop(track_id, None)
        self._track_meta.pop(track_id, None)
        self._track_dropped.pop(track_id, None)
        self._track_last_ts.pop(track_id, None)

    # ── Reading ────────────────────────────────────────────────

    def _cursor(self):
        """Caller holds the lock."""
        return {
            "gen": self._gen,
            "detections": {
                name: self._det_dropped[name] + len(self._det[name]["t"])
                for name in CLASSES
            },
            "tracks": {
                tid: self._track_dropped.get(tid, 0) + len(cols["t"])
                for tid, cols in self._tracks.items()
            },
        }

    def cursor(self):
        with self._lock:
            return self._cursor()

    def snapshot(self, window_s=None, now_ms=None):
        """Everything a consumer needs to draw from cold, plus the cursor to
        continue from.

        `window_s` narrows what is served without touching what is held. The
        cursor is always the end of everything, not the end of the window, so
        deltas continue from now however much history was asked for.
        """
        if window_s is not None and now_ms is None:
            now_ms = int(time.time() * 1000)
        with self._lock:
            cutoff = None if window_s is None else now_ms - window_s * 1000

            detections = {}
            for name in CLASSES:
                cols = self._det[name]
                start = 0 if cutoff is None else _first_at_or_after(cols["t"], cutoff)
                detections[name] = _slice(cols, start)

            tracks = {}
            for tid, cols in self._tracks.items():
                start = 0 if cutoff is None else _first_at_or_after(cols["t"], cutoff)
                if start >= len(cols["t"]):
                    continue
                tracks[tid] = dict(_slice(cols, start), meta=self._track_meta.get(tid, {}))

            payload = {
                "gen": self._gen,
                "window_s": window_s,
                "detections": detections,
                "tracks": tracks,
            }
            return payload, self._cursor()

    def since(self, cursor):
        """Whatever has been appended since `cursor`.

        Returns (payload, new_cursor). payload is None when the cursor cannot
        be honoured — a different generation, meaning clear() ran — and the
        caller should send a fresh snapshot rather than try to reconcile.

        A track the cursor has never seen comes back whole, which is what
        makes a newly promoted track arrive with the history the tracker
        backfilled on promotion rather than truncated at the moment the
        consumer happened to connect.
        """
        if not cursor or "gen" not in cursor:
            return None, None
        with self._lock:
            if cursor["gen"] != self._gen:
                return None, None

            seen_det = cursor.get("detections") or {}
            detections = {}
            for name in CLASSES:
                cols = self._det[name]
                start = max(0, seen_det.get(name, 0) - self._det_dropped[name])
                detections[name] = _slice(cols, start)

            seen_tracks = cursor.get("tracks") or {}
            tracks = {}
            for tid, cols in self._tracks.items():
                start = max(0, seen_tracks.get(tid, 0) - self._track_dropped.get(tid, 0))
                if start >= len(cols["t"]):
                    continue
                tracks[tid] = dict(_slice(cols, start), meta=self._track_meta.get(tid, {}))

            payload = {"gen": self._gen, "detections": detections, "tracks": tracks}
            return payload, self._cursor()

    # ── Introspection ──────────────────────────────────────────

    def stats(self):
        """Point counts and the memory they occupy, for /health."""
        with self._lock:
            per_class = {name: len(self._det[name]["t"]) for name in CLASSES}
            track_points = sum(len(cols["t"]) for cols in self._tracks.values())
            points = sum(per_class.values()) + track_points
            return {
                "detections": per_class,
                "tracks": len(self._tracks),
                "track_points": track_points,
                "points": points,
                # 20 bytes a point: an int64 and three float32s.
                "approx_bytes": points * 20,
            }


class TeeEventWriter:
    """Fans track events out to several writers.

    The tracker takes one event_writer, and there are now two things that want
    the events: the JSONL file, which live_score reads, and the in-memory
    history the page is served from. Ordering matters — the file's writer is
    what deduplicates detections, so it goes first and the history sees what
    was actually written.
    """

    def __init__(self, *writers):
        self._writers = [w for w in writers if w is not None]

    def write_event(self, *args, **kwargs):
        for writer in self._writers:
            writer.write_event(*args, **kwargs)

    def close(self):
        for writer in self._writers:
            if hasattr(writer, "close"):
                writer.close()


def start_pruner(history, interval_s=60, stop_event=None):
    """Drop what has aged out, on its own cadence.

    Independent of whether anyone is watching, and of whether frames are
    arriving: a node that stops receiving should still let its window empty
    rather than holding four hours of stale points indefinitely.
    """
    def loop():
        while True:
            if stop_event is not None:
                if stop_event.wait(interval_s):
                    return
            else:
                time.sleep(interval_s)
            history.prune(int(time.time() * 1000))

    thread = threading.Thread(target=loop, daemon=True, name="history-pruner")
    thread.start()
    return thread
