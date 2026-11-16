"""Records of confirmed tracks that went beyond OUTSIDE_DOPPLER_HZ, kept in a size-capped SQLite file.

retire() runs on the frame thread and only queues. flush() runs on the writer thread and does every
history read and disk write. Neither may raise into its caller. See docs/outside-store.md.
"""

import json
import os
import sqlite3
import sys
import threading
import time
import zlib
from collections import deque

from .config import get_config
from .history import DOPPLER_DP
from .track import OUTSIDE_DOPPLER_HZ

SCHEMA = 1
MARGIN_MS = 10_000
MAX_TRACK_MS = 5 * 60 * 1000
MAX_WAIT_S = 60.0
MAX_PENDING = 1000
MAX_BYTES = 256 * 1024 * 1024
MIN_FREE_BYTES = 1024 * 1024 * 1024
MAX_RECORD_POINTS = 50_000
POLL_S = 1.0
LIST_LIMIT = 50
MAX_LIST_LIMIT = 500
WAL_LIMIT_BYTES = 4 * 1024 * 1024
COMPRESS_LEVEL = 6
CORRUPT = ("SQLITE_CORRUPT", "SQLITE_NOTADB")

SUMMARY = (
    "id",
    "run",
    "track_id",
    "stitch_id",
    "adsb_hex",
    "birth_ms",
    "last_ms",
    "stored_ms",
    "peak_doppler_hz",
    "n_outside",
    "n_points",
    "n_neighbours",
    "complete",
    "bytes",
)

CREATE = """
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run TEXT NOT NULL,
    track_id TEXT NOT NULL,
    stitch_id TEXT,
    adsb_hex TEXT,
    birth_ms INTEGER NOT NULL,
    last_ms INTEGER NOT NULL,
    stored_ms INTEGER NOT NULL,
    peak_doppler_hz REAL NOT NULL,
    n_outside INTEGER NOT NULL,
    n_points INTEGER NOT NULL,
    n_neighbours INTEGER NOT NULL,
    complete INTEGER NOT NULL,
    bytes INTEGER NOT NULL,
    payload BLOB NOT NULL,
    UNIQUE (run, track_id)
)
"""

OLDEST = "SELECT id, bytes FROM records WHERE id != ? ORDER BY id LIMIT 1"
INSERT = """
INSERT OR IGNORE INTO records (
    run, track_id, stitch_id, adsb_hex, birth_ms, last_ms, stored_ms,
    peak_doppler_hz, n_outside, n_points, n_neighbours, complete, bytes, payload
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
LIST = """
SELECT id, run, track_id, stitch_id, adsb_hex, birth_ms, last_ms, stored_ms,
    peak_doppler_hz, n_outside, n_points, n_neighbours, complete, bytes
FROM records WHERE id < ? ORDER BY id DESC LIMIT ?
"""
NO_ID_ABOVE = 2**63 - 1


class _Unusable(Exception):
    pass


class _Newer(Exception):
    pass


class _NoRoom(Exception):
    pass


def _file_bytes(db):
    """(bytes of pages in use, bytes of the whole database) as this connection sees it, uncommitted changes included."""
    page_size, pages, free = (
        db.execute(f"PRAGMA {name}").fetchone()[0] for name in ("page_size", "page_count", "freelist_count")
    )
    return (pages - free) * page_size, pages * page_size


def _disk_free(path):
    """Bytes free to an unprivileged writer on the filesystem holding `path`, or None if it cannot be read."""
    try:
        stat = os.statvfs(os.path.dirname(os.path.abspath(path)))
    except OSError:
        return None
    return stat.f_bavail * stat.f_frsize


def _corrupt(error):
    return isinstance(error, _Unusable) or getattr(error, "sqlite_errorname", None) in CORRUPT


def _now_ms():
    return int(time.time() * 1000)


def _count(cols):
    return len(cols["t"])


def _stride(n):
    return max(1, -(-n // MAX_RECORD_POINTS))


def _without(cols, own):
    """cols less the points in own, matched on (t, delay, doppler)."""
    keep = [i for i, point in enumerate(zip(cols["t"], cols["delay"], cols["doppler"])) if point not in own]
    return {k: [v[i] for i in keep] if isinstance(v, list) else v for k, v in cols.items()}


def _thin(cols, stride):
    return {k: v[::stride] if isinstance(v, list) else v for k, v in cols.items()}


def _summary(track, run_id):
    return {
        "run": run_id,
        "track_id": track.id,
        "stitch_id": track.stitch_id or track.id,
        "adsb_hex": track.adsb_hex,
        "birth_ms": int(track.birth_timestamp),
        "last_ms": int(track.last_update_ms),
        "n_outside": track.n_outside,
        "peak_doppler_hz": round(float(track.peak_doppler_hz), DOPPLER_DP),
        "n_detections": track.n_observations,
        "n_frames": track.n_frames,
        "mean_snr": round(track.total_snr / max(track.n_observations, 1), 1),
        "max_velocity_ms": round(float(track.max_velocity_ms), 1),
        "shadow_fraction": round(track.shadow_fraction(), 3),
        "interference_fraction": round(track.interference_fraction(), 3),
    }


class OutsideStore:
    """Keeps one record per confirmed track with a detection beyond OUTSIDE_DOPPLER_HZ.

    The oldest are dropped once the database file's pages in use pass max_bytes, or sooner when the
    filesystem has less than min_free_bytes free. The file is opened and checked by open() on the writer
    thread, never at construction, so a large store cannot delay ingest; retire() queues meanwhile.

    Every database access holds _db_lock; the queue and counters hold _lock. _lock is never held while
    taking _db_lock or the history's lock.
    """

    def __init__(
        self, path, history, max_bytes=MAX_BYTES, min_free_bytes=MIN_FREE_BYTES, clock=time.monotonic, disk_free=None
    ):
        self.path = path
        self.history = history
        self.max_bytes = int(max_bytes)
        self.min_free_bytes = int(min_free_bytes)
        self._clock = clock
        self._disk_free = disk_free or (lambda: _disk_free(path))
        self.free_bytes = None
        self._pending = deque()
        self._lock = threading.Lock()
        self._db_lock = threading.Lock()
        self.n_queued = 0
        self.n_dropped = 0
        self.n_written = 0
        self.n_failed = 0
        self.n_rolled = 0
        self.n_skipped = 0
        self.n_too_long = 0
        self.error = None
        self._db = None

    # ── Frame thread ───────────────────────────────────────────

    def retire(self, track, run_id):
        """Queue a retired track if it went outside. Never touches disk and never raises."""
        try:
            if not track.id or not track.n_outside:
                return
            if track.last_update_ms - track.birth_timestamp > MAX_TRACK_MS:
                with self._lock:
                    self.n_too_long += 1
                return
            item = _summary(track, run_id)
            item["queued_at"] = self._clock()
        except Exception as e:
            self._fail(f"could not summarise track {getattr(track, 'id', None)}", e)
            return
        with self._lock:
            if self.error is not None or len(self._pending) >= MAX_PENDING:
                self.n_dropped += 1
                return
            self._pending.append(item)
            self.n_queued += 1

    # ── Writer thread ──────────────────────────────────────────

    def open(self):
        """Open and check the file if not yet done. Idempotent; True once the store can be written."""
        with self._db_lock:
            if self._db is None and self.error is None:
                self._db = self._open()
            return self._db is not None

    @property
    def state(self):
        if self.error is not None:
            return "disabled"
        return "open" if self._db is not None else "opening"

    def flush(self, force=False):
        """Write each queued record whose neighbour window has settled, or every one when forced. Returns the count.

        Opens the store first if it is not open yet. While it is disabled, queued records are dropped and counted.
        """
        if not self.open():
            with self._lock:
                self.n_dropped += len(self._pending)
                self._pending.clear()
            return 0
        written = 0
        while True:
            released = self.history.released_ms()
            with self._lock:
                if not self._pending or not (force or self._settled(self._pending[0], released)):
                    return written
                item = self._pending.popleft()
            if self._store(item):
                written += 1

    def _settled(self, item, released):
        if released is not None and released >= item["last_ms"] + MARGIN_MS:
            return True
        return self._clock() - item["queued_at"] >= MAX_WAIT_S

    def _store(self, item):
        try:
            self._write(self._build(item))
        except _NoRoom as e:
            with self._lock:
                self.n_skipped += 1
            print(f"Outside store: skipped track {item['run']}/{item['track_id']}: {e}", file=sys.stderr)
            return False
        except Exception as e:
            self._fail(f"could not store track {item['run']}/{item['track_id']}", e)
            return False
        with self._lock:
            self.n_written += 1
        return True

    def _build(self, item):
        points = self.history.track(item["track_id"]) or {k: [] for k in ("t", "delay", "doppler", "snr", "lat", "lon")}
        points.pop("meta", None)
        start, end = item["birth_ms"] - MARGIN_MS, item["last_ms"] + MARGIN_MS
        around = self.history.window(start, end, exclude=item["track_id"])
        own = set(zip(points["t"], points["delay"], points["doppler"]))
        detections = {k: _without(c, own) for k, c in around["detections"].items()}
        tracks = around["tracks"]
        n_points = _count(points)
        n_neighbours = sum(_count(c) for c in detections.values()) + sum(_count(c) for c in tracks.values())
        track_stride, stride = _stride(n_points), _stride(n_neighbours)
        config = get_config()
        record = {k: v for k, v in item.items() if k != "queued_at"}
        record.update(
            schema=SCHEMA,
            limit_hz=OUTSIDE_DOPPLER_HZ,
            stored_ms=_now_ms(),
            n_points=n_points,
            complete=n_points >= item["n_detections"],
            track=dict(_thin(points, track_stride), stride=track_stride),
            neighbours={
                "start_ms": start,
                "end_ms": end,
                "total": n_neighbours,
                "stride": stride,
                "detections": {k: _thin(c, stride) for k, c in detections.items()},
                "tracks": {k: _thin(c, stride) for k, c in tracks.items()},
            },
            radar=dict(config.get("radar") or {}),
            location=config.get("location"),
        )
        return record

    def _write(self, record):
        payload = zlib.compress(json.dumps(record, separators=(",", ":")).encode("utf-8"), COMPRESS_LEVEL)
        row = (
            record["run"],
            record["track_id"],
            record["stitch_id"],
            record["adsb_hex"],
            record["birth_ms"],
            record["last_ms"],
            record["stored_ms"],
            record["peak_doppler_hz"],
            record["n_outside"],
            record["n_points"],
            record["neighbours"]["total"],
            int(record["complete"]),
            len(payload),
            payload,
        )
        with self._db_lock:
            db = self._db
            if db is None:
                raise sqlite3.OperationalError(self.error or "store is closed")
            db.execute("BEGIN IMMEDIATE")
            try:
                cap = self._cap(_file_bytes(db)[1])
                page_size = db.execute("PRAGMA page_size").fetchone()[0]
                rolled = self._roll(db, cap, need=len(payload) + page_size)
                cursor = db.execute(INSERT, row)
                if cursor.rowcount:
                    rolled += self._roll(db, cap, keep_id=cursor.lastrowid)
                db.execute("COMMIT")
            except BaseException:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise
        with self._lock:
            self.n_rolled += rolled

    def _cap(self, file_size):
        """Bytes of pages the file may have in use: max_bytes, less whatever would take the disk below the floor."""
        self.free_bytes = self._disk_free()
        if self.free_bytes is None:
            return self.max_bytes
        return min(self.max_bytes, file_size + max(0, self.free_bytes - self.min_free_bytes))

    def _roll(self, db, cap, need=0, keep_id=None):
        """Delete the oldest records until the pages in use plus `need` fit cap.

        Called before the insert with the record's size, so freed pages are reused rather than the file growing,
        then after it with keep_id, where the exact check is made and _NoRoom raised if only that record is left.
        """
        rolled = 0
        while (used := _file_bytes(db)[0]) + need > cap:
            oldest = db.execute(OLDEST, (-1 if keep_id is None else keep_id,)).fetchone()
            if oldest is None:
                if keep_id is None:
                    return rolled
                raise _NoRoom(f"{used} bytes in use with nothing older to delete, room for {cap}")
            db.execute("DELETE FROM records WHERE id = ?", (oldest[0],))
            rolled += 1
        return rolled

    # ── Opening ────────────────────────────────────────────────

    def _open(self):
        """Open the file. Set it aside only if SQLite calls it corrupt; any other failure disables the store."""
        try:
            return self._connect()
        except Exception as e:
            if not _corrupt(e):
                return self._disable(e)
            print(f"Outside store {self.path} is corrupt ({e}); moving it to {self.path}.bad", file=sys.stderr)
        try:
            self._set_aside()
            return self._connect()
        except Exception as e:
            return self._disable(e)

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5.0, isolation_level=None, check_same_thread=False)
        try:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA:
                raise _Newer(f"schema {version} is newer than this tracker's {SCHEMA}; leaving the file untouched")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute(f"PRAGMA journal_size_limit={WAL_LIMIT_BYTES}")
            check = db.execute("PRAGMA quick_check").fetchone()[0]
            if check != "ok":
                raise _Unusable(f"quick_check: {check}")
            db.execute(CREATE)
            db.execute(f"PRAGMA user_version={SCHEMA}")
        except BaseException:
            db.close()
            raise
        return db

    def _set_aside(self):
        for suffix in ("", "-wal", "-shm"):
            bad = f"{self.path}.bad{suffix}"
            if os.path.exists(bad):
                os.remove(bad)
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(self.path + suffix):
                os.replace(self.path + suffix, f"{self.path}.bad{suffix}")

    def _disable(self, error):
        self.error = str(error)
        print(f"Outside store disabled: {self.path}: {error}", file=sys.stderr)
        return None

    def _fail(self, what, error):
        with self._lock:
            self.n_failed += 1
        print(f"Outside store: {what}: {error!r}", file=sys.stderr)

    # ── Reading ────────────────────────────────────────────────

    def records(self, before=None, limit=LIST_LIMIT):
        """Summaries newest first, only ids below `before` when given. None when the store is disabled."""
        args = (NO_ID_ABOVE if before is None else int(before), max(1, min(int(limit), MAX_LIST_LIMIT)))
        with self._db_lock:
            if self._db is None:
                return None
            rows = self._db.execute(LIST, args).fetchall()
        return [{**dict(zip(SUMMARY, row)), "complete": bool(row[SUMMARY.index("complete")])} for row in rows]

    def record(self, record_id):
        """One full record, or None if it is not held."""
        with self._db_lock:
            if self._db is None:
                return None
            row = self._db.execute("SELECT id, payload FROM records WHERE id = ?", (int(record_id),)).fetchone()
        if row is None:
            return None
        return {**json.loads(zlib.decompress(row[1])), "id": row[0]}

    def stats(self):
        """Counters and the stored size, for /health."""
        with self._lock:
            out = {
                "state": self.state,
                "error": self.error,
                "pending": len(self._pending),
                "queued": self.n_queued,
                "written": self.n_written,
                "dropped": self.n_dropped,
                "failed": self.n_failed,
                "rolled": self.n_rolled,
                "skipped": self.n_skipped,
                "too_long": self.n_too_long,
                "max_bytes": self.max_bytes,
                "min_free_bytes": self.min_free_bytes,
                "free_bytes": self.free_bytes,
            }
        try:
            with self._db_lock:
                if self._db is not None:
                    out["records"], out["payload_bytes"] = self._db.execute(
                        "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM records"
                    ).fetchone()
                    out["used_bytes"], out["file_bytes"] = _file_bytes(self._db)
        except sqlite3.Error as e:
            out["error"] = str(e)
        return out

    def close(self):
        with self._db_lock:
            if self._db is not None:
                self._db.close()
                self._db = None


def start_writer(store, interval_s=POLL_S, stop_event=None):
    """Flush on a cadence, so the frame path never waits on the SD card."""

    def loop():
        while True:
            if stop_event is not None:
                if stop_event.wait(interval_s):
                    return
            else:
                time.sleep(interval_s)
            try:
                store.flush()
            except Exception as e:
                print(f"Outside store writer: {e!r}", file=sys.stderr)

    thread = threading.Thread(target=loop, daemon=True, name="outside-writer")
    thread.start()
    return thread
