"""Tests for the outside store: confirmed tracks that went beyond +/-300 Hz, kept on disk with their neighbours.

See docs/outside-store.md.
"""

import json
import os
import random
import sqlite3
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from retina_tracker import outside
from retina_tracker.config import get_config, load_config, set_config
from retina_tracker.control import start_control_server
from retina_tracker.history import DetectionHistory
from retina_tracker.kalman import KalmanFilter, doppler_to_range_rate
from retina_tracker.outside import (
    MARGIN_MS,
    MAX_TRACK_MS,
    MAX_WAIT_S,
    NEAR_DELAY_KM,
    NEAR_DOPPLER_HZ,
    OutsideStore,
)
from retina_tracker.track import OUTSIDE_DOPPLER_HZ, Track
from retina_tracker.tracker import Tracker

FRAME_MS = 500
FAST = 350.0
SLOW = 100.0


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeTrack:
    """Just what OutsideStore.retire reads from a Track."""

    def __init__(self, track_id, last_ms=10_000, n_outside=3):
        self.id = track_id
        self.stitch_id = None
        self.adsb_hex = None
        self.birth_timestamp = last_ms - 5_000
        self.last_update_ms = last_ms
        self.n_outside = n_outside
        self.peak_doppler_hz = -420.0
        self.n_frames = 10
        self.total_snr = 120.0
        self.n_observations = 10
        self.max_velocity_ms = 300.0

    def shadow_fraction(self):
        return 0.0

    def interference_fraction(self):
        return 0.0


def make_store(tmp_path, history=None, **kwargs):
    store = OutsideStore(str(tmp_path / "outside.sqlite"), history or DetectionHistory(), **kwargs)
    store.open()
    return store


def fill(store, first, count):
    for i in range(first, first + count):
        store.retire(FakeTrack(f"261114-{i:06X}"), "run")
    return store.flush(force=True)


def make_pipeline(tmp_path):
    set_config(load_config(None))
    history = DetectionHistory()
    store = make_store(tmp_path, history)
    tracker = Tracker(event_writer=history, detection_sink=history, retire_sink=store)
    return tracker, history, store


def clutter(i):
    return {"delay": 5.0 + (i * 7.3) % 50.0, "doppler": -900.0 + (i * 137.0) % 1800.0, "snr": 6.0}


def fly(tracker, start_frame, frames, doppler, after=30, replica=False):
    """A target at constant Doppler with one clutter detection per frame, then empty frames until long after it dies.

    replica adds a weaker multipath copy 2 km behind and 30 Hz below the target, which the tracker shadows.
    """
    step = doppler_to_range_rate(doppler) * FRAME_MS / 1000.0
    for i in range(frames):
        target = {"delay": 30.0 + i * step, "doppler": doppler, "snr": 15.0}
        dets = [target, clutter(start_frame + i)]
        if replica:
            dets.append({"delay": target["delay"] + 2.0, "doppler": doppler - 30.0, "snr": 9.0})
        tracker.process_frame(dets, (start_frame + i) * FRAME_MS)
    for i in range(frames, frames + after):
        tracker.process_frame([], (start_frame + i) * FRAME_MS)


# ── The trigger ────────────────────────────────────────────────


def test_a_track_counts_its_detections_beyond_the_limit():
    set_config(load_config(None))
    track = Track({"delay": 20.0, "doppler": FAST, "snr": 15.0}, 0, KalmanFilter())
    assert (track.n_outside, track.peak_doppler_hz) == (1, FAST)
    track.update({"delay": 20.1, "doppler": SLOW, "snr": 15.0}, 500)
    track.update({"delay": 20.2, "doppler": -400.0, "snr": 15.0}, 1000)
    track.update({"delay": 20.3, "doppler": OUTSIDE_DOPPLER_HZ, "snr": 15.0}, 1500)
    assert (track.n_outside, track.peak_doppler_hz) == (2, -400.0)


def test_a_confirmed_track_that_went_outside_is_stored_with_its_points_and_neighbours(tmp_path):
    tracker, _, store = make_pipeline(tmp_path)
    fly(tracker, 0, 20, FAST, replica=True)
    assert store.flush() == 1

    [summary] = store.records()
    assert summary["run"] == tracker.run_id
    assert summary["n_outside"] == 20 and summary["peak_doppler_hz"] == FAST
    assert summary["complete"] is True

    record = store.record(summary["id"])
    assert record["track_id"] == summary["track_id"]
    assert record["limit_hz"] == OUTSIDE_DOPPLER_HZ
    assert len(record["track"]["t"]) == record["n_points"] >= 20
    assert set(record["track"]["doppler"]) == {FAST}
    window = record["neighbours"]
    assert window["start_ms"] == record["birth_ms"] - MARGIN_MS
    assert window["end_ms"] == record["last_ms"] + MARGIN_MS
    assert window["in_window"] == 60
    assert window["detections"]["associated"]["t"] == []
    kept = window["detections"]["unassociated"]
    assert sum(1 for d in kept["doppler"] if d == FAST - 30.0) == 20
    assert len(kept["t"]) < 40
    for t, delay, doppler in zip(kept["t"], kept["delay"], kept["doppler"]):
        at_delay, at_doppler = outside._position(record["track"], t)
        assert abs(delay - at_delay) <= NEAR_DELAY_KM and abs(doppler - at_doppler) <= NEAR_DOPPLER_HZ
    assert summary["track_id"] not in window["tracks"]
    assert record["radar"]["center_frequency"] == get_config()["radar"]["center_frequency"]


def test_a_track_that_stayed_inside_is_not_stored(tmp_path):
    tracker, _, store = make_pipeline(tmp_path)
    fly(tracker, 0, 20, SLOW)
    assert store.flush(force=True) == 0
    assert store.stats()["queued"] == 0


def test_a_track_that_never_confirmed_is_not_stored(tmp_path):
    tracker, _, store = make_pipeline(tmp_path)
    tracker.process_frame([{"delay": 30.0, "doppler": FAST, "snr": 15.0}], 0)
    for i in range(1, 20):
        tracker.process_frame([], i * FRAME_MS)
    assert store.stats()["queued"] == 0


# ── Settling ───────────────────────────────────────────────────


def test_a_record_waits_until_its_neighbour_window_has_been_released(tmp_path):
    tracker, history, store = make_pipeline(tmp_path)
    fly(tracker, 0, 20, FAST, after=12)
    assert store.stats()["pending"] == 1
    assert history.released_ms() < 19 * FRAME_MS + MARGIN_MS
    assert store.flush() == 0
    for i in range(32, 45):
        tracker.process_frame([], i * FRAME_MS)
    assert store.flush() == 1


def test_a_track_longer_than_the_limit_is_counted_not_stored(tmp_path):
    store = make_store(tmp_path)
    at_limit = FakeTrack("261114-000001", last_ms=MAX_TRACK_MS + 1_000)
    at_limit.birth_timestamp = 1_000
    too_long = FakeTrack("261114-000002", last_ms=MAX_TRACK_MS + 1_001)
    too_long.birth_timestamp = 1_000
    store.retire(at_limit, "run")
    store.retire(too_long, "run")
    stats = store.stats()
    assert (stats["queued"], stats["too_long"]) == (1, 1)
    store.flush(force=True)
    assert [r["track_id"] for r in store.records()] == ["261114-000001"]


def test_a_stalled_feed_still_writes_after_the_longest_wait(tmp_path):
    clock = Clock()
    store = make_store(tmp_path, clock=clock)
    store.retire(FakeTrack("261114-000001"), "run")
    assert store.flush() == 0
    clock.now += MAX_WAIT_S
    assert store.flush() == 1


# ── Overflow ───────────────────────────────────────────────────


def test_the_oldest_records_roll_off_once_the_pages_in_use_pass_max_bytes(tmp_path):
    store = make_store(tmp_path)
    fill(store, 0, 1)
    store.max_bytes = store.stats()["used_bytes"] + 4 * 4096

    assert fill(store, 1, 60) == 60
    stats = store.stats()
    assert stats["used_bytes"] <= store.max_bytes and stats["rolled"] > 0
    ids = [r["id"] for r in store.records(limit=500)]
    assert ids == list(range(61, 61 - len(ids), -1)) and len(ids) < 61


def test_the_cap_counts_the_file_not_the_payload(tmp_path):
    store = make_store(tmp_path)
    fill(store, 0, 20)
    stats = store.stats()
    assert stats["file_bytes"] >= stats["used_bytes"] > stats["payload_bytes"]


def test_a_record_that_cannot_fit_is_skipped_and_the_store_left_alone(tmp_path):
    store = make_store(tmp_path)
    fill(store, 1, 1)
    store.max_bytes = 1
    assert fill(store, 2, 1) == 0
    assert [r["track_id"] for r in store.records()] == ["261114-000001"]
    assert (store.stats()["skipped"], store.stats()["failed"]) == (1, 0)


def test_a_disk_at_its_floor_rolls_the_oldest_instead_of_growing(tmp_path):
    free = {"bytes": 10**12}
    store = make_store(tmp_path, min_free_bytes=10**9, disk_free=lambda: free["bytes"])
    fill(store, 0, 40)
    size = store.stats()["file_bytes"]

    free["bytes"] = 10**9
    assert fill(store, 40, 40) == 40
    stats = store.stats()
    assert stats["rolled"] > 0 and stats["free_bytes"] == 10**9
    assert stats["file_bytes"] <= size + 4096
    assert store.records()[0]["track_id"] == "261114-00004F"


def test_a_record_that_would_grow_the_file_on_a_full_disk_is_skipped(tmp_path):
    history = history_with_track("261114-000001")
    rng = random.Random(3)
    for i in range(4_000):
        near = {"delay": 10.0 + rng.uniform(-4, 4), "doppler": 350.0 + rng.uniform(-90, 90), "snr": rng.uniform(4, 9)}
        history.write_detections(5_000 + i, [], [near], [])
    store = make_store(tmp_path, history, min_free_bytes=10**9, disk_free=lambda: 10**6)
    store.retire(FakeTrack("261114-000001"), "run")
    assert store.flush(force=True) == 0
    assert store.records() == [] and store.stats()["skipped"] == 1


def test_free_space_that_cannot_be_read_leaves_only_the_size_cap(tmp_path):
    store = make_store(tmp_path, disk_free=lambda: None)
    store.retire(FakeTrack("261114-000001"), "run")
    assert store.flush(force=True) == 1


def test_a_full_queue_drops_and_counts_rather_than_growing(tmp_path, monkeypatch):
    monkeypatch.setattr(outside, "MAX_PENDING", 2)
    store = make_store(tmp_path)
    for i in range(3):
        store.retire(FakeTrack(f"261114-{i:06X}"), "run")
    stats = store.stats()
    assert (stats["pending"], stats["dropped"]) == (2, 1)


def history_with_track(track_id, start_ms=5_000, end_ms=10_000, delay=10.0, delay_rate=0.0, doppler=350.0):
    """A history holding one straight track, one point every 500 ms."""
    history = DetectionHistory()
    dets = [
        {"timestamp": t, "delay": delay + delay_rate * (t - start_ms) / 1000, "doppler": doppler, "snr": 12.0}
        for t in range(start_ms, end_ms + 1, 500)
    ]
    history.write_event(track_id, end_ms, len(dets), dets)
    return history


def stored_neighbours(store, track):
    store.retire(track, "run")
    store.flush(force=True)
    return store.record(store.records()[0]["id"])["neighbours"]


def test_only_detections_near_the_track_at_the_time_are_kept(tmp_path):
    history = history_with_track("261114-000001")
    history.write_detections(
        7_500,
        [],
        [
            {"delay": 14.9, "doppler": 449.0, "snr": 6.0},
            {"delay": 15.2, "doppler": 350.0, "snr": 6.0},
            {"delay": 10.0, "doppler": 249.0, "snr": 6.0},
            {"delay": 10.0, "doppler": -350.0, "snr": 6.0},
        ],
        [],
    )
    neighbours = stored_neighbours(make_store(tmp_path, history), FakeTrack("261114-000001"))
    kept = neighbours["detections"]["unassociated"]
    assert list(zip(kept["delay"], kept["doppler"])) == [(14.9, 449.0)]
    assert (neighbours["in_window"], neighbours["total"]) == (4, 1)
    assert (neighbours["delay_km"], neighbours["doppler_hz"]) == (NEAR_DELAY_KM, NEAR_DOPPLER_HZ)


def test_before_and_after_the_track_the_tube_follows_its_line(tmp_path):
    history = history_with_track("261114-000001", delay_rate=0.5)
    before = {"delay": 10.0 - 0.5 * 4 + 4.5, "doppler": 350.0, "snr": 6.0}
    left_behind = {"delay": 10.0 + 4.0, "doppler": 350.0, "snr": 6.0}
    history.write_detections(1_000, [], [before, left_behind], [])
    after = {"delay": 12.5 + 0.5 * 6 - 4.5, "doppler": 350.0, "snr": 6.0}
    history.write_detections(16_000, [], [after], [])
    neighbours = stored_neighbours(make_store(tmp_path, history), FakeTrack("261114-000001"))
    kept = neighbours["detections"]["unassociated"]
    assert list(zip(kept["t"], kept["delay"])) == [(1_000, before["delay"]), (16_000, after["delay"])]


def test_other_tracks_keep_only_their_points_near_the_track(tmp_path):
    history = history_with_track("261114-000001")
    near = [{"timestamp": 6_000, "delay": 11.0, "doppler": 330.0, "snr": 9.0}]
    far = [{"timestamp": 6_500, "delay": 40.0, "doppler": 330.0, "snr": 9.0}]
    history.write_event("261114-000002", 6_500, 2, near + far)
    history.write_event("261114-000003", 6_500, 1, [{"timestamp": 6_500, "delay": 10.0, "doppler": -80.0, "snr": 9.0}])
    neighbours = stored_neighbours(make_store(tmp_path, history), FakeTrack("261114-000001"))
    assert list(neighbours["tracks"]) == ["261114-000002"]
    assert neighbours["tracks"]["261114-000002"]["t"] == [6_000]


def test_a_track_the_history_no_longer_holds_keeps_no_neighbours(tmp_path):
    history = DetectionHistory()
    history.write_detections(7_500, [], [clutter(1)], [])
    neighbours = stored_neighbours(make_store(tmp_path, history), FakeTrack("261114-000001"))
    assert (neighbours["in_window"], neighbours["total"]) == (1, 0)


def test_neighbours_are_thinned_to_the_record_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(outside, "MAX_RECORD_POINTS", 10)
    history = history_with_track("261114-000001")
    for i in range(100):
        history.write_detections(5_000 + i * 50, [], [{"delay": 10.5, "doppler": 360.0, "snr": 6.0}], [])
    store = make_store(tmp_path, history)
    store.retire(FakeTrack("261114-000001", last_ms=10_000), "run")
    store.flush(force=True)
    neighbours = store.record(store.records()[0]["id"])["neighbours"]
    assert (neighbours["total"], neighbours["stride"]) == (100, 10)
    assert len(neighbours["detections"]["unassociated"]["t"]) == 10


# ── Safety ─────────────────────────────────────────────────────


def test_records_survive_reopening_and_ids_are_never_reused(tmp_path):
    store = make_store(tmp_path)
    fill(store, 1, 2)
    store.close()
    with sqlite3.connect(tmp_path / "outside.sqlite") as db:
        db.execute("DELETE FROM records WHERE id = 2")

    reopened = make_store(tmp_path)
    assert [r["id"] for r in reopened.records()] == [1]
    fill(reopened, 3, 1)
    assert [r["id"] for r in reopened.records()] == [3, 1]


FIXTURES = Path(__file__).parent / "fixtures"


def test_a_schema_1_store_from_the_first_release_opens_with_its_records(tmp_path):
    """Every later release must pass this: the store lives on across upgrades, migrated in place, never set aside."""
    with sqlite3.connect(tmp_path / "outside.sqlite") as db:
        db.executescript((FIXTURES / "outside_store_schema1.sql").read_text())
    store = make_store(tmp_path)
    assert store.state == "open" and not os.path.exists(tmp_path / "outside.sqlite.bad")
    assert [(r["id"], r["track_id"]) for r in store.records()] == [(2, "261114-000002"), (1, "261114-000001")]
    first = store.record(1)
    assert first["schema"] == 1 and first["n_points"] == len(first["track"]["t"]) == 11
    assert first["neighbours"]["total"] == 1
    fill(store, 3, 1)
    assert [r["id"] for r in store.records()] == [3, 2, 1]


def test_a_store_from_a_newer_schema_is_left_untouched(tmp_path):
    path = tmp_path / "outside.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=99")
        db.execute("CREATE TABLE future (x)")
    before = path.read_bytes()
    store = make_store(tmp_path)
    assert store.state == "disabled" and "newer" in store.error
    assert path.read_bytes() == before and not os.path.exists(tmp_path / "outside.sqlite.bad")


def test_constructing_the_store_touches_no_disk_and_retirements_queue_while_it_opens(tmp_path):
    store = OutsideStore(str(tmp_path / "outside.sqlite"), DetectionHistory())
    assert not os.path.exists(tmp_path / "outside.sqlite") and store.state == "opening"
    store.retire(FakeTrack("261114-000001"), "run")
    assert store.stats()["pending"] == 1
    assert store.flush(force=True) == 1
    assert store.state == "open" and os.path.exists(tmp_path / "outside.sqlite")


def test_a_corrupt_file_is_set_aside_and_a_fresh_store_started(tmp_path):
    path = tmp_path / "outside.sqlite"
    path.write_bytes(b"not a database, just bytes " * 100)
    store = make_store(tmp_path)
    assert store.state == "open"
    assert (tmp_path / "outside.sqlite.bad").read_bytes().startswith(b"not a database")
    store.retire(FakeTrack("261114-000001"), "run")
    assert store.flush(force=True) == 1


def test_a_path_that_cannot_be_opened_disables_the_store_without_raising(tmp_path):
    store = OutsideStore(str(tmp_path / "missing" / "outside.sqlite"), DetectionHistory())
    store.retire(FakeTrack("261114-000001"), "run")
    assert store.flush(force=True) == 0
    stats = store.stats()
    assert stats["state"] == "disabled" and stats["error"] and stats["dropped"] == 1
    store.retire(FakeTrack("261114-000002"), "run")
    assert store.stats()["dropped"] == 2
    assert not os.path.exists(tmp_path / "missing")


def test_a_failing_write_is_counted_and_never_raises(tmp_path, monkeypatch):
    store = make_store(tmp_path)

    def broken(record):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store, "_write", broken)
    store.retire(FakeTrack("261114-000001"), "run")
    assert store.flush(force=True) == 0
    assert store.stats()["failed"] == 1


def test_a_track_cleared_from_the_history_is_stored_as_incomplete(tmp_path):
    tracker, history, store = make_pipeline(tmp_path)
    fly(tracker, 0, 20, FAST, after=12)
    history.clear()
    store.flush(force=True)
    [summary] = store.records()
    assert summary["complete"] is False and summary["n_points"] == 0


def test_the_history_window_is_inclusive_and_leaves_out_the_track_asked_about():
    history = DetectionHistory()
    for t in (1_000, 2_000, 3_000):
        history.write_detections(t, [], [clutter(t)], [])
        for tid in ("a", "b"):
            history.write_event(tid, t, 1, [{"timestamp": t, "delay": 1.0, "doppler": 1.0, "snr": 1.0}])
    window = history.window(1_000, 2_000, exclude="a")
    assert window["detections"]["unassociated"]["t"] == [1_000, 2_000]
    assert list(window["tracks"]) == ["b"] and window["tracks"]["b"]["t"] == [1_000, 2_000]
    assert history.released_ms() == 3_000


# ── Reading ────────────────────────────────────────────────────


@pytest.fixture
def served(tmp_path):
    store = make_store(tmp_path)
    for i in range(1, 4):
        store.retire(FakeTrack(f"261114-{i:06X}"), "run")
    store.flush(force=True)
    server = start_control_server(Tracker(config=get_config()), threading.Lock(), port=0, outside=store)
    try:
        yield f"http://127.0.0.1:{server.port}"
    finally:
        server.shutdown()
        server.server_close()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def test_the_list_is_newest_first_and_pages_by_id(served):
    status, body = get(served + "/outside?limit=2")
    assert status == 200 and body["limit_hz"] == OUTSIDE_DOPPLER_HZ and body["schema"] == 1
    assert [r["id"] for r in body["records"]] == [3, 2]
    _, body = get(served + "/outside?before=2")
    assert [r["id"] for r in body["records"]] == [1]


def test_one_record_is_served_whole(served):
    status, body = get(served + "/outside/2")
    assert status == 200 and body["id"] == 2 and body["track_id"] == "261114-000002"
    assert "neighbours" in body and "radar" in body


@pytest.mark.parametrize(
    ("path", "status"),
    [("/outside/99", 404), ("/outside/abc", 400), ("/outside?before=x", 400), ("/outside?limit=0", 400)],
)
def test_bad_or_missing_records_are_answered_not_raised(served, path, status):
    assert get(served + path)[0] == status


def test_health_reports_the_store(served):
    _, body = get(served + "/health")
    assert body["outside"]["records"] == 3 and body["outside"]["state"] == "open"


def test_a_store_still_opening_is_a_503(tmp_path):
    store = OutsideStore(str(tmp_path / "outside.sqlite"), DetectionHistory())
    server = start_control_server(Tracker(config=get_config()), threading.Lock(), port=0, outside=store)
    try:
        status, body = get(f"http://127.0.0.1:{server.port}/outside")
        assert status == 503 and body["error"] == "outside store opening"
    finally:
        server.shutdown()
        server.server_close()


def test_no_store_is_a_503():
    server = start_control_server(Tracker(config=get_config()), threading.Lock(), port=0)
    try:
        assert get(f"http://127.0.0.1:{server.port}/outside")[0] == 503
    finally:
        server.shutdown()
        server.server_close()
