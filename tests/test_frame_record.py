"""Tests for the per-frame track record served on GET /frame.

events.jsonl is written per association, so a coasting track emits nothing
there and a deletion is silent. retina-telemetry has to say, with every
detection frame it sends, which confirmed tracks are alive after that frame and
which detection each took, and this record is the only place that answer lives.

What matters most is `hit`. The consumer holds the frame as blah2-api sent it,
so the index has to point into those arrays, not into whatever the tracker
kept after rejecting and partitioning.
"""

import copy
import json
import re
import threading
import urllib.error
import urllib.request

import pytest

from retina_tracker.config import get_config, set_config
from retina_tracker.control import start_control_server
from retina_tracker.history import DetectionHistory, TeeEventWriter
from retina_tracker.output import InnovationWriter, TrackEventWriter
from retina_tracker.server import process_streaming_frame
from retina_tracker.tracker import MAX_FRAME_RECORDS, Tracker

T0 = 1_758_800_000_000
DT_MS = 500


def request(url, method="GET"):
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def ts(i):
    return T0 + i * DT_MS


def target_frame(i):
    """One aircraft, preceded in the arrays by a detection the tracker rejects
    (non-finite delay) and one it partitions out as below SNR."""
    return {
        "timestamp": ts(i),
        "delay": [float("nan"), 30.0, 10.0 + 0.05 * i],
        "doppler": [0.0, -100.0, 50.0],
        "snr": [20.0, 1.0, 15.0],
    }


def empty_frame(i):
    return {"timestamp": ts(i), "delay": [], "doppler": [], "snr": []}


def confirmed(tracker, frames=5):
    for i in range(frames):
        process_streaming_frame(tracker, target_frame(i))
    return tracker.frame_record(ts(frames - 1))


@pytest.fixture
def tracker():
    return Tracker(config=get_config())


def test_an_active_track_names_its_detection_in_the_arrays_as_received(tracker):
    record = confirmed(tracker)
    assert tracker.n_detections_rejected > 0
    [track] = record["tracks"]
    assert track["state"] == "active"
    assert track["hit"] == 2
    assert track["n_missed"] == 0
    assert track["born_timestamp"] == ts(0)
    assert record["timestamp"] == ts(4)
    assert record["run"] == tracker.run_id


def test_tentative_tracks_never_appear(tracker):
    process_streaming_frame(tracker, target_frame(0))
    assert tracker.tracks
    assert tracker.frame_record(ts(0))["tracks"] == []


def test_a_track_promoted_by_tracklet_while_associating_is_active_with_its_hit(tracker):
    for i in range(4):
        process_streaming_frame(tracker, target_frame(i))
        if tracker.frame_record(ts(i))["tracks"]:
            break
    [track] = tracker.frame_record(ts(i))["tracks"]
    assert track["state"] == "active"
    assert track["hit"] == 2


def test_a_track_promoted_on_a_frame_it_missed_is_coasting():
    """M-of-N promotes on the frame count, whether or not the track associated
    in that frame. Its internal state reads ACTIVE; the record must not."""
    config = copy.deepcopy(get_config())
    config.setdefault("tracklet", {})["max_time_span"] = 0.0
    set_config(config)
    tracker = Tracker(config=config)
    for i in range(4):
        process_streaming_frame(tracker, target_frame(i))
    assert tracker.frame_record(ts(3))["tracks"] == []
    for i in (4, 5):
        process_streaming_frame(tracker, empty_frame(i))
    assert tracker.frame_record(ts(4))["tracks"] == []
    [track] = tracker.frame_record(ts(5))["tracks"]
    assert track["state"] == "coasting"
    assert track["hit"] is None


def test_a_missed_frame_is_coasting_with_no_hit(tracker):
    confirmed(tracker)
    process_streaming_frame(tracker, empty_frame(5))
    [track] = tracker.frame_record(ts(5))["tracks"]
    assert track["state"] == "coasting"
    assert track["hit"] is None
    assert track["n_missed"] == 1


def test_a_confirmed_track_is_listed_deleted_exactly_once(tracker):
    track_id = confirmed(tracker)["tracks"][0]["id"]
    seen = []
    for i in range(5, 25):
        process_streaming_frame(tracker, empty_frame(i))
        seen.append([(t["id"], t["state"], t["hit"]) for t in tracker.frame_record(ts(i))["tracks"]])
    died = next(n for n, tracks in enumerate(seen) if tracks != [(track_id, "coasting", None)])
    assert seen[died] == [(track_id, "deleted", None)]
    assert all(tracks == [] for tracks in seen[died + 1 :])
    assert tracker.tracks == []


def test_a_detection_without_a_frame_index_gives_a_null_hit(tracker):
    for i in range(5):
        tracker.process_frame([{"delay": 10.0 + 0.05 * i, "doppler": 50.0, "snr": 15.0}], ts(i))
    [track] = tracker.frame_record(ts(4))["tracks"]
    assert track["state"] == "active"
    assert track["hit"] is None


def test_a_non_finite_timestamp_records_nothing(tracker):
    tracker.process_frame([], float("nan"))
    assert tracker.frame_record() is None


def test_a_repeated_timestamp_replaces_the_earlier_record(tracker):
    confirmed(tracker)
    process_streaming_frame(tracker, empty_frame(4))
    assert tracker.frame_record(ts(4))["tracks"][0]["state"] == "coasting"
    assert list(tracker.frame_records).count(ts(4)) == 1
    assert tracker.frame_record() is tracker.frame_record(ts(4))


def test_the_record_is_bounded(tracker):
    for i in range(MAX_FRAME_RECORDS + 10):
        process_streaming_frame(tracker, empty_frame(i))
    assert len(tracker.frame_records) == MAX_FRAME_RECORDS == 32
    assert tracker.frame_record(ts(9)) is None
    assert tracker.frame_record(ts(10)) is not None


def test_the_run_id_is_well_formed_and_distinct_per_tracker(tracker):
    assert re.fullmatch(r"[0-9A-Za-z._-]{1,64}", tracker.run_id)
    assert Tracker(config=get_config()).run_id != tracker.run_id


def test_frame_index_reaches_no_output(tmp_path):
    """The key exists only so the record can say where a detection sat. Every
    writer the live server runs picks its keys explicitly, and this holds
    them to it."""
    events_path = tmp_path / "events.jsonl"
    innovations_path = tmp_path / "innovations.jsonl"
    history = DetectionHistory()
    events = TrackEventWriter(str(events_path), max_bytes=0)
    innovations = InnovationWriter(str(innovations_path), max_bytes=0)
    tracker = Tracker(
        event_writer=TeeEventWriter(events, history),
        config=get_config(),
        detection_sink=history,
        innovation_writer=innovations,
    )
    for i in range(8):
        process_streaming_frame(tracker, target_frame(i))
    events.close()
    innovations.close()

    assert events_path.read_text().strip()
    assert innovations_path.read_text().strip()
    assert "frame_index" not in events_path.read_text()
    assert "frame_index" not in innovations_path.read_text()
    snapshot = history.snapshot()[0]
    assert snapshot["tracks"]
    assert "frame_index" not in json.dumps(snapshot)
    assert "frame_index" not in json.dumps(tracker.to_dict())


# ── Over HTTP ──────────────────────────────────────────────


@pytest.fixture
def served():
    tracker = Tracker(config=get_config())
    lock = threading.Lock()
    server = start_control_server(tracker, lock, host="127.0.0.1", port=0)
    try:
        yield tracker, f"http://127.0.0.1:{server.port}"
    finally:
        server.shutdown()
        server.server_close()


def not_held(tracker, latest):
    return 404, {"error": "frame not held", "run": tracker.run_id, "latest": latest}


def test_a_tracker_never_fed_says_nothing_is_held(served):
    """latest null: nothing will arrive, so a caller should not wait."""
    tracker, base = served
    assert request(base + "/frame") == not_held(tracker, None)
    assert request(base + f"/frame?timestamp={ts(0)}") == not_held(tracker, None)


def test_a_frame_not_yet_processed_names_an_older_latest(served):
    """latest older than asked: probably in flight, worth a short retry."""
    tracker, base = served
    confirmed(tracker)
    assert request(base + f"/frame?timestamp={ts(5)}") == not_held(tracker, ts(4))
    process_streaming_frame(tracker, empty_frame(5))
    status, body = request(base + f"/frame?timestamp={ts(5)}")
    assert (status, body["timestamp"]) == (200, ts(5))


def test_an_evicted_frame_names_a_newer_latest(served):
    """latest newer than asked: aged out, so give up now."""
    tracker, base = served
    last = MAX_FRAME_RECORDS + 4
    for i in range(last + 1):
        process_streaming_frame(tracker, empty_frame(i))
    assert request(base + f"/frame?timestamp={ts(0)}") == not_held(tracker, ts(last))


def test_a_frame_is_served_by_its_timestamp(served):
    tracker, base = served
    confirmed(tracker)
    process_streaming_frame(tracker, empty_frame(5))

    status, body = request(base + f"/frame?timestamp={ts(4)}")
    assert status == 200
    assert body["run"] == tracker.run_id
    assert body["timestamp"] == ts(4)
    [track] = body["tracks"]
    assert set(track) == {
        "id",
        "state",
        "hit",
        "n_associated",
        "n_missed",
        "adsb_hex",
        "is_anomalous",
        "anomaly_types",
        "max_velocity_ms",
        "born_timestamp",
        "avg_snr",
        "shadow_fraction",
        "interference_fraction",
    }
    assert (track["state"], track["hit"]) == ("active", 2)


def test_no_timestamp_serves_the_latest_frame(served):
    tracker, base = served
    confirmed(tracker)
    process_streaming_frame(tracker, empty_frame(5))
    status, body = request(base + "/frame/")
    assert status == 200
    assert body["timestamp"] == ts(5)
    assert body["tracks"][0]["state"] == "coasting"


def test_an_unheld_timestamp_is_404(served):
    tracker, base = served
    confirmed(tracker)
    assert request(base + f"/frame?timestamp={ts(99)}") == not_held(tracker, ts(4))


@pytest.mark.parametrize("raw", ["abc", "1.5", ""])
def test_an_unparseable_timestamp_is_400(served, raw):
    _tracker, base = served
    assert request(base + f"/frame?timestamp={raw}") == (400, {"error": "bad timestamp"})


def test_a_reset_clears_the_frames_but_keeps_the_run(served):
    tracker, base = served
    confirmed(tracker)
    _status, before = request(base + "/health")

    request(base + "/reset", method="POST")

    assert request(base + "/frame") == not_held(tracker, None)
    assert request(base + f"/frame?timestamp={ts(4)}") == not_held(tracker, None)
    _status, after = request(base + "/health")
    assert before["run"] == after["run"] == tracker.run_id
