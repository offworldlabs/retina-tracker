"""The events file carries only detections it has not already written.

Each track event used to repeat the track's whole rolling window
(Track.get_recent_detections, up to detection_window points) to communicate
the one point that was new, multiplying the file by roughly the window size.

The reason that was safe to change is that neither consumer reads an event as
"the track's state now". live_score.load_tracks unions detections by
timestamp across every event mentioning a track, and retina-gui's buffer
appends only timestamps it has not seen. Both reconstruct the same history
from a delta stream as from a repeating one, which is what these pin down.
"""

import json

import pytest

from retina_tracker.live_score import load_tracks
from retina_tracker.output import EMITTED_MEMORY, TrackEventWriter

BASE_TS = 1718747745000


def detection(i):
    return {"timestamp": BASE_TS + i * 500, "delay": 16.1 + i * 0.01,
            "doppler": 134.5 - i * 0.2, "snr": 16.2, "adsb": None}


def rolling_window(upto, size=20):
    """What get_recent_detections(n=size) returns after `upto` points."""
    start = max(0, upto - size)
    return [detection(i) for i in range(start, upto)]


def read_events(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def emit_track_life(writer, track_id, points, window=20, **meta):
    """One event per point, each carrying the rolling window, as the tracker
    does for a track that is associated on every frame."""
    for n in range(1, points + 1):
        writer.write_event(track_id, BASE_TS + n * 500, n,
                           rolling_window(n, window), **meta)


def test_the_first_event_for_a_track_carries_everything_it_has(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = TrackEventWriter(str(path), max_bytes=0)
    writer.write_event("T1", BASE_TS, 3, rolling_window(3))
    writer.close()

    assert [d["timestamp"] for d in read_events(path)[0]["detections"]] == \
        [BASE_TS, BASE_TS + 500, BASE_TS + 1000]


def test_later_events_carry_only_what_is_new(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = TrackEventWriter(str(path), max_bytes=0)
    emit_track_life(writer, "T1", 5)
    writer.close()

    events = read_events(path)
    counts = [len(e["detections"]) for e in events]
    assert counts == [1, 1, 1, 1, 1], counts
    # ...and between them they still describe every point, in order.
    seen = [d["timestamp"] for e in events for d in e["detections"]]
    assert seen == [BASE_TS + i * 500 for i in range(5)]


def test_an_event_is_still_written_when_nothing_is_new(tmp_path):
    """length, the anomaly flags and shadow_fraction move over a track's
    life. A consumer that missed those updates would hold a stale opinion of
    a live track, so the event goes out with an empty detections list."""
    path = tmp_path / "events.jsonl"
    writer = TrackEventWriter(str(path), max_bytes=0)
    writer.write_event("T1", BASE_TS, 1, rolling_window(1), is_anomalous=False)
    writer.write_event("T1", BASE_TS + 10, 1, rolling_window(1),
                       is_anomalous=True, anomaly_types=["sustained_orbit"])
    writer.close()

    events = read_events(path)
    assert len(events) == 2
    assert events[1]["detections"] == []
    assert events[1]["is_anomalous"] is True
    assert events[1]["anomaly_types"] == ["sustained_orbit"]


def test_tracks_are_independent_of_each_other(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = TrackEventWriter(str(path), max_bytes=0)
    writer.write_event("T1", BASE_TS, 3, rolling_window(3))
    writer.write_event("T2", BASE_TS, 3, rolling_window(3))
    writer.close()

    events = read_events(path)
    assert len(events[0]["detections"]) == 3
    assert len(events[1]["detections"]) == 3, "T2 was filtered against T1's history"


def test_live_score_reconstructs_the_same_history(tmp_path):
    """The compatibility case. A delta stream and a repeating stream must
    collapse to identical per-track detections, because load_tracks unions
    by timestamp rather than trusting any single event."""
    delta_path = tmp_path / "delta.jsonl"
    writer = TrackEventWriter(str(delta_path), max_bytes=0)
    emit_track_life(writer, "T1", 50, adsb_hex="4CA2D1", shadow_fraction=0.25)
    writer.close()

    # The same life, written the old way: every event repeating its window.
    repeat_path = tmp_path / "repeat.jsonl"
    with open(repeat_path, "w") as f:
        for n in range(1, 51):
            f.write(json.dumps({
                "track_id": "T1", "adsb_hex": "4CA2D1", "adsb_initialized": False,
                "timestamp": BASE_TS + n * 500, "length": n,
                "detections": rolling_window(n), "is_anomalous": False,
                "max_velocity_ms": 0.0, "anomaly_types": [], "shadow_fraction": 0.25,
            }) + "\n")

    delta = load_tracks(str(delta_path))
    repeat = load_tracks(str(repeat_path))

    assert delta == repeat
    assert len(delta["T1"]["detections"]) == 50
    assert delta["T1"]["adsb_hex"] == "4CA2D1"
    assert delta["T1"]["shadow_fraction"] == 0.25


def test_a_retina_gui_style_union_also_reconstructs(tmp_path):
    """retina-gui appends only timestamps it has not seen, which is the same
    reconstruction by a different route."""
    path = tmp_path / "events.jsonl"
    writer = TrackEventWriter(str(path), max_bytes=0)
    emit_track_life(writer, "T1", 30)
    writer.close()

    held = []
    last_seen = None
    for event in read_events(path):
        for det in event["detections"]:
            if last_seen is not None and det["timestamp"] <= last_seen:
                continue
            held.append(det["timestamp"])
            last_seen = det["timestamp"]

    assert held == [BASE_TS + i * 500 for i in range(30)]


def test_the_file_is_dramatically_smaller(tmp_path):
    """The whole point. A 50-point track through a 20-point window."""
    delta_path = tmp_path / "delta.jsonl"
    writer = TrackEventWriter(str(delta_path), max_bytes=0)
    emit_track_life(writer, "T1", 50)
    writer.close()

    repeat_bytes = sum(
        len(json.dumps({"track_id": "T1", "detections": rolling_window(n)}) + "\n")
        for n in range(1, 51)
    )
    delta_bytes = delta_path.stat().st_size
    # Conservative: the delta file still carries per-event metadata the
    # comparison above leaves out, and still wins by a wide margin.
    assert delta_bytes < repeat_bytes / 3, (delta_bytes, repeat_bytes)


def test_the_high_water_map_is_bounded(tmp_path):
    """Track ids are unique for the life of a run, so remembering every one
    would be an unbounded dict on a process that runs for weeks."""
    path = tmp_path / "events.jsonl"
    writer = TrackEventWriter(str(path), max_bytes=0)
    for i in range(EMITTED_MEMORY + 50):
        writer.write_event(f"T{i}", BASE_TS, 1, rolling_window(1))
    writer.close()

    assert len(writer._emitted_through) == EMITTED_MEMORY


def test_an_evicted_track_repeats_rather_than_loses(tmp_path):
    """Eviction costs a repeated window, never a dropped detection. Both
    consumers dedupe, so a repeat is free and an omission would not be."""
    path = tmp_path / "events.jsonl"
    writer = TrackEventWriter(str(path), max_bytes=0)
    writer.write_event("OLD", BASE_TS, 2, rolling_window(2))
    for i in range(EMITTED_MEMORY + 10):
        writer.write_event(f"T{i}", BASE_TS, 1, rolling_window(1))
    writer.write_event("OLD", BASE_TS, 3, rolling_window(3))
    writer.close()

    revisit = [e for e in read_events(path) if e["track_id"] == "OLD"][-1]
    assert len(revisit["detections"]) == 3, "an evicted track must resend, not skip"


@pytest.mark.parametrize("window", [1, 5, 20])
def test_reconstruction_holds_at_any_window_size(tmp_path, window):
    path = tmp_path / "events.jsonl"
    writer = TrackEventWriter(str(path), max_bytes=0)
    emit_track_life(writer, "T1", 25, window=window)
    writer.close()

    tracks = load_tracks(str(path))
    assert [d["timestamp"] for d in tracks["T1"]["detections"]] == \
        [BASE_TS + i * 500 for i in range(25)]
