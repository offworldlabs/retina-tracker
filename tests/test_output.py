"""Tests for size-bounded rotation of the streaming JSONL events file.

The events file is written continuously for the whole life of the process and is
only ever truncated by a restart, so without a size bound it grows until the node
runs out of disk. Rotation has to stay compatible with the only consumer, which
tails the live path from EOF and re-reads from offset 0 whenever the file shrinks.
"""

import json
import os

from retina_tracker.output import TrackEventWriter


def make_detections(n=20):
    return [
        {"timestamp": 1718747745000 + i, "delay": 16.103456, "doppler": 134.5012, "snr": 16.23, "adsb": None}
        for i in range(n)
    ]


def write_events(writer, count, start=0):
    for i in range(start, start + count):
        writer.write_event(f"250618-{i:06d}", 1718747750000 + i, 20, make_detections())


def read_ids(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line)["track_id"] for line in f if line.strip()]


def event_size():
    return len(
        json.dumps(
            {
                "track_id": "250618-000000",
                "adsb_hex": None,
                "adsb_initialized": False,
                "timestamp": 1718747750000,
                "length": 20,
                "detections": make_detections(),
                "is_anomalous": False,
                "max_velocity_ms": 0.0,
                "anomaly_types": [],
            }
        )
        + "\n"
    )


def test_rotates_at_size_bound(tmp_path):
    path = tmp_path / "events.jsonl"
    max_bytes = event_size() * 3
    writer = TrackEventWriter(str(path), max_bytes=max_bytes, backup_count=1)

    write_events(writer, 3)
    assert not os.path.exists(f"{path}.1")

    write_events(writer, 1, start=3)
    writer.close()

    assert os.path.exists(f"{path}.1")
    assert read_ids(f"{path}.1") == ["250618-000000", "250618-000001", "250618-000002"]
    assert read_ids(path) == ["250618-000003"]


def test_no_events_lost_across_rotation(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = TrackEventWriter(str(path), max_bytes=event_size() * 2, backup_count=1)

    write_events(writer, 4)
    writer.close()

    assert read_ids(f"{path}.1") + read_ids(path) == [f"250618-{i:06d}" for i in range(4)]


def test_footprint_stays_bounded(tmp_path):
    path = tmp_path / "events.jsonl"
    max_bytes = event_size() * 2
    writer = TrackEventWriter(str(path), max_bytes=max_bytes, backup_count=1)

    write_events(writer, 200)
    writer.close()

    segments = [p for p in tmp_path.iterdir() if p.name.startswith("events.jsonl")]
    assert len(segments) == 2
    assert sum(p.stat().st_size for p in segments) <= max_bytes * 2


def test_keeps_backup_count_segments(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = TrackEventWriter(str(path), max_bytes=event_size(), backup_count=2)

    write_events(writer, 5)
    writer.close()

    assert read_ids(f"{path}.2") == ["250618-000002"]
    assert read_ids(f"{path}.1") == ["250618-000003"]
    assert read_ids(path) == ["250618-000004"]
    assert not os.path.exists(f"{path}.3")


def test_live_path_shrinks_so_tailers_reset(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = TrackEventWriter(str(path), max_bytes=event_size() * 3, backup_count=1)

    write_events(writer, 3)
    offset = os.path.getsize(path)

    write_events(writer, 1, start=3)
    writer.close()

    assert os.path.getsize(path) < offset


def test_rotation_disabled_when_max_bytes_zero(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = TrackEventWriter(str(path), max_bytes=0, backup_count=1)

    write_events(writer, 20)
    writer.close()

    assert not os.path.exists(f"{path}.1")
    assert len(read_ids(path)) == 20


def test_stdout_is_never_rotated(capsys):
    writer = TrackEventWriter("-", max_bytes=1, backup_count=1)

    write_events(writer, 3)
    writer.close()

    assert len(capsys.readouterr().out.strip().split("\n")) == 3
