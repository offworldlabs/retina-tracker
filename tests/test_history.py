"""The rolling record: what it holds, what it costs, and what it serves.

Two things here are load-bearing beyond the obvious. Memory is bounded twice,
by the window and by a hard point ceiling, because a window alone makes the
footprint a function of how busy the sky is and a node cannot promise that.
And a consumer's position is a monotonic count rather than an index, because
pruning drops from the front.
"""

import json

import pytest

from retina_tracker.history import CLASSES, DetectionHistory

BASE = 1789030000000


def det(delay=10.0, doppler=50.0, snr=15.0):
    return {"delay": delay, "doppler": doppler, "snr": snr}


def track_det(ts, delay=10.0, doppler=50.0, snr=15.0):
    return {"timestamp": ts, "delay": delay, "doppler": doppler, "snr": snr}


@pytest.fixture
def history():
    return DetectionHistory()


# ── what it holds ───────────────────────────────────────────────────────────

def test_detections_land_in_the_class_they_were_given(history):
    history.write_detections(BASE, [det(1.0)], [det(2.0)], [det(3.0)])
    payload, _ = history.snapshot()

    assert payload["detections"]["associated"]["delay"] == [1.0]
    assert payload["detections"]["unassociated"]["delay"] == [2.0]
    assert payload["detections"]["below_snr"]["delay"] == [3.0]


def test_every_class_is_present_even_when_empty(history):
    """A consumer should not have to special-case a quiet node."""
    payload, _ = history.snapshot()
    for name in CLASSES:
        assert payload["detections"][name] == {"t": [], "delay": [], "doppler": [], "snr": []}


def test_track_points_and_metadata_are_kept(history):
    history.write_event("T1", BASE, 3, [track_det(BASE), track_det(BASE + 500)],
                        adsb_hex="4CA2D1", is_anomalous=True,
                        anomaly_types=["sustained_orbit"], max_velocity_ms=231.55,
                        shadow_fraction=0.6432)
    payload, _ = history.snapshot()

    assert payload["tracks"]["T1"]["t"] == [BASE, BASE + 500]
    assert payload["tracks"]["T1"]["meta"] == {
        "adsb_hex": "4CA2D1", "length": 3, "max_velocity_ms": 231.6,
        "is_anomalous": True, "anomaly_types": ["sustained_orbit"],
        "shadow_fraction": 0.643,
    }


def test_adsb_initialized_is_the_one_field_dropped_on_purpose(history):
    history.write_event("T1", BASE, 1, [track_det(BASE)],
                        adsb_initialized=True, something_new=7)
    payload, _ = history.snapshot()
    assert "adsb_initialized" not in payload["tracks"]["T1"]["meta"]
    assert "something_new" not in payload["tracks"]["T1"]["meta"]


def test_a_track_never_goes_backwards(history):
    """Every read relies on points being in timestamp order."""
    history.write_event("T1", BASE, 2, [track_det(BASE), track_det(BASE + 500)])
    history.write_event("T1", BASE, 2, [track_det(BASE), track_det(BASE + 500),
                                        track_det(BASE + 1000)])
    payload, _ = history.snapshot()
    assert payload["tracks"]["T1"]["t"] == [BASE, BASE + 500, BASE + 1000]


# ── the wire form ───────────────────────────────────────────────────────────

def test_values_are_rounded_to_the_precision_they_have(history):
    """float32 storage reads 16.1 back as 16.100000381469727, which is 18
    bytes on the wire for four bytes of meaning."""
    history.write_detections(BASE, [det(16.1, -45.678912, 12.3456)], [], [])
    payload, _ = history.snapshot()
    assoc = payload["detections"]["associated"]

    assert assoc["delay"] == [16.1]
    assert assoc["doppler"] == [-45.68]
    assert assoc["snr"] == [12.3]


def test_the_payload_is_json_serialisable(history):
    """array.array is not, so the slice has to materialise lists."""
    history.write_detections(BASE, [det()], [det()], [det()])
    history.write_event("T1", BASE, 1, [track_det(BASE)])
    payload, _ = history.snapshot()
    json.dumps(payload)  # raises if not


def test_rounding_keeps_the_payload_compact(history):
    """The reason rounding happens here rather than being left to the
    consumer: a float32 read back unrounded serialises to 17 significant
    digits, and there are three of them per point."""
    for i in range(200):
        history.write_detections(BASE + i, [det(16.1 + i * 0.01, -45.6, 12.3)], [], [])
    payload, _ = history.snapshot()
    rounded = payload["detections"]["associated"]

    cols = history._det["associated"]
    raw = {k: list(cols[k]) for k in cols}

    encode = lambda obj: len(json.dumps(obj, separators=(",", ":")))  # noqa: E731
    assert encode(rounded) < encode(raw) / 2, (encode(rounded), encode(raw))


# ── the view window ─────────────────────────────────────────────────────────

def test_a_window_narrows_what_is_served_without_touching_what_is_held(history):
    history.write_detections(BASE, [det(1.0)], [], [])
    history.write_detections(BASE + 100_000, [det(2.0)], [], [])
    history.write_event("OLD", BASE, 1, [track_det(BASE)])
    history.write_event("NEW", BASE + 100_000, 1, [track_det(BASE + 100_000)])

    payload, _ = history.snapshot(window_s=60, now_ms=BASE + 100_000)

    assert payload["detections"]["associated"]["delay"] == [2.0]
    assert list(payload["tracks"]) == ["NEW"]
    # Retention untouched: the window is a display concern.
    assert len(history._det["associated"]["t"]) == 2
    assert set(history._tracks) == {"OLD", "NEW"}


def test_the_cursor_covers_everything_not_just_the_window(history):
    """Otherwise a windowed consumer's first delta would re-send the history
    the window had just excluded."""
    history.write_detections(BASE, [det(1.0)], [], [])
    history.write_detections(BASE + 100_000, [det(2.0)], [], [])

    payload, cursor = history.snapshot(window_s=60, now_ms=BASE + 100_000)

    assert payload["detections"]["associated"]["delay"] == [2.0]
    assert cursor["detections"]["associated"] == 2

    delta, _ = history.since(cursor)
    assert delta["detections"]["associated"]["t"] == []


# ── deltas ──────────────────────────────────────────────────────────────────

def test_since_returns_only_what_was_appended(history):
    history.write_detections(BASE, [det(1.0)], [], [])
    _, cursor = history.snapshot()

    history.write_detections(BASE + 500, [det(2.0)], [det(3.0)], [])
    delta, cursor2 = history.since(cursor)

    assert delta["detections"]["associated"]["delay"] == [2.0]
    assert delta["detections"]["unassociated"]["delay"] == [3.0]
    assert cursor2["detections"]["associated"] == 2


def test_since_is_empty_when_nothing_changed(history):
    history.write_detections(BASE, [det()], [], [])
    _, cursor = history.snapshot()

    delta, _ = history.since(cursor)

    assert all(delta["detections"][name]["t"] == [] for name in CLASSES)
    assert delta["tracks"] == {}


def test_since_sends_a_track_the_cursor_has_never_seen_whole(history):
    """A track promoted after the consumer connected arrives with the
    history the tracker backfilled, not truncated at the join."""
    _, cursor = history.snapshot()
    history.write_event("T1", BASE, 3, [track_det(BASE), track_det(BASE + 500),
                                        track_det(BASE + 1000)])

    delta, _ = history.since(cursor)
    assert delta["tracks"]["T1"]["t"] == [BASE, BASE + 500, BASE + 1000]


def test_since_omits_tracks_with_no_new_points(history):
    history.write_event("T1", BASE, 1, [track_det(BASE)])
    _, cursor = history.snapshot()

    history.write_event("T2", BASE + 500, 1, [track_det(BASE + 500)])
    delta, _ = history.since(cursor)

    assert list(delta["tracks"]) == ["T2"]


def test_since_survives_a_prune_that_dropped_unseen_points(history):
    """A position is a monotonic count, not an index, because pruning drops
    from the front."""
    h = DetectionHistory(window_s=10)
    h.write_detections(BASE, [det(1.0)], [], [])
    _, cursor = h.snapshot()

    h.write_detections(BASE + 50_000, [det(2.0)], [], [])
    h.prune(now_ms=BASE + 51_000)

    delta, _ = h.since(cursor)
    assert delta["detections"]["associated"]["delay"] == [2.0]


def test_since_refuses_a_cursor_from_before_a_clear(history):
    history.write_detections(BASE, [det()], [], [])
    _, cursor = history.snapshot()

    history.clear()

    delta, new_cursor = history.since(cursor)
    assert delta is None and new_cursor is None


def test_a_fresh_snapshot_works_again_after_a_clear(history):
    history.write_detections(BASE, [det()], [], [])
    history.clear()

    _, cursor = history.snapshot()
    history.write_detections(BASE + 500, [det(9.0)], [], [])

    delta, _ = history.since(cursor)
    assert delta["detections"]["associated"]["delay"] == [9.0]


# ── bounds ──────────────────────────────────────────────────────────────────

def test_pruning_drops_what_is_older_than_the_window():
    h = DetectionHistory(window_s=10)
    h.write_detections(BASE, [det(1.0)], [], [])
    h.write_detections(BASE + 50_000, [det(2.0)], [], [])
    h.write_event("OLD", BASE, 1, [track_det(BASE)])
    h.write_event("NEW", BASE + 50_000, 1, [track_det(BASE + 50_000)])

    h.prune(now_ms=BASE + 51_000)

    payload, _ = h.snapshot()
    assert payload["detections"]["associated"]["delay"] == [2.0]
    assert list(payload["tracks"]) == ["NEW"]


def test_the_point_ceiling_bounds_memory_whatever_the_rate():
    """A window alone makes the footprint a function of how busy the sky is.
    This is the promise underneath it."""
    h = DetectionHistory(window_s=4 * 3600, max_points=100)
    for i in range(1000):
        h.write_detections(BASE + i, [det(float(i))], [], [])

    assert len(h._det["associated"]["t"]) == 100
    # The newest are kept, the oldest dropped.
    payload, _ = h.snapshot()
    assert payload["detections"]["associated"]["delay"][-1] == 999.0


def test_the_ceiling_does_not_break_a_cursor():
    h = DetectionHistory(max_points=100)
    h.write_detections(BASE, [det(1.0)], [], [])
    _, cursor = h.snapshot()

    for i in range(1, 500):
        h.write_detections(BASE + i, [det(float(i))], [], [])

    delta, _ = h.since(cursor)
    # Everything still held that the cursor had not seen, and nothing twice.
    assert delta["detections"]["associated"]["delay"] == [float(i) for i in range(400, 500)]


def test_tracks_are_bounded_too():
    h = DetectionHistory(max_tracks=5)
    for i in range(20):
        h.write_event(f"T{i}", BASE + i * 1000, 1, [track_det(BASE + i * 1000)])

    assert len(h._tracks) == 5


def test_track_eviction_takes_the_stalest_first():
    h = DetectionHistory(max_tracks=2)
    h.write_event("OLD", BASE, 1, [track_det(BASE)])
    h.write_event("LIVE", BASE + 10_000, 1, [track_det(BASE + 10_000)])
    h.write_event("LIVE", BASE + 20_000, 2, [track_det(BASE + 20_000)])

    h.write_event("NEW", BASE + 30_000, 1, [track_det(BASE + 30_000)])

    assert "OLD" not in h._tracks
    assert set(h._tracks) == {"LIVE", "NEW"}


def test_stats_reports_what_is_held_and_what_it_costs(history):
    for i in range(100):
        history.write_detections(BASE + i, [det()], [det()], [det()])
    history.write_event("T1", BASE, 1, [track_det(BASE)])

    stats = history.stats()
    assert stats["detections"]["associated"] == 100
    assert stats["tracks"] == 1
    assert stats["points"] == 301
    assert stats["approx_bytes"] == 301 * 20


def test_the_footprint_is_what_it_claims():
    """20 bytes a point, against the ~192 a list of 4-float tuples costs."""
    import sys

    h = DetectionHistory()
    n = 20_000
    for i in range(n):
        h.write_detections(BASE + i, [det()], [], [])

    cols = h._det["associated"]
    actual = sum(sys.getsizeof(cols[k]) for k in cols)
    per_point = actual / n
    assert per_point < 32, per_point
    assert h.stats()["approx_bytes"] == n * 20
