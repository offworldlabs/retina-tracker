"""The ADS-B-proven coverage wedge and the delay arcs cut to it.

Every position is invented: a receiver at 10 N 20 E and a transmitter 28 km
from it on a bearing of 250 degrees.
"""

import json
import threading
import urllib.error
import urllib.request
from argparse import Namespace

import numpy as np
import pytest

from retina_tracker.cli import coverage_state_path
from retina_tracker.config import get_config, load_blah2_config
from retina_tracker.control import start_control_server
from retina_tracker.coverage import (
    ARC_MIN_DIFFERENTIAL_KM,
    Coverage,
    Geometry,
    bearing_deg,
    destination,
    ellipse_range_km,
    ground_range_km,
    replay_arcs,
    verified_position,
)
from retina_tracker.history import DetectionHistory
from retina_tracker.tracker import Tracker

RX = (10.0, 20.0)
TX_BEARING = 250.0
BASELINE_KM = 28.0
TX = tuple(float(v) for v in destination(*RX, TX_BEARING, BASELINE_KM))
GEOMETRY = Geometry(RX[0], RX[1], TX[0], TX[1])
NOW = 1_800_000_000_000
DAY_MS = 86_400_000


def at(bearing, range_km):
    lat, lon = destination(*RX, bearing, range_km)
    return float(lat), float(lon)


def detection(ts, hex_, bearing, range_km, age=1.0, delay=40.0):
    lat, lon = at(bearing, range_km)
    return {
        "timestamp": ts,
        "delay": delay,
        "doppler": 50.0,
        "snr": 15.0,
        "adsb": {"hex": hex_, "lat": lat, "lon": lon, "position_age": age},
    }


def fly(coverage, track_id, hex_, bearing, range_km, n=3, start=NOW - 60_000):
    detections = [detection(start + i * 1000, hex_, bearing, range_km) for i in range(n)]
    coverage.write_event(track_id, detections[-1]["timestamp"], n, detections, adsb_hex=hex_)


def fresh(**kwargs):
    return Coverage(GEOMETRY, bin_deg=5.0, hold_s=86400, min_pass_detections=3, now_ms=NOW - DAY_MS, **kwargs)


# ── what counts as a verified position ──────────────────────────────────────


def test_a_fresh_fix_on_the_tracks_own_aircraft_is_verified():
    det = detection(NOW, "abc123", 60.0, 30.0)
    assert verified_position(det, "abc123", max_age_s=10) == pytest.approx(at(60.0, 30.0))


@pytest.mark.parametrize(
    "change",
    [
        {"hex": "def456"},
        {"position_age": 11.0},
        {"position_age": None},
        {"lat": 91.0},
        {"lon": float("nan")},
    ],
)
def test_a_fix_that_cannot_stand_for_the_track_is_refused(change):
    det = detection(NOW, "abc123", 60.0, 30.0)
    det["adsb"].update(change)
    assert verified_position(det, "abc123", max_age_s=10) is None


def test_no_position_without_a_track_hex():
    assert verified_position(detection(NOW, "abc123", 60.0, 30.0), None) is None


# ── geometry ────────────────────────────────────────────────────────────────


def test_geometry_comes_from_blah2s_location_block():
    location = {
        "rx": {"latitude": RX[0], "longitude": RX[1], "altitude": 50},
        "tx": {"latitude": TX[0], "longitude": TX[1], "altitude": 200},
    }
    g = Geometry.from_location(location)
    assert g.baseline_km == pytest.approx(BASELINE_KM, abs=1e-6)
    assert g.tx_bearing_deg == pytest.approx(TX_BEARING, abs=1e-6)


@pytest.mark.parametrize(
    "location",
    [
        {},
        {"rx": {"latitude": None, "longitude": None}, "tx": {"latitude": None, "longitude": None}},
        {"rx": {"latitude": 0.0, "longitude": 0.0}, "tx": {"latitude": 10.0, "longitude": 20.0}},
        {"rx": {"latitude": "10", "longitude": 20.0}, "tx": {"latitude": 10.1, "longitude": 20.1}},
    ],
)
def test_an_unplaced_node_has_no_geometry(location):
    assert Geometry.from_location(location) is None


def test_the_ellipse_reaches_half_the_differential_plus_the_baseline_towards_the_transmitter():
    d = 40.0
    assert ellipse_range_km(d, BASELINE_KM, 0.0) == pytest.approx(d / 2 + BASELINE_KM)
    assert ellipse_range_km(d, BASELINE_KM, 180.0) == pytest.approx(d / 2)


def test_every_point_on_the_ellipse_has_the_differential_it_was_drawn_for():
    d = 40.0
    psi = np.arange(0.0, 360.0, 15.0)
    r = ellipse_range_km(d, BASELINE_KM, psi)
    to_tx = np.sqrt(r**2 + BASELINE_KM**2 - 2 * r * BASELINE_KM * np.cos(np.radians(psi)))
    assert np.allclose(r + to_tx - BASELINE_KM, d)


# ── the wedge ───────────────────────────────────────────────────────────────


def test_one_aircraft_proves_nothing():
    coverage = fresh()
    fly(coverage, "t1", "aaa001", 62.0, 40.0)
    assert coverage.held(NOW)[12] == 0.0


def test_a_slice_holds_the_second_best_range_among_aircraft():
    coverage = fresh()
    fly(coverage, "t1", "aaa001", 62.0, 30.0)
    fly(coverage, "t2", "aaa002", 61.0, 50.0)
    fly(coverage, "t3", "aaa003", 63.0, 40.0)
    assert coverage.held(NOW)[12] == pytest.approx(40.0, abs=1e-6)


def test_two_passes_of_one_aircraft_are_still_one_aircraft():
    coverage = fresh()
    fly(coverage, "t1", "aaa001", 62.0, 30.0)
    fly(coverage, "t2", "aaa001", 62.0, 50.0)
    assert coverage.held(NOW)[12] == 0.0


def test_a_pass_needs_enough_detections():
    coverage = fresh()
    fly(coverage, "t1", "aaa001", 62.0, 30.0, n=2)
    fly(coverage, "t2", "aaa002", 62.0, 50.0)
    assert coverage.held(NOW)[12] == 0.0


def test_a_repeated_event_window_is_not_counted_twice():
    coverage = fresh()
    detections = [detection(NOW - 60_000 + i * 1000, "aaa001", 62.0, 30.0) for i in range(2)]
    for _ in range(5):
        coverage.write_event("t1", NOW, 2, detections, adsb_hex="aaa001")
    fly(coverage, "t2", "aaa002", 62.0, 50.0)
    assert coverage.held(NOW)[12] == 0.0


def test_a_detection_tagged_with_another_aircraft_is_ignored():
    coverage = fresh()
    detections = [detection(NOW - 60_000 + i * 1000, "bbb999", 62.0, 30.0) for i in range(3)]
    coverage.write_event("t1", NOW, 3, detections, adsb_hex="aaa001")
    fly(coverage, "t2", "aaa002", 62.0, 50.0)
    assert coverage.held(NOW)[12] == 0.0


def test_the_hold_slides():
    coverage = fresh()
    fly(coverage, "t1", "aaa001", 62.0, 30.0)
    fly(coverage, "t2", "aaa002", 62.0, 50.0)
    assert coverage.held(NOW)[12] > 0
    assert coverage.held(NOW + DAY_MS + 1)[12] == 0.0


def test_a_reset_clears_the_wedge_and_says_when():
    coverage = fresh()
    fly(coverage, "t1", "aaa001", 62.0, 30.0)
    fly(coverage, "t2", "aaa002", 62.0, 50.0)
    coverage.reset(now_ms=NOW)
    state = coverage.state(NOW)
    assert state["proven_bins"] == 0
    assert state["reset_ms"] == NOW
    assert state["reset_reason"] == "manual"
    assert state["since_ms"] == NOW
    assert state["now_ms"] == NOW


def test_the_state_reports_lobes_that_start_and_end_at_the_receiver():
    coverage = fresh()
    for bearing in (52.0, 57.0, 62.0):
        fly(coverage, f"a{bearing}", "aaa001", bearing, 30.0)
        fly(coverage, f"b{bearing}", "aaa002", bearing, 50.0)
    state = coverage.state(NOW)
    assert state["proven_bins"] == 3
    assert state["aircraft"] == 2
    [ring] = state["lobes"]
    assert ring[0] == ring[-1] == [round(RX[0], 5), round(RX[1], 5)]
    edge = ring[1:-1]
    assert np.allclose(ground_range_km(RX[0], RX[1], *np.array(edge).T), 30.0, atol=0.01)
    bearings = bearing_deg(RX[0], RX[1], *np.array(edge).T)
    assert bearings.min() == pytest.approx(50.0, abs=0.01)
    assert bearings.max() == pytest.approx(65.0, abs=0.01)


# ── arcs ────────────────────────────────────────────────────────────────────


def test_an_arc_is_faint_everywhere_before_the_wedge_exists():
    arc = fresh().arc(40.0, NOW)
    assert arc["solid"] == []
    assert sum(len(line) for line in arc["faint"]) > 100


def test_an_arc_is_solid_only_inside_the_wedge():
    coverage = fresh()
    for bearing in (52.0, 57.0, 62.0):
        fly(coverage, f"a{bearing}", "aaa001", bearing, 60.0)
        fly(coverage, f"b{bearing}", "aaa002", bearing, 60.0)
    arc = coverage.arc(40.0, NOW)
    assert arc["solid"]
    points = np.array([p for line in arc["solid"] for p in line])
    bearings = bearing_deg(RX[0], RX[1], points[:, 0], points[:, 1])
    assert bearings.min() >= 50.0 - 0.01
    assert bearings.max() <= 65.0 + 0.01


def test_an_arc_lies_on_its_ellipse():
    arc = fresh().arc(40.0, NOW)
    points = np.array([p for line in arc["faint"] for p in line])
    to_rx = ground_range_km(RX[0], RX[1], points[:, 0], points[:, 1])
    to_tx = ground_range_km(TX[0], TX[1], points[:, 0], points[:, 1])
    assert np.allclose(to_rx + to_tx - BASELINE_KM, 40.0, atol=0.05)


def test_a_near_baseline_arc_is_not_drawn():
    assert fresh().arc(ARC_MIN_DIFFERENTIAL_KM - 0.1, NOW) is None


def test_no_arcs_without_geometry():
    assert Coverage(None, bin_deg=5.0, hold_s=86400, min_pass_detections=3).arc(40.0, NOW) is None


def test_a_replay_spreads_a_bounded_number_of_arcs_over_the_track():
    track = {"t": list(range(1000)), "delay": [40.0 + i * 0.01 for i in range(1000)]}
    arcs = replay_arcs(fresh(), track, NOW)
    assert len(arcs) == 240
    assert arcs[0]["t"] == 0
    assert arcs[-1]["t"] == 999


# ── persistence ─────────────────────────────────────────────────────────────


def test_the_wedge_survives_a_restart(tmp_path):
    path = str(tmp_path / "coverage.json")
    coverage = fresh(state_path=path)
    fly(coverage, "t1", "aaa001", 62.0, 30.0)
    fly(coverage, "t2", "aaa002", 62.0, 50.0)
    coverage.save()

    restored = fresh(state_path=path)
    restored.load(NOW)
    assert restored.held(NOW)[12] == pytest.approx(30.0, abs=1e-6)


def test_moving_the_receiver_starts_the_wedge_again(tmp_path):
    path = str(tmp_path / "coverage.json")
    coverage = fresh(state_path=path)
    fly(coverage, "t1", "aaa001", 62.0, 30.0)
    fly(coverage, "t2", "aaa002", 62.0, 50.0)
    coverage.save()

    moved = Coverage(
        Geometry(RX[0] + 0.1, RX[1], TX[0], TX[1]), bin_deg=5.0, hold_s=86400, min_pass_detections=3, state_path=path
    )
    moved.load(NOW)
    state = moved.state(NOW)
    assert state["proven_bins"] == 0
    assert state["reset_reason"] == "geometry"


def test_an_unreadable_state_file_is_ignored(tmp_path):
    path = tmp_path / "coverage.json"
    path.write_text("{not json")
    coverage = fresh(state_path=str(path))
    coverage.load(NOW)
    assert coverage.state(NOW)["proven_bins"] == 0


# ── the history carries verified positions ──────────────────────────────────


def test_track_points_carry_their_verified_position():
    history = DetectionHistory()
    good = detection(NOW, "aaa001", 62.0, 30.0)
    stale = detection(NOW + 1000, "aaa001", 62.0, 31.0, age=30.0)
    history.write_event("t1", NOW + 1000, 2, [good, stale], adsb_hex="aaa001")

    track = history.track("t1")
    assert track["lat"][0] == pytest.approx(at(62.0, 30.0)[0], abs=1e-4)
    assert track["lat"][1] is None
    assert track["lon"][1] is None


def test_latest_names_each_tracks_newest_point():
    history = DetectionHistory()
    history.write_event("t1", NOW, 1, [detection(NOW, "aaa001", 62.0, 30.0, delay=41.0)], adsb_hex="aaa001")
    history.write_event("t2", NOW, 1, [{"timestamp": NOW, "delay": 55.0, "doppler": 10.0, "snr": 12.0}])
    latest = sorted(history.latest(window_s=60, now_ms=NOW))
    assert latest == [("t1", NOW, 41.0, "aaa001"), ("t2", NOW, 55.0, None)]


# ── config and CLI ──────────────────────────────────────────────────────────


def test_blah2s_location_is_read(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text(
        "capture:\n  fc: 195000000\nlocation:\n"
        "  rx: {latitude: 10.0, longitude: 20.0, altitude: 50}\n"
        "  tx: {latitude: 9.9, longitude: 19.8, altitude: 200}\n"
    )
    found = load_blah2_config(str(path))
    assert found["location"]["tx"]["latitude"] == 9.9


@pytest.mark.parametrize(
    "stream,explicit,expected",
    [
        ("/app/output/events.jsonl", None, "/app/output/coverage.json"),
        ("-", None, None),
        (None, None, None),
        ("/app/output/events.jsonl", "/elsewhere/c.json", "/elsewhere/c.json"),
    ],
)
def test_coverage_is_saved_beside_the_events_file(stream, explicit, expected):
    assert coverage_state_path(Namespace(stream_output=stream, coverage_state=explicit)) == expected


# ── the control surface ─────────────────────────────────────────────────────


@pytest.fixture
def served():
    history = DetectionHistory()
    coverage = Coverage(GEOMETRY, bin_deg=5.0, hold_s=86400, min_pass_detections=3)
    server = start_control_server(
        Tracker(config=get_config()), threading.Lock(), host="127.0.0.1", port=0, history=history, coverage=coverage
    )
    try:
        yield history, coverage, f"http://127.0.0.1:{server.port}"
    finally:
        server.shutdown()
        server.server_close()


def request(url, method="GET"):
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def test_the_wedge_is_served(served):
    _history, _coverage, base = served
    status, body = request(base + "/coverage")
    assert status == 200
    assert body["enabled"] is True
    assert body["geometry"]["baseline_km"] == pytest.approx(BASELINE_KM, abs=1e-3)
    assert len(body["held_km"]) == 72


def test_latest_arcs_are_served_for_unlabelled_tracks_only(served):
    history, _coverage, base = served
    now = NOW
    history.write_event("t1", now, 1, [detection(now, "aaa001", 62.0, 30.0)], adsb_hex="aaa001")
    history.write_event("t2", now, 1, [{"timestamp": now, "delay": 55.0, "doppler": 10.0, "snr": 12.0}])
    status, body = request(base + "/coverage/arcs")
    assert status == 200
    assert [a["id"] for a in body["arcs"]] == ["t2"]
    assert body["arcs"][0]["arc"]["faint"]


def test_a_tracks_replay_arcs_are_served(served):
    history, _coverage, base = served
    points = [{"timestamp": NOW + i * 1000, "delay": 40.0 + i, "doppler": 10.0, "snr": 12.0} for i in range(5)]
    history.write_event("261017-00001A", NOW + 4000, 5, points)
    status, body = request(base + "/coverage/track/261017-00001A")
    assert status == 200
    assert [a["delay"] for a in body["arcs"]] == [40.0, 41.0, 42.0, 43.0, 44.0]


@pytest.mark.parametrize("path,code", [("/coverage/track/nope", 404), ("/coverage/track/a%2F..%2Fb", 400)])
def test_an_unknown_or_unsafe_track_is_refused(served, path, code):
    _history, _coverage, base = served
    assert request(base + path)[0] == code


def test_the_wedge_can_be_reset(served):
    _history, coverage, base = served
    fly(coverage, "t1", "aaa001", 62.0, 30.0)
    fly(coverage, "t2", "aaa002", 62.0, 50.0)
    status, body = request(base + "/coverage/reset", method="POST")
    assert (status, body) == (200, {"ok": True})
    assert coverage.state()["reset_reason"] == "manual"


def test_coverage_routes_say_so_when_it_is_off():
    server = start_control_server(Tracker(config=get_config()), threading.Lock(), host="127.0.0.1", port=0)
    try:
        assert request(f"http://127.0.0.1:{server.port}/coverage")[0] == 503
    finally:
        server.shutdown()
        server.server_close()
