"""Scoring a node's own output without truth data.

The synthetic scorer is exact but single-site: every constant behind it was
fitted to one capture, and one of them has since been measured 5x wrong. This
scorer is weaker - it only sees cooperative traffic - but it runs on any node,
which is the only way to tell a real improvement from one fitted to one site.
"""

import json

import pytest

from retina_tracker.live_score import (
    fragmentation,
    kinematic_consistency,
    load_blah2_fc,
    load_tracks,
    manoeuvre_rates,
    near_origin,
    replica_offsets,
    score,
    shadow_guardrail,
    wavelength_km,
)

FC = 213000000.0
LAM_KM = wavelength_km(FC)


def detection(ts_ms, delay, doppler, snr=12.0):
    return {"timestamp": ts_ms, "delay": delay, "doppler": doppler, "snr": snr}


def event(track_id, detections, adsb_hex=None, shadow_fraction=None):
    payload = {"track_id": track_id, "adsb_hex": adsb_hex, "detections": detections}
    if shadow_fraction is not None:
        payload["shadow_fraction"] = shadow_fraction
    return json.dumps(payload)


def write(tmp_path, lines):
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def consistent_track(track_id, doppler, start_delay=10.0, n=8, adsb_hex=None):
    """A track whose delay advances exactly as its Doppler demands."""
    rate = -LAM_KM * doppler
    rows = [detection(i * 1000, start_delay + rate * i, doppler) for i in range(n)]
    return track_id, rows, adsb_hex


class TestLoading:
    def test_a_track_is_rebuilt_from_the_events_that_carry_it(self, tmp_path):
        """Each event holds only recent detections, so history spans many of them."""
        path = write(
            tmp_path,
            [
                event("t1", [detection(0, 10.0, -50.0), detection(1000, 10.07, -50.0)]),
                event("t1", [detection(1000, 10.07, -50.0), detection(2000, 10.14, -50.0)]),
            ],
        )
        tracks = load_tracks(path)
        assert [d["timestamp"] for d in tracks["t1"]["detections"]] == [0, 1000, 2000]

    def test_an_adsb_hex_seen_on_any_event_sticks_to_the_track(self, tmp_path):
        path = write(tmp_path, [event("t1", [detection(0, 10.0, -50.0)]), event("t1", [], adsb_hex="abc123")])
        assert load_tracks(path)["t1"]["adsb_hex"] == "abc123"

    def test_malformed_lines_are_skipped_rather_than_fatal(self, tmp_path):
        path = write(tmp_path, ["{not json", event("t1", [detection(0, 10.0, -50.0)])])
        assert set(load_tracks(path)) == {"t1"}

    def test_carrier_frequency_is_read_from_a_blah2_config(self, tmp_path):
        config = tmp_path / "config.yml"
        config.write_text("capture:\n  fs: 2000000\n  fc: 213000000\nprocess:\n  fc: 999\n")
        assert load_blah2_fc(str(config)) == 213000000.0

    def test_a_config_without_a_capture_section_yields_nothing(self, tmp_path):
        config = tmp_path / "config.yml"
        config.write_text("process:\n  fc: 999\n")
        assert load_blah2_fc(str(config)) is None


class TestKinematicConsistency:
    def _tracks(self, specs):
        return {tid: {"detections": rows, "adsb_hex": hexid, "shadow_fraction": None} for tid, rows, hexid in specs}

    def test_a_track_matching_its_own_doppler_is_consistent(self):
        tracks = self._tracks([consistent_track("t1", -80.0)])
        assert kinematic_consistency(tracks, LAM_KM)["impossible"] == 0

    def test_a_fixed_delay_under_a_large_doppler_is_impossible(self):
        """The signature of a fixed clutter line chained across delay."""
        rows = [detection(i * 1000, 9.0, -60.0) for i in range(8)]
        tracks = self._tracks([("t1", rows, None)])
        assert kinematic_consistency(tracks, LAM_KM)["impossible"] == 1

    def test_range_moving_the_wrong_way_is_impossible(self):
        rows = [detection(i * 1000, 10.0 + 0.1 * i, 60.0) for i in range(8)]
        tracks = self._tracks([("t1", rows, None)])
        assert kinematic_consistency(tracks, LAM_KM)["impossible"] == 1

    def test_a_track_too_short_to_judge_is_not_counted(self):
        rows = [detection(i * 1000, 9.0, -60.0) for i in range(3)]
        tracks = self._tracks([("t1", rows, None)])
        assert kinematic_consistency(tracks, LAM_KM)["tracks_measured"] == 0

    def test_a_track_spanning_too_little_time_is_not_counted(self):
        rows = [detection(i * 100, 9.0, -60.0) for i in range(8)]
        tracks = self._tracks([("t1", rows, None)])
        assert kinematic_consistency(tracks, LAM_KM)["tracks_measured"] == 0


class TestFragmentation:
    def test_two_tracks_for_one_aircraft_is_a_split(self):
        tracks = {
            "t1": {"detections": [], "adsb_hex": "abc", "shadow_fraction": None},
            "t2": {"detections": [], "adsb_hex": "abc", "shadow_fraction": None},
            "t3": {"detections": [], "adsb_hex": "def", "shadow_fraction": None},
        }
        result = fragmentation(tracks)
        assert result["aircraft_identified"] == 2
        assert result["mean"] == pytest.approx(1.5)
        assert result["split"] == ["abc"]

    def test_unlabelled_tracks_do_not_count(self):
        tracks = {"t1": {"detections": [], "adsb_hex": None, "shadow_fraction": None}}
        assert fragmentation(tracks)["aircraft_identified"] == 0


class TestShadowGuardrail:
    def test_it_reports_nothing_for_a_build_that_predates_the_field(self):
        tracks = {"t1": {"detections": [], "adsb_hex": "abc", "shadow_fraction": None}}
        assert shadow_guardrail(tracks) == {"reported": False}

    def test_identified_aircraft_over_the_threshold_are_counted(self):
        """The whole point: a site where the thresholds suppress real traffic."""
        tracks = {
            "t1": {"detections": [], "adsb_hex": "abc", "shadow_fraction": 0.9},
            "t2": {"detections": [], "adsb_hex": "def", "shadow_fraction": 0.0},
            "t3": {"detections": [], "adsb_hex": None, "shadow_fraction": 0.7},
        }
        result = shadow_guardrail(tracks)
        assert result["identified_over_threshold"] == 1
        assert result["identified_max"] == pytest.approx(0.9)
        assert result["unlabelled_median"] == pytest.approx(0.7)


class TestNearOrigin:
    def test_tracks_below_the_ground_traffic_range_are_counted(self):
        tracks = {
            "t1": {
                "detections": [detection(i * 1000, 1.0, -19.0) for i in range(4)],
                "adsb_hex": None,
                "shadow_fraction": None,
            },
            "t2": {
                "detections": [detection(i * 1000, 40.0, -19.0) for i in range(4)],
                "adsb_hex": None,
                "shadow_fraction": None,
            },
        }
        result = near_origin(tracks)
        assert result["near_origin"] == 1
        assert result["rate"] == pytest.approx(0.5)


class TestReplicaOffsets:
    def test_an_unlabelled_return_behind_a_brighter_aircraft_is_measured(self):
        tracks = {
            "air": {"detections": [detection(0, 10.0, -50.0, snr=18.0)], "adsb_hex": "abc", "shadow_fraction": None},
            "ghost": {"detections": [detection(0, 11.2, -20.0, snr=11.0)], "adsb_hex": None, "shadow_fraction": None},
        }
        result = replica_offsets(tracks)
        assert result["pairs"] == 1
        assert result["behind_parent_rate"] == pytest.approx(1.0)
        assert result["delay_offset_median_km"] == pytest.approx(1.2)
        assert result["snr_deficit_median_db"] == pytest.approx(-7.0)

    def test_returns_in_different_frames_are_never_paired(self):
        """Doppler slews fast enough during a pass that cross-frame pairing
        invents offsets that are not there."""
        tracks = {
            "air": {"detections": [detection(0, 10.0, -50.0, snr=18.0)], "adsb_hex": "abc", "shadow_fraction": None},
            "ghost": {
                "detections": [detection(5000, 11.2, -20.0, snr=11.0)],
                "adsb_hex": None,
                "shadow_fraction": None,
            },
        }
        assert replica_offsets(tracks)["pairs"] == 0


class TestManoeuvreRates:
    def test_it_measures_the_slew_the_process_noise_has_to_cover(self):
        rows = [detection(i * 1000, 10.0, -50.0 + 4.0 * i) for i in range(6)]
        tracks = {"t1": {"detections": rows, "adsb_hex": "abc", "shadow_fraction": None}}
        result = manoeuvre_rates(tracks, LAM_KM)
        assert result["doppler_rate_median_hz_s"] == pytest.approx(4.0)
        assert result["range_accel_median_ms2"] == pytest.approx(4.0 * LAM_KM * 1000, rel=1e-3)

    def test_unidentified_tracks_are_excluded(self):
        """Replicas and clutter would otherwise pollute the aircraft dynamics."""
        rows = [detection(i * 1000, 10.0, -50.0 + 4.0 * i) for i in range(6)]
        tracks = {"t1": {"detections": rows, "adsb_hex": None, "shadow_fraction": None}}
        assert manoeuvre_rates(tracks, LAM_KM)["samples"] == 0


def test_score_runs_end_to_end(tmp_path):
    tid, rows, _ = consistent_track("t1", -80.0, adsb_hex="abc")
    path = write(tmp_path, [event(tid, rows, adsb_hex="abc", shadow_fraction=0.0)])
    result = score(load_tracks(path), FC)
    assert result["wavelength_m"] == pytest.approx(1.4075, abs=1e-4)
    assert result["kinematics"]["impossible"] == 0
    assert result["fragmentation"]["aircraft_identified"] == 1
    assert result["shadow"]["reported"] is True
