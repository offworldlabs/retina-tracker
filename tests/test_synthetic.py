"""The synthetic generator has to keep looking like the node it was built from.

Every assertion here compares generated output against a constant in
`synthetic.profile` that was measured from a live capture. The fitted knobs in
that module have no independent justification, so these tests are what stops
someone retuning one until the dataset stops resembling real data.

Statistics are averaged over several seeds and compared with a tolerance,
because a 118 s window is genuinely noisy: the reference itself is one sample.
"""

import json
import math
from collections import Counter, defaultdict

import numpy as np
import pytest

from synthetic import profile
from synthetic.generate import continuity, generate, summarise
from synthetic.score import score
from synthetic.world import Site, bistatic

SEEDS = range(20260908, 20260914)
REFERENCE_DURATION_S = 118.0

LIVE_ADSB_KEYS = {
    "hex",
    "lat",
    "lon",
    "alt",
    "gs",
    "track",
    "expected_delay",
    "expected_doppler",
    "delay_residual",
    "doppler_residual",
}


@pytest.fixture(scope="module")
def runs():
    return [generate(duration_s=REFERENCE_DURATION_S, seed=seed) for seed in SEEDS]


@pytest.fixture(scope="module")
def stats(runs):
    rows = [{**summarise(f, t), **continuity(t)} for f, t in runs]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k], (int, float))}


def close_to(value, reference, tolerance=0.15):
    return abs(value - reference) <= tolerance * abs(reference)


def test_frame_has_exactly_the_live_wire_shape(runs):
    frames, _ = runs[0]
    for frame in frames:
        assert set(frame) == {"timestamp", "delay", "doppler", "snr", "adsb"}
        assert isinstance(frame["timestamp"], int)
        n = len(frame["delay"])
        assert len(frame["doppler"]) == n
        assert len(frame["snr"]) == n
        assert len(frame["adsb"]) == n


def test_adsb_blocks_carry_the_keys_live_data_carries(runs):
    frames, _ = runs[0]
    blocks = [a for frame in frames for a in frame["adsb"] if a]
    assert blocks
    for block in blocks:
        assert set(block) == LIVE_ADSB_KEYS
    # The wire sends `alt`; retina-tracker reads `alt_baro`. Emitting the key
    # the tracker wants would hide that mismatch instead of exercising it.
    assert not any("alt_baro" in block for block in blocks)


def test_values_stay_inside_the_map_and_survive_rounding(runs):
    frames, _ = runs[0]
    for frame in frames:
        for delay, doppler, snr in zip(frame["delay"], frame["doppler"], frame["snr"]):
            assert profile.DELAY_FLOOR_KM <= delay <= profile.DELAY_MAX_KM
            assert profile.DOPPLER_BLANK_HZ <= abs(doppler) <= profile.DOPPLER_MAX_HZ
            assert snr >= profile.SNR_FLOOR_DB
            assert round(delay, 2) == delay
            assert round(doppler, 2) == doppler
            assert round(snr, 2) == snr


def test_frames_arrive_at_the_measured_cadence(runs):
    frames, _ = runs[0]
    gaps = np.diff([f["timestamp"] for f in frames]) / 1000.0
    assert close_to(float(np.median(gaps)), profile.FRAME_INTERVAL_S, 0.05)
    assert gaps.max() > profile.FRAME_INTERVAL_S * 1.5


def test_detections_are_mostly_doppler_ascending(runs):
    frames, _ = runs[0]
    sorted_frames = sum(1 for f in frames if f["doppler"] == sorted(f["doppler"]))
    assert 0.6 <= sorted_frames / len(frames) <= 0.95


def test_detection_load_matches_the_reference(stats):
    assert close_to(stats["detections_per_frame"], profile.DETECTIONS_PER_FRAME)
    assert close_to(stats["target_per_frame"], profile.TARGET_DETECTIONS_PER_FRAME)
    assert close_to(stats["clutter_per_frame"], profile.CLUTTER_PER_FRAME)


def test_snr_populations_match_the_reference(stats):
    assert close_to(stats["target_snr_median"], profile.TARGET_SNR_MEDIAN_DB)
    assert close_to(stats["target_snr_p95"], profile.TARGET_SNR_P95_DB)
    assert close_to(stats["clutter_snr_median"], profile.CLUTTER_SNR_MEDIAN_DB)
    assert close_to(stats["clutter_snr_p95"], profile.CLUTTER_SNR_P95_DB)
    assert stats["target_snr_median"] > stats["clutter_snr_median"] + 3.0


def test_adsb_match_rate_and_residuals_match_the_reference(stats):
    assert close_to(stats["adsb_match_rate"], profile.ADSB_MATCH_RATE)
    assert close_to(stats["delay_residual_median"], profile.DELAY_RESIDUAL_MEDIAN_KM)
    # Tail statistics are noisier than the medians on a 118 s window.
    assert close_to(stats["delay_residual_p95"], profile.DELAY_RESIDUAL_P95_KM, 0.22)
    assert close_to(stats["doppler_residual_median"], profile.DOPPLER_RESIDUAL_MEDIAN_HZ)
    assert close_to(stats["doppler_residual_p95"], profile.DOPPLER_RESIDUAL_P95_HZ, 0.22)


def test_targets_drop_out_in_bursts_not_independently(stats):
    assert close_to(stats["continuity_median"], profile.CONTINUITY)
    assert close_to(stats["longest_run_median"], profile.LONGEST_RUN_MEDIAN, 0.25)


def test_clutter_is_one_shot_not_persistent(runs):
    frames, truths = runs[0]
    cells = defaultdict(set)
    for frame, truth in zip(frames, truths):
        for delay, doppler, source in zip(frame["delay"], frame["doppler"], truth["sources"]):
            if source is None:
                cells[(round(delay), round(doppler / 5) * 5)].add(frame["timestamp"])
    single = sum(1 for seen in cells.values() if len(seen) == 1)
    assert single / len(cells) >= 0.85


def test_truth_lines_up_with_every_frame(runs):
    frames, truths = runs[0]
    assert len(frames) == len(truths)
    for frame, truth in zip(frames, truths):
        assert frame["timestamp"] == truth["timestamp"]
        assert len(truth["sources"]) == len(frame["delay"])
        assert len(truth["matched"]) == len(frame["delay"])
        for source in truth["sources"]:
            assert source is None or source in truth["targets"]


def test_adsb_is_sometimes_stale_enough_to_look_frozen(runs):
    """The mechanism behind live position_mismatch flags must be present."""
    frames, _ = runs[0]
    last: dict[str, tuple] = {}
    frozen = 0
    for frame in frames:
        for block in frame["adsb"]:
            if not block:
                continue
            key = (block["lat"], block["lon"])
            if last.get(block["hex"]) == key and block["gs"] >= 50:
                frozen += 1
            last[block["hex"]] = key
    assert frozen > 0


def test_generation_is_deterministic():
    first = generate(duration_s=30.0, seed=7)
    second = generate(duration_s=30.0, seed=7)
    assert json.dumps(first[0]) == json.dumps(second[0])
    assert json.dumps(first[1]) == json.dumps(second[1])


def test_bistatic_geometry_is_self_consistent():
    site = Site()
    stationary = np.array([0.0, 0.0, 0.0])
    delay, doppler = bistatic(site, site.rx_ecef + np.array([0.0, 0.0, 10000.0]), stationary)
    assert delay > 0
    assert doppler == pytest.approx(0.0, abs=1e-9)


def test_a_target_on_the_baseline_has_near_zero_bistatic_range():
    site = Site()
    midpoint = (site.rx_ecef + site.tx_ecef) / 2
    delay, _ = bistatic(site, midpoint, np.array([0.0, 0.0, 0.0]))
    assert delay == pytest.approx(0.0, abs=1e-6)


def test_expected_values_agree_with_the_emitted_measurement(runs):
    frames, _ = runs[0]
    for frame in frames:
        for delay, doppler, block in zip(frame["delay"], frame["doppler"], frame["adsb"]):
            if not block:
                continue
            assert abs(delay - block["expected_delay"]) < profile.ADSB_DELAY_TOLERANCE_KM + 0.01
            assert abs(doppler - block["expected_doppler"]) < profile.ADSB_DOPPLER_TOLERANCE_HZ + 0.01
            assert block["delay_residual"] == pytest.approx(delay - block["expected_delay"], abs=0.011)


def test_scorer_gives_a_perfect_tracker_full_marks(runs):
    frames, truths = runs[0]
    by_source = defaultdict(lambda: {"timestamps": [], "delays": [], "dopplers": []})
    for frame, truth in zip(frames, truths):
        for delay, doppler, source in zip(frame["delay"], frame["doppler"], truth["sources"]):
            if source is None:
                continue
            entry = by_source[source]
            entry["timestamps"].append(frame["timestamp"])
            entry["delays"].append(delay)
            entry["dopplers"].append(doppler)

    tracks = [
        {
            "id": f"260908-{hexid.upper()}",
            "n_associated": len(entry["timestamps"]),
            "is_anomalous": False,
            "anomaly_types": [],
            "history": entry,
        }
        for hexid, entry in by_source.items()
        if len(entry["timestamps"]) >= 3
    ]

    result = score(truths, frames, tracks)
    assert result["track_recall"] == 1.0
    assert result["purity_mean"] == 1.0
    assert result["id_switches"] == 0
    assert result["false_tracks"] == 0
    assert result["mislabelled_track_ids"] == 0
    assert result["anomaly_false_positive_rate"] == 0.0


def test_scorer_notices_a_track_that_swaps_identity(runs):
    frames, truths = runs[0]
    picks = []
    for frame, truth in zip(frames, truths):
        for delay, doppler, source in zip(frame["delay"], frame["doppler"], truth["sources"]):
            if source is not None:
                picks.append((frame["timestamp"], delay, doppler, source))
    assert len({p[3] for p in picks}) > 1
    first = picks[0][3]
    other = next(p for p in picks if p[3] != first)
    chosen = [p for p in picks if p[3] == first][:5] + [p for p in picks if p[3] == other[3]][:5]
    track = {
        "id": f"260908-{first.upper()}",
        "n_associated": len(chosen),
        "is_anomalous": False,
        "anomaly_types": [],
        "history": {
            "timestamps": [p[0] for p in chosen],
            "delays": [p[1] for p in chosen],
            "dopplers": [p[2] for p in chosen],
        },
    }
    result = score(truths, frames, [track])
    assert result["id_switches"] >= 1
    assert result["purity_mean"] < 1.0


def test_clutter_only_track_is_scored_as_false(runs):
    frames, truths = runs[0]
    clutter = [
        (frame["timestamp"], delay, doppler)
        for frame, truth in zip(frames, truths)
        for delay, doppler, source in zip(frame["delay"], frame["doppler"], truth["sources"])
        if source is None
    ][:6]
    track = {
        "id": "260908-00000A",
        "n_associated": len(clutter),
        "is_anomalous": False,
        "anomaly_types": [],
        "history": {
            "timestamps": [c[0] for c in clutter],
            "delays": [c[1] for c in clutter],
            "dopplers": [c[2] for c in clutter],
        },
    }
    result = score(truths, frames, [track])
    assert result["false_tracks"] == 1
    assert result["false_track_rate"] == 1.0


def test_aircraft_kinematics_stay_physical(runs):
    frames, truths = runs[0]
    tracks = defaultdict(list)
    for truth in truths:
        for hexid, state in truth["targets"].items():
            tracks[hexid].append((truth["t"], state["delay"], state["doppler"]))
    rates = []
    for points in tracks.values():
        for (t0, d0, _), (t1, d1, _) in zip(points, points[1:]):
            if 0.5 < t1 - t0 < 2.0:
                rates.append(abs(d1 - d0) / (t1 - t0))
    assert rates
    median_ms = float(np.median(rates)) * 1000.0
    assert median_ms < 2 * profile.BISTATIC_RANGE_RATE_P95_MS
    assert math.isfinite(median_ms)


def test_counts_do_not_drift_with_window_length():
    short = summarise(*generate(duration_s=118.0, seed=99))
    long_run = summarise(*generate(duration_s=472.0, seed=99))
    assert close_to(long_run["detections_per_frame"], short["detections_per_frame"], 0.2)
    assert close_to(long_run["clutter_per_frame"], short["clutter_per_frame"], 0.25)


def test_a_minority_of_aircraft_carry_no_adsb(runs):
    equipped = Counter(
        state["adsb_equipped"] for _, truths in runs for truth in truths for state in truth["targets"].values()
    )
    assert equipped[False] > 0
    assert equipped[True] > equipped[False] * 5
