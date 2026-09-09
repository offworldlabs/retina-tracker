"""Score a node's own track output, with no truth data and no synthetic harness.

synthetic/score.py answers "was that track right" by consulting a generator that
knows which aircraft produced each detection. That makes it exact and makes it a
single site: every constant in synthetic/profile.py was fitted to one capture
from one node, and at least one of them has since been measured 5x wrong against
live data.

This reads the events.jsonl a node already writes and scores it against the
ADS-B annotation blah2 attaches to each detection. That is a weaker label - it
covers only cooperative traffic, matches within a tolerance, and misses roughly
a quarter of real target detections - so the numbers here are not comparable to
the synthetic ones. What it buys is that they can be produced on any node, which
is the only way to tell a real improvement from one fitted to a single site.

Deliberately stdlib-only: it is meant to be copied onto a node and run under
whatever python3 is there, as well as inside the container.

    python -m retina_tracker.live_score events.jsonl --fc 213000000
    python -m retina_tracker.live_score events.jsonl --blah2-config config.yml
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict

SPEED_OF_LIGHT = 299792458.0

# A track whose range moved less than this share of what its own Doppler
# demanded is not describing a physical trajectory.  Well below 1 because the
# delay measurement is coarse and the fit is over few points.
CONSISTENCY_RATIO = 0.33

# Below this bistatic range a return is far more likely to be traffic near the
# receiver than an aircraft: measured populations there run at road speeds, in
# fixed cells, on a daily cycle decoupled from air traffic.
NEAR_ORIGIN_KM = 3.0

MIN_DETECTIONS = 4
MIN_SPAN_S = 5.0


def wavelength_km(fc_hz):
    return SPEED_OF_LIGHT / fc_hz / 1000.0


def load_blah2_fc(path):
    """Read capture.fc out of a blah2 config without a YAML dependency."""
    section = None
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not line.startswith((" ", "\t")) and stripped.endswith(":"):
                section = stripped[:-1]
            elif section == "capture" and stripped.startswith("fc:"):
                return float(stripped.split(":", 1)[1].strip())
    return None


def load_tracks(path):
    """Collapse an events stream into one record per track.

    Each event carries only the most recent detections, so a long track appears
    across many events; the union over timestamp recovers its full history.
    """
    detections = defaultdict(dict)
    adsb = {}
    reported_shadow = {}
    with open(path, errors="replace") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            track_id = event.get("track_id")
            if track_id is None:
                continue
            for detection in event.get("detections") or []:
                detections[track_id][detection["timestamp"]] = detection
            if event.get("adsb_hex"):
                adsb[track_id] = event["adsb_hex"]
            if event.get("shadow_fraction") is not None:
                reported_shadow[track_id] = event["shadow_fraction"]
    return {
        track_id: {
            "detections": [rows[ts] for ts in sorted(rows)],
            "adsb_hex": adsb.get(track_id),
            "shadow_fraction": reported_shadow.get(track_id),
        }
        for track_id, rows in detections.items()
    }


def _slope(xs, ys):
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def _measurable(track):
    rows = track["detections"]
    if len(rows) < MIN_DETECTIONS:
        return None
    times = [r["timestamp"] / 1000.0 for r in rows]
    if times[-1] - times[0] < MIN_SPAN_S:
        return None
    return times


def kinematic_consistency(tracks, lam_km):
    """How many tracks move the way their own Doppler says they must.

    Doppler is the bistatic range rate, so a track's delay must advance by
    -wavelength * doppler. Nothing enforces that if the filter treats the two
    axes as independent, and roughly half the tracks of such a build do not.
    """
    consistent = impossible = 0
    worst = []
    for track_id, track in tracks.items():
        times = _measurable(track)
        if times is None:
            continue
        rows = track["detections"]
        observed = _slope(times, [r["delay"] for r in rows])
        if observed is None:
            continue
        implied = -lam_km * (sum(r["doppler"] for r in rows) / len(rows))
        if implied and observed / implied >= CONSISTENCY_RATIO:
            consistent += 1
        else:
            impossible += 1
        worst.append((abs(observed - implied), track_id, observed, implied))
    total = consistent + impossible
    worst.sort(reverse=True)
    return {
        "tracks_measured": total,
        "consistent": consistent,
        "impossible": impossible,
        "impossible_rate": round(impossible / total, 3) if total else None,
        "worst": [
            {"track_id": t, "observed_km_s": round(o, 4), "implied_km_s": round(i, 4)} for _, t, o, i in worst[:5]
        ],
    }


def fragmentation(tracks):
    """Distinct track ids per identified aircraft.

    The live stand-in for the synthetic fragmentation metric. It sees only
    ADS-B-equipped traffic, so it is a floor rather than the whole picture.
    """
    by_hex = defaultdict(set)
    for track_id, track in tracks.items():
        if track["adsb_hex"]:
            by_hex[track["adsb_hex"]].add(track_id)
    counts = [len(ids) for ids in by_hex.values()]
    return {
        "aircraft_identified": len(by_hex),
        "tracks_for_them": sum(counts),
        "mean": round(statistics.fmean(counts), 2) if counts else None,
        "max": max(counts) if counts else None,
        "split": sorted(h for h, ids in by_hex.items() if len(ids) > 1),
    }


def shadow_guardrail(tracks):
    """Whether the multipath thresholds still fit this site.

    They were fitted where identified aircraft never exceeded 0.10 while replica
    tracks ran a median of 0.64. Identified aircraft drifting off zero here means
    the thresholds are suppressing real targets at this site.
    """
    identified = [t["shadow_fraction"] for t in tracks.values() if t["adsb_hex"] and t["shadow_fraction"] is not None]
    unlabelled = [
        t["shadow_fraction"] for t in tracks.values() if not t["adsb_hex"] and t["shadow_fraction"] is not None
    ]
    if not identified and not unlabelled:
        return {"reported": False}
    return {
        "reported": True,
        "identified_max": round(max(identified), 3) if identified else None,
        "identified_mean": round(statistics.fmean(identified), 3) if identified else None,
        "unlabelled_median": round(statistics.median(unlabelled), 3) if unlabelled else None,
        "identified_over_threshold": sum(1 for f in identified if f >= 0.5),
    }


def near_origin(tracks):
    """Share of tracks sitting where ground traffic lives."""
    near = []
    total = 0
    for track_id, track in tracks.items():
        rows = track["detections"]
        if len(rows) < 3:
            continue
        total += 1
        mean_delay = sum(r["delay"] for r in rows) / len(rows)
        if mean_delay < NEAR_ORIGIN_KM:
            near.append((track_id, mean_delay, sum(r["doppler"] for r in rows) / len(rows)))
    return {
        "tracks": total,
        "near_origin": len(near),
        "rate": round(len(near) / total, 3) if total else None,
        "with_adsb": sum(1 for t, _, _ in near if tracks[t]["adsb_hex"]),
    }


def replica_offsets(tracks):
    """Where unlabelled detections sit relative to the brightest aircraft present.

    Multipath and sidelobe copies appear at longer bistatic range and lower SNR
    than the target that cast them. Same-frame comparison only, because Doppler
    slews fast enough during a pass that interpolating across frames invents
    offsets that are not there.
    """
    frames = defaultdict(list)
    for track_id, track in tracks.items():
        for row in track["detections"]:
            frames[row["timestamp"]].append((track_id, row))
    delays, snrs = [], []
    for rows in frames.values():
        labelled = [(t, r) for t, r in rows if tracks[t]["adsb_hex"]]
        unlabelled = [(t, r) for t, r in rows if not tracks[t]["adsb_hex"]]
        if not labelled or not unlabelled:
            continue
        _, parent = max(labelled, key=lambda pair: pair[1]["snr"])
        for _, row in unlabelled:
            delays.append(row["delay"] - parent["delay"])
            snrs.append(row["snr"] - parent["snr"])
    if not delays:
        return {"pairs": 0}
    return {
        "pairs": len(delays),
        "behind_parent_rate": round(sum(1 for d in delays if d > 0) / len(delays), 3),
        "weaker_than_parent_rate": round(sum(1 for s in snrs if s < 0) / len(snrs), 3),
        "delay_offset_median_km": round(statistics.median(delays), 2),
        "snr_deficit_median_db": round(statistics.median(snrs), 2),
    }


def manoeuvre_rates(tracks, lam_km):
    """Observed Doppler slew, which is what the process noise has to cover.

    synthetic/profile.py fixes this for the generator, so measuring it here is
    how that generator gets held to something real.
    """
    rates = []
    for track in tracks.values():
        rows = track["detections"]
        if not track["adsb_hex"] or len(rows) < MIN_DETECTIONS:
            continue
        for before, after in zip(rows, rows[1:]):
            dt = (after["timestamp"] - before["timestamp"]) / 1000.0
            if dt > 0:
                rates.append(abs(after["doppler"] - before["doppler"]) / dt)
    if not rates:
        return {"samples": 0}
    ordered = sorted(rates)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    median = statistics.median(ordered)
    return {
        "samples": len(rates),
        "doppler_rate_median_hz_s": round(median, 3),
        "doppler_rate_p95_hz_s": round(p95, 3),
        "range_accel_median_ms2": round(median * lam_km * 1000, 2),
        "range_accel_p95_ms2": round(p95 * lam_km * 1000, 2),
    }


def score(tracks, fc_hz):
    lam_km = wavelength_km(fc_hz)
    stamps = [r["timestamp"] for t in tracks.values() for r in t["detections"]]
    return {
        "center_frequency_hz": fc_hz,
        "wavelength_m": round(lam_km * 1000, 4),
        "tracks": len(tracks),
        "span_s": round((max(stamps) - min(stamps)) / 1000.0, 1) if stamps else 0.0,
        "kinematics": kinematic_consistency(tracks, lam_km),
        "fragmentation": fragmentation(tracks),
        "shadow": shadow_guardrail(tracks),
        "near_origin": near_origin(tracks),
        "replicas": replica_offsets(tracks),
        "manoeuvre": manoeuvre_rates(tracks, lam_km),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("events", help="events.jsonl written by the tracker")
    parser.add_argument("--fc", type=float, help="Carrier frequency in Hz")
    parser.add_argument("--blah2-config", help="blah2 config.yml to read capture.fc from")
    args = parser.parse_args()

    fc_hz = args.fc
    if fc_hz is None and args.blah2_config:
        fc_hz = load_blah2_fc(args.blah2_config)
    if fc_hz is None:
        parser.error("give --fc or --blah2-config: Doppler means nothing without the wavelength")

    tracks = load_tracks(args.events)
    if not tracks:
        print(f"No tracks in {args.events}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(score(tracks, fc_hz), indent=2))


if __name__ == "__main__":
    main()
