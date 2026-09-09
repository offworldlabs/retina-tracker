"""Score tracker output against synthetic ground truth.

Answers the questions live data cannot: which detections were really the same
aircraft, whether a track held one identity throughout, and whether an anomaly
flag was earned. The generated traffic is entirely ordinary, so every anomaly
raised against it is a false positive by construction.

Detections are matched back to truth on (timestamp, delay, doppler), which is
exact: both sides carry the same 2 dp values the wire format uses.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_source_index(truths: list[dict], frames: list[dict]) -> dict[tuple, str | None]:
    index: dict[tuple, str | None] = {}
    for frame, truth in zip(frames, truths):
        for delay, doppler, source in zip(frame["delay"], frame["doppler"], truth["sources"]):
            index[(frame["timestamp"], delay, doppler)] = source
    return index


def track_sources(track: dict, index: dict[tuple, str | None]) -> list[str | None]:
    history = track["history"]
    out = []
    for timestamp, delay, doppler in zip(history["timestamps"], history["delays"], history["dopplers"]):
        if delay is None or doppler is None:
            continue
        out.append(index.get((timestamp, delay, doppler), "UNKNOWN"))
    return out


def score(truths: list[dict], frames: list[dict], tracks: list[dict], min_associations: int = 3) -> dict:
    index = build_source_index(truths, frames)

    detectable: dict[str, int] = Counter()
    for truth in truths:
        for hexid, state in truth["targets"].items():
            if state["detected"]:
                detectable[hexid] += 1

    total_target_detections = sum(1 for truth in truths for s in truth["sources"] if s is not None)
    total_clutter = sum(1 for truth in truths for s in truth["sources"] if s is None)

    considered = [t for t in tracks if t["n_associated"] >= min_associations]
    per_aircraft: dict[str, list[dict]] = defaultdict(list)
    false_tracks = []
    covered_target_detections = 0
    covered_clutter = 0
    id_switches = 0
    purities = []
    mislabelled = 0

    for track in considered:
        sources = track_sources(track, index)
        if not sources:
            continue
        counts = Counter(sources)
        real = {k: v for k, v in counts.items() if k not in (None, "UNKNOWN")}
        covered_target_detections += sum(real.values())
        covered_clutter += counts.get(None, 0)

        seen = [s for s in sources if s not in (None, "UNKNOWN")]
        id_switches += sum(1 for a, b in zip(seen, seen[1:]) if a != b)

        if not real:
            false_tracks.append(track)
            continue
        dominant, dominant_n = max(real.items(), key=lambda kv: kv[1])
        purities.append(dominant_n / len(sources))
        per_aircraft[dominant].append(track)

        track_id = track.get("id") or ""
        suffix = track_id.split("-")[-1].lower()
        if suffix and suffix != dominant.lower():
            mislabelled += 1

    detected_aircraft = {h for h, n in detectable.items() if n >= min_associations}
    found = {h for h in detected_aircraft if h in per_aircraft}
    fragmentation = [len(per_aircraft[h]) for h in found]

    anomalous = [t for t in considered if t.get("is_anomalous")]
    anomaly_types = Counter(a for t in considered for a in t.get("anomaly_types", []))

    return {
        "frames": len(frames),
        "aircraft_detectable": len(detected_aircraft),
        "aircraft_tracked": len(found),
        "aircraft_missed": sorted(detected_aircraft - found),
        "track_recall": round(len(found) / max(len(detected_aircraft), 1), 3),
        "tracks_considered": len(considered),
        "false_tracks": len(false_tracks),
        "false_track_rate": round(len(false_tracks) / max(len(considered), 1), 3),
        "fragmentation_mean": round(sum(fragmentation) / max(len(fragmentation), 1), 2),
        "fragmentation_max": max(fragmentation, default=0),
        "purity_mean": round(sum(purities) / max(len(purities), 1), 3),
        "purity_min": round(min(purities), 3) if purities else 0.0,
        "id_switches": id_switches,
        "mislabelled_track_ids": mislabelled,
        "detection_recall": round(covered_target_detections / max(total_target_detections, 1), 3),
        "clutter_absorbed": covered_clutter,
        "clutter_absorbed_rate": round(covered_clutter / max(total_clutter, 1), 3),
        "anomalous_tracks": len(anomalous),
        "anomaly_false_positive_rate": round(len(anomalous) / max(len(considered), 1), 3),
        "anomaly_types": dict(anomaly_types),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score tracker output against synthetic truth")
    parser.add_argument("detections", help="The .detection file that was tracked")
    parser.add_argument("tracks", help="tracks.json written by retina-tracker -o")
    parser.add_argument("-t", "--truth", help="Truth sidecar (default: <detections>.truth.jsonl)")
    parser.add_argument("--min-assoc", type=int, default=3, help="Ignore tracks below this many associations")
    args = parser.parse_args()

    frames = load_jsonl(args.detections)
    truths = load_jsonl(args.truth or f"{args.detections}.truth.jsonl")
    with open(args.tracks) as f:
        tracks = json.load(f)["tracks"]

    print(json.dumps(score(truths, frames, tracks, min_associations=args.min_assoc), indent=2))


if __name__ == "__main__":
    main()
