"""Generate a synthetic detection stream and its ground truth.

Writes two files. The `.detection` file is byte-shaped exactly like a capture
from a live node, so it can be replayed through the tracker with no special
casing. The `.truth.jsonl` sidecar carries, per frame, which aircraft produced
each detection and where every aircraft actually was, which is the thing live
data can never tell you and the reason this dataset exists.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import profile
from .sensor import Sensor
from .world import Site, build_fleet

DEFAULT_EPOCH_MS = 1_788_800_000_000


def generate(
    duration_s: float = 600.0,
    seed: int = 20260908,
    site: Site | None = None,
    concurrent: int | None = None,
    clutter_per_frame: float | None = None,
    epoch_ms: int = DEFAULT_EPOCH_MS,
) -> tuple[list[dict], list[dict]]:
    site = site or Site()
    rng = np.random.default_rng(seed)
    fleet = build_fleet(site, duration_s, rng, concurrent=concurrent)
    sensor = Sensor(site, rng, clutter_per_frame=clutter_per_frame)

    frames: list[dict] = []
    truths: list[dict] = []
    t = 0.0
    while t <= duration_s:
        timestamp = epoch_ms + int(round(t * 1000.0))
        frame, truth = sensor.frame(fleet, t, timestamp)
        frames.append(frame)
        truths.append(truth)
        step = profile.FRAME_INTERVAL_S + rng.normal(0.0, profile.FRAME_INTERVAL_JITTER_S)
        if rng.random() < profile.FRAME_SKIP_PROBABILITY:
            step += profile.FRAME_INTERVAL_S
        t += max(step, 0.5)
    return frames, truths


def summarise(frames: list[dict], truths: list[dict]) -> dict:
    counts = [len(f["delay"]) for f in frames]
    snr = np.array([s for f in frames for s in f["snr"]])
    delay = np.array([d for f in frames for d in f["delay"]])
    doppler = np.array([d for f in frames for d in f["doppler"]])
    adsb = [a for f in frames for a in f["adsb"]]
    matched = [a for a in adsb if a]
    target_snr = np.array(
        [s for f, tr in zip(frames, truths) for s, src in zip(f["snr"], tr["sources"]) if src is not None]
    )
    clutter_snr = np.array(
        [s for f, tr in zip(frames, truths) for s, src in zip(f["snr"], tr["sources"]) if src is None]
    )
    n_target = sum(1 for tr in truths for s in tr["sources"] if s is not None)
    correct = sum(
        1 for tr in truths for src, mat in zip(tr["sources"], tr["matched"]) if src is not None and mat == src
    )
    stolen = sum(
        1 for tr in truths for src, mat in zip(tr["sources"], tr["matched"]) if src is None and mat is not None
    )
    return {
        "frames": len(frames),
        "duration_s": round((frames[-1]["timestamp"] - frames[0]["timestamp"]) / 1000.0, 1),
        "detections": int(sum(counts)),
        "detections_per_frame": round(float(np.mean(counts)), 2),
        "target_per_frame": round(n_target / len(frames), 2),
        "clutter_per_frame": round((sum(counts) - n_target) / len(frames), 2),
        "adsb_match_rate": round(len(matched) / max(len(adsb), 1), 3),
        "adsb_correct_rate": round(correct / max(n_target, 1), 3),
        "clutter_false_match": stolen,
        "snr_median": round(float(np.median(snr)), 2),
        "target_snr_median": round(float(np.median(target_snr)), 2),
        "target_snr_p95": round(float(np.percentile(target_snr, 95)), 2),
        "clutter_snr_median": round(float(np.median(clutter_snr)), 2),
        "clutter_snr_p95": round(float(np.percentile(clutter_snr, 95)), 2),
        "delay_range": [round(float(delay.min()), 2), round(float(delay.max()), 2)],
        "doppler_range": [round(float(doppler.min()), 2), round(float(doppler.max()), 2)],
        "aircraft": len({s for tr in truths for s in tr["sources"] if s}),
        "delay_residual_median": round(float(np.median([abs(a["delay_residual"]) for a in matched])), 3),
        "delay_residual_p95": round(float(np.percentile([abs(a["delay_residual"]) for a in matched], 95)), 3),
        "doppler_residual_median": round(float(np.median([abs(a["doppler_residual"]) for a in matched])), 3),
        "doppler_residual_p95": round(float(np.percentile([abs(a["doppler_residual"]) for a in matched], 95)), 3),
    }


def continuity(truths: list[dict]) -> dict:
    seen: dict[str, list[bool]] = {}
    for tr in truths:
        for hexid, state in tr["targets"].items():
            if state["in_coverage"]:
                seen.setdefault(hexid, []).append(state["detected"])
    hits = [sum(v) for v in seen.values() if sum(v)]
    runs = []
    for v in seen.values():
        best = cur = 0
        for x in v:
            cur = cur + 1 if x else 0
            best = max(best, cur)
        if best:
            runs.append(best)
    frac = [sum(v) / len(v) for v in seen.values() if v]
    return {
        "targets_in_coverage": len(seen),
        "continuity_median": round(float(np.median(frac)), 3) if frac else 0.0,
        "hits_median": int(np.median(hits)) if hits else 0,
        "hits_max": int(max(hits)) if hits else 0,
        "longest_run_median": int(np.median(runs)) if runs else 0,
        "longest_run_max": int(max(runs)) if runs else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a life-like synthetic detection dataset")
    parser.add_argument("-o", "--output", default="data/synthetic.detection", help="Detection stream output")
    parser.add_argument("-t", "--truth", help="Ground truth output (default: <output>.truth.jsonl)")
    parser.add_argument("-d", "--duration", type=float, default=600.0, help="Seconds of data (default: 600)")
    parser.add_argument("-s", "--seed", type=int, default=20260908)
    parser.add_argument("--concurrent", type=int, help="Aircraft airborne at any instant")
    parser.add_argument("--clutter", type=float, help="False alarms per frame (default: measured 2.54)")
    parser.add_argument("--fc", type=float, help="Centre frequency in Hz")
    parser.add_argument("--stats", action="store_true", help="Print measured statistics of the result")
    args = parser.parse_args()

    site = Site(fc_hz=args.fc) if args.fc else Site()
    frames, truths = generate(
        duration_s=args.duration,
        seed=args.seed,
        site=site,
        concurrent=args.concurrent,
        clutter_per_frame=args.clutter,
    )

    truth_path = args.truth or f"{args.output}.truth.jsonl"
    with open(args.output, "w") as f:
        for frame in frames:
            f.write(json.dumps(frame) + "\n")
    with open(truth_path, "w") as f:
        for truth in truths:
            f.write(json.dumps(truth) + "\n")

    print(f"Wrote {len(frames)} frames to {args.output}", file=sys.stderr)
    print(f"Wrote ground truth to {truth_path}", file=sys.stderr)
    if args.stats:
        print(json.dumps({**summarise(frames, truths), **continuity(truths)}, indent=2))


if __name__ == "__main__":
    main()
