"""Fit R and Q from recorded innovations, separating two constants that cancel.

R and Q are both wrong today, in opposite directions. R is around 45x too large
in delay, while the model error it is standing in for grows as the square of the
prediction interval rather than the fifth power white jerk implies. The sum comes
out plausible, which is why nothing has complained, and it is why neither can be
fitted alone: a single scalar like NIS moves one way for an oversized R and the
other for an undersized Q.

What separates them is the prediction interval. The innovation covariance is
HPH' + R, so scatter measured against how long the filter had been predicting is
a constant plus a growing term:

    var(nu | L) = R + a * L^p

R is the intercept, and (a, p) are the shape and size of the model error. A grid
over p makes the rest a two-parameter linear fit, which is all this needs.

Deliberately stdlib-only, like live_score: it is meant to be copied onto a node
and run under whatever python3 is there.

    python -m retina_tracker.calibrate innovations.jsonl --blah2-config config.yml
    python -m retina_tracker.calibrate jn1.jsonl ffb.jsonl --fc 177000000
"""

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict

SPEED_OF_LIGHT = 299792458.0

# Median of chi-squared with two degrees of freedom. A matched filter's NIS
# should sit here; the plan measured 1.083 live, which is the signature of the
# oversized R holding the whole thing down.
CHI2_2DOF_MEDIAN = 2.0 * math.log(2.0)

# 1.4826 * MAD estimates sigma for a Gaussian, and unlike a standard deviation
# it does not let one mis-association set the answer. Calibration data comes
# from a live node, so it has mis-associations in it.
MAD_TO_SIGMA = 1.4826

# Prediction intervals are quantised by the frame rate anyway; rounding groups
# the ones that differ only by clock jitter.
INTERVAL_QUANTUM_S = 0.05

MIN_PER_INTERVAL = 30
MIN_INTERVALS = 3

# A track whose Doppler never leaves one bin is the fallback test for
# interference, used only on records written before the map carried its verdict.
FIXED_DOPPLER_BINS = 1.0
MIN_RECORDS_TO_JUDGE_A_TRACK = 8


def wavelength_km(fc_hz):
    return SPEED_OF_LIGHT / fc_hz / 1000.0


def delay_cell_km(fs_hz):
    return SPEED_OF_LIGHT / fs_hz / 1000.0


def load_blah2_capture(path):
    """Read fc, fs and cpi out of a blah2 config without a YAML dependency."""
    found = {}
    section = []
    with open(path) as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            stripped = line.strip()
            depth = indent // 2
            section = section[:depth]
            if stripped.endswith(":"):
                section.append(stripped[:-1])
                continue
            key, _, value = stripped.partition(":")
            value = value.strip()
            if section[:1] == ["capture"] and key in ("fc", "fs"):
                found[key] = float(value)
            elif section[:2] == ["process", "data"] and key == "cpi":
                found["cpi"] = float(value)
    return found


def load_records(path):
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def interfering_tracks(records):
    """Tracks whose Doppler never left one bin, for records with no verdict.

    Only a fallback. It costs the genuine constant-Doppler aircraft along with
    the tone, which biases what is left toward manoeuvring targets, so it is
    worth something for R and worth less for Q. Records written by a tracker
    carrying the occupancy map say so per detection instead.
    """
    by_track = defaultdict(list)
    for record in records:
        doppler = record.get("doppler")
        if doppler is not None:
            by_track[record.get("birth")].append(doppler)
    return {
        birth
        for birth, dopplers in by_track.items()
        if len(dopplers) >= MIN_RECORDS_TO_JUDGE_A_TRACK and max(dopplers) - min(dopplers) < FIXED_DOPPLER_BINS
    }


def usable(records):
    """Records that describe a real measurement of a real target.

    Drops the interference, which is most of the record at an interfered node,
    and the degenerate updates, whose innovation is NaN because a singular
    covariance meant the measurement was skipped.
    """
    fallback = interfering_tracks(records) if not any("interfering" in r for r in records) else set()
    kept = []
    for record in records:
        if record.get("interfering") or record.get("birth") in fallback:
            continue
        innovation = record.get("innovation") or []
        if len(innovation) != 2 or any(x is None or not math.isfinite(x) for x in innovation):
            continue
        if not record.get("dt"):
            continue
        kept.append(record)
    return kept


def prediction_interval(record):
    """How long the filter had been predicting when this innovation was taken.

    dt is the interval of the last predict and n_missed the frames coasted
    before this association landed, so this assumes a steady frame rate across
    a coast. It is the quantity R and Q are separated along, and getting it
    from the record is the whole reason the tracker emits n_missed.
    """
    return record["dt"] * (1 + record.get("n_missed", 0))


def mad_sigma(values):
    centre = statistics.median(values)
    return MAD_TO_SIGMA * statistics.median([abs(v - centre) for v in values])


def group_by_interval(records, component):
    """Scatter of one innovation component against prediction interval."""
    buckets = defaultdict(list)
    for record in records:
        interval = round(prediction_interval(record) / INTERVAL_QUANTUM_S) * INTERVAL_QUANTUM_S
        buckets[round(interval, 3)].append(record["innovation"][component])
    return sorted(
        (interval, len(values), mad_sigma(values) ** 2)
        for interval, values in buckets.items()
        if interval > 0 and len(values) >= MIN_PER_INTERVAL
    )


def fit_noise_model(points, exponents=None):
    """Least squares for var = R + a * L^p, weighted by sample count.

    Linear in R and a once p is fixed, so a grid over p reduces the whole thing
    to a 2x2 solve per candidate. p is what says whether Q has the right shape:
    white jerk, which is what the filter implements, would be 5.
    """
    if len(points) < MIN_INTERVALS:
        return None
    if exponents is None:
        exponents = [0.5 + 0.05 * i for i in range(91)]

    best = None
    for p in exponents:
        s_w = s_x = s_xx = s_y = s_xy = 0.0
        for interval, count, variance in points:
            x = interval**p
            w = float(count)
            s_w += w
            s_x += w * x
            s_xx += w * x * x
            s_y += w * variance
            s_xy += w * x * variance
        determinant = s_w * s_xx - s_x * s_x
        if abs(determinant) < 1e-30:
            continue
        r = (s_y * s_xx - s_x * s_xy) / determinant
        a = (s_w * s_xy - s_x * s_y) / determinant
        if r < 0 or a < 0:
            continue
        cost = sum(count * (variance - (r + a * interval**p)) ** 2 for interval, count, variance in points)
        if best is None or cost < best[0]:
            best = (cost, r, a, p)

    if best is None:
        return None
    _cost, r, a, p = best
    return {"R": r, "a": a, "exponent": p}


def normalised_scatter(records, component):
    """Actual scatter over what the filter claimed, per axis.

    NIS collapses both axes into one number, which is how an oversized R and an
    undersized Q have been hiding in it. Above one means the filter is more
    confident than it has earned on that axis; below one means the reverse.
    """
    ratios = [
        record["innovation"][component] / math.sqrt(record["s_diag"][component])
        for record in records
        if record.get("s_diag") and record["s_diag"][component] > 0
    ]
    return mad_sigma(ratios) if len(ratios) >= MIN_PER_INTERVAL else None


def calibrate(records, fc_hz, fs_hz, cpi_s):
    kept = usable(records)
    report = {
        "n_records": len(records),
        "n_used": len(kept),
        "n_interfering": sum(1 for r in records if r.get("interfering")),
        "n_tracks": len({r.get("birth") for r in kept}),
        "delay_cell_km": delay_cell_km(fs_hz),
        "doppler_bin_hz": 1.0 / cpi_s,
        "wavelength_km": wavelength_km(fc_hz),
        "nis_median": statistics.median([r["nis"] for r in kept if r.get("nis") is not None]) if kept else None,
        "axes": {},
    }
    if not kept:
        return report

    for name, component in (("delay", 0), ("doppler", 1)):
        fit = fit_noise_model(group_by_interval(kept, component))
        axis = {"scatter_vs_claimed": normalised_scatter(kept, component)}
        if fit:
            sigma = math.sqrt(fit["R"])
            if name == "delay":
                axis["sigma"] = sigma
                axis["unit"] = "km"
                axis["cells"] = sigma / report["delay_cell_km"]
            else:
                # The innovation is range rate, which is what R and S are in.
                sigma_hz = sigma / report["wavelength_km"]
                axis["sigma"] = sigma_hz
                axis["unit"] = "Hz"
                axis["cells"] = sigma_hz / report["doppler_bin_hz"]
            axis["exponent"] = fit["exponent"]
            # At p = 2 the growing term is a velocity error held for the whole
            # interval, which is the shape the live data showed.
            axis["error_rate"] = math.sqrt(fit["a"])
        report["axes"][name] = axis

    return report


def format_report(name, report):
    lines = [f"{name}"]
    lines.append(
        f"  records {report['n_used']} used of {report['n_records']} "
        f"({report['n_interfering']} interfering), {report['n_tracks']} tracks"
    )
    lines.append(f"  cell    {report['delay_cell_km']:.4f} km x {report['doppler_bin_hz']:.2f} Hz")
    if report["nis_median"] is not None:
        lines.append(f"  NIS     median {report['nis_median']:.3f}, matched {CHI2_2DOF_MEDIAN:.3f}")

    for axis_name in ("delay", "doppler"):
        axis = report["axes"].get(axis_name)
        if not axis:
            lines.append(f"  {axis_name:<8}not enough spread in prediction interval to separate R from Q")
            continue
        if "sigma" in axis:
            lines.append(
                f"  {axis_name:<8}R = {axis['sigma']:.4f} {axis['unit']} = {axis['cells']:.3f} cells, "
                f"model error grows as L^{axis['exponent']:.2f} at {axis['error_rate']:.4g}"
            )
        if axis["scatter_vs_claimed"] is not None:
            lines.append(f"  {'':<8}scatter is {axis['scatter_vs_claimed']:.2f}x what the filter claimed")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", nargs="+", help="innovations.jsonl written by --innovations")
    parser.add_argument("--blah2-config", help="blah2 config.yml to read fc, fs and cpi from")
    parser.add_argument("--fc", type=float, help="Centre frequency in Hz")
    parser.add_argument("--fs", type=float, default=2000000.0, help="Sample rate in Hz (default: 2e6)")
    parser.add_argument("--cpi", type=float, default=0.5, help="Coherent processing interval in s (default: 0.5)")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON")
    args = parser.parse_args(argv)

    capture = load_blah2_capture(args.blah2_config) if args.blah2_config else {}
    fc = args.fc or capture.get("fc")
    if not fc:
        parser.error("centre frequency unknown: pass --fc or --blah2-config")
    fs = capture.get("fs", args.fs)
    cpi = capture.get("cpi", args.cpi)

    reports = {}
    for path in args.files:
        reports[path] = calibrate(load_records(path), fc, fs, cpi)

    if args.json:
        print(json.dumps(reports, indent=2))
        return 0

    for path, report in reports.items():
        print(format_report(path, report))
        print()

    cells = [
        report["axes"][axis]["cells"]
        for report in reports.values()
        for axis in ("delay", "doppler")
        if report["axes"].get(axis, {}).get("cells") is not None
    ]
    if len(cells) > 1:
        spread = max(cells) / min(cells)
        print(f"k across {len(cells)} axis-site pairs: {min(cells):.3f} to {max(cells):.3f} cells, {spread:.2f}x apart")
        print("One k is only shippable if that spread is small. Two axes agreeing is time-bandwidth duality holding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
