# synthetic

Life-like blah2 detection frames with ground truth, for measuring tracker
performance.

Live data cannot tell you whether two detections were the same aircraft, so it
cannot tell you whether a track was right. This package generates a stream that
looks like a real node's and records what actually happened, so track quality
becomes a number instead of a judgement.

## Use

```bash
# 15 minutes of traffic, plus the truth sidecar
python -m synthetic.generate -o data/synthetic.detection -d 900 --stats

# replay it through the tracker exactly as if it came off a node
python -m retina_tracker.track_detections data/synthetic.detection \
    -c config.yaml -o data/synthetic.tracks.json

# score the result against truth
python -m synthetic.score data/synthetic.detection data/synthetic.tracks.json
```

`generate` writes two files. The `.detection` file is shaped exactly like a live
capture, including the `alt` key that blah2-api really sends (retina-tracker
reads `alt_baro`, so this dataset exercises that mismatch rather than papering
over it). The `.truth.jsonl` sidecar records, per frame, which aircraft produced
each detection and where every aircraft actually was.

Useful switches: `-d` duration, `-s` seed, `--concurrent` traffic density,
`--clutter` false alarms per frame, `--fc` centre frequency.

## Calibration

Every constant in `profile.py` marked as measured came from a live capture of
the Atlanta reference node on the standard ±300 Hz envelope: 125 frames over
118 s, 1387 detections, 35 aircraft. `tests/test_synthetic.py` asserts the
generator still reproduces them, which is what stops the fitted knobs in the
same module drifting somewhere unrealistic.

Fourteen of sixteen reference statistics land within 15%, most within 5%:
detections per frame and their target/clutter split, both SNR populations,
ADS-B match rate and residuals, continuity and burst length, frame cadence.

Three properties matter more than the headline counts and are easy to get wrong:

- **Dropouts come in bursts.** Continuity is 0.37 with a mean detected run of
  several frames, modelled as a two-state Markov chain. An independent
  per-frame draw at the same rate gives runs of 2-3 and makes association far
  easier than it is in the field.
- **Clutter is one-shot.** 90% of false alarms occupy a given 1 km × 5 Hz cell
  in exactly one frame. The clutter filter has already removed the zero-Doppler
  returns that would otherwise track.
- **ADS-B is not uniformly fresh.** Most aircraft refresh near blah2-api's 1 s
  cache, but a minority go many seconds between updates. That minority supplies
  the heavy tail of the residual distribution and is the only way a lat/lon
  repeats across consecutive frames, which is what `position_mismatch` reads as
  a frozen GPS.

### Known deviation

The generator reproduces the reference's *mean* detections per aircraft but not
its skew: real aircraft entered and left the matched population faster than
bistatic geometry alone explains. Individual tracks therefore live somewhat
longer here than in the field, making track initiation marginally less frequent
than reality. `profile.py` records the numbers.

### Site defaults

The receiver is the Atlanta position committed in `retina_tracker/config.yaml`.
The transmitter and `fc` are **not** exposed by any node API, so the defaults
are a documented stand-in in the VHF-high broadcast band chosen to match the
observed Doppler spread. Pass real values to `Site` when they are known; nothing
outside the defaults assumes Atlanta.

## What the scorer reports

`track_recall` and `detection_recall` for what was found; `fragmentation` for
how many tracks each aircraft was broken into; `purity` and `id_switches` for
whether a track held one identity; `false_track_rate` and `clutter_absorbed` for
what the tracker invented; `mislabelled_track_ids` for tracks whose ID no longer
matches the aircraft they are following.

The generated traffic is entirely ordinary, so `anomaly_false_positive_rate` is
exactly that: every anomaly raised against this dataset is false by
construction.
