# Anomaly detection

The tracker currently has **no anomaly detection**. The legacy detector was
removed on 2026-10-09 so that a new one can be built from scratch on the
improved tracker.

## Why the old detector went

- **It judged ADS-B, not the radar.** Nine of its ten checks read ADS-B fields
  (reported speed, heading, position, altitude, hex). Only the supersonic check
  used the radar, through a fixed Doppler threshold. A target without a
  transponder, which is what a passive radar is uniquely placed to see, got one
  test.
- **On live data it only caught the tracker's own mistakes.** Over 34.5 hours of
  v0.4.0 on jonathan-node-1, 4 of 1,315 tracks were flagged, all as identity
  swaps on ADS-B aircraft: tracks that had jumped onto a second aircraft. Once the
  gate was sized from the resolution cell (docs/measurement-noise.md), it flagged
  nothing in 8.8 hours.
- **It latched on fixed thresholds.** Three net observations raised a flag for
  the track's lifetime, so its false-alarm rate was whatever the thresholds
  happened to give at each site.

## What stays on the wire, and why

`events.jsonl`, `GET /frame` and the history record still carry the three
fields consumers expect, because retina-telemetry requires them on every track
and they are required in its `node-ingest-v1` schema:

| Field | Value now |
| --- | --- |
| `is_anomalous` | always `false` |
| `anomaly_types` | always `[]` |
| `max_velocity_ms` | the largest speed floor seen, `|range rate| / 2`, in m/s |

`max_velocity_ms` used to mix ADS-B ground speed with a Doppler-derived value.
It is now radar only. Bistatic range rate can never exceed twice the target's
speed, so `|range rate| / 2` is a floor on the target's speed in any geometry.

The writers (`TrackEventWriter.write_event`, `DetectionHistory`) still accept
real values for the flags, so the new detector can set them without a wire
change.

## What was removed

All of the per-track checks and their state in `Track` (supersonic from Doppler
and from ADS-B speed, instant acceleration, 10 g acceleration, instant turn,
sustained orbit, frozen position, altitude jump, identity swap, long hover),
their thresholds in `config.py`, the synthetic scorer's anomaly metrics, the
anomaly test suites (`test_anomaly_detection.py`, `test_anomaly_mofn.py`,
`test_integration_synthetic.py` and its `motion_patterns.py`), and
`docker-compose.integration-test.yml`, which existed to confirm anomaly flags
and no longer ran (it built directories this repo does not contain).

One behaviour change follows from removing the identity-swap check: a track's
`adsb_hex` is now the first valid hex it was labelled with, and no longer
switches when a second aircraft's hex appears on it.

## The rebuild

The proposal under discussion, in order:

1. **Prerequisites.** Calibrated `R` and `Q`, so the filter's uncertainties are
   honest; retroactive stitching, so a fragmented ordinary aircraft cannot look
   like an impossible manoeuvre; and a physics check at confirmation, so
   artefact tracks are never judged.
2. **Target anomalies from the radar alone**, using bounds the geometry
   guarantees: supersonic only if `(|range rate| - 3 sigma) / 2` exceeds Mach 1.
   Acceleration checks must first clear the geometric term that grows near the
   transmitter and receiver, which needs each node's antenna positions.
3. **Integrity anomalies**: a persistent disagreement between a radar track and
   the delay and Doppler its own ADS-B report predicts, which blah2-api already
   computes. This is the check only a radar can make.
4. **ADS-B feed checks** (frozen position, altitude jumps) reported as feed
   quality, not as anomalous targets.
5. **Evidence accumulated with a known false-alarm rate**, a cumulative sum of
   log-likelihood ratios with a threshold chosen for a target interval between
   false alarms, the same on every node.
6. **Stitching must not hide anomalies.** Two tracks that meet in delay but need
   more acceleration than any airframe should become an anomaly candidate, not
   two unrelated tracks.

Validation: false alarms counted per aircraft-hour on ADS-B-verified ordinary
traffic from every node, and detections measured by injecting synthetic
supersonic and high-g targets into real recorded frames.
