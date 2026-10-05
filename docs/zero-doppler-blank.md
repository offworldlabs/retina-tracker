# Holding tracks through the zero-Doppler blank

blah2's CFAR detector discards every cell within `process.detection.minDoppler`
of zero Doppler (`CfarDetector1D.cpp`). On the fleet that is 15 Hz. A target
whose bistatic range rate passes through zero, which every aircraft does at its
closest point to the bistatic ellipse, is therefore invisible for as long as the
crossing takes, however strong its signal.

## Why tracks used to die there

The crossing is slow. Near zero Doppler the bistatic range is at an extremum, so
the time spent inside the blank is `2 * v_blank / a`, where `v_blank` is the
blank's half-width in range rate and `a` is the bistatic range acceleration.

| Traffic | Blank (m/s) | Acceleration (m/s^2) | Time in blank |
| --- | --- | --- | --- |
| jonathan-node-1, median | 21.1 | 1.72 | 24.5 s |
| jonathan-node-1, p95 | 21.1 | 10.79 | 3.9 s |
| Atlanta reference, median | 25.4 | 0.58 | 87.6 s |
| Atlanta reference, p95 | 25.4 | 3.72 | 13.7 s |

A confirmed track was deleted after `n_delete` = 10 missed frames, 5 s at 2 Hz.
Only the fastest crossings, close to the baseline, could coast through. On the
synthetic harness 1% of crossings kept their track at Atlanta geometry and 6% at
close geometry. This was not a gate problem: the track was gone before the
target came back.

## What the tracker does now

A confirmed track's miss does not count towards deletion when the blank can
explain it (`Track.is_occluded`):

- its predicted Doppler is inside the blank, widened by one Doppler bin because
  the centroider can report a peak slightly inside the edge;
- or its prediction has crossed zero since its last detection;
- or its association gate, `sqrt(gate_threshold)` standard deviations of
  predicted Doppler, reaches into the blank.

Once the prediction has entered or crossed the blank this latches until the next
association. The exit time depends on the acceleration estimated going in, and
the harness showed that estimate is too poor to trust: its sign was wrong in
4 of 20 failed crossings inspected, and its magnitude was often several times
the true slew. A track whose prediction leaves the far side early would otherwise
start counting misses while the target is still hidden.

Covariance keeps growing while the track is held, because the `n_coast` freeze
reads `n_missed`, which held misses do not increment. The gate on exit therefore
reflects how long the target was unobserved.

A track is held for no longer than it had been observed before the blank
(`Track.max_occluded_s`: last update minus birth). This is what keeps the hold
from manufacturing false tracks. On jonathan-node-1, an unbounded hold let
tracklets of 5-7 detections, sitting just outside the blank close to the
receiver, coast for 27-51 s and then snap onto unrelated detections 3-11 km and
60-220 Hz away. A held prediction travels on an acceleration estimate, and a
track observed for 4 s has no business extrapolating for 30. The genuine
crossings on the same data came from tracks observed for minutes. The rule has
no tuned number in it: the ratio of hold to observation is 1.

It is also capped at `MAX_OCCLUDED_S` = 120 s, so a long track that never
reacquires cannot live forever. That cap is not a tuned threshold: harness
results were identical to three decimal places at 60, 120 and 240 s.

Freezing the gate's growth while held (counting held frames towards `n_coast`)
was tried as the alternative fix and made jn1 worse, 10 implausible tracks
against 6: the gate's width was not the problem, the prediction's travel was.

Tentative tracks are never held. Their lifetime is `n_window` frames regardless.

## Where the blank comes from

`radar.doppler_blank`, read from blah2's `process.detection.minDoppler` when the
tracker is started with `--blah2-config`. Unset means unknown, and then nothing
is held, which is the old behaviour. No value is shipped in `config.yaml`: a
guessed blank would hold tracks at a node that does not blank, so it follows the
same rule as the ambiguity bounds.

## Measured effect

### jonathan-node-1, replayed

The tracker's in-memory history from 2.5 h of live jn1 operation (17,930 frames,
31,373 detections, every one it received with its classification) was rebuilt
into frames, with ADS-B labels joined back from that run's events.jsonl, and
replayed through both versions under jn1's own configuration. The old code
reproduces the live run exactly (80 confirmed tracks, 22 aircraft), which is the
check that the replay is faithful.

| | Old | Held, unbounded | Held, bounded by observed time |
| --- | --- | --- | --- |
| Labelled crossings held | 0 of 10 | 9 of 10 | 8 of 10 |
| Kinematically implausible tracks | 1 of 60 | 3 of 49 | 1 of 48 |
| Tracks per identified aircraft | 1.82 | 1.41 | 1.50 |
| Confirmed tracks | 80 | 59 | 65 |
| Aircraft identified | 22 | 22 | 22 |

Implausible is live_score's kinematic check, after the fix below. The one
implausible track in the bounded column is the same 4-detection track as the
old one. Labels are joined from the old run's output, so a detection only the
new code tracked carries no label, which biases the crossing count against the
new code.

Genuine crossings on jn1 look exactly as the physics says: they enter at about
+16 Hz, leave at about -16 Hz 6-25 s later, and the delay barely moves, usually
under 0.1 km, because the range is at an extremum.

### Synthetic harness

Synthetic harness, 3 seeds of 30 minutes each, at Atlanta geometry and at close
geometry (`spawn_radius_km=12`, jonathan-node-1-like). Crossings held is the
scorer's `blank_crossing_rate`: of the crossings whose entry detection was in a
track, the share where the same track holds the aircraft on the far side.

With `R` as shipped:

| | Crossings held | ID switches | Clutter absorbed | Detection recall | Track recall |
| --- | --- | --- | --- | --- | --- |
| Atlanta, before | 0.9% | 262 | 8.3% | 0.912 | 0.997 |
| Atlanta, after | 29.2% | 275 | 9.7% | 0.915 | 0.997 |
| Close, before | 5.8% | 1193 | 10.6% | 0.890 | 0.980 |
| Close, after | 29.9% | 1309 | 11.6% | 0.897 | 0.970 |

With delay `R` set to the harness's true scatter (0.04 km^2), to show how this
combines with the Phase 2 calibration:

| | Crossings held | ID switches | Clutter absorbed |
| --- | --- | --- | --- |
| Atlanta, before | 0.4% | 96 | 2.0% |
| Atlanta, after | 33.1% | 106 | 2.6% |
| Close, before | 5.2% | 457 | 3.1% |
| Close, after | 33.6% | 506 | 4.0% |

Without the observed-time bound the harness held 38-50%, but that version made
false tracks on real data, so the harness's numbers are not the deciding ones.
The remaining cost is a held track occasionally taking a detection that is not
its own: ID switches rise 2-10% and clutter absorption under 1.5 points. A
calibrated `R` shrinks that cost along with everything else.

## live_score's kinematic check across a crossing

live_score fitted one slope to a track's delay and compared it with the track's
mean Doppler. Across a zero-Doppler crossing both are near zero with signs that
need not agree, so every track that held a crossing read as impossible. It now
judges each side of a crossing separately, and a track is impossible if either
side is. Before that fix the bounded version above read 4 implausible tracks,
two of them ADS-B-labelled aircraft with clean crossings.

## What is still lost, and why

About two thirds of reached crossings still fail on the harness. Instrumenting them on the close
harness, with `R` calibrated:

- **Died before exit.** The prediction never entered the blank and drifted away
  from it, usually because the acceleration estimate had the wrong sign. These
  are the largest remaining group.
- **Hijacked while held.** The widening gate takes another aircraft's detection
  or clutter, which ends the hold. With `R` as shipped this outnumbered the
  tracks reacquiring their own target.
- **Missed at exit.** The target emerged outside the gate.

## Tried and not shipped

**Identity handover across the blank.** Under constant acceleration the range
covered between two times is exactly the mean of the two range rates times the
interval: `r_b - r_a = T (v_a + v_b) / 2`. That links a track's last update
before the blank to a new track confirmed after it, with no acceleration
estimate. On raw detections it separates true crossings (median residual
0.22 km, p95 0.9 km) from false pairings (median 2.4-3.8 km). As an identity
handover at confirmation, with parked tracks excluded from ordinary association,
it raised Atlanta crossings to 64-70%. In dense close traffic it cost track
recall, 0.97 down to 0.94-0.95, which means it merged different aircraft. The
gate's model-error term, `q T^5 / 120` from the filter's own jerk floor, is too
loose at long gaps while `Q` is uncalibrated. Worth revisiting after Phase 2.

**Log-determinant association cost.** Replacing the SNR-weighted Mahalanobis
cost with `d^2 + ln|S|`, which stops the widest gate winning by default, moved
crossings by 2 points and nothing else measurably.

## Universality

Nothing here is per-site. The blank comes from the node's own blah2 config, the
widening is one Doppler bin from its own CPI, the gate reach uses the existing
`gate_threshold`, and the cap does not affect results. Every change was checked
at two geometries 5x apart in median range and at two measurement noise levels,
and kept only where it helped both.

`R` has since been sized from the resolution cell (docs/measurement-noise.md),
which cost one real crossing on jn1: a track formed later under the tighter gate,
and the observation limit would not hold it across the gap.

The harness is fitted to one site (Atlanta) and its noise to that site's ADS-B
residuals, which overstate radar noise about 9x. The trends above are evidence;
the absolute numbers are not a prediction for any node.
