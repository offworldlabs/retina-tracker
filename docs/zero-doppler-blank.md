# The zero-Doppler blank

blah2's CFAR detector discards every cell within `process.detection.minDoppler`
of zero Doppler (`CfarDetector1D.cpp`). On the fleet that is 15 Hz. A target
whose bistatic range rate passes through zero, which every aircraft does at its
closest point to the bistatic ellipse, is therefore invisible for as long as the
crossing takes, however strong its signal.

The crossing is slow. Near zero Doppler the bistatic range is at an extremum, so
the time spent inside the blank is `2 * v_blank / a`, where `v_blank` is the
blank's half-width in range rate and `a` is the bistatic range acceleration.

| Traffic | Blank (m/s) | Acceleration (m/s^2) | Time in blank |
| --- | --- | --- | --- |
| jonathan-node-1, median | 21.1 | 1.72 | 24.5 s |
| jonathan-node-1, p95 | 21.1 | 10.79 | 3.9 s |
| Atlanta reference, median | 25.4 | 0.58 | 87.6 s |
| Atlanta reference, p95 | 25.4 | 3.72 | 13.7 s |

Genuine crossings on jn1 look exactly as the physics says: they enter at about
+16 Hz, leave at about -16 Hz 6-25 s later, and the delay barely moves, usually
under 0.1 km, because the range is at an extremum.

## What the tracker does at a crossing

Nothing special. A confirmed track that stops being detected is deleted after
`n_delete` missed frames, inside the blank or not, and the target starts a new
track when it reappears on the far side. Joining the two is stitching's job, not
the filter's: for ADS-B aircraft, by their ICAO hex (see "Stitching" below).
Aircraft without ADS-B show as two tracks per crossing until the physics-based
stitch is designed.

## Why the tracker no longer holds tracks through the blank

From 5 to 19 October 2026 a confirmed track was held through the blank: misses
the blank could explain did not count towards deletion, for up to the time the
track had been observed and at most 120 s. It kept most crossings as one track
(on jn1, 20-25 labelled crossings of 26-33 against 2-4 without it). It was
removed because it was the physical weak point of the tracker.

**What went wrong.** Once v0.5.0 put the hold on every node, users reported
massive jumps across the blank joining two detections. On 17 October jn1 showed
5 in 6.5 h and jn2 7 in 47 h. Every one was a held track that, after 25-60 s,
took a single detection hundreds of hertz from where it entered, then died, so
that detection was its last point. One was a real aircraft (jn1 track
261017-000208, ADS-B ab6a39): 85 detections slewing steadily from +50 Hz to
+15.5 Hz, then after 30 s held, a 5.9 dB detection at +71 Hz on the side it came
from, about 110 m/s from where its own acceleration put it.

**Why.** A held track decides on one detection, the weakest evidence there is,
when its gate is at its widest. White jerk grows the range-rate variance as
`q T^3 / 3`: at the fitted `q` a 40 s hold has a gate about 3 x 146 m/s wide.
With no SNR threshold there are always some 4-6 dB detections in it. `Q` was
fitted on prediction intervals up to about 6 s and the hold extrapolated it to a
minute.

**Fixes tried first.** None removed the problem without cost:

- A trapezoid check, `r_b - r_a = T (rdot_a + rdot_b) / 2` against the last
  detection, stopped every jump over 0.5 km and refused none of 89 ADS-B-confirmed
  reacquisitions. It cannot stop range-rate jumps: after a long hold the gate is a
  long thin ellipse whose axis lies close to the trapezoid line, and ab6a39 passed
  it. Kept on the unmerged branch `20261017-reacquisition-check`.
- Holding only tracks with more detections fails: the tracks that jumped were the
  longer ones (median 66 detections against 34 for genuine crossings).
- Bounding how long the straight-line Doppler path spends outside the blank
  refused 6 of 89 genuine reacquisitions, because blah2 also misses aircraft a
  few hertz outside the blank.
- Holding only tracks heading into the blank stops most range-rate jumps (the
  false ones hover at the edge, a median 0.03 Hz/s against 3.4 Hz/s), but not
  ab6a39, which was heading in.
- Sizing the gate during a hold from data. Over 84 ADS-B reacquisitions the
  reappearing range rate lands a median 4 m/s (90% 15 m/s) from the constant
  acceleration prediction, and the false ones a median 148 m/s. But no labelled
  aircraft was held longer than 40 s, and every false jump came from a 20-110 s
  hold, so the tolerance would be extrapolated exactly where it matters.

**Measured effect of removing it.** v0.5.0 replayed with and without the hold on
six datasets, the tracker's own 4 h history from jn1, fairforest B and jn2 on 11
and 17 October, with ADS-B labels joined from the same runs:

| Dataset | Jumps over 0.5 km | Range-rate jumps over 100 m/s | Labelled crossings in one track | In two tracks | Labelled detections tracked | Tracks per aircraft |
| --- | --- | --- | --- | --- | --- | --- |
| jn1 17 Oct, hold / none | 2 / 0 | 4 / 0 | 25 / 2 | 8 / 25 | 1.000 / 0.993 | 1.58 / 1.89 |
| ffb 17 Oct, hold / none | 20 / 0 | 13 / 0 | 14 / 0 | 23 / 36 | 0.976 / 0.975 | 2.53 / 2.76 |
| jn2 17 Oct, hold / none | 0 / 0 | 2 / 0 | 7 / 5 | 1 / 2 | 1.000 / 0.995 | 1.05 / 1.10 |
| jn1 11 Oct, hold / none | 9 / 0 | 8 / 0 | 20 / 4 | 6 / 22 | 0.995 / 0.993 | 1.56 / 1.91 |
| ffb 11 Oct, hold / none | 2 / 0 | 3 / 0 | 19 / 0 | 36 / 46 | 0.988 / 0.984 | 2.27 / 2.37 |
| jn2 11 Oct, hold / none | 2 / 0 | 4 / 0 | 6 / 5 | 0 / 0 | 0.950 / 0.943 | 1.07 / 1.07 |

Without the hold there is no jump of either kind on any dataset. Most crossings
split into two tracks instead, and in every split labelled crossing both sides
are confirmed tracks, which is what stitching needs. Labelled detections tracked
fall 0.1-0.7 points, the detections near the blank edge that no longer join a
track. jn2's crossings mostly survive either way: at UHF the blank is 8 m/s of
range rate, so crossings are short enough to coast through.

## live_score's kinematic check across a crossing

live_score fitted one slope to a track's delay and compared it with the track's
mean Doppler. Across a zero-Doppler crossing both are near zero with signs that
need not agree, so every track that held a crossing read as impossible. It judges
each side of a crossing separately, and a track is impossible if either side is.
That still matters for a stitched chain scored as one track.

## Stitching

We want to retroactively stitch tracks together when we think they belong to the
same aircraft, so that tracks already reported under separate IDs end up as one.
This serves two separate goals.

1. **An ADS-B-verified aircraft shows as a single track.** Fragmented tracks
   carrying the same ICAO hex are one aircraft, however many times the tracker
   broke it. The hex is used rather than the callsign because it is the stable
   identifier and the one the tracker stores. **This is the zero-crossing stitch:**
   when a new track forms on the far side of the blank with the same hex, the two
   are stitched.
2. **The filter only has to model well-constrained physics.** It should not
   widen its gates or process noise to survive a dramatic change of velocity or
   direction. When an aircraft does that, the track may break, and stitching
   rejoins it afterwards: two tracks at clearly different Doppler, one ending at
   roughly the time the other begins, at a continuous delay. No ADS-B is needed
   for this case. Still to be designed.

The second goal is a design rule for the filter, not only a feature. It is why
`R` can be one resolution cell (docs/measurement-noise.md), why `Q` should not be
raised to catch manoeuvres (a wider `Q` brought the jumps back on fairforest B),
and why the blank hold was removed.

What it has to get right: a join between two different aircraft, or between an
aircraft and a CW-tone track, is worse than leaving the fragments apart. Each
fragment keeps its own track id, so fragmentation stays measurable and the event
writer's per-id bookkeeping stays sound (PR #31); a stitch adds a link between
ids. retina-gui has no notion of a link yet: it treats one id as one aircraft.
Today only dead tracks are merged (`Tracker._merge_tracks`: 5 s, 5 km, 50 Hz),
internally, and a merge is never emitted.

**Evidence for the physics stitch.** With the hold off, a fragment's range rate
and acceleration fitted over its last 10 s, and the next fragment's over its
first 10 s, meet under constant jerk (`rdot_b - rdot_a = T (a_a + a_b) / 2`,
and the cubic form for range) for same-hex pairs to 90% within 12-14 m/s and
70 m at gaps under 20 s, and 34 m/s and 200 m at 20-40 s. The nearest pair of
different aircraft missed by 16-42 m/s and 87-218 m below 20 s. Beyond 40 s
manoeuvring dominates and only the hex can join. An earlier attempt, identity
reclaim at confirmation by dead-reckoning the dead track through the filter
(branch `20260913-track-stitching`, parked), gated on the same `T^3` covariance
that made the hold's jumps.
