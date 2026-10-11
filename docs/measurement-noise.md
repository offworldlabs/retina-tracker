# Measurement noise

## The problem: tracks jumping kilometres

A confirmed track would take one stray detection several kilometres off its
path and come straight back, for example 49.16 km to 54.0 km and back to
49.16 km within 2.5 s. Between two consecutive detections in a track, constant
acceleration fixes how far the delay can move:

    delay_b - delay_a = T * (v_a + v_b) / 2,    v = -wavelength * doppler

Measured on jn1 replays, ordinary steps miss that by a median of 15 m and a 90th
percentile of 47 m, which is the radar's real noise. Jumps miss it by kilometres.
Every jump found was a single outlier rather than a sustained drift, usually at
4-6 dB SNR, and about half were in ADS-B-labelled aircraft tracks. None of them
crossed the zero-Doppler blank: the old code made them at the same rate (23 in
13 tracks against 22 in 12 on the same jn1 replay), so they predate the blank
work.

The cause was `R`. Delay measurement noise was a fixed 1 km standard deviation,
about 7 delay cells, and `measurement_noise_scale` inflates it at low SNR, to
about 2 km at 4 dB. The association gate was therefore about +/-6 km in delay
for exactly the detections most likely to be false alarms.

## The change, in two stages

### 5 October: one resolution cell

`R` was set to one resolution cell on each axis, `(c / fs)^2` in delay (150 m at
2 MHz) and `(1 / cpi)^2` in Doppler (2 Hz at a 0.5 s CPI), with the SNR scaling
unchanged. Innovation recordings from jn1 (17 Sep) and fairforest B (22-23 Sep)
both said `R` was 25-33x too large in delay, but disagreed on a measured value by
3.7x, so one cell was the conservative bound that needed no fitting. That is
what removed the jumps in the replays below.

### 11 October: measured on three nodes

All three nodes recorded innovations under the same tracker code for 26 hours
(10-11 October), and the noise was measured from those recordings. The dataset,
scripts and outputs are in the git-ignored `data/calibration/`.

**Method.** The calibration tool's approach, extrapolating innovation scatter
back to zero prediction interval, cannot separate `R` from `Q` at 2 frames per
second: no interval is shorter than one frame, and scatter already rises by half
between 0.5 s and 1 s, so the intercept moved from 2 m to 18 m with the choice of
tracks. Measurement noise was instead taken from smoother residuals: a quadratic
through seven consecutive detections of the same track, centred on each one, so
the motion is absorbed by the fit and what remains is the measurement. Tracks were
either ADS-B-labelled or long confirmed tracks whose delay moves as their own
Doppler says; both filters gave the same answer.

| Node | Band | Delay noise | Doppler noise |
| --- | --- | --- | --- |
| jonathan-node-1 | VHF 213 MHz | 14.7-15.7 m (0.10 cells) | 0.28-0.32 Hz = 0.40-0.44 m/s (0.15 bins) |
| fairforest B | VHF 213 MHz | 12.1-12.2 m (0.08 cells) | 0.29-0.30 Hz = 0.41-0.42 m/s (0.15 bins) |
| jonathan-node-2 | UHF 545 MHz | 14.8-15.9 m (0.10 cells) | 0.63-0.69 Hz = 0.34-0.38 m/s (0.33 bins) |

By SNR, in 2 dB bands across all three nodes, delay variance falls as
`SNR^-0.40`, giving 17.5 m at 10 dB; the old model's `SNR^-1` fell 2.5x too
steeply. Doppler noise barely moves with SNR (fitted exponent 0.06), staying at
about 0.4 m/s from 8 dB to 40 dB within 1.5x on every node.

**What the tracker now uses** (`config.py`):

```
delay      sigma = 0.12 cells at 10 dB (17.5 m at 2 MHz), variance scaled by max(SNR/10 dB, 0.1)^-0.4
range rate sigma = 0.4 m/s, not scaled by SNR
```

Delay noise stays in cells because it held at a tenth of a cell on every node.
Doppler noise is a range rate, not bins, because it held at about 0.4 m/s on both
bands while differing 2.2x in bins. That conclusion rests on a single UHF node.
One SNR-scaling function, `kalman.delay_noise_scale`, serves both the update and
the association gate.

Against the one-cell bound this is about 10x tighter in delay and 7x (VHF) or 3x
(UHF) tighter in Doppler.

### Process noise

**5 October.** With `R` at one cell, `Q` gained a range-rate random walk of
1e-5 km^2/s^3 on top of white jerk, chosen from 0, 1e-5 and 1e-4 on the replays
(table below): without it fairforest B lost crossings, at 1e-4 the jumps came back.

**11 October, fitted against measured `R`.** Each node's ADS-B-labelled tracks
were replayed offline through the filter with `R` fixed at the measured values,
over a grid of white jerk (0 to 1e-4 km^2/s^5) and range-rate random walk (0 to
3e-4 km^2/s^3). The best `Q` is the one whose median NIS sits at its matched
value, 2 ln 2 = 1.39, across prediction intervals, weighted by how many updates
each interval holds. The prediction interval is the true time since the track's
previous update, because the recorded interval omits frames held through the
zero-Doppler blank.

| Node | Traffic | Best white jerk (km^2/s^5) | Best range-rate walk | Median NIS at 0.5 s / 1 s / 1.5-2.5 s / 3-6 s / over 6 s |
| --- | --- | --- | --- | --- |
| jn1 | close, VHF | 1e-5 | 0 | 1.32 / 1.92 / 2.15 / 1.78 / 0.48 |
| fairforest B | VHF | 1e-5 | 0 | 1.33 / 2.21 / 2.59 / 2.08 / 0.63 |
| jn2 | far, UHF | 1e-6 (3e-6 next) | 0 | 1.37 / 1.46 / 1.73 / 1.70 / too few |

Three conclusions:

- **The range-rate walk is not supported once `R` is right.** Every node's best
  fit had it at zero. On 5 October it was compensating for an `R` ten times too
  large, so it is removed.
- **The size of `Q` depends on the site**, 10x between close-traffic jn1 and
  far-traffic jn2, as expected of a quantity dominated by geometry. The floor is
  set at 1e-6 and the adaptive scaling, which widens `Q` by however far the
  filter's own NIS says the motion model is failing, carries it towards 1e-5 at
  close-traffic sites. Its ceiling of 25x covers that. Nothing is set per site.
- **No single Gaussian `Q` matches every interval.** Even at the best fit the
  filter is right at one frame, overconfident at 1-6 s and overcautious beyond
  6 s. That is the signature of aircraft that mostly fly smoothly and
  occasionally manoeuvre. Rather than inflating `Q` to catch the manoeuvres,
  which widens every gate for every track, the filter stays tight and the breaks
  are left to adaptive widening and to stitching tracks afterwards
  (docs/zero-doppler-blank.md).

### ADS-B no longer changes where a track starts

A track whose first detection carried ADS-B used to start with the
`adsb.initial_covariance.position` uncertainty, 100 m. Its state always came from
the radar detection, never the ADS-B position, so that uncertainty had no basis,
and against the measured 18 m it would make labelled tracks start less certain
than unlabelled ones. It is removed, along with the setting; ADS-B labels a track
and earns it association priority, nothing more.

## Measured effect

Each node's own tracker history (every detection it received, classified) was
rebuilt into frames and replayed through both versions under that node's own
configuration, with ADS-B labels joined back from its events.jsonl. The three
nodes differ where it matters: jn1 has close traffic, fairforest B is dominated
by a CW tone and runs `min_snr` 7, and jn2 is UHF at 545 MHz against the others'
213 MHz, so a Doppler cell is 2.6x less range rate there.

| Replay | Jumps over 1 km | Tracks with jumps | Implausible tracks (live_score) | Aircraft tracked | Tracks per aircraft | Labelled crossings held |
| --- | --- | --- | --- | --- | --- | --- |
| jn1, 2.5 h, before | 24 | 14 | 1 of 48 | 22 | 1.50 | 8 of 10 |
| jn1, after | 0 | 0 | 0 of 44 | 21 | 1.57 | 7 of 10 |
| fairforest B, 4 h, before | 328 | 116 | 38 of 507 | 189 | 2.91 | 29 of 134 |
| fairforest B, after | 27 | 22 | 24 of 502 | 188 | 2.76 | 30 of 131 |
| jn2 (UHF), 4 h, before | 30 | 19 | 8 of 57 | no labels | | |
| jn2, after | 0 | 0 | 3 of 49 | no labels | | |

What the "aircraft tracked" changes are:

- jn1 loses a3bdac, a 15-detection track at 53-58 km whose delay jumped about
  4 km back and forth. It existed because the wide gate stitched scattered
  detections together.
- fairforest B loses three "aircraft" of 800-3,300 detections that sat at
  16-26 Hz while sweeping 30-56 km of delay. That is its CW tone, given an ADS-B
  label by coincidental matches. The one-cell gate stops the tone stitching
  across delay. It loses one 3-detection track and gains two aircraft it did not
  track before.

The one real cost found is the jn1 crossing: aircraft a7e1d3 enters the blank on
a track only 1.5 s old, because the tighter gate formed it later, and the
observation limit from docs/zero-doppler-blank.md will not hold a 1.5 s track
across a 15 s gap.

On the synthetic harness, both geometries, 3 x 30 minutes:

| | ID switches | Clutter absorbed | Purity | False tracks | Crossings held | Detection recall |
| --- | --- | --- | --- | --- | --- | --- |
| Far, before | 275 | 9.7% | 0.93 | 1.3 | 29% | 0.915 |
| Far, after | 137 | 2.6% | 0.97 | 0 | 35% | 0.849 |
| Close, before | 1309 | 11.6% | 0.85 | 1.3 | 30% | 0.897 |
| Close, after | 602 | 3.8% | 0.92 | 0 | 33% | 0.822 |

The detection recall cost is the harness's own noise, which is fitted to ADS-B
residuals and runs at about 1.3 cells, so some of its simulated detections fall
outside a one-cell gate. Real radar noise is a fraction of a cell, and no node
replay lost a real aircraft that way.

## Choosing range_rate, 5 October (superseded)

Tried at 0, 1e-5 and 1e-4 km^2/s^3 with one-cell `R`; removed on 11 October:

| | jn1 jumps | fairforest B jumps | fairforest B crossings | Harness close ID switches |
| --- | --- | --- | --- | --- |
| 0 | 0 | 22 | 22 of 131 | not run |
| 1e-5 | 0 | 27 | 30 of 131 | 602 |
| 1e-4 | 0 | 87 | 42 of 133 | 1111 |

Without it fairforest B lost crossings it held before; at 1e-4 the gate widens
enough for jumps to come back. 1e-5 is the only value that kept every node at or
better than before on both jumps and crossings. A half-cell `R` with the same
term was also tried and was no better than one cell anywhere.

## Fixtures that only passed because the gate was huge

Fourteen tests failed when `R` shrank, and every one was a fixture that was
physically impossible: a delay moving against its own Doppler, often several
times too fast, or a filter updated without being predicted forward. The old
gate accepted them. They now move as their own Doppler says. The intermittent
ADS-B integration test, previously a strict expected failure for exactly this
reason, now takes its Doppler from the bistatic range rate and passes.

## Open

- jn2 carries no ADS-B labels at all over 4 hours, which looks like the
  wrong-port fault fairforest B had until September. Its aircraft recall cannot
  be checked until that is fixed.
- A measured `k` agreeing across sites would tighten this further. The UHF node
  has no innovations recording yet; one would show whether `k` in cells holds
  across carriers, which is the claim this design rests on.
- The adaptive process-noise scale still multiplies only the jerk term.
