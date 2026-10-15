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

**Method.** The former calibration tool's approach (`retina_tracker.calibrate`, since
removed), extrapolating innovation scatter
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

The delay scaling has no floor at high SNR, and the data does not ask for one up
to 32 dB, the strongest band with enough detections (fairforest B, 392 between
28 and 32 dB). In every band from 8 to 32 dB on every node, the robust width
stays at 0.38-0.58 of the width `R` gives that SNR. The share of residuals past
3 sigma of `R` falls from about 1% at 8-12 dB to none above 24 dB. Above 32 dB
the scale is an extrapolation: at 40 dB it gives 9 m. If a node with stronger
returns shows its strongest aircraft refused by the gate, floor the scale there.

**Heavy tails, and sizing `R` for the gate.** The widths above are robust
estimates (median absolute deviation), which describe the typical detection and
ignore the tails. Real radar noise has heavy tails. Used directly as `R`, the
chi-squared gate of 9, which should admit 98.9% of genuine measurements, refused
far more:

| Genuine aircraft measurements outside the gate | Delay beyond 3 sigma | Range rate beyond 3 sigma | Joint gate |
| --- | --- | --- | --- |
| Gaussian noise | 0.3% | 0.3% | 1.1% |
| jn1 | 3.1% | 4.0% | 7.9% |
| fairforest B | 4.2% | 2.4% | 7.7% |
| jn2 | 4.0% | 7.2% | 12.2% |

Replayed, that cost 5-7% of ADS-B-labelled detections against the deployed code,
and raising `Q` recovered little of it, which is how the cause was pinned to `R`.
The scale on both axes that makes the gate admit 98.9% of real measurements was
2.00x at jn1, 2.05x at fairforest B and 2.10x at jn2: the same at every node and
on both bands, so it is a property of blah2's measurements, not of a site.

**What the tracker now uses** (`config.py`):

```
delay      sigma = 0.25 cells at 10 dB (37 m at 2 MHz), variance scaled by max(SNR/10 dB, 0.1)^-0.4
range rate sigma = 0.8 m/s, not scaled by SNR
```

That is the robust width, 0.12 cells and 0.4 m/s, times the coverage factor of
two. Delay noise stays in cells because it held at a tenth of a cell on every
node. Doppler noise is a range rate, not bins, because it held at about 0.4 m/s
on both bands while differing 2.2x in bins; that conclusion rests on a single UHF
node. One SNR-scaling function, `kalman.delay_noise_scale`, serves both the
update and the association gate.

Against the one-cell bound this is about 4x tighter in delay and 3.5x (VHF) or
1.4x (UHF) tighter in Doppler.

The cost of sizing `R` for the gate is that the filter weights measurements as if
they were noisier than the typical one: its median NIS sits at 0.35-0.75 rather
than the matched 1.39. A filter that models the heavy tails directly could have
both, and is a larger change for later.

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
- **The fitted size of `Q` depends on the site**, 10x between close-traffic jn1
  and far-traffic jn2, as expected of a quantity dominated by geometry. The floor
  is set at 1e-6. The adaptive scaling was expected to widen it towards 1e-5 at
  close-traffic sites, but in replay it barely engaged with the robust-width `R`
  (90th percentile 1.3-1.6x) and not at all with the final `R` (1.0x). It did not
  need to: replayed with the floor at 3e-6 or 1e-5, every node recovered little
  recall and got more jumps. Nothing is set per site.
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

### No SNR threshold

`tracker.min_snr` is removed. It set detections below a per-node SNR aside before
the tracker saw them, and was set by hand per node: 4 dB on jn1 and jn2, 7 dB on
fairforest B. That is a per-site tuning, and with measured noise the gate and the
physics checks are what reject noise, so every detection with a finite SNR is now
tracked. Low-SNR detections are not trusted blindly: their delay noise is larger,
by the measured `SNR^-0.4` scaling, so the gate admits them less readily.

A detection with a non-finite SNR is rejected and counted in
`n_detections_rejected`, because SNR now scales the measurement noise and a NaN
would reach the covariance.

The `below_snr` detection class stays on the wire, always empty, because
retina-gui draws it as a layer. A config that still sets `min_snr` is loaded and
ignored. Follow-ups outside this repo: retina-gui's config editor still offers
the field and describes it as discarding detections, and retina-node's
`default.yml` still sets it.

### Rerunning the calibration

Calibration is an offline job, run once per change of hardware or band rather than
per node. The old `retina_tracker.calibrate` was removed because its method gives
confident wrong answers at 2 frames per second. To repeat the 11 October fit:

1. **Record.** Start the tracker with `--innovations <file>`: one line per Kalman
   update with the prediction, the measurement and its SNR. Record a full day,
   with daytime traffic, on every node being calibrated, keeping each node's
   events file for the ADS-B labels. The writer truncates its file when the
   tracker restarts, so copy it off regularly.
2. **Select aircraft.** Keep records from ADS-B-labelled tracks, joining labels on
   `track_id` only from events after the tracker started (ids restart every run),
   and separately from long confirmed tracks whose delay moves as their Doppler
   says. Drop records the occupancy map marked `interfering`.
3. **Measure `R` from smoother residuals.** For each run of seven consecutive
   detections of one track, fit a quadratic in time to delay and to Doppler and
   take the centre detection's residual. Robust sigma (1.4826 x median absolute
   deviation), inflated by `sqrt(7 / 4)` for the fit's three parameters, in SNR
   bands to fit the SNR exponent.
4. **Fit `Q` by replay.** With `R` fixed, run each aircraft track's own detections
   through the filter offline over a grid of process noise, and keep the `Q` whose
   median NIS sits at 2 ln 2 = 1.39 across prediction intervals. Use the record's
   `interval`, the true time since that track's previous update; `dt` and
   `n_missed` miss frames held through the zero-Doppler blank.
5. **Validate by replay** of every node's recorded history before shipping.

The scripts used on 11 October are kept with the dataset in the git-ignored
`data/calibration/fit/`, so they exist only on the machine that ran them.

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

## Validation on recorded histories, 11 October

Each node's last 4 hours of history (every detection the tracker received,
including those below its old `min_snr`) were replayed through three versions
under that node's own configs, with ADS-B labels from the same run:

- **A**, v0.4.0: 1 km delay noise, 5 s deletion, `min_snr`;
- **B**, the code deployed for the recordings: blank hold, one-cell `R`;
- **final**, this change: measured `R` sized for the gate, fitted `Q`, no `min_snr`.

| | jn1: A / B / final | fairforest B: A / B / final | jn2 (UHF): A / B / final |
| --- | --- | --- | --- |
| Jumps over 1 km | 88 / 6 / 4 | 28 / 8 / 4 | 79 / 4 / 2 |
| ADS-B-labelled detections tracked | 99.7% / 100% / 99.5% | 99.4% / 99.9% / 98.8% | 98.6% / 100% / 95.0% |
| Aircraft mostly tracked | 45 / 45 / 45 of 45 | 137 / 137 / 135 of 137 | 15 / 15 / 14 of 15 |
| Labelled crossings held | 5/26 / 20/26 / 20/26 | 1/53 / 19/55 / 17/50 | 5/6 / 6/6 / 6/6 |
| Tracks per ADS-B aircraft | 1.87 / 1.58 / 1.56 | 2.32 / 2.19 / 2.27 | 1.07 / 1.07 / 1.07 |
| Median NIS at one frame (matched 1.39) | 0.03 / 0.04 / 0.45 | 0.02 / 0.04 / 0.35 | 0.17 / 0.06 / 0.45 |

Most of the improvement over v0.4.0 came with B. Against B, this change halves
the remaining jumps and removes the per-site SNR threshold, for slightly lower
recall at fairforest B and jn2. The three aircraft no longer mostly tracked are
marginal: one has a 454 m jump between consecutive detections, which the gate is
right to refuse; one is three detections over one second, too short to confirm;
one is six sparse detections whose Doppler changes fast, which is stitching's job.
Labels come from the run made by B, which slightly favours B.

**The synthetic harness cannot validate this `R`.** Its noise was fitted to ADS-B
residuals and runs at about 1.3 delay cells with heavy tails, roughly nine times
real radar noise, so on it this change keeps 20% of detections. It needs its
noise model refitted to the measurements here before it can serve as a
regression check on the filter again. Until then the repo has no end-to-end
tracker test: `test_integration_synthetic.py` was removed with the anomaly checks
(see `docs/anomaly-detection.md`) and nothing in `tests/` replaces it. The
end-to-end check is the replay of recorded node history described under
"Rerunning the calibration", which runs locally against data that is not in the
repo.

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

- jn2's ADS-B labels work again since the 6 October rollout, but its tar1090 sees
  about 24 aircraft against 273 at jn1, so its labelled sample stays thin.
- The Doppler-noise finding, a fixed range rate rather than a fixed share of a
  bin, rests on one UHF node. A second UHF site would settle it.
- The synthetic harness's noise model needs refitting to the measured radar
  noise, heavy tails included, before it can check the filter again.
- A filter that models the heavy tails directly (for example a Student-t
  measurement model) would let `R` describe the typical detection and still
  admit the tail.
- retina-gui's config editor still offers `min_snr` and describes it as
  discarding detections, and retina-node's `default.yml` still sets it.
