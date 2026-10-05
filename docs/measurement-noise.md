# Measurement noise from the resolution cell

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

## The change

`R` is one resolution cell on each axis, derived from the node's own capture
config (`MEASUREMENT_NOISE_CELLS` in `config.py`):

- delay: `(c / fs)^2`, 150 m at 2 MHz;
- Doppler: `(1 / cpi)^2` in Hz, 2 Hz at a 0.5 s CPI, converted to range rate at
  the node's own wavelength.

The SNR scaling is unchanged: it is relative to the 10 dB reference and matches
the estimator's variance going as 1/SNR.

`Q` gains a range-rate random walk, `process_noise.range_rate` = 1e-5 km^2/s^3,
on top of the existing white jerk. The plan measured model error growing as
roughly `L^2` in the prediction interval where white jerk alone gives `L^5`, and
`R` was the term hiding that. A tighter `R` without a `Q` that grows faster over
short intervals loses aircraft where Doppler slews hard. This term is a property
of aircraft motion in range rate, not of the radar, so it is the same at every
site.

The ADS-B start covariance is now `(position / 1000)^2` km^2. It had been
`position / 1000`, a standard deviation used as a variance: 0.1 km^2 for the
100 m default, which was tighter than the old 1 km^2 radar start but would now
be looser than the new one-cell radar start.

## Why one cell, rather than a measured k

The plan asks for `R = (k * cell)^2` with `k` measured once and shown to agree
across sites. `retina_tracker.calibrate` on two VHF nodes' innovations:

| Node | Delay k (cells) | Doppler k (cells) | Claimed vs real delay scatter | Model error growth (delay, Doppler) |
| --- | --- | --- | --- | --- |
| jonathan-node-1, 17 Sep, 30k records | 0.044 | 0.022 | 25x too large | L^1.25, L^1.90 |
| fairforest B, 22-23 Sep, 206k records | 0.029 | 0.083 | 33x too large | L^0.95, L^1.25 |

Both say `R` was far too large, by similar factors. They do not agree on a `k`:
the four values span 3.7x, and the growth exponents disagree too. By the tool's
own rule that is not shippable as a measured constant. One cell is above every
measured value, so it is conservative everywhere, and it needs no fitting: a
detection is reported to within its resolution cell. A measured `k` can replace
it once one value holds across sites, and nothing else needs to change.

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

## Choosing range_rate

Tried at 0, 1e-5 and 1e-4 km^2/s^3 with one-cell `R`:

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
