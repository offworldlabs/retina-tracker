"""Measured statistics of a real blah2 detection stream.

Provenance: a live capture of the Atlanta reference node (radar3.retnode.com,
ADS-B enabled, on the standard +/-300 Hz envelope) taken 2026-09-07: 125 frames
over 118 s, 1387 detections, 35 distinct aircraft. Every number below was
measured from that capture, and `tests/test_synthetic.py` asserts the generator
reproduces them.

Two properties matter more than the rest and are easy to get wrong:

* A real target is missing from about two frames in three, and its dropouts
  come in bursts, not independently. Continuity is 0.37 with a mean detected
  run of 7 frames. A per-frame Bernoulli draw at p=0.37 gives runs of 2-3 and
  makes association far easier than it is in the field.
* Unmatched detections are one-shot false alarms, not persistent ghosts. 90%
  of them occupy a given 1 km x 5 Hz cell in exactly one frame; the clutter
  filter has already removed the zero-Doppler returns that would otherwise
  track.
"""

FRAME_INTERVAL_S = 0.94
FRAME_INTERVAL_JITTER_S = 0.02
FRAME_SKIP_PROBABILITY = 0.02

CPI_S = 0.5
SAMPLE_RATE_HZ = 2_000_000
DELAY_BIN_KM = 0.149896
DELAY_MIN_KM = -1.499
DELAY_MAX_KM = 59.958
DELAY_FLOOR_KM = 0.749
DOPPLER_MAX_HZ = 300.0
DOPPLER_BIN_HZ = 2.0
DOPPLER_BLANK_HZ = 15.0

DETECTIONS_PER_FRAME = 11.1
TARGET_DETECTIONS_PER_FRAME = 8.56
CLUTTER_PER_FRAME = 2.54

CONTINUITY = 0.37
LONGEST_RUN_MEDIAN = 7
LONGEST_RUN_MAX = 49

SNR_FLOOR_DB = 3.5
TARGET_SNR_MEDIAN_DB = 11.09
TARGET_SNR_P95_DB = 16.48
CLUTTER_SNR_MEDIAN_DB = 6.38
CLUTTER_SNR_P95_DB = 13.85

ADSB_MATCH_RATE = 0.77
ADSB_DELAY_TOLERANCE_KM = 2.0
ADSB_DOPPLER_TOLERANCE_HZ = 5.0

DELAY_RESIDUAL_MEDIAN_KM = 0.15
DELAY_RESIDUAL_P95_KM = 0.59
DOPPLER_RESIDUAL_MEDIAN_HZ = 0.66
DOPPLER_RESIDUAL_P95_HZ = 3.27


CLUTTER_DELAY_DECADE_WEIGHTS = (0.287, 0.142, 0.174, 0.110, 0.208, 0.079)
CLUTTER_DOPPLER_BAND_WEIGHTS = (
    0.0757,
    0.0726,
    0.1041,
    0.0662,
    0.2050,
    0.1924,
    0.0915,
    0.0599,
    0.0694,
    0.0631,
)
CLUTTER_DOPPLER_BAND_HZ = 60.0

DOPPLER_SORT_INVERSION_RATE = 0.22

# Known deviation. The generator reproduces the reference's mean detections
# per aircraft, but not its skew: real aircraft entered and left the matched
# population faster than bistatic geometry alone explains (reference median 22
# hits per aircraft over 118 s against roughly 37 here, and 35 distinct
# aircraft against roughly 27). Coverage out to 60 km of bistatic range means
# a 200 m/s target is genuinely available for several minutes, so closing this
# would take breaking a metric that does match. The effect is that individual
# tracks live slightly longer here than in the field, which makes track
# initiation marginally less frequent than reality.
TARGET_HITS_MEDIAN = 22
TARGET_HITS_MAX = 110
BISTATIC_RANGE_RATE_MEDIAN_MS = 97.0
BISTATIC_RANGE_RATE_P95_MS = 352.0
DOPPLER_RATE_MEDIAN_HZ_S = 0.570
DOPPLER_RATE_P95_HZ_S = 2.329


# ---------------------------------------------------------------------------
# Fitted, not measured.
#
# These have no direct counterpart in the capture. They are the knobs of the
# generator's own models, chosen so that the statistics it produces land on the
# measured values above. `tests/test_synthetic.py` is what holds them honest:
# change one and the assertions on the measured constants are what fail.
#
# DETECTED_RUN_MEAN is the mean of the Markov chain's detected runs, and is
# deliberately shorter than LONGEST_RUN_MEDIAN, which is the median across
# targets of each target's single longest run.
# ---------------------------------------------------------------------------

DETECTED_RUN_MEAN = 2.9
CHAIN_CONTINUITY = 0.375
CONCURRENT_AIRCRAFT = 32
SPAWN_RADIUS_KM = 46.0
HEADING_SPREAD_DEG = 28.0

# A real tar1090 feed reaches far past radar coverage (the nodes run a 120 nm
# adsb.lol fallback), so most of the ADS-B the matcher sees belongs to aircraft
# the radar cannot possibly detect. That surplus is what lets clutter steal a
# plausible identity, and it is the mechanism behind the false position_mismatch
# and identity_swap flags measured on live data.
ADSB_FEED_AIRCRAFT = 240
ADSB_EQUIPPED_RATE = 0.98

# blah2-api refetches tar1090 at most once a second (CACHE_INTERVAL) and
# tar1090 itself refreshes a given aircraft only every few seconds, so frames
# arriving ~0.94 s apart routinely carry a repeated lat/lon against a non-zero
# ground speed. That staleness is what retina-tracker's position_mismatch check
# reads as a frozen GPS, and it was the single largest source of false anomaly
# flags on both live nodes. Part of the measured ADS-B residual is this, not
# radar measurement error.
# The feed is not uniformly fresh. Most aircraft refresh close to the 1 s API
# cache, but a minority (weak reception, MLAT, edge of the receiver's range)
# go many seconds between updates. That minority does two things at once: it
# supplies the heavy tail of the ADS-B residual distribution, and it is the
# only way a lat/lon repeats across two consecutive frames, which is what
# position_mismatch reads as a frozen GPS.
ADSB_REFRESH_FAST_S = (0.4, 2.0)
ADSB_REFRESH_SLOW_S = (3.0, 8.0)
ADSB_STALE_FRACTION = 0.05
ADSB_FEED_RADIUS_KM = 220.0
TARGET_STRENGTH_SIGMA_DB = 1.0
RANGE_TILT_DB_PER_DECADE = 1.5
RANGE_TILT_REFERENCE_KM = 20.0
SNR_SELECTION_OFFSET_DB = -0.35

DELAY_NOISE_DF = 5.0
DELAY_NOISE_SCALE_KM = 0.158
DOPPLER_NOISE_DF = 2.0
DOPPLER_NOISE_SCALE_HZ = 0.66
NOISE_SCALE_CLIP = (0.45, 1.9)
