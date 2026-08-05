"""M-of-N anomaly latching.

Flags used to be set by a single observation: one clutter Doppler spike
mis-associated into a track branded it supersonic for life (observed on
staging: an ADS-B-seeded subsonic track flagged supersonic while its
current Doppler read 0.0 Hz).  These tests pin the replacement contract:

- per-frame streams need ANOMALY_RAISE_N net anomalous observations to
  latch, so a lone noisy datapoint never flags;
- clean observations dilute pre-latch evidence, so scattered outliers
  never accumulate to the threshold;
- once latched, the flag is permanent — an anomalous object stays
  anomalous after it stops exhibiting the behaviour;
- discrete debounced streams (identity_swap, long_hover) latch on first
  occurrence — their checks already span multiple frames.
"""

from retina_tracker.config import set_config
from retina_tracker.track import Track

TS = 1_700_000_000_000


def make_track():
    # fc=100 MHz -> Mach-1 Doppler threshold = 2 * 343 * 1e8 / c ~= 228.8 Hz
    set_config({"radar": {"center_frequency": 100_000_000}, "adsb": {"enabled": False}})
    det = {"delay": 50.0, "doppler": 10.0, "snr": 15.0}
    return Track(det, TS, kf=None)


def spike(track):
    """One supersonic-Doppler observation (well above the 228.8 Hz threshold)."""
    track._check_doppler_anomaly({"delay": 50.0, "doppler": 500.0, "snr": 15.0})


def clean(track):
    """One clean-Doppler observation."""
    track._check_doppler_anomaly({"delay": 50.0, "doppler": 10.0, "snr": 15.0})


class TestMofNLatch:
    def test_single_spike_does_not_flag(self):
        """The staging false positive: one clutter Doppler spike in an
        otherwise-clean track must not brand it supersonic."""
        t = make_track()
        spike(t)
        assert not t.is_anomalous
        assert t.anomaly_types == set()

    def test_scattered_outliers_never_accumulate(self):
        """Clean observations dilute pre-latch evidence: sparse spikes with
        clean frames between them must not latch, however many arrive."""
        t = make_track()
        for _ in range(20):
            spike(t)
            clean(t)
            clean(t)
        assert not t.is_anomalous

    def test_sustained_supersonic_latches(self):
        t = make_track()
        for _ in range(Track.ANOMALY_RAISE_N):
            spike(t)
        assert t.is_anomalous
        assert t.anomaly_types == {"supersonic"}

    def test_latched_flag_is_permanent(self):
        """An anomalous object stays anomalous — slowing down (or losing
        the evidence stream entirely) must not clear the flag."""
        t = make_track()
        for _ in range(Track.ANOMALY_RAISE_N):
            spike(t)
        assert t.is_anomalous
        for _ in range(100):
            clean(t)
        assert t.is_anomalous
        assert t.anomaly_types == {"supersonic"}

    def test_altitude_jumps_need_mofn_too(self):
        """Per-frame event checks obey the same rule: one impossible
        altitude jump (a mis-associated frame) does not latch; repeated
        jumps do."""
        t = make_track()

        def alt(alt_ft, ts):
            t._check_altitude_anomaly(
                {"delay": 50.0, "doppler": 10.0, "snr": 15.0, "adsb": {"alt_baro": alt_ft}},
                ts,
            )

        alt(10_000, TS)
        alt(30_000, TS + 1000)  # one 20k ft jump — noisy datapoint
        assert not t.is_anomalous
        alt(10_000, TS + 2000)  # jump 2
        alt(30_000, TS + 3000)  # jump 3
        assert t.is_anomalous
        assert t.anomaly_types == {"altitude_jump"}

    def test_identity_swap_latches_on_first_occurrence(self):
        """A genuine transponder swap happens exactly once; the check
        already debounces over 2 consecutive frames, so a single confirmed
        occurrence latches."""
        set_config(
            {
                "radar": {"center_frequency": 100_000_000},
                "adsb": {"enabled": True, "reference_location": [34.85, -82.4]},
            }
        )
        det = {
            "delay": 50.0,
            "doppler": 10.0,
            "snr": 15.0,
            "adsb": {"hex": "aaa111", "lat": 34.8, "lon": -82.4, "alt_baro": 10000, "gs": 250.0, "track": 90.0},
        }
        t = Track(det, TS, kf=None)
        swapped = {"delay": 50.0, "doppler": 10.0, "snr": 15.0, "adsb": {"hex": "bbb222"}}
        t._check_identity_change_anomaly(swapped, TS + 1000)
        assert not t.is_anomalous  # debounce: first mismatched frame
        t._check_identity_change_anomaly(swapped, TS + 2000)
        assert t.is_anomalous
        assert t.anomaly_types == {"identity_swap"}

    def test_observation_log_records_prelatch_events(self):
        """anomaly_detections logs every occurrence, including ones that
        never latch a flag."""
        t = make_track()

        def alt(alt_ft, ts):
            t._check_altitude_anomaly(
                {"delay": 50.0, "doppler": 10.0, "snr": 15.0, "adsb": {"alt_baro": alt_ft}},
                ts,
            )

        alt(10_000, TS)
        alt(30_000, TS + 1000)
        assert not t.is_anomalous
        events = [e for e in t.anomaly_detections if e["type"] == "altitude_jump"]
        assert len(events) == 1
        assert events[0]["altitude_change_ft"] == 20_000
