"""Tests for stitching an ADS-B aircraft's fragments into one chain.

Each fragment keeps its own track id. A confirmed fragment carrying an ADS-B hex
is linked to that aircraft's previous fragment by stitch_id, the id of the
chain's first fragment, which every event carries. See docs/stitching.md.
"""

from retina_tracker.config import load_config, set_config
from retina_tracker.history import WINDOW_S
from retina_tracker.kalman import doppler_to_range_rate
from retina_tracker.tracker import Tracker

FRAME_MS = 500
DOPPLER = -80.0


class Recorder:
    def __init__(self):
        self.events = []

    def write_event(self, track_id, timestamp, length, detections, **fields):
        self.events.append({"track_id": track_id, **fields})


def make_tracker():
    set_config(load_config(None))
    recorder = Recorder()
    return Tracker(event_writer=recorder), recorder


def fly(tracker, start_frame, frames, hex_from=0, adsb_hex="abc123", delay=20.0):
    """One fragment of an aircraft, labelled from its hex_from-th detection, then lost for good."""
    step = doppler_to_range_rate(DOPPLER) * FRAME_MS / 1000.0
    for i in range(frames):
        adsb = {"hex": adsb_hex} if adsb_hex and i >= hex_from else None
        det = {"delay": delay + i * step, "doppler": DOPPLER, "snr": 15.0, "adsb": adsb}
        tracker.process_frame([det], (start_frame + i) * FRAME_MS)
    track = next(t for t in tracker.tracks if t.birth_timestamp == start_frame * FRAME_MS)
    for i in range(frames, frames + 12):
        tracker.process_frame([], (start_frame + i) * FRAME_MS)
    return track


def test_a_later_fragment_of_the_same_aircraft_continues_its_chain():
    tracker, recorder = make_tracker()
    first = fly(tracker, 0, 20)
    second = fly(tracker, 40, 20)
    assert second.id != first.id
    assert first.stitch_id == first.id and second.stitch_id == first.id
    assert {e["stitch_id"] for e in recorder.events if e["track_id"] == second.id} == {first.id}


def test_every_fragment_of_a_chain_names_its_first():
    tracker, _ = make_tracker()
    first = fly(tracker, 0, 20)
    fly(tracker, 40, 20)
    third = fly(tracker, 80, 20)
    assert third.stitch_id == first.id


def test_a_fragment_that_is_labelled_late_joins_the_chain_then():
    tracker, recorder = make_tracker()
    first = fly(tracker, 0, 20)
    second = fly(tracker, 40, 20, hex_from=10)
    stitch_ids = [e["stitch_id"] for e in recorder.events if e["track_id"] == second.id]
    assert stitch_ids[0] == second.id and stitch_ids[-1] == first.id


def test_an_unlabelled_fragment_between_does_not_break_the_chain():
    tracker, _ = make_tracker()
    first = fly(tracker, 0, 20)
    unlabelled = fly(tracker, 40, 20, adsb_hex=None)
    third = fly(tracker, 80, 20)
    assert unlabelled.stitch_id is None and unlabelled.to_dict()["stitch_id"] == unlabelled.id
    assert third.stitch_id == first.id


def test_different_aircraft_are_never_linked():
    tracker, _ = make_tracker()
    fly(tracker, 0, 20)
    other = fly(tracker, 40, 20, adsb_hex="def456")
    assert other.stitch_id == other.id


def test_fragments_alive_at_the_same_time_are_not_linked():
    tracker, _ = make_tracker()
    step = doppler_to_range_rate(DOPPLER) * FRAME_MS / 1000.0
    for i in range(20):
        dets = [
            {"delay": d + i * step, "doppler": DOPPLER, "snr": 15.0, "adsb": {"hex": "abc123"}} for d in (20.0, 40.0)
        ]
        tracker.process_frame(dets, i * FRAME_MS)
    assert len(tracker.tracks) == 2
    assert all(t.stitch_id == t.id for t in tracker.tracks)


def test_a_fragment_beyond_the_history_window_starts_a_new_chain():
    tracker, _ = make_tracker()
    fly(tracker, 0, 20)
    later = fly(tracker, (WINDOW_S * 1000) // FRAME_MS + 100, 20)
    assert later.stitch_id == later.id


def test_a_single_stray_label_never_links():
    tracker, _ = make_tracker()
    fly(tracker, 0, 20)
    stray = fly(tracker, 40, 20, hex_from=19)
    assert stray.adsb_hex == "abc123" and stray.stitch_id is None


def detection(i, delay, adsb_hex="abc123"):
    step = doppler_to_range_rate(DOPPLER) * FRAME_MS / 1000.0
    return {"delay": delay + i * step, "doppler": DOPPLER, "snr": 15.0, "adsb": {"hex": adsb_hex}}


def test_a_fragment_born_while_its_predecessor_coasts_links_once_it_dies():
    tracker, _ = make_tracker()
    for i in range(20):
        tracker.process_frame([detection(i, 20.0)], i * FRAME_MS)
    first = tracker.tracks[0]
    for i in range(22, 50):
        tracker.process_frame([detection(i, 40.0)], i * FRAME_MS)
    second = next(t for t in tracker.tracks if t is not first)
    assert first not in tracker.tracks
    assert second.stitch_id == first.id


def test_a_fragment_born_while_its_predecessor_coasts_stays_apart_if_it_returns():
    tracker, _ = make_tracker()
    for i in range(20):
        tracker.process_frame([detection(i, 20.0)], i * FRAME_MS)
    first = tracker.tracks[0]
    for i in range(22, 40):
        dets = [detection(i, 40.0)] + ([detection(i, 20.0)] if i >= 26 else [])
        tracker.process_frame(dets, i * FRAME_MS)
    second = next(t for t in tracker.tracks if t is not first)
    assert first in tracker.tracks
    assert second.stitch_id == second.id
