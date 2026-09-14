"""A track's identity is its own, never the aircraft's.

The id used to be built from adsb_hex whenever the aircraft was known, so every
pass of that aircraft in a day answered to a single identity. That hid the
failure we most need to see: live_score.fragmentation counts distinct ids per
hex, and an id derived from the hex can only ever yield one, so a site splitting
an aircraft four ways still reported close to one. It also let two tracks of one
aircraft share TrackEventWriter's per-track high-water mark, where whichever
emitted first suppressed the other's earlier detections outright.

The label is not lost by this. It travels in the event's own adsb_hex field,
which is where every consumer already reads it.
"""

import json
import re

import pytest

from retina_tracker.live_score import fragmentation, load_tracks
from retina_tracker.output import TrackEventWriter
from retina_tracker.track import Track

COUNTER_ID = re.compile(r"^\d{6}-[0-9A-F]{6}$")
BASE_TS = 1718747745000
DAY_MS = 86400000


@pytest.fixture(autouse=True)
def _fresh_counter():
    """The counter is class state, so it outlives a test without this."""
    Track._daily_counter = 0
    Track._last_date = None
    yield
    Track._daily_counter = 0
    Track._last_date = None


def detection(ts, delay, doppler=-120.0):
    return {"timestamp": ts, "delay": delay, "doppler": doppler, "snr": 16.0, "adsb": None}


class TestEveryTrackGetsItsOwnIdentity:
    def test_successive_tracks_never_share_an_id(self):
        ids = [Track._generate_id(BASE_TS) for _ in range(5)]
        assert len(set(ids)) == 5

    def test_two_tracks_born_in_the_same_millisecond_still_differ(self):
        """Concurrent tracks of one aircraft are the case that used to collide."""
        assert Track._generate_id(BASE_TS) != Track._generate_id(BASE_TS)

    def test_an_id_is_always_drawn_from_the_counter(self):
        assert COUNTER_ID.match(Track._generate_id(BASE_TS))

    def test_the_counter_resets_on_a_new_day(self):
        first = Track._generate_id(BASE_TS)
        second = Track._generate_id(BASE_TS + DAY_MS)

        assert first.endswith("-000000")
        assert second.endswith("-000000")
        assert first.split("-")[0] != second.split("-")[0]

    def test_the_date_still_leads_the_id(self):
        """Operators read the date off the id, and the archive sorts on it."""
        assert Track._generate_id(BASE_TS).split("-")[0].isdigit()


class TestTwoTracksOfOneAircraft:
    """The fragmentation case: one aircraft, two identities, both intact."""

    def _events(self, tmp_path):
        path = tmp_path / "events.jsonl"
        writer = TrackEventWriter(str(path), max_bytes=0)
        first = Track._generate_id(BASE_TS)
        second = Track._generate_id(BASE_TS)

        writer.write_event(
            first,
            BASE_TS + 4000,
            2,
            [detection(BASE_TS + 3500, 18.0), detection(BASE_TS + 4000, 18.2)],
            adsb_hex="abc123",
        )
        writer.write_event(
            second,
            BASE_TS + 4000,
            2,
            [detection(BASE_TS + 1000, 21.0), detection(BASE_TS + 1500, 21.2)],
            adsb_hex="abc123",
        )
        writer.close()
        return path, first, second

    def test_both_identities_reach_the_events_file(self, tmp_path):
        path, first, second = self._events(tmp_path)

        assert set(load_tracks(str(path))) == {first, second}

    def test_the_later_track_does_not_suppress_the_earlier_one(self, tmp_path):
        """A shared id would put these behind one high-water mark, and the
        second track's older detections would never be written."""
        path, _, second = self._events(tmp_path)

        assert [d["timestamp"] for d in load_tracks(str(path))[second]["detections"]] == [
            BASE_TS + 1000,
            BASE_TS + 1500,
        ]

    def test_fragmentation_can_finally_see_the_split(self, tmp_path):
        path, _, _ = self._events(tmp_path)

        result = fragmentation(load_tracks(str(path)))

        assert result["aircraft_identified"] == 1
        assert result["mean"] == pytest.approx(2.0)
        assert result["split"] == ["abc123"]

    def test_the_label_still_reaches_the_consumer(self, tmp_path):
        path, first, _ = self._events(tmp_path)

        with open(path) as f:
            events = [json.loads(line) for line in f if line.strip()]

        assert {e["adsb_hex"] for e in events} == {"abc123"}
        assert load_tracks(str(path))[first]["adsb_hex"] == "abc123"
