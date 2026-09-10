"""Every detection the tracker is given comes back out, classified.

The tracker consumed detections and only ever reported the ones that ended up
inside a confirmed track. Two exits were silent: anything below MIN_SNR was
dropped at the gate before the tracker looked at it, and anything that started
a tentative track which never promoted disappeared with it. Nothing
downstream could tell "the tracker rejected what it saw" from "there was
nothing to see".

The subtlety these pin down is *when* a verdict is final. "Associated" means
the detection ended up in a track that was confirmed, and that is not knowable
at the moment the detection arrives.
"""

import pytest

from retina_tracker.config import get_config, set_config
from retina_tracker.server import process_streaming_frame
from retina_tracker.tracker import Tracker


class RecordingSink:
    """Stands in for whatever consumes classified detections."""

    def __init__(self):
        self.calls = []

    def write_detections(self, timestamp, associated, unassociated, below_snr):
        self.calls.append(
            {
                "timestamp": timestamp,
                "associated": associated,
                "unassociated": unassociated,
                "below_snr": below_snr,
            }
        )

    # Convenience views over everything released so far.
    def snrs(self, bucket):
        return [d["snr"] for call in self.calls for d in call[bucket]]

    def count(self, bucket):
        return sum(len(call[bucket]) for call in self.calls)

    @property
    def timestamps(self):
        return [call["timestamp"] for call in self.calls]


@pytest.fixture
def sink():
    return RecordingSink()


def make_tracker(sink, **kwargs):
    return Tracker(config=get_config(), detection_sink=sink, **kwargs)


def frame(tracker, timestamp, points):
    """points: list of (delay, doppler, snr)."""
    process_streaming_frame(
        tracker,
        {
            "timestamp": timestamp,
            "delay": [p[0] for p in points],
            "doppler": [p[1] for p in points],
            "snr": [p[2] for p in points],
        },
    )


def steady_target(tracker, frames, delay=10.0, doppler=50.0, snr=15.0, start=1000, step=500):
    for i in range(frames):
        frame(tracker, start + i * step, [(delay + i * 0.05, doppler - i * 0.2, snr)])


def drain(tracker, start_ts=900000, max_frames=120):
    """Run empty frames until nothing is pending.

    A frame is held until its tracks settle, so a test that sends a handful
    and asserts immediately is measuring the queue rather than the
    classification. Empty frames let tentative tracks miss their way to
    deletion, which settles them."""
    ts = start_ts
    for _ in range(max_frames):
        if not tracker._pending_classification:
            return
        ts += 500
        frame(tracker, ts, [])


# ── the SNR gate ────────────────────────────────────────────────────────────


def test_detections_below_the_gate_are_reported_not_discarded(sink):
    """The blind spot this exists to close."""
    min_snr = get_config()["tracker"]["min_snr"]
    tracker = make_tracker(sink)
    frame(tracker, 1000, [(10.0, 50.0, min_snr + 5), (20.0, -30.0, min_snr - 1)])
    drain(tracker)

    assert sink.count("below_snr") == 1
    assert sink.snrs("below_snr") == [min_snr - 1]


def test_a_detection_exactly_on_the_gate_is_kept(sink):
    """`>=`, matching the filter this replaced."""
    min_snr = get_config()["tracker"]["min_snr"]
    tracker = make_tracker(sink)
    frame(tracker, 1000, [(10.0, 50.0, min_snr)])
    drain(tracker)

    assert sink.count("below_snr") == 0
    assert sink.count("associated") + sink.count("unassociated") == 1


def test_a_nan_snr_lands_in_exactly_one_bucket(sink):
    """NaN fails both comparisons, so two comprehensions would have dropped
    it from the accounting entirely."""
    tracker = make_tracker(sink)
    frame(tracker, 1000, [(10.0, 50.0, float("nan"))])
    drain(tracker)

    total = sink.count("associated") + sink.count("unassociated") + sink.count("below_snr")
    assert total == 1


def test_every_detection_is_accounted_for(sink):
    min_snr = get_config()["tracker"]["min_snr"]
    tracker = make_tracker(sink)
    given = 0
    for i in range(12):
        points = [
            (10.0 + i * 0.05, 50.0, min_snr + 8),
            (200.0, -100.0, min_snr - 2),
            (55.0 + i * 3.0, 20.0, min_snr + 1),
        ]
        given += len(points)
        frame(tracker, 1000 + i * 500, points)
    drain(tracker)

    seen = sink.count("associated") + sink.count("unassociated") + sink.count("below_snr")
    assert seen == given, (seen, given)


# ── when a verdict becomes final ────────────────────────────────────────────


def test_a_detection_that_ends_up_in_a_confirmed_track_reads_associated(sink):
    tracker = make_tracker(sink)
    steady_target(tracker, 12)

    assert any(t.ever_confirmed for t in tracker.tracks), "no track confirmed; test is vacuous"
    assert sink.count("associated") > 0


def test_a_detection_is_not_called_unassociated_before_its_track_can_promote(sink):
    """The trap. A detection that starts a tentative track is unassociated at
    that instant and becomes associated a few frames later if the track
    promotes. Answering on arrival would be wrong for exactly the detections
    that matter."""
    tracker = make_tracker(sink)
    # One frame: the detection starts a tentative track that cannot possibly
    # have promoted yet, so nothing may be released about it.
    frame(tracker, 1000, [(10.0, 50.0, 15.0)])

    assert sink.calls == [], "a verdict was published before it could be known"


def test_the_held_frame_is_released_once_its_track_confirms(sink):
    tracker = make_tracker(sink)
    steady_target(tracker, 12)

    assert sink.calls, "the frame was never released"
    assert sink.timestamps[0] == 1000
    assert sink.count("associated") > 0


def test_clutter_that_never_confirms_reads_unassociated(sink):
    """Each detection is somewhere new, so every one starts a track that dies
    without promoting."""
    tracker = make_tracker(sink)
    for i in range(40):
        frame(tracker, 1000 + i * 500, [(20.0 + i * 25.0, -200.0 + i * 9.0, 15.0)])
    drain(tracker)

    assert sink.count("unassociated") > 0
    assert sink.count("associated") == 0


def test_a_detection_joining_an_established_track_is_not_delayed(sink):
    """Once a track is confirmed its verdict is already settled, so a
    steady feed is classified without lag."""
    tracker = make_tracker(sink)
    steady_target(tracker, 12)
    released_before = len(sink.calls)

    steady_target(tracker, 1, start=1000 + 12 * 500)
    assert len(sink.calls) == released_before + 1


# ── ordering and bounds ─────────────────────────────────────────────────────


def test_a_rejected_detection_waits_for_its_frame(sink):
    """Deliberate. A below-gate detection's verdict is known the moment it
    arrives, but releasing it ahead of the rest of its frame would mean
    emitting the same frame twice and out of order. One call per frame, in
    order, is worth a few frames of lag to the consumer that buffers these.
    """
    min_snr = get_config()["tracker"]["min_snr"]
    tracker = make_tracker(sink)
    frame(tracker, 1000, [(10.0, 50.0, min_snr + 5), (20.0, -30.0, min_snr - 1)])

    assert sink.calls == [], "the rejected detection outran its frame"

    drain(tracker)
    assert sink.calls[0]["timestamp"] == 1000
    assert len(sink.calls[0]["below_snr"]) == 1


def test_frames_are_released_in_order(sink):
    """A consumer buffers these and wants to rely on append order being
    timestamp order, so an unsettled frame blocks rather than being skipped."""
    tracker = make_tracker(sink)
    for i in range(30):
        frame(tracker, 1000 + i * 500, [(10.0 + i * 0.05, 50.0, 15.0), (300.0 - i * 7.0, -150.0, 15.0)])

    assert sink.timestamps == sorted(sink.timestamps)


def test_a_frame_that_never_settles_is_forced_out(sink):
    """A tentative track that keeps associating but never clears promotion
    would otherwise hold the whole queue behind it forever."""
    tracker = make_tracker(sink, max_pending_classification_frames=5)
    frame(tracker, 1000, [(10.0, 50.0, 15.0)])
    assert sink.calls == []

    # Frames elsewhere, so the first frame's track neither promotes nor is
    # touched by anything that would resolve it quickly.
    for i in range(1, 8):
        frame(tracker, 1000 + i * 500, [(400.0 + i * 20.0, 300.0, 15.0)])

    assert sink.calls, "the queue never drained"
    assert sink.timestamps[0] == 1000


def test_the_pending_queue_does_not_grow_without_bound(sink):
    tracker = make_tracker(sink, max_pending_classification_frames=10)
    for i in range(200):
        frame(tracker, 1000 + i * 500, [(10.0 + i * 0.05, 50.0, 15.0)])

    assert len(tracker._pending_classification) <= 10


def test_reset_drops_anything_still_pending(sink):
    tracker = make_tracker(sink)
    frame(tracker, 1000, [(10.0, 50.0, 15.0)])
    assert tracker._pending_classification

    tracker.reset()

    assert len(tracker._pending_classification) == 0


# ── cost when nobody is listening ───────────────────────────────────────────


def test_no_sink_means_no_bookkeeping():
    """The CLI and any node without a consumer must not pay for this."""
    tracker = Tracker(config=get_config())
    steady_target(tracker, 12)
    assert len(tracker._pending_classification) == 0


def test_the_gate_still_filters_what_the_tracker_sees(sink):
    """Reporting the rejects must not mean tracking them."""
    original = get_config()
    try:
        set_config({**original, "tracker": {**original["tracker"], "min_snr": 10.0}})
        tracker = make_tracker(sink)
        for i in range(12):
            frame(tracker, 1000 + i * 500, [(10.0 + i * 0.05, 50.0, 2.0)])
        assert tracker.tracks == [], "a sub-threshold detection was tracked"
        assert sink.count("below_snr") == 12
    finally:
        set_config(original)
