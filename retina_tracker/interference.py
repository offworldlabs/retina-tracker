"""A learned map of which Doppler bins hold an interferer rather than traffic."""

import math
from collections import defaultdict, deque

import numpy as np

from .config import (
    CPI_S,
    DELAY_CELL_KM,
    DOPPLER_BIN_HZ,
    INTERFERENCE_MAX_DELAY_SPREAD_CELLS,
    INTERFERENCE_MIN_FRAME_FRACTION,
    INTERFERENCE_MIN_SAMPLES,
    INTERFERENCE_WINDOW_S,
)
from .kalman import doppler_to_range_rate

SPREAD_PERCENTILES = (10, 90)

# How often a bin's delay spread is recomputed, in frames. Occupancy is a
# counter and is exact every frame; the spread is a percentile over the whole
# window and is the only part whose cost grows with the number of bins a node
# can see. A window turns over one frame at a time, so a verdict is stable
# across far more than this: at a 60 s window and a 0.5 s CPI, ten frames is
# five seconds of staleness against sixty seconds of evidence. Bins are
# staggered by index so each frame refreshes a tenth of them rather than all
# of them at once, which is what keeps the per-frame cost flat as the Doppler
# span widens.
SPREAD_REFRESH_FRAMES = 10


class DopplerOccupancy:
    """Which Doppler bins are behaving like a tone rather than like traffic.

    Interference is site-specific and cannot be shipped: one node's dominant
    tone sits at +27.9 Hz and another's at a pair around -60 Hz, so a notch
    tuned on either misses the other. What can be shipped is the shape of the
    evidence, which is the same at both.

    Two statistics over a rolling window, and a bin has to be high on both:

    **In what share of recent frames the bin held a detection.** A tone is
    present in every frame; an aircraft crosses a 2 Hz bin in seconds. This is
    the statistic that has to be gathered over many frames rather than within
    one, because CFAR thins a continuous ridge to one to three peaks per frame.
    A per-frame test was tried against recorded data and caught 14.4% of the
    artefact where this catches 45%. Persistence is the signal.

    **How far its detections scatter in delay, after the drift its own Doppler
    mandates.** This is the condition protecting a real target that holds
    near-constant bistatic Doppler, and it is load-bearing: extent alone would
    not do it. A target sitting in one Doppler bin has, by definition, a
    near-constant bistatic range rate, so its delay marches steadily and its
    raw extent grows without limit. Removing the drift that the bin's own
    Doppler implies leaves an aircraft inside a resolution cell or two, while a
    tone, whose peaks land wherever the ridge is brightest that frame, stays
    scattered across the whole delay axis.

    Nothing here is tuned per site. The bins come from the capture config and
    the thresholds are fleet-wide, so a node with no interference learns an
    empty map and suppresses nothing.
    """

    def __init__(
        self,
        doppler_bin_hz,
        delay_cell_km,
        window_frames,
        min_frame_fraction,
        max_delay_spread_cells,
        min_samples,
    ):
        self.doppler_bin_hz = doppler_bin_hz
        self.delay_cell_km = delay_cell_km
        self.window_frames = max(int(window_frames), 1)
        self.min_frame_fraction = min_frame_fraction
        self.max_delay_spread_cells = max_delay_spread_cells
        self.min_samples = min_samples
        self.clear()

    @classmethod
    def from_config(cls):
        return cls(
            doppler_bin_hz=DOPPLER_BIN_HZ(),
            delay_cell_km=DELAY_CELL_KM(),
            window_frames=round(INTERFERENCE_WINDOW_S() / CPI_S()),
            min_frame_fraction=INTERFERENCE_MIN_FRAME_FRACTION(),
            max_delay_spread_cells=INTERFERENCE_MAX_DELAY_SPREAD_CELLS(),
            min_samples=INTERFERENCE_MIN_SAMPLES(),
        )

    def clear(self):
        # Per frame, the bin -> how many of its detections that frame held, so
        # an evicted frame can give back exactly the samples it contributed.
        self._frames = deque()
        self._frames_occupied = defaultdict(int)
        self._samples = defaultdict(deque)
        self._interfering = frozenset()
        self._spread_exceeded = {}
        self._frame_index = 0

    def _bin(self, doppler):
        return math.floor(doppler / self.doppler_bin_hz + 0.5)

    def observe(self, detections, timestamp):
        """Fold one frame in, evict whatever left the window, re-judge."""
        frame = defaultdict(list)
        for det in detections:
            doppler = det.get("doppler")
            delay = det.get("delay")
            if doppler is None or delay is None:
                continue
            if not (math.isfinite(doppler) and math.isfinite(delay)):
                continue
            frame[self._bin(doppler)].append((timestamp, delay))

        for bin_index, samples in frame.items():
            self._frames_occupied[bin_index] += 1
            self._samples[bin_index].extend(samples)
        self._frames.append({bin_index: len(s) for bin_index, s in frame.items()})

        while len(self._frames) > self.window_frames:
            for bin_index, n in self._frames.popleft().items():
                self._frames_occupied[bin_index] -= 1
                samples = self._samples[bin_index]
                for _ in range(n):
                    samples.popleft()
                if not self._frames_occupied[bin_index]:
                    del self._frames_occupied[bin_index]
                    del self._samples[bin_index]

        self._frame_index += 1
        self._interfering = self._judge()

    def _judge(self):
        # A partly filled window would read every bin it has seen as fully
        # occupied, so nothing is suppressed until there is a window to judge
        # against. The cost is that a restarted tracker suppresses nothing for
        # its first window, and whatever the interferer starts in that time is
        # established and passes through from then on. That is one window's
        # worth of artefact tracks per restart against continuous ones, and it
        # is the safe direction to fail in: the alternative is convicting a bin
        # on evidence too thin to have convicted anything.
        if len(self._frames) < self.window_frames:
            return frozenset()

        needed = self.min_frame_fraction * self.window_frames
        self._spread_exceeded = {b: v for b, v in self._spread_exceeded.items() if b in self._frames_occupied}
        return frozenset(
            bin_index
            for bin_index, occupied in self._frames_occupied.items()
            if occupied >= needed and self._spread_exceeded_in(bin_index)
        )

    def _spread_exceeded_in(self, bin_index):
        """Whether this bin's delay scatter is too wide, recomputed on a
        stagger. A bin is measured the first time it is asked about, so a new
        interferer is never judged on a stale verdict it does not have.

        Only bins that are occupied enough to be judged get here at all, so a
        bin whose occupancy oscillates around the floor can hold its verdict
        for longer than the refresh interval. That is harmless: on every frame
        it spends below the floor it is not convicted whatever its spread says,
        and the frame it comes back up is a frame its verdict is asked for
        again.
        """
        due = (self._frame_index + bin_index) % SPREAD_REFRESH_FRAMES == 0
        if due or bin_index not in self._spread_exceeded:
            self._spread_exceeded[bin_index] = self.delay_spread_cells(bin_index) > self.max_delay_spread_cells
        return self._spread_exceeded[bin_index]

    def delay_spread_cells(self, bin_index):
        """Scatter in delay about the drift this bin's Doppler mandates.

        A percentile range rather than the full extent: one stray detection in
        a bin an aircraft otherwise owns would push the extent over the
        threshold, and that is the direction that costs an aircraft.
        """
        samples = self._samples.get(bin_index)
        if not samples or len(samples) < self.min_samples:
            return 0.0

        times = np.fromiter((t for t, _ in samples), float, len(samples))
        delays = np.fromiter((d for _, d in samples), float, len(samples))
        range_rate = doppler_to_range_rate(bin_index * self.doppler_bin_hz)
        drift = range_rate * (times - times[0]) / 1000.0
        low, high = np.percentile(delays - drift, SPREAD_PERCENTILES)
        return float(high - low) / self.delay_cell_km

    def frame_fraction(self, bin_index):
        """Share of the frames held in the window in which this bin was busy."""
        if not self._frames:
            return 0.0
        return self._frames_occupied.get(bin_index, 0) / len(self._frames)

    def is_interfering(self, doppler):
        try:
            if doppler is None or not math.isfinite(doppler):
                return False
        except TypeError:
            return False
        return self._bin(doppler) in self._interfering

    def interfering_bins(self):
        return self._interfering
