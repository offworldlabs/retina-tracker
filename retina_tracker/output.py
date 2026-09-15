"""TrackEventWriter for streaming JSONL output."""

import json
import os
import sys
from collections import OrderedDict

DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 1

# How many tracks to remember having written detections for. Only live tracks
# emit, so this is an LRU over "recently emitting" rather than over every track
# of the run. Evicting one costs a repeated window, not a lost detection, so a
# generous cap is cheap: a track id and an integer apiece.
EMITTED_MEMORY = 512


class TrackEventWriter:
    """Writes track lifecycle events in JSONL (JSON Lines) format.

    Each event is a single JSON object on its own line, enabling streaming consumption.

    File output is size-bounded: before the write that would take the current file past
    max_bytes it is renamed to <path>.1 (shifting any older segments up to backup_count)
    and a new file is started, so the on-disk footprint stays under
    max_bytes * (backup_count + 1).
    Consumers tail the live path and already reset to offset 0 when it shrinks, which
    is the same shrink they see when this process restarts and truncates.
    Set max_bytes to 0 to disable rotation. stdout output is never rotated.
    """

    def __init__(self, output_file, max_bytes=DEFAULT_MAX_BYTES, backup_count=DEFAULT_BACKUP_COUNT):
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.bytes_written = 0
        # track_id -> newest detection timestamp already written for it.
        self._emitted_through = OrderedDict()

        if output_file == "-":
            self.path = None
            self.output = sys.stdout
            self._is_stdout = True
        else:
            self.path = output_file
            # Long-lived handle, released by close(); not a context-manager case.
            self.output = open(output_file, "w")  # noqa: SIM115
            self._is_stdout = False

    def _new_detections(self, track_id, detections):
        """The detections of this event not already written for this track.

        Each event carries a rolling window of the track's most recent points
        (Track.get_recent_detections), of which typically one is new. Repeating
        the other nineteen every time multiplied this file by roughly twenty
        for no consumer's benefit: live_score.load_tracks unions its
        detections by timestamp, and so does retina-gui's buffer, so neither
        can tell a delta stream from a repeating one.

        A high-water mark is sufficient because a track's history only grows
        forwards for as long as it can emit. Merging is the one thing that
        splices older points into a track, and it operates on all_tracks,
        the post-mortem archive, after the track has been deleted from
        self.tracks and can no longer produce an event.
        """
        through = self._emitted_through.get(track_id)
        if through is not None:
            self._emitted_through.move_to_end(track_id)
            detections = [d for d in detections if d["timestamp"] > through]
        if detections:
            self._emitted_through[track_id] = max(d["timestamp"] for d in detections)
            self._emitted_through.move_to_end(track_id)
            while len(self._emitted_through) > EMITTED_MEMORY:
                self._emitted_through.popitem(last=False)
        return detections

    def write_event(
        self,
        track_id,
        timestamp,
        length,
        detections,
        adsb_hex=None,
        adsb_initialized=False,
        is_anomalous=False,
        max_velocity_ms=0.0,
        anomaly_types=None,
        shadow_fraction=0.0,
        interference_fraction=0.0,
    ):
        # Only what is new. The event is still written when nothing is —
        # length, the anomaly flags and shadow_fraction all move over a
        # track's life, and a consumer that missed those updates would be
        # reading a stale opinion of a live track.
        detections = self._new_detections(track_id, detections)

        event = {
            "track_id": track_id,
            "adsb_hex": adsb_hex,
            "adsb_initialized": adsb_initialized,
            "timestamp": timestamp,
            "length": length,
            "detections": detections,
            "is_anomalous": is_anomalous,
            "max_velocity_ms": max_velocity_ms,
            "anomaly_types": sorted(anomaly_types) if anomaly_types else [],
            "shadow_fraction": shadow_fraction,
            # What the occupancy map thought of this track's detections, which
            # is how it is held to account: a track carrying an adsb_hex is an
            # aircraft whatever the map says, so this reading above zero on one
            # is the map reaching for something it must not have.
            "interference_fraction": interference_fraction,
        }
        self._write_line(json.dumps(event) + "\n")

    def _write_line(self, line):
        if not self._is_stdout and self.max_bytes:
            size = len(line.encode("utf-8"))
            if self.bytes_written and self.bytes_written + size > self.max_bytes:
                self._rotate()
            self.bytes_written += size

        self.output.write(line)
        self.output.flush()

    def _rotate(self):
        self.output.close()

        for i in range(self.backup_count, 0, -1):
            source = self.path if i == 1 else f"{self.path}.{i - 1}"
            if os.path.exists(source):
                os.replace(source, f"{self.path}.{i}")

        self.output = open(self.path, "w")  # noqa: SIM115
        self.bytes_written = 0

    def close(self):
        if not self._is_stdout:
            self.output.close()


class InnovationWriter(TrackEventWriter):
    """Writes one record per Kalman update, for calibrating R and Q.

    Off unless a path is given. R and Q cannot be fitted from the events file:
    it carries the detections that were associated, not what the filter
    predicted before seeing them, and the difference between those is the only
    quantity either constant answers to.

    Reconstructing innovations offline by replaying recorded detections works
    but cannot see the filter's real state, because a replay has to guess at
    the covariance, the adaptive process-noise scale and the coasting history
    that shaped each prediction. This records them instead of inferring them.

    Inherits the size bound so a node left recording cannot fill its disk.
    """

    def write_residual(self, track_id, timestamp, track, detection):
        residual = track.last_residual
        if residual is None:
            return

        record = {
            "track_id": track_id,
            "birth": track.birth_timestamp,
            "timestamp": timestamp,
            "dt": track.last_dt,
            "snr": detection.get("snr"),
            # The measurement, so a record can be filtered on its own. Most
            # tracks at an interfered site are built on a fixed-Doppler tone
            # and must be excluded before anything is fitted to them; without
            # the Doppler here that can only be done by joining back to the
            # events file, which a track that never confirmed is absent from.
            "delay": detection.get("delay"),
            "doppler": detection.get("doppler"),
            "n_missed": track.last_n_missed,
            "q_scale": track.last_q_scale,
            "innovation": [float(x) for x in residual.innovation],
            "s_diag": [float(residual.S[0][0]), float(residual.S[1][1])],
            "nis": residual.nis,
        }
        self._write_line(json.dumps(record) + "\n")
