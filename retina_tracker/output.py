"""TrackEventWriter for streaming JSONL output."""

import json
import os
import sys

DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 1


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

        if output_file == "-":
            self.path = None
            self.output = sys.stdout
            self._is_stdout = True
        else:
            self.path = output_file
            # Long-lived handle, released by close(); not a context-manager case.
            self.output = open(output_file, "w")  # noqa: SIM115
            self._is_stdout = False

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
    ):
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
        }
        line = json.dumps(event) + "\n"

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
