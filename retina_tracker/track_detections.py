#!/usr/bin/env python3
"""
Radar tracker for bistatic radar detection data from blah2.

This module re-exports from the split submodules for backward compatibility.
New code should import directly from:
  tracker.config, tracker.kalman, tracker.track, tracker.tracker,
  tracker.output, tracker.server, tracker.cli
"""

# Re-export everything that was previously importable from this module.

# Configuration
# CLI / file processing
from .cli import (
    load_detections,
    main,
    process_detections,
    save_tracks,
    visualize_tracks,
)
from .config import (
    GATE_THRESHOLD,
    M_THRESHOLD,
    MEASUREMENT_NOISE_DELAY_KM2,
    MEASUREMENT_NOISE_RANGE_RATE_KM2,
    N_COAST,
    N_DELETE,
    N_WINDOW,
    PROCESS_NOISE_JERK,
    SPEED_OF_LIGHT,
    TRACKLET_MAX_DELAY_RESIDUAL,
    TRACKLET_MAX_DOPPLER_RESIDUAL,
    TRACKLET_MAX_TIME_SPAN,
    get_config,
    load_blah2_config,
    load_config,
    set_config,
)

# Kalman filter
from .kalman import KalmanFilter

# Output
from .output import TrackEventWriter

# Server / streaming
from .server import process_streaming_frame, run_tcp_server

# Track
from .track import Track, TrackState

# Tracker
from .tracker import Tracker

__all__ = [
    # Config
    "load_config",
    "load_blah2_config",
    "get_config",
    "set_config",
    "M_THRESHOLD",
    "N_WINDOW",
    "N_DELETE",
    "N_COAST",
    "GATE_THRESHOLD",
    "PROCESS_NOISE_JERK",
    "MEASUREMENT_NOISE_DELAY_KM2",
    "MEASUREMENT_NOISE_RANGE_RATE_KM2",
    "TRACKLET_MAX_DELAY_RESIDUAL",
    "TRACKLET_MAX_DOPPLER_RESIDUAL",
    "TRACKLET_MAX_TIME_SPAN",
    "SPEED_OF_LIGHT",
    # Core classes
    "KalmanFilter",
    "TrackState",
    "Track",
    "TrackEventWriter",
    "Tracker",
    # Processing
    "load_detections",
    "process_detections",
    "save_tracks",
    "process_streaming_frame",
    "run_tcp_server",
    "visualize_tracks",
    "main",
]


if __name__ == "__main__":
    main()
