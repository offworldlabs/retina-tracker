"""
retina-tracker: Multi-target tracker for bistatic passive radar.

Supports two deployment modes:
- Node mode: Tracker runs locally on each radar node
- Server mode: Central server aggregates detections from multiple nodes

Usage:
    from tracker import Tracker, TrackEventWriter, load_config

    # Configure
    config = load_config('config.yaml')

    # Create tracker
    writer = TrackEventWriter('-')  # stdout
    tracker = Tracker(event_writer=writer, config=config)

    # Process frames
    tracker.process_frame(detections, timestamp)

    # Get current tracks
    tracks = tracker.get_tracks()
"""

from .track_detections import (
    # Core classes
    Tracker,
    Track,
    TrackEventWriter,
    KalmanFilter,
    TrackState,
    # Configuration
    load_config,
    get_config,
    set_config,
    # Frame processing
    process_streaming_frame,
)

__all__ = [
    'Tracker',
    'Track',
    'TrackEventWriter',
    'KalmanFilter',
    'TrackState',
    'load_config',
    'get_config',
    'set_config',
    'process_streaming_frame',
]
