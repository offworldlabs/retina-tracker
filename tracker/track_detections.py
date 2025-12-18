#!/usr/bin/env python3
"""
Radar tracker for bistatic radar detection data from blah2.
Implements Kalman filtering and GNN data association for multi-target tracking.
Based on blah2's tracker architecture with enhanced filtering.
"""

import json
import os
import sys
from datetime import datetime
import numpy as np
from scipy.optimize import linear_sum_assignment
from enum import Enum
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import yaml


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(config_path=None):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, looks for config.yaml in:
                    1. Current directory
                    2. Parent directory (for running from tracker/)

    Returns:
        Dict with configuration values, or defaults if no config found.
    """
    default_config = {
        'tracker': {
            'm_threshold': 4,
            'n_window': 6,
            'n_delete': 10,
            'min_snr': 7.0,
            'gate_threshold': 2.0,
            'detection_window': 20
        },
        'process_noise': {
            'delay': 0.1,
            'doppler': 0.5
        },
        'tracklet': {
            'max_delay_residual': 2.0,
            'max_doppler_residual': 10.0,
            'max_time_span': 3.0
        },
        'adsb': {
            'enabled': False,
            'priority': True,
            'reference_location': None,
            'initial_covariance': {
                'position': 100.0,
                'velocity': 5.0
            }
        }
    }

    if config_path is None:
        # Search for config.yaml
        search_paths = ['config.yaml', '../config.yaml']
        for path in search_paths:
            if os.path.exists(path):
                config_path = path
                break

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            loaded = yaml.safe_load(f)
            # Merge with defaults
            for key in default_config:
                if key in loaded:
                    default_config[key].update(loaded[key])
        print(f"Loaded config from {config_path}", file=sys.stderr)

    return default_config


# Global config (loaded at module import or via set_config)
_config = None

def get_config():
    """Get current configuration, loading defaults if needed."""
    global _config
    if _config is None:
        _config = load_config()
    return _config

def set_config(config):
    """Set configuration dict."""
    global _config
    _config = config


# ============================================================================
# TRACK EVENT WRITER (JSONL STREAMING OUTPUT)
# ============================================================================

class TrackEventWriter:
    """
    Writes track lifecycle events in JSONL (JSON Lines) format.
    Each event is a single JSON object on its own line, enabling streaming consumption.
    """

    def __init__(self, output_file):
        """
        Initialize event writer.

        Args:
            output_file: Path to output file, or '-' for stdout
        """
        if output_file == '-':
            self.output = sys.stdout
            self._is_stdout = True
        else:
            self.output = open(output_file, 'w')
            self._is_stdout = False

    def write_event(self, track_id, timestamp, length, detections, adsb_hex=None, adsb_initialized=False):
        """
        Write a single track update to the output stream.

        Args:
            track_id: Hex ID of the track (YYMMDD-XXXXXX format)
            timestamp: Timestamp in milliseconds
            length: Total number of detections associated with this track
            detections: List of detection dicts with timestamp, delay, doppler, snr
            adsb_hex: Optional ICAO hex from ADS-B
            adsb_initialized: Whether track was initialized with ADS-B data
        """
        event = {
            'track_id': track_id,
            'adsb_hex': adsb_hex,
            'adsb_initialized': adsb_initialized,
            'timestamp': timestamp,
            'length': length,
            'detections': detections
        }
        self.output.write(json.dumps(event) + '\n')
        self.output.flush()  # Important for streaming

    def close(self):
        """Close the output file (if not stdout)."""
        if not self._is_stdout:
            self.output.close()

# ============================================================================
# TUNABLE PARAMETERS (loaded from config)
# ============================================================================

def _get_param(section, key, default=None):
    """Get parameter from config."""
    config = get_config()
    return config.get(section, {}).get(key, default)

# Track confirmation (M-of-N logic)
def M_THRESHOLD(): return _get_param('tracker', 'm_threshold', 4)
def N_WINDOW(): return _get_param('tracker', 'n_window', 6)

# Track management
def N_DELETE(): return _get_param('tracker', 'n_delete', 10)
N_COAST = 3  # Frames to coast before deletion (not in config)

# Gating and association
def GATE_THRESHOLD(): return _get_param('tracker', 'gate_threshold', 2.0)
def MIN_SNR(): return _get_param('tracker', 'min_snr', 7.0)

# Kalman filter parameters
def PROCESS_NOISE_DELAY(): return _get_param('process_noise', 'delay', 0.1)
def PROCESS_NOISE_DOPPLER(): return _get_param('process_noise', 'doppler', 0.5)
MEASUREMENT_NOISE_DELAY = 1.0   # Base measurement noise for delay
MEASUREMENT_NOISE_DOPPLER = 5.0 # Base measurement noise for doppler

# Tracklet parameters
def TRACKLET_MAX_DELAY_RESIDUAL(): return _get_param('tracklet', 'max_delay_residual', 2.0)
def TRACKLET_MAX_DOPPLER_RESIDUAL(): return _get_param('tracklet', 'max_doppler_residual', 10.0)
def TRACKLET_MAX_TIME_SPAN(): return _get_param('tracklet', 'max_time_span', 3.0)


# ============================================================================
# TRACK STATE ENUMERATION
# ============================================================================

class TrackState(Enum):
    """Track states following blah2 architecture."""
    TENTATIVE = 0   # New track, not yet confirmed
    ASSOCIATED = 1  # Tentative track with association this frame
    ACTIVE = 2      # Confirmed track
    COASTING = 3    # Active track without recent association


# ============================================================================
# KALMAN FILTER CLASS
# ============================================================================

class KalmanFilter:
    """
    2D Kalman filter for constant velocity model in delay-Doppler space.
    State vector: [delay, delay_rate, doppler, doppler_rate]
    """

    def __init__(self, dt=0.5):
        """
        Initialize Kalman filter.

        Args:
            dt: Time step between measurements (seconds)
        """
        self.dt = dt
        self.dim_state = 4
        self.dim_meas = 2

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, dt, 0,  0],
            [0,  1, 0,  0],
            [0,  0, 1, dt],
            [0,  0, 0,  1]
        ])

        # Measurement matrix (observe delay and doppler only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])

        # Process noise covariance
        q_delay = PROCESS_NOISE_DELAY()
        q_doppler = PROCESS_NOISE_DOPPLER()
        self.Q = np.array([
            [q_delay * dt**3 / 3, q_delay * dt**2 / 2, 0, 0],
            [q_delay * dt**2 / 2, q_delay * dt, 0, 0],
            [0, 0, q_doppler * dt**3 / 3, q_doppler * dt**2 / 2],
            [0, 0, q_doppler * dt**2 / 2, q_doppler * dt]
        ])

        # Measurement noise covariance (base, can be scaled by SNR)
        self.R = np.diag([MEASUREMENT_NOISE_DELAY, MEASUREMENT_NOISE_DOPPLER])

    def predict(self, state, covariance):
        """
        Predict next state and covariance.

        Args:
            state: Current state vector (4,)
            covariance: Current covariance matrix (4, 4)

        Returns:
            Predicted state and covariance
        """
        state_pred = self.F @ state
        cov_pred = self.F @ covariance @ self.F.T + self.Q

        # Constrain delay to be non-negative (physical constraint)
        if state_pred[0] < 0:
            state_pred[0] = 0.0
            # Also limit negative velocity to prevent further negative predictions
            if state_pred[1] < 0:
                state_pred[1] = 0.0

        return state_pred, cov_pred

    def update(self, state, covariance, measurement, snr=None):
        """
        Update state with measurement using Kalman gain.

        Args:
            state: Predicted state vector (4,)
            covariance: Predicted covariance matrix (4, 4)
            measurement: Measurement vector [delay, doppler] (2,)
            snr: Signal-to-noise ratio (optional, for adaptive R)

        Returns:
            Updated state and covariance
        """
        # Adaptive measurement noise based on SNR
        R = self.R.copy()
        if snr is not None:
            # Convert SNR from dB to linear scale: SNR_linear = 10^(SNR_dB/10)
            # Scale measurement noise inversely with SNR:
            # - High SNR (e.g., 20 dB → linear=100) → scale = 1/(100/10) = 0.1 (low noise)
            # - Low SNR (e.g., 8 dB → linear=6.3) → scale = 1/(6.3/10) = 1.59 (high noise)
            # max() prevents division by very small values when SNR is extremely low
            snr_linear = 10 ** (snr / 10)
            noise_scale = 1.0 / max(snr_linear / 10, 0.1)
            R = R * noise_scale

        # Innovation (measurement residual)
        z_pred = self.H @ state
        innovation = measurement - z_pred

        # Innovation covariance
        S = self.H @ covariance @ self.H.T + R

        # Kalman gain - protect against singular matrix
        try:
            K = covariance @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular matrix - innovation covariance is not invertible
            # This can happen with numerical issues or degenerate measurements
            # Return state unchanged (no update)
            print("Warning: Singular innovation covariance in Kalman update, "
                  "skipping measurement", file=sys.stderr)
            return state, covariance

        # Update state and covariance
        state_upd = state + K @ innovation
        cov_upd = (np.eye(self.dim_state) - K @ self.H) @ covariance

        return state_upd, cov_upd

    def get_innovation_covariance(self, covariance):
        """
        Get innovation covariance S for gating.

        Args:
            covariance: State covariance matrix (4, 4)

        Returns:
            Innovation covariance (2, 2)
        """
        S = self.H @ covariance @ self.H.T + self.R
        return S


# ============================================================================
# TRACK CLASS
# ============================================================================

class Track:
    """
    Represents a single radar track with state history.
    Based on blah2's Track class with Kalman filtering.
    """

    _daily_counter = 0
    _last_date = None

    def __init__(self, detection, timestamp, kf, frame=0, config=None):
        """
        Initialize track from first detection.

        Args:
            detection: Dict with 'delay', 'doppler', 'snr', and optional 'adsb'
            timestamp: Timestamp of detection (ms)
            kf: KalmanFilter instance
            frame: Frame number for this detection
            config: Configuration dict (optional)
        """
        self.id = None  # ID assigned at promotion only
        self.state_status = TrackState.TENTATIVE
        self.kf = kf
        self.adsb_hex = None
        self.adsb_initialized = False

        # Check for ADS-B data
        adsb_config = config.get('adsb', {}) if config else {}
        if (detection.get('adsb') and
            adsb_config.get('enabled') and
            adsb_config.get('reference_location')):
            self._init_from_adsb(detection, adsb_config)
        else:
            self._init_from_delay_doppler(detection)


        # Track history
        self.history = {
            'timestamps': [timestamp],
            'frames': [frame],
            'states': [self.state.copy()],
            'measurements': [detection],
            'state_status': [self.state_status.name]
        }

        # M/N logic counters
        self.n_frames = 1
        self.n_associated = 0
        self.n_missed = 0

        # Track quality metrics
        self.total_snr = detection['snr']
        self.birth_timestamp = timestamp
        self.death_timestamp = timestamp

    @staticmethod
    def _validate_adsb_data(adsb):
        """Validate ADS-B data fields are within reasonable ranges.

        Args:
            adsb: Dictionary containing ADS-B data

        Returns:
            True if data is valid, False otherwise
        """
        if not isinstance(adsb, dict):
            return False

        # Validate latitude
        if 'lat' in adsb:
            lat = adsb['lat']
            if not isinstance(lat, (int, float)) or np.isnan(lat) or not (-90 <= lat <= 90):
                return False

        # Validate longitude
        if 'lon' in adsb:
            lon = adsb['lon']
            if not isinstance(lon, (int, float)) or np.isnan(lon) or not (-180 <= lon <= 180):
                return False

        # Validate altitude (barometric)
        if 'alt_baro' in adsb:
            alt = adsb['alt_baro']
            if not isinstance(alt, (int, float)) or np.isnan(alt) or not (-1000 <= alt <= 60000):
                return False

        # Validate ground speed
        if 'gs' in adsb:
            gs = adsb['gs']
            if not isinstance(gs, (int, float)) or np.isnan(gs) or not (0 <= gs <= 1000):
                return False

        # Validate track angle
        if 'track' in adsb:
            track = adsb['track']
            if not isinstance(track, (int, float)) or np.isnan(track) or not (0 <= track < 360):
                return False

        return True

    def _init_from_adsb(self, detection, adsb_config):
        """
        Initialize track using ADS-B position and velocity data.

        Args:
            detection: Detection dict with 'adsb' field
            adsb_config: ADS-B configuration dict with reference_location and initial_covariance
        """
        from . import geometry

        adsb = detection['adsb']

        # Validate ADS-B data before using it
        if not self._validate_adsb_data(adsb):
            # Invalid ADS-B data, fall back to radar-only initialization
            self._init_from_delay_doppler(detection)
            return

        self.adsb_hex = adsb.get('hex')
        self.adsb_initialized = True

        # Get reference location
        ref = adsb_config['reference_location']

        # Use measured delay and Doppler for state
        # (keeping state in delay-Doppler space for consistency with Kalman filter)
        self.state = np.array([
            detection['delay'],
            0.0,
            detection['doppler'],
            0.0
        ])

        # But use ADS-B velocity to estimate better initial velocity
        if adsb.get('gs') is not None and adsb.get('track') is not None:
            # Validate ADS-B velocity data is reasonable
            gs = adsb['gs']
            track = adsb['track']
            if not (0 <= gs <= 1000 and 0 <= track < 360 and not np.isnan(gs) and not np.isnan(track)):
                # Invalid velocity data, skip velocity initialization
                pass
            else:
                # This gives us a rough velocity estimate
                # In reality, delay/Doppler rates depend on geometry, but this is better than zero
                vel_east, vel_north, vel_up = geometry.enu_velocity_from_adsb(
                    gs, track, adsb.get('geom_rate', 0)
                )
                # Validate computed velocities are not NaN
                if (np.isnan(vel_east) or np.isnan(vel_north) or np.isnan(vel_up)):
                    # Computed velocities are NaN, skip velocity initialization
                    pass
                else:
                    # Use velocity magnitude as a proxy for delay/Doppler rates
                    vel_horiz = np.sqrt(vel_east**2 + vel_north**2)
                    # Validate vel_horiz is not NaN or infinite
                    if np.isnan(vel_horiz) or np.isinf(vel_horiz):
                        # Skip velocity initialization
                        pass
                    else:
                        # Rough estimate: delay rate ~ velocity / speed of light * 1000 (km/s)
                        delay_rate_est = vel_horiz / 299792.458  # Very rough approximation
                        if not (np.isnan(delay_rate_est) or np.isinf(delay_rate_est)):
                            self.state[1] = delay_rate_est

        # Lower covariance for ADS-B initialization
        pos_unc = adsb_config['initial_covariance']['position']
        vel_unc = adsb_config['initial_covariance']['velocity']
        # Convert position uncertainty to delay units (meters to km)
        delay_unc = pos_unc / 1000.0
        # Use configured uncertainties
        self.covariance = np.diag([delay_unc, vel_unc/1000, 20.0, 10.0])

    def _init_from_delay_doppler(self, detection):
        """
        Initialize track from delay-Doppler measurements only (fallback mode).

        Args:
            detection: Detection dict with 'delay', 'doppler', 'snr'
        """
        # Initialize state [delay, delay_rate, doppler, doppler_rate]
        # Start with zero velocity assumption
        self.state = np.array([
            detection['delay'],
            0.0,
            detection['doppler'],
            0.0
        ])

        # Initialize covariance with high uncertainty
        self.covariance = np.diag([10.0, 5.0, 20.0, 10.0])

    @classmethod
    def _generate_id(cls, timestamp_ms, adsb_hex=None):
        """
        Generate track ID in YYMMDD-XXXXXX format.

        Args:
            timestamp_ms: Timestamp in milliseconds
            adsb_hex: Optional ICAO hex identifier from ADS-B

        Returns:
            Track ID string: YYMMDD-ICAOHEX or YYMMDD-XXXXXX
        """
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0)
        date_str = dt.strftime('%y%m%d')

        # Use ICAO hex if available
        if adsb_hex:
            return f"{date_str}-{adsb_hex.upper()}"

        # Otherwise use sequential counter
        if cls._last_date != date_str:
            cls._daily_counter = 0
            cls._last_date = date_str

        track_id = f"{date_str}-{cls._daily_counter:06X}"
        cls._daily_counter += 1
        return track_id

    def predict(self, dt):
        """
        Predict track state to next time step.

        Args:
            dt: Time delta (seconds)
        """
        self.kf.dt = dt
        self.state, self.covariance = self.kf.predict(self.state, self.covariance)

    def update(self, detection, timestamp, frame=0):
        """
        Update track with associated detection.

        Args:
            detection: Dict with 'delay', 'doppler', 'snr'
            timestamp: Timestamp of detection (ms)
            frame: Frame number for this detection
        """
        measurement = np.array([detection['delay'], detection['doppler']])
        self.state, self.covariance = self.kf.update(
            self.state, self.covariance, measurement, detection.get('snr')
        )

        # Update history
        self.history['timestamps'].append(timestamp)
        self.history['frames'].append(frame)
        self.history['states'].append(self.state.copy())
        self.history['measurements'].append(detection)
        self.history['state_status'].append(self.state_status.name)

        # Update M/N counters
        self.n_associated += 1
        self.n_missed = 0
        self.n_frames += 1

        # Update quality metrics
        self.total_snr += detection.get('snr', 0)
        self.death_timestamp = timestamp

    def mark_missed(self, timestamp, frame=0):
        """Mark track as not associated this frame."""
        self.n_missed += 1
        self.n_frames += 1

        # Append predicted state to history
        self.history['timestamps'].append(timestamp)
        self.history['frames'].append(frame)
        self.history['states'].append(self.state.copy())
        self.history['measurements'].append(None)
        self.history['state_status'].append(self.state_status.name)

        # Update death timestamp for duration tracking
        self.death_timestamp = timestamp

    def get_predicted_measurement(self):
        """Get predicted measurement [delay, doppler]."""
        return self.kf.H @ self.state

    def get_innovation_covariance(self):
        """Get innovation covariance for gating."""
        return self.kf.get_innovation_covariance(self.covariance)

    def promote_if_ready(self):
        """Apply M-of-N logic to promote TENTATIVE to ACTIVE."""
        if self.state_status == TrackState.TENTATIVE:
            if self.n_frames >= N_WINDOW():
                if self.n_associated >= M_THRESHOLD():
                    self.state_status = TrackState.ACTIVE
                    return True
        return False

    def get_quality_score(self):
        """
        Calculate track quality score (0-100).
        Higher score = better quality track.
        """
        if self.n_frames == 0:
            return 0.0

        # Continuity score (40 points): percentage of frames with associations
        continuity = (self.n_associated / self.n_frames) * 40.0

        # SNR quality (30 points): average SNR normalized to 0-30
        avg_snr = self.total_snr / max(self.n_associated, 1)
        snr_score = min((avg_snr / 15.0) * 30.0, 30.0)  # Normalize: 15dB = full score

        # Duration (20 points): track lifetime in seconds
        duration_sec = (self.death_timestamp - self.birth_timestamp) / 1000.0
        duration_score = min((duration_sec / 60.0) * 20.0, 20.0)  # 60 sec = full score

        # Association count (10 points): number of associations
        assoc_score = min((self.n_associated / 50.0) * 10.0, 10.0)  # 50 assoc = full score

        total_score = continuity + snr_score + duration_score + assoc_score
        return total_score

    def is_high_quality(self):
        """Check if track meets minimum quality thresholds."""
        if self.n_associated < 3:
            return False

        # Minimum continuity: 40%
        continuity = self.n_associated / max(self.n_frames, 1)
        if continuity < 0.4:
            return False

        # Minimum average SNR: 8 dB
        avg_snr = self.total_snr / max(self.n_associated, 1)
        if avg_snr < 8.0:
            return False

        # Minimum duration: 5 seconds
        duration_sec = (self.death_timestamp - self.birth_timestamp) / 1000.0
        if duration_sec < 5.0:
            return False

        return True

    def should_delete(self):
        """Check if track should be deleted."""
        if self.state_status == TrackState.TENTATIVE:
            # Delete tentative tracks after N_WINDOW if not promoted
            return self.n_frames > N_WINDOW()
        else:
            # Delete active/coasting tracks after N_DELETE missed
            return self.n_missed > N_DELETE()

    def get_length_bucket(self):
        """Categorize track by length into short/medium/long."""
        if self.n_associated < 10:
            return 'short'
        elif self.n_associated < 50:
            return 'medium'
        else:
            return 'long'

    def get_recent_detections(self, n=20):
        """Get last N detections with timestamps and frame numbers.

        Args:
            n: Maximum number of detections to return (default: 20)

        Returns:
            List of detection dicts with timestamp, frame, delay, doppler, snr
        """
        detections = []
        for i, m in enumerate(self.history['measurements']):
            if m is not None:
                detections.append({
                    'timestamp': self.history['timestamps'][i],
                    'delay': m['delay'],
                    'doppler': m['doppler'],
                    'snr': m['snr']
                })
        return detections[-n:] if n else detections

    def to_dict(self):
        """Convert track to dictionary for JSON serialization."""
        duration_sec = (self.death_timestamp - self.birth_timestamp) / 1000.0
        avg_snr = self.total_snr / max(self.n_associated, 1)
        continuity = self.n_associated / max(self.n_frames, 1)

        return {
            'id': self.id,
            'adsb_hex': self.adsb_hex,
            'adsb_initialized': self.adsb_initialized,
            'state_status': self.state_status.name,
            'n_frames': self.n_frames,
            'n_associated': self.n_associated,
            'length_bucket': self.get_length_bucket(),
            'quality_score': self.get_quality_score(),
            'avg_snr': avg_snr,
            'duration_sec': duration_sec,
            'continuity': continuity,
            'birth_timestamp': self.birth_timestamp,
            'death_timestamp': self.death_timestamp,
            'history': {
                'timestamps': self.history['timestamps'],
                'states': [s.tolist() for s in self.history['states']],
                'delays': [m['delay'] if m else None for m in self.history['measurements']],
                'dopplers': [m['doppler'] if m else None for m in self.history['measurements']],
                'snrs': [m['snr'] if m else None for m in self.history['measurements']],
                'state_status': self.history['state_status']
            }
        }


# ============================================================================
# TRACKER CLASS
# ============================================================================

class Tracker:
    """
    Multi-target tracker using Kalman filtering and GNN data association.
    Based on blah2 architecture with enhanced filtering.
    """

    def __init__(self, event_writer=None, detection_window=20, config=None):
        """Initialize tracker.

        Args:
            event_writer: Optional TrackEventWriter for streaming JSONL output
            detection_window: Number of detections to include in sliding window (default: 20)
            config: Configuration dict (optional)
        """
        self.kf = KalmanFilter()
        self.tracks = []
        self.all_tracks = []  # Keep history of all tracks
        self.last_timestamp = None
        self.detection_window = detection_window
        self.frame_count = 0
        self.event_writer = event_writer
        self.config = config if config else get_config()

    def process_frame(self, detections, timestamp):
        """
        Process one frame of detections.

        Args:
            detections: List of detection dicts
            timestamp: Timestamp (ms)
        """
        self.frame_count += 1

        # Calculate time delta
        if self.last_timestamp is not None:
            dt = (timestamp - self.last_timestamp) / 1000.0  # Convert to seconds
        else:
            dt = 0.5  # Default for first frame

        # Filter detections by SNR
        detections = [d for d in detections if d['snr'] >= MIN_SNR()]

        # 1. Predict all tracks
        for track in self.tracks:
            track.predict(dt)

        # 2. Associate detections to tracks (GNN)
        associations = self._associate(detections)

        # 3. Update tracks with associations
        associated_tracks = set()
        associated_detections = set()

        for track_idx, det_idx in associations:
            track = self.tracks[track_idx]
            det = detections[det_idx]
            track.update(det, timestamp, frame=self.frame_count)
            # Transition COASTING -> ACTIVE when track gets detection again
            if track.state_status == TrackState.COASTING:
                track.state_status = TrackState.ACTIVE
            associated_tracks.add(track_idx)
            associated_detections.add(det_idx)

            # Emit track update (only for promoted tracks with IDs)
            if track.id and self.event_writer:
                detections_window = track.get_recent_detections(n=self.detection_window)
                self.event_writer.write_event(track.id, timestamp, track.n_associated, detections_window,
                                              adsb_hex=track.adsb_hex, adsb_initialized=track.adsb_initialized)

        # 4. Mark missed tracks
        for i, track in enumerate(self.tracks):
            if i not in associated_tracks:
                track.mark_missed(timestamp, frame=self.frame_count)
                # Update coasting state
                if track.state_status == TrackState.ACTIVE:
                    track.state_status = TrackState.COASTING

        # 5. Promote tracks (M/N logic)
        for track in self.tracks:
            promoted = track.promote_if_ready()
            if promoted:
                # Assign ID at promotion time (use ICAO hex if available)
                track.id = Track._generate_id(timestamp, adsb_hex=track.adsb_hex)
                if self.event_writer:
                    # Include all promotion detections
                    detections_list = track.get_recent_detections(n=track.n_associated)
                    self.event_writer.write_event(track.id, timestamp, track.n_associated, detections_list,
                                                  adsb_hex=track.adsb_hex, adsb_initialized=track.adsb_initialized)

        # 6. Initiate new tracks from unassociated detections
        for i, det in enumerate(detections):
            if i not in associated_detections:
                new_track = Track(det, timestamp, self.kf, frame=self.frame_count, config=self.config)
                self.tracks.append(new_track)
                # No event emitted here - track_init fires at promotion only

        # 6b. Promote tentative tracks with consistent linear motion (tracklet-based initiation)
        self._initiate_tracklets(timestamp)

        # 7. Delete old tracks (but save them to history)
        deleted_tracks = [t for t in self.tracks if t.should_delete()]
        for track in deleted_tracks:
            if track.state_status == TrackState.ACTIVE or track.n_associated >= M_THRESHOLD():
                self.all_tracks.append(track)
        self.tracks = [t for t in self.tracks if not t.should_delete()]

        # 8. Merge compatible tracks (post-processing on historical tracks)
        if len(self.all_tracks) > 1:
            self._merge_tracks()

        self.last_timestamp = timestamp

    def _associate(self, detections):
        """
        Associate detections to tracks using GNN (Hungarian algorithm).

        Args:
            detections: List of detection dicts

        Returns:
            List of (track_idx, detection_idx) tuples
        """
        if not self.tracks or not detections:
            return []

        # Build cost matrix (Mahalanobis distance)
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        cost_matrix = np.full((n_tracks, n_dets), 1e6)

        for i, track in enumerate(self.tracks):
            z_pred = track.get_predicted_measurement()
            S = track.get_innovation_covariance()

            # Compute inverse of innovation covariance - protect against singular matrix
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                # Singular innovation covariance - skip this track in association
                print(f"Warning: Singular innovation covariance for track {track.id}, "
                      f"skipping association", file=sys.stderr)
                continue

            # Dynamic gating: expand gate for coasting tracks
            gate_threshold = GATE_THRESHOLD()
            if track.state_status == TrackState.COASTING and track.n_missed > 0:
                # Expand gate by 10% per missed frame, up to 1.2x (VERY_LOW config)
                expansion = min(1.0 + 0.1 * track.n_missed, 1.2)
                gate_threshold = GATE_THRESHOLD() * expansion

            for j, det in enumerate(detections):
                z = np.array([det['delay'], det['doppler']])
                innovation = z - z_pred

                # Mahalanobis distance
                mahal_dist = innovation.T @ S_inv @ innovation

                # Gate: only consider if within threshold
                if mahal_dist < gate_threshold:
                    # Base cost: Mahalanobis distance
                    cost = mahal_dist

                    # SNR weighting: prefer high-SNR detections
                    snr = det.get('snr', 10.0)
                    snr_weight = 20.0 / max(snr, 5.0)  # Higher SNR = lower cost
                    cost *= snr_weight

                    # ADS-B priority: prefer ADS-B-initialized tracks
                    if self.config.get('adsb', {}).get('priority') and track.adsb_initialized:
                        cost *= 0.8  # 20% cost reduction for ADS-B tracks

                    cost_matrix[i, j] = cost

        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter out associations with infinite cost
        associations = [
            (r, c) for r, c in zip(row_ind, col_ind)
            if cost_matrix[r, c] < 1e6
        ]

        return associations

    def _initiate_tracklets(self, timestamp):
        """
        Promote tentative tracks with consistent linear motion to ACTIVE.

        Scans TENTATIVE tracks with >= 3 associations and checks if the last
        3 measurements form a consistent linear trajectory. If so, immediately
        promotes to ACTIVE and initializes velocity from linear fit.

        This prevents track loss due to early deletion before M/N promotion.

        Args:
            timestamp: Current frame timestamp (ms) for event emission
        """
        for track in self.tracks:
            # Only process tentative tracks with sufficient associations
            if track.state_status != TrackState.TENTATIVE:
                continue
            if track.n_associated < 3:
                continue

            # Extract last 3 associated measurements
            measurements = track.history['measurements']
            timestamps = track.history['timestamps']

            # Find indices of last 3 associations (skip None entries)
            assoc_indices = [i for i, m in enumerate(measurements) if m is not None]
            if len(assoc_indices) < 3:
                continue

            # Get last 3 associations
            last_3_indices = assoc_indices[-3:]

            # Extract delay, doppler, and time for linear fit
            delays = []
            dopplers = []
            times = []

            for idx in last_3_indices:
                m = measurements[idx]
                t = timestamps[idx]
                delays.append(m['delay'])
                dopplers.append(m['doppler'])
                times.append(t / 1000.0)  # Convert to seconds

            # Check if measurements span reasonable time (allow gaps, but check total window)
            dt_total = times[-1] - times[0]
            if dt_total > TRACKLET_MAX_TIME_SPAN():
                continue
            if dt_total < 0.1:  # Too short time window for reliable velocity estimate
                continue

            delay_velocity = (delays[-1] - delays[0]) / dt_total
            doppler_velocity = (dopplers[-1] - dopplers[0]) / dt_total

            # Check linearity: compute residuals
            delay_residuals = []
            doppler_residuals = []

            for i in range(len(times)):
                dt = times[i] - times[0]
                pred_delay = delays[0] + delay_velocity * dt
                pred_doppler = dopplers[0] + doppler_velocity * dt

                delay_residuals.append(abs(delays[i] - pred_delay))
                doppler_residuals.append(abs(dopplers[i] - pred_doppler))

            # Check if residuals are small (good linear fit)
            max_delay_residual = max(delay_residuals)
            max_doppler_residual = max(doppler_residuals)

            # Check against configured thresholds
            if (max_delay_residual < TRACKLET_MAX_DELAY_RESIDUAL() and
                max_doppler_residual < TRACKLET_MAX_DOPPLER_RESIDUAL()):
                # Good linear fit - promote immediately
                track.state_status = TrackState.ACTIVE

                # Update state with fitted velocity
                track.state[1] = delay_velocity    # delay_rate
                track.state[3] = doppler_velocity  # doppler_rate

                # Assign ID at promotion time (use ICAO hex if available)
                track.id = Track._generate_id(timestamp, adsb_hex=track.adsb_hex)

                print(f"Track {track.id} promoted to ACTIVE via tracklet (linear fit: "
                      f"v_delay={delay_velocity:.2f}, v_doppler={doppler_velocity:.2f})",
                      file=sys.stderr)

                # Emit track at promotion
                if self.event_writer:
                    # Include all promotion detections
                    detections_list = track.get_recent_detections(n=track.n_associated)
                    self.event_writer.write_event(track.id, timestamp, track.n_associated, detections_list,
                                                  adsb_hex=track.adsb_hex, adsb_initialized=track.adsb_initialized)

    def _merge_tracks(self):
        """
        Merge tracks that are likely from the same aircraft.
        Looks for temporal continuity: track A ends, track B starts nearby.
        """
        if len(self.all_tracks) < 2:
            return

        merged_indices = set()

        for i in range(len(self.all_tracks)):
            if i in merged_indices:
                continue

            track_a = self.all_tracks[i]

            for j in range(i + 1, len(self.all_tracks)):
                if j in merged_indices:
                    continue

                track_b = self.all_tracks[j]

                # Check temporal proximity
                time_gap = abs(track_a.death_timestamp - track_b.birth_timestamp)
                if time_gap > 5000:  # Max 5 second gap
                    continue

                # Get end state of track_a and start state of track_b
                end_state_a = track_a.history['states'][-1]
                start_state_b = track_b.history['states'][0]

                # Calculate distance in delay-Doppler space
                delay_diff = abs(end_state_a[0] - start_state_b[0])
                doppler_diff = abs(end_state_a[2] - start_state_b[2])

                # Merge if close enough (within 5 delay units and 50 Hz)
                if delay_diff < 5.0 and doppler_diff < 50.0:
                    # Merge track_b into track_a
                    self._merge_track_pair(track_a, track_b)
                    merged_indices.add(j)
                    break  # Only merge once per track_a

        # Remove merged tracks
        self.all_tracks = [t for i, t in enumerate(self.all_tracks) if i not in merged_indices]

    def _merge_track_pair(self, track_a, track_b):
        """Merge track_b into track_a."""
        # Append track_b's history to track_a
        track_a.history['timestamps'].extend(track_b.history['timestamps'])
        track_a.history['states'].extend(track_b.history['states'])
        track_a.history['measurements'].extend(track_b.history['measurements'])
        track_a.history['state_status'].extend(track_b.history['state_status'])

        # Update counters
        track_a.n_frames += track_b.n_frames
        track_a.n_associated += track_b.n_associated
        track_a.total_snr += track_b.total_snr
        track_a.death_timestamp = track_b.death_timestamp

        # Update final state
        track_a.state = track_b.state
        track_a.covariance = track_b.covariance

    def get_active_tracks(self):
        """Get all ACTIVE tracks."""
        return [t for t in self.tracks if t.state_status == TrackState.ACTIVE]

    def get_all_tracks(self):
        """Get all current tracks."""
        return self.tracks

    def get_confirmed_tracks(self):
        """Get all tracks that were ever confirmed (active, coasting, or historical)."""
        confirmed = []
        # Current active and coasting tracks (COASTING tracks were previously ACTIVE)
        confirmed.extend([t for t in self.tracks if t.state_status in (TrackState.ACTIVE, TrackState.COASTING)])
        # Historical tracks
        confirmed.extend(self.all_tracks)
        return confirmed

    def to_dict(self):
        """Convert all tracks to dictionary."""
        all_confirmed = self.get_confirmed_tracks()
        return {
            'tracks': [t.to_dict() for t in all_confirmed],
            'n_tracks': len(all_confirmed),
            'n_active': len(self.get_active_tracks())
        }


# ============================================================================
# MAIN TRACKING SCRIPT
# ============================================================================

def load_detections(filepath):
    """Load detection data from JSON or JSONL file."""
    with open(filepath, 'r') as f:
        content = f.read().strip()

        # Try parsing as single JSON array first
        try:
            detections = json.loads(content)
            if isinstance(detections, list):
                return detections
        except json.JSONDecodeError:
            pass

        # Try parsing as JSONL (one JSON object per line)
        detections = []
        lines = content.split('\n')
        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            if line:
                try:
                    detections.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num} in {filepath}: {e}",
                          file=sys.stderr)
                    print(f"  Line content: {line[:100]}{'...' if len(line) > 100 else ''}",
                          file=sys.stderr)

        return detections

    return []


def process_detections(detections_file, event_writer=None, detection_window=20):
    """Process all detections and generate tracks."""
    # Use stderr for progress if stdout is used for streaming
    output = sys.stderr if event_writer and event_writer._is_stdout else sys.stdout

    # Load detections
    print(f"Loading detections from {detections_file}...", file=output)
    detection_frames = load_detections(detections_file)
    print(f"Loaded {len(detection_frames)} detection frames", file=output)

    # Initialize tracker (config loaded automatically via get_config())
    tracker = Tracker(event_writer=event_writer, detection_window=detection_window, config=get_config())

    # Process each frame
    for i, frame in enumerate(detection_frames):
        timestamp = frame['timestamp']
        delays = frame.get('delay', [])
        dopplers = frame.get('doppler', [])
        snrs = frame.get('snr', [])
        adsb_list = frame.get('adsb', [])

        # Convert to detection list
        detections = []
        for idx, (delay, doppler, snr) in enumerate(zip(delays, dopplers, snrs)):
            detection = {
                'delay': delay,
                'doppler': doppler,
                'snr': snr
            }
            # Add ADS-B data if available for this detection
            if adsb_list and idx < len(adsb_list) and adsb_list[idx] is not None:
                detection['adsb'] = adsb_list[idx]
            detections.append(detection)

        # Process frame
        tracker.process_frame(detections, timestamp)

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(detection_frames)} frames, "
                  f"{len(tracker.tracks)} tracks ({len(tracker.get_active_tracks())} active)",
                  file=output)

    return tracker


def save_tracks(tracker, output_file):
    """Save tracks to JSON file."""
    data = tracker.to_dict()
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {data['n_tracks']} tracks to {output_file}", file=sys.stderr)


def visualize_tracks(tracker, detections_file, output_image, tracks_only=False, min_associations=0, length_bucket='all'):
    """Visualize tracks overlaid on detections or tracks only."""
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Load detections data (needed for background plot)
    if not tracks_only:
        detection_frames = load_detections(detections_file)
        all_delays = []
        all_dopplers = []
        all_snrs = []

        for frame in detection_frames:
            all_delays.extend(frame.get('delay', []))
            all_dopplers.extend(frame.get('doppler', []))
            all_snrs.extend(frame.get('snr', []))

    # Plot tracks FIRST (lower z-order, so detections appear on top)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    # Always use all confirmed tracks for plotting
    plot_tracks = tracker.get_confirmed_tracks()

    # Filter by minimum associations
    if min_associations > 0:
        plot_tracks = [t for t in plot_tracks if t.n_associated >= min_associations]

    # Filter by length bucket
    if length_bucket != 'all':
        plot_tracks = [t for t in plot_tracks if t.get_length_bucket() == length_bucket]

    print(f"Plotting {len(plot_tracks)} tracks (min_assoc >= {min_associations}, length_bucket = {length_bucket})", file=sys.stderr)

    for i, track in enumerate(plot_tracks):
        color = colors[i % 20]

        # Extract actual measurements (not predicted states)
        measurements = track.history['measurements']
        delays = [m['delay'] for m in measurements if m is not None]
        dopplers = [m['doppler'] for m in measurements if m is not None]

        if len(delays) == 0:
            continue

        # Plot thin black connecting line first (so it's behind dots)
        ax.plot(delays, dopplers, '-', color='black', linewidth=0.5,
                alpha=0.3, zorder=1)

        # Plot detections as scatter points colored by track
        ax.scatter(delays, dopplers, c=[color]*len(delays), s=35,
                  alpha=0.8, edgecolors='none', zorder=2, label=f'Track {track.id}')

    # Plot all detections ON TOP of tracks (higher z-order)
    if not tracks_only:
        scatter = ax.scatter(all_delays, all_dopplers, c=all_snrs,
                            cmap='coolwarm', s=5, alpha=0.4, zorder=3, label='Detections')
        cbar = plt.colorbar(scatter, ax=ax, label='SNR (dB)')

    # Labels and title
    ax.set_xlabel('Delay', fontsize=12)
    ax.set_ylabel('Doppler (Hz)', fontsize=12)
    title = 'Radar Tracks Only' if tracks_only else 'Radar Tracks'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Legend for tracks (limit to avoid clutter)
    if len(plot_tracks) > 0:
        if len(plot_tracks) <= 15:
            ax.legend(loc='upper right', fontsize=8, ncol=2)
        else:
            # Just show a sample
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:15], labels[:15], loc='upper right', fontsize=8, ncol=2)

    plt.tight_layout()
    fig.savefig(output_image, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_image}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='Track bistatic radar detections')
    parser.add_argument('file', help='Path to .detection file')
    parser.add_argument('-o', '--output', default='tracks.json',
                       help='Output JSON file for tracks')
    parser.add_argument('-v', '--visualize', help='Output image file for visualization')
    parser.add_argument('--tracks-only', action='store_true',
                       help='Visualize tracks only (no detection background)')
    parser.add_argument('--min-assoc', type=int, default=0,
                       help='Minimum associations to plot a track (default: 0)')
    parser.add_argument('--length-bucket', choices=['short', 'medium', 'long', 'all'],
                       default='all', help='Filter tracks by length bucket (short:<10, medium:10-49, long:>=50)')
    parser.add_argument('-s', '--stream-output', type=str,
                       help='Output file for streaming JSONL events (use - for stdout)')
    parser.add_argument('--detection-window', type=int, default=20,
                       help='Number of detections to include in sliding window (default: 20)')
    parser.add_argument('-c', '--config', type=str,
                       help='Path to configuration file (default: config.yaml)')

    args = parser.parse_args()

    # Load configuration
    if args.config:
        set_config(load_config(args.config))

    # Setup event writer if streaming output requested
    event_writer = None
    if args.stream_output:
        event_writer = TrackEventWriter(args.stream_output)

    # Process detections
    tracker = process_detections(args.file, event_writer=event_writer,
                                 detection_window=args.detection_window)

    # Close event writer
    if event_writer:
        event_writer.close()

    # Save tracks
    save_tracks(tracker, args.output)

    # Visualize if requested
    if args.visualize:
        visualize_tracks(tracker, args.file, args.visualize,
                        tracks_only=args.tracks_only, min_associations=args.min_assoc,
                        length_bucket=args.length_bucket)


if __name__ == '__main__':
    main()
