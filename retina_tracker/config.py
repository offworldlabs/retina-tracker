"""Configuration loading and defaults for retina-tracker."""

import os
import sys

import yaml


def load_blah2_config(blah2_config_path):
    """Load center frequency from blah2 config file.

    Args:
        blah2_config_path: Path to blah2 config.yml file

    Returns:
        Center frequency in Hz, or None if not found
    """
    try:
        with open(blah2_config_path) as f:
            config = yaml.safe_load(f)
        fc = config.get("capture", {}).get("fc")
        if fc is not None:
            print(
                f"Loaded center frequency {fc / 1e6:.1f} MHz from {blah2_config_path}",
                file=sys.stderr,
            )
        return fc
    except (OSError, yaml.YAMLError) as e:
        print(
            f"Warning: Failed to load blah2 config from {blah2_config_path}: {e}",
            file=sys.stderr,
        )
        return None


def load_config(config_path=None):
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, looks for config.yaml in:
                    1. Current directory
                    2. Parent directory (for running from tracker/)

    Returns:
        Dict with configuration values, or defaults if no config found.
    """
    default_config = {
        "mode": "node",
        "tracker": {
            "m_threshold": 3,
            "n_window": 6,
            "n_delete": 10,
            "n_coast": 3,
            "min_snr": 4.0,
            "gate_threshold": 9.0,
            "detection_window": 20,
        },
        "process_noise": {"range_jerk": 1e-7},
        "tracklet": {"max_delay_residual": 2.0, "max_doppler_residual": 10.0, "max_time_span": 3.0},
        "shadow": {"enabled": True, "delay_km": 3.0, "snr_margin_db": 3.0, "min_fraction": 0.5},
        "adsb": {
            "enabled": False,
            "priority": True,
            "reference_location": None,
            "initial_covariance": {"position": 100.0, "velocity": 5.0},
        },
        "radar": {
            "blah2_config": None,
            "center_frequency": 200000000,
        },
        "tcp": {
            "host": "0.0.0.0",
            "port": 3012,
        },
        "output": {
            "max_bytes": 67108864,
            "backup_count": 1,
        },
    }

    if config_path is None:
        search_paths = ["config.yaml", "../config.yaml"]
        for path in search_paths:
            if os.path.exists(path):
                config_path = path
                break

    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            loaded = yaml.safe_load(f)
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


def _get_param(section, key, default=None):
    """Get parameter from config."""
    config = get_config()
    return config.get(section, {}).get(key, default)


# Track confirmation (M-of-N logic)
def M_THRESHOLD():
    return _get_param("tracker", "m_threshold", 3)


def N_WINDOW():
    return _get_param("tracker", "n_window", 6)


def N_DELETE():
    return _get_param("tracker", "n_delete", 10)


def N_COAST():
    return _get_param("tracker", "n_coast", 3)


def GATE_THRESHOLD():
    return _get_param("tracker", "gate_threshold", 9.0)


def MIN_SNR():
    return _get_param("tracker", "min_snr", 4.0)


def OUTPUT_MAX_BYTES():
    return _get_param("output", "max_bytes", 67108864)


def OUTPUT_BACKUP_COUNT():
    return _get_param("output", "backup_count", 1)


MEASUREMENT_NOISE_DELAY = 1.0
MEASUREMENT_NOISE_DOPPLER = 5.0
INITIAL_RANGE_ACCEL_VARIANCE = 1e-4


def CENTER_FREQUENCY_HZ():
    return _get_param("radar", "center_frequency", 200000000)


def WAVELENGTH_KM():
    return SPEED_OF_LIGHT / CENTER_FREQUENCY_HZ() / 1000.0


def PROCESS_NOISE_JERK():
    return _get_param("process_noise", "range_jerk", 1e-7)


def SHADOW_ENABLED():
    return _get_param("shadow", "enabled", True)


def SHADOW_DELAY_KM():
    return _get_param("shadow", "delay_km", 3.0)


def SHADOW_SNR_MARGIN_DB():
    return _get_param("shadow", "snr_margin_db", 3.0)


def SHADOW_MIN_FRACTION():
    return _get_param("shadow", "min_fraction", 0.5)


def TRACKLET_MAX_DELAY_RESIDUAL():
    return _get_param("tracklet", "max_delay_residual", 2.0)


def TRACKLET_MAX_DOPPLER_RESIDUAL():
    return _get_param("tracklet", "max_doppler_residual", 10.0)


def TRACKLET_MAX_TIME_SPAN():
    return _get_param("tracklet", "max_time_span", 3.0)


# Anomaly detection constants
SPEED_OF_LIGHT = 299792458.0
MACH_1_MS = 343.0
KNOTS_TO_MS = 0.514444
MAX_NORMAL_ACCEL_MS2 = 15.0
MAX_DIRECTION_CHANGE_DEG_PER_SEC = 30.0

# Sustained orbit detection
ORBIT_HEADING_WINDOW = 4  # consecutive frames to accumulate
ORBIT_MIN_CUMULATIVE_DEG = 270.0  # total |heading change| over window to flag

# GPS spoof detection (position mismatch)
SPOOF_POSITION_EPSILON_DEG = 0.002  # ~220 m — below this ADS-B is "frozen"
SPOOF_MIN_SPEED_KTS = 50  # aircraft must report moving
SPOOF_MIN_FROZEN_FRAMES = 2  # consecutive frozen frames to flag

# Altitude anomaly
ALTITUDE_JUMP_THRESHOLD_FT = 8000.0  # impossible alt change in one frame

# Anomalous acceleration (extreme, >10g)
ANOMALOUS_ACCEL_MS2 = 98.1  # 10g × 9.81 m/s²

# Long hover detection
LONG_HOVER_POSITION_EPSILON_DEG = 0.001  # ~111 m — position "frozen" threshold
LONG_HOVER_MIN_DURATION_S = 900.0  # 15 minutes


def get_mach1_doppler_threshold():
    """Calculate Doppler threshold for Mach 1 based on center frequency.

    Uses worst-case bistatic geometry (TX/RX collocated): f_d = 2 * v * fc / c
    """
    return 2 * MACH_1_MS * CENTER_FREQUENCY_HZ() / SPEED_OF_LIGHT
