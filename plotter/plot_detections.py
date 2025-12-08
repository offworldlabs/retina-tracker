#!/usr/bin/env python3
"""
Plot bistatic radar detection data from blah2 software.
Displays delay-Doppler map with SNR color coding.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Range limits (set to None for auto-range)
DELAY_MIN = None
DELAY_MAX = None
DOPPLER_MIN = None
DOPPLER_MAX = None

def load_detections(filepath):
    """Load detection data from JSON file."""
    with open(filepath, 'r') as f:
        content = f.read().strip()
        # Try to parse as array of objects first
        try:
            detections = json.loads(content)
            if isinstance(detections, list):
                return detections
        except json.JSONDecodeError:
            pass

        # Fall back to line-by-line parsing
        detections = []
        for line in content.split('\n'):
            line = line.strip()
            if line:
                detections.append(json.loads(line))
    return detections

def extract_data(detections):
    """Extract delay, doppler, and SNR arrays from all detections."""
    all_delays = []
    all_dopplers = []
    all_snrs = []

    for detection in detections:
        delays = detection.get('delay', [])
        dopplers = detection.get('doppler', [])
        snrs = detection.get('snr', [])

        # Ensure all arrays have same length
        min_len = min(len(delays), len(dopplers), len(snrs))

        all_delays.extend(delays[:min_len])
        all_dopplers.extend(dopplers[:min_len])
        all_snrs.extend(snrs[:min_len])

    return np.array(all_delays), np.array(all_dopplers), np.array(all_snrs)

def plot_detections(delays, dopplers, snrs):
    """Create scatter plot of detections."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Apply range limits if specified
    mask = np.ones(len(delays), dtype=bool)
    if DELAY_MIN is not None:
        mask &= delays >= DELAY_MIN
    if DELAY_MAX is not None:
        mask &= delays <= DELAY_MAX
    if DOPPLER_MIN is not None:
        mask &= dopplers >= DOPPLER_MIN
    if DOPPLER_MAX is not None:
        mask &= dopplers <= DOPPLER_MAX

    delays = delays[mask]
    dopplers = dopplers[mask]
    snrs = snrs[mask]

    # Create scatter plot with SNR color mapping (blue to red)
    scatter = ax.scatter(delays, dopplers, c=snrs,
                        cmap='coolwarm', s=20, alpha=0.6)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='SNR (dB)')

    # Labels and title
    ax.set_xlabel('Delay', fontsize=12)
    ax.set_ylabel('Doppler (Hz)', fontsize=12)
    ax.set_title('Bistatic Radar Detection Map', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Plot bistatic radar detection data')
    parser.add_argument('file', help='Path to .detection file')
    parser.add_argument('-o', '--output', help='Save plot to file instead of displaying')

    args = parser.parse_args()

    # Load and process data
    print(f"Loading detections from {args.file}...")
    detections = load_detections(args.file)
    print(f"Loaded {len(detections)} detection frames")

    delays, dopplers, snrs = extract_data(detections)
    print(f"Total detections: {len(delays)}")

    # Create plot
    fig = plot_detections(delays, dopplers, snrs)

    # Save or display
    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {args.output}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
