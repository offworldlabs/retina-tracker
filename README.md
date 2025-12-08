# retina-tracker

Multi-target tracker for bistatic passive radar detection data from [blah2](https://github.com/30hours/blah2).

## Overview

This tracker processes delay-Doppler detections to create continuous tracks of targets (typically commercial aircraft) in high-clutter environments. It uses:

- **Kalman filtering** with constant-velocity motion model in delay-Doppler space
- **Global Nearest Neighbor (GNN)** data association with Hungarian algorithm
- **M-of-N confirmation** logic for track promotion
- **Tracklet-based initiation** for robust track creation

## Installation

```bash
git clone https://github.com/yourusername/retina-tracker.git
cd retina-tracker
pip install -r requirements.txt
```

## Usage

### Tracker

Process detection data and output tracks:

```bash
# Basic usage
python -m tracker.track_detections data/input.detection -o output/tracks.json

# With streaming JSONL output
python -m tracker.track_detections data/input.detection -s output/events.jsonl

# With visualization
python -m tracker.track_detections data/input.detection -o output/tracks.json -v output/tracks.png

# With custom config
python -m tracker.track_detections data/input.detection -c custom_config.yaml -o output/tracks.json
```

### Plotter

Visualize raw detection data:

```bash
python -m plotter.plot_detections data/input.detection -o output/detections.png
```

## Configuration

Edit `config.yaml` to adjust tracker parameters:

```yaml
tracker:
  m_threshold: 4          # Associations needed to confirm track
  n_window: 6             # Window for M-of-N logic
  n_delete: 10            # Missed frames before deletion
  min_snr: 7.0            # Detection threshold (dB)
  gate_threshold: 9.0     # Chi-squared gate (99% confidence)
  detection_window: 20    # Rolling detection window size

process_noise:
  delay: 0.1
  doppler: 0.5
```

## Output Formats

### Streaming JSONL (`-s` flag)

Each line is a self-contained track update:

```json
{"track_id": "250618-00A3F1", "timestamp": 1718747750000, "length": 15, "detections": [
  {"timestamp": 1718747745000, "delay": 16.10, "doppler": 134.50, "snr": 16.2},
  {"timestamp": 1718747746000, "delay": 15.95, "doppler": 134.55, "snr": 17.1}
]}
```

- `track_id`: Unique ID in YYMMDD-XXXXXX format
- `timestamp`: Event timestamp (ms)
- `length`: Total detections associated with this track
- `detections`: Rolling window of recent measurements (up to 20)

### Tracks JSON (`-o` flag)

Complete track history for offline analysis.

## Input Format

Detection files from blah2 (`.detection`):

```json
{"timestamp": 1718747745000, "delay": [16.1, 22.3], "doppler": [134.5, -50.2], "snr": [16.2, 12.1]}
{"timestamp": 1718747745500, "delay": [16.0, 22.2], "doppler": [134.6, -50.1], "snr": [15.8, 11.9]}
```

## Data Directory

Place your `.detection` files in the `data/` directory. This directory is gitignored to avoid committing large data files.

## License

MIT
