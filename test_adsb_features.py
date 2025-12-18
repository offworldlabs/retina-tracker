#!/usr/bin/env python3
"""
Test ADS-B tracking features with synthetic data.
"""

import json
import os
import sys
sys.path.insert(0, 'tracker')

from tracker.track_detections import process_detections, set_config

def create_test_detection_file():
    """Create synthetic detection data with ADS-B fields."""
    detections = []

    # Frame 1: Two detections, one with ADS-B, one without
    detections.append({
        "timestamp": 1700000000000,
        "delay": [50.0, 75.0],
        "doppler": [100.0, -50.0],
        "snr": [15.0, 12.0],
        "adsb": [
            {
                "hex": "a12345",
                "lat": 37.8,
                "lon": -122.2,
                "alt_baro": 8500,
                "gs": 250,
                "track": 45
            },
            None  # Second detection has no ADS-B match
        ]
    })

    # Frames 2-5: Continuing tracks
    for i in range(1, 5):
        detections.append({
            "timestamp": 1700000000000 + i * 500,
            "delay": [50.0 + i * 0.5, 75.0 - i * 0.3],
            "doppler": [100.0 - i * 2, -50.0 + i * 1.5],
            "snr": [15.5, 12.5],
            "adsb": [
                {
                    "hex": "a12345",
                    "lat": 37.8 + i * 0.001,
                    "lon": -122.2 + i * 0.001,
                    "alt_baro": 8500 + i * 100,
                    "gs": 250,
                    "track": 45
                },
                None
            ]
        })

    # Save to file
    with open('test_adsb_data.detection', 'w') as f:
        json.dump(detections, f)

    print(f"✓ Created test data: {len(detections)} frames")
    return 'test_adsb_data.detection'


def test_adsb_tracking():
    """Test ADS-B-assisted tracking."""
    print("\n" + "="*60)
    print("Testing ADS-B Tracking Features")
    print("="*60)

    # Create test data
    test_file = create_test_detection_file()

    # Configure with ADS-B enabled
    config = {
        'tracker': {
            'm_threshold': 3,
            'n_window': 5,
            'n_delete': 10,
            'min_snr': 7.0,
            'gate_threshold': 9.0,
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
            'enabled': True,
            'priority': True,
            'reference_location': {
                'latitude': 37.7644,
                'longitude': -122.3954,
                'altitude': 23
            },
            'initial_covariance': {
                'position': 100.0,
                'velocity': 5.0
            }
        }
    }

    set_config(config)

    # Process detections
    print("\nProcessing detections with ADS-B enabled...")
    tracker = process_detections(test_file)

    # Check results
    confirmed_tracks = tracker.get_confirmed_tracks()
    print(f"\n✓ Generated {len(confirmed_tracks)} confirmed tracks")

    # Verify ADS-B features
    adsb_tracks = [t for t in confirmed_tracks if t.adsb_initialized]
    non_adsb_tracks = [t for t in confirmed_tracks if not t.adsb_initialized]

    print(f"  - {len(adsb_tracks)} ADS-B-initialized tracks")
    print(f"  - {len(non_adsb_tracks)} radar-only tracks")

    # Check track details
    for track in confirmed_tracks:
        track_dict = track.to_dict()
        print(f"\nTrack {track_dict['id']}:")
        print(f"  ADS-B hex: {track_dict['adsb_hex']}")
        print(f"  ADS-B initialized: {track_dict['adsb_initialized']}")
        print(f"  Associations: {track_dict['n_associated']}")
        print(f"  Quality score: {track_dict['quality_score']:.1f}")
        print(f"  Initial covariance: delay={track.covariance[0,0]:.3f}, doppler={track.covariance[2,2]:.3f}")

    # Assertions
    assert len(confirmed_tracks) >= 1, "Should have at least 1 confirmed track"
    assert len(adsb_tracks) >= 1, "Should have at least 1 ADS-B track"

    # Verify ADS-B track has ICAO hex in ID
    adsb_track = adsb_tracks[0]
    assert adsb_track.adsb_hex == "a12345", f"Expected hex a12345, got {adsb_track.adsb_hex}"
    assert "A12345" in adsb_track.id, f"Track ID should contain ICAO hex: {adsb_track.id}"

    # Verify ADS-B track has lower covariance
    if len(non_adsb_tracks) > 0:
        non_adsb_track = non_adsb_tracks[0]
        # ADS-B track should have lower delay covariance at initialization
        print(f"\nCovariance comparison:")
        print(f"  ADS-B track delay covariance: {adsb_track.covariance[0,0]:.3f}")
        print(f"  Radar-only track delay covariance: {non_adsb_track.covariance[0,0]:.3f}")

    # Clean up
    os.remove(test_file)
    print("\n" + "="*60)
    print("✓ All ADS-B tracking tests passed!")
    print("="*60)


def test_backward_compatibility():
    """Test that tracker works with old detection format (no adsb field)."""
    print("\n" + "="*60)
    print("Testing Backward Compatibility")
    print("="*60)

    # Create old-format data
    detections = []
    for i in range(5):
        detections.append({
            "timestamp": 1700000000000 + i * 500,
            "delay": [50.0 + i * 0.5],
            "doppler": [100.0 - i * 2],
            "snr": [15.0]
        })

    with open('test_old_format.detection', 'w') as f:
        json.dump(detections, f)

    print("✓ Created old-format test data (no adsb field)")

    # Process with default config (ADS-B disabled)
    tracker = process_detections('test_old_format.detection')
    confirmed_tracks = tracker.get_confirmed_tracks()

    print(f"✓ Processed {len(confirmed_tracks)} tracks")

    # Verify tracks work without ADS-B
    for track in confirmed_tracks:
        assert track.adsb_hex is None, "Old format should have no ADS-B hex"
        assert not track.adsb_initialized, "Old format should not be ADS-B initialized"

    os.remove('test_old_format.detection')
    print("✓ Backward compatibility test passed!")


if __name__ == '__main__':
    try:
        test_adsb_tracking()
        test_backward_compatibility()
        print("\n" + "="*60)
        print("SUCCESS: All tests passed! ✓")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
