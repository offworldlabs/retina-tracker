#!/usr/bin/env python3
"""
Tests for geometry coordinate conversion functions.
"""

import numpy as np
import sys
sys.path.insert(0, 'tracker')
from tracker import geometry


def test_ft2m():
    """Test feet to meters conversion."""
    result = geometry.ft2m(5000)
    expected = 1524.0
    assert abs(result - expected) < 0.1, f"Expected {expected}, got {result}"
    print(f"✓ test_ft2m passed: {result:.2f}m")


def test_knots_to_ms():
    """Test knots to meters per second conversion."""
    result = geometry.knots_to_ms(100)
    expected = 51.4444
    assert abs(result - expected) < 0.01, f"Expected {expected}, got {result}"
    print(f"✓ test_knots_to_ms passed: {result:.4f} m/s")


def test_lla2ecef():
    """Test LLA to ECEF conversion with San Francisco coordinates."""
    lat, lon, alt = 37.7644, -122.3954, 23
    x, y, z = geometry.lla2ecef(lat, lon, alt)

    # Verify coordinates are in reasonable range for Earth surface
    assert abs(x) > 1e6 and abs(x) < 1e7, f"X coordinate {x} out of range"
    assert abs(y) > 1e6 and abs(y) < 1e7, f"Y coordinate {y} out of range"
    assert abs(z) > 1e6 and abs(z) < 1e7, f"Z coordinate {z} out of range"

    # Verify radius is approximately Earth radius
    # Note: Earth is ellipsoid, so radius varies by latitude
    # At SF latitude (37.76°), radius is ~7-8km less than equatorial
    radius = np.sqrt(x**2 + y**2 + z**2)
    earth_radius = 6378137.0
    assert abs(radius - earth_radius) < 10000, f"Radius {radius} far from Earth radius"

    print(f"✓ test_lla2ecef passed: ECEF=({x:.0f}, {y:.0f}, {z:.0f})")


def test_ecef_lla_roundtrip():
    """Test that ECEF -> LLA -> ECEF roundtrip is accurate."""
    # San Francisco coordinates
    lat_orig, lon_orig, alt_orig = 37.7644, -122.3954, 23

    # Convert to ECEF and back
    x, y, z = geometry.lla2ecef(lat_orig, lon_orig, alt_orig)
    lat_new, lon_new, alt_new = geometry.ecef2lla(x, y, z)

    # Check roundtrip accuracy (should be within mm for position)
    lat_err = abs(lat_new - lat_orig)
    lon_err = abs(lon_new - lon_orig)
    alt_err = abs(alt_new - alt_orig)

    assert lat_err < 1e-6, f"Latitude error {lat_err} too large"
    assert lon_err < 1e-6, f"Longitude error {lon_err} too large"
    assert alt_err < 0.01, f"Altitude error {alt_err}m too large"

    print(f"✓ test_ecef_lla_roundtrip passed: errors=(lat:{lat_err:.2e}°, lon:{lon_err:.2e}°, alt:{alt_err:.3f}m)")


def test_lla2enu():
    """Test LLA to ENU conversion with known reference point."""
    # Reference: San Francisco receiver
    ref_lat, ref_lon, ref_alt = 37.7644, -122.3954, 23

    # Point: Aircraft roughly 50km northeast and 1500m up
    lat, lon, alt = 38.0, -122.0, 1523

    east, north, up = geometry.lla2enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)

    # Verify ENU coordinates make sense
    assert north > 0, f"Aircraft should be north, got {north}m"
    assert east > 0, f"Aircraft should be east, got {east}m"
    assert up > 0, f"Aircraft should be up, got {up}m"
    assert up > 1000 and up < 2000, f"Altitude {up}m unexpected"

    # Verify horizontal distance is reasonable (roughly 50-60km)
    horiz_dist = np.sqrt(east**2 + north**2) / 1000
    assert 40 < horiz_dist < 70, f"Horizontal distance {horiz_dist}km unexpected"

    print(f"✓ test_lla2enu passed: ENU=({east:.0f}, {north:.0f}, {up:.0f}m), horiz={horiz_dist:.1f}km")


def test_enu_lla_roundtrip():
    """Test that ENU -> LLA -> ENU roundtrip is accurate."""
    ref_lat, ref_lon, ref_alt = 37.7644, -122.3954, 23

    # ENU coordinates: 10km east, 20km north, 1500m up
    east_orig, north_orig, up_orig = 10000, 20000, 1500

    # Convert to LLA and back
    lat, lon, alt = geometry.enu2lla(east_orig, north_orig, up_orig, ref_lat, ref_lon, ref_alt)
    east_new, north_new, up_new = geometry.lla2enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)

    # Check roundtrip accuracy
    east_err = abs(east_new - east_orig)
    north_err = abs(north_new - north_orig)
    up_err = abs(up_new - up_orig)

    assert east_err < 0.01, f"East error {east_err}m too large"
    assert north_err < 0.01, f"North error {north_err}m too large"
    assert up_err < 0.01, f"Up error {up_err}m too large"

    print(f"✓ test_enu_lla_roundtrip passed: errors=(E:{east_err:.3f}m, N:{north_err:.3f}m, U:{up_err:.3f}m)")


def test_enu_velocity_from_adsb():
    """Test conversion of ADS-B velocity to ENU components."""
    # Aircraft heading 045° (northeast) at 250 knots
    ground_speed = 250
    track = 45
    vertical_rate = 1000  # ft/min climbing

    vel_east, vel_north, vel_up = geometry.enu_velocity_from_adsb(
        ground_speed, track, vertical_rate
    )

    # Verify velocity magnitude matches ground speed
    horiz_speed = np.sqrt(vel_east**2 + vel_north**2)
    expected_speed = geometry.knots_to_ms(ground_speed)
    assert abs(horiz_speed - expected_speed) < 0.01, \
        f"Horizontal speed {horiz_speed} != {expected_speed}"

    # At 45° heading, east and north components should be equal
    assert abs(vel_east - vel_north) < 0.01, \
        f"For 45° heading, east={vel_east} should equal north={vel_north}"

    # Verify vertical component is positive (climbing)
    assert vel_up > 0, f"Climbing aircraft should have vel_up > 0, got {vel_up}"

    print(f"✓ test_enu_velocity_from_adsb passed: vel=({vel_east:.2f}, {vel_north:.2f}, {vel_up:.2f}) m/s")


def test_enu_velocity_cardinal_directions():
    """Test velocity conversions for cardinal directions."""
    ground_speed = 100  # knots
    expected_ms = geometry.knots_to_ms(ground_speed)

    # North (0°)
    e, n, u = geometry.enu_velocity_from_adsb(ground_speed, 0)
    assert abs(e) < 0.01, f"North heading should have ~0 east velocity"
    assert abs(n - expected_ms) < 0.01, f"North heading should have full north velocity"

    # East (90°)
    e, n, u = geometry.enu_velocity_from_adsb(ground_speed, 90)
    assert abs(e - expected_ms) < 0.01, f"East heading should have full east velocity"
    assert abs(n) < 0.01, f"East heading should have ~0 north velocity"

    # South (180°)
    e, n, u = geometry.enu_velocity_from_adsb(ground_speed, 180)
    assert abs(e) < 0.01, f"South heading should have ~0 east velocity"
    assert abs(n + expected_ms) < 0.01, f"South heading should have negative north velocity"

    # West (270°)
    e, n, u = geometry.enu_velocity_from_adsb(ground_speed, 270)
    assert abs(e + expected_ms) < 0.01, f"West heading should have negative east velocity"
    assert abs(n) < 0.01, f"West heading should have ~0 north velocity"

    print(f"✓ test_enu_velocity_cardinal_directions passed")


def test_norm():
    """Test 3D vector norm calculation."""
    # 3-4-5 triangle in 3D: (3, 4, 0) has norm 5
    result = geometry.norm(3, 4, 0)
    assert abs(result - 5.0) < 0.001, f"Expected 5.0, got {result}"

    # Unit vector in all dimensions
    result = geometry.norm(1, 1, 1)
    expected = np.sqrt(3)
    assert abs(result - expected) < 0.001, f"Expected {expected}, got {result}"

    print(f"✓ test_norm passed")


def test_lla2enu_origin():
    """Test that reference point converts to (0, 0, 0) in ENU."""
    ref_lat, ref_lon, ref_alt = 37.7644, -122.3954, 23

    # Converting reference point to itself should give origin
    east, north, up = geometry.lla2enu(ref_lat, ref_lon, ref_alt, ref_lat, ref_lon, ref_alt)

    assert abs(east) < 0.01, f"Reference point should have east=0, got {east}"
    assert abs(north) < 0.01, f"Reference point should have north=0, got {north}"
    assert abs(up) < 0.01, f"Reference point should have up=0, got {up}"

    print(f"✓ test_lla2enu_origin passed: ENU=({east:.6f}, {north:.6f}, {up:.6f})")


def test_bistatic_geometry_consistency():
    """Test that geometry functions match blah2-arm bistatic.js results."""
    # Use same test coordinates as blah2-arm integration test
    rx = {'latitude': 37.7644, 'longitude': -122.3954, 'altitude': 23}
    tx = {'latitude': 37.49917, 'longitude': -121.87222, 'altitude': 783}
    aircraft = {'lat': 37.6, 'lon': -122.1, 'alt_baro': 5000}

    # Convert aircraft to ENU relative to RX
    alt_m = geometry.ft2m(aircraft['alt_baro'])
    east, north, up = geometry.lla2enu(
        aircraft['lat'], aircraft['lon'], alt_m,
        rx['latitude'], rx['longitude'], rx['altitude']
    )

    # Verify aircraft is roughly in expected position
    # Southwest and up from receiver
    assert east > 0, f"Aircraft should be east"
    assert north < 0, f"Aircraft should be south"
    assert up > 0, f"Aircraft should be up"

    # Calculate distances using ECEF (same as bistatic.js)
    ac_x, ac_y, ac_z = geometry.lla2ecef(aircraft['lat'], aircraft['lon'], alt_m)
    rx_x, rx_y, rx_z = geometry.lla2ecef(rx['latitude'], rx['longitude'], rx['altitude'])
    tx_x, tx_y, tx_z = geometry.lla2ecef(tx['latitude'], tx['longitude'], tx['altitude'])

    d_rx_ac = geometry.norm(rx_x - ac_x, rx_y - ac_y, rx_z - ac_z)
    d_tx_ac = geometry.norm(tx_x - ac_x, tx_y - ac_y, tx_z - ac_z)
    d_rx_tx = geometry.norm(rx_x - tx_x, rx_y - tx_y, rx_z - tx_z)

    bistatic_range = (d_rx_ac + d_tx_ac - d_rx_tx) / 1000  # km

    # Bistatic range should be reasonable (within 0-500 km)
    assert 0 < bistatic_range < 500, f"Bistatic range {bistatic_range}km out of range"

    print(f"✓ test_bistatic_geometry_consistency passed: bistatic_range={bistatic_range:.2f}km")


def run_tests():
    """Run all geometry tests."""
    print("Running geometry module tests...\n")

    tests = [
        test_ft2m,
        test_knots_to_ms,
        test_lla2ecef,
        test_ecef_lla_roundtrip,
        test_lla2enu,
        test_enu_lla_roundtrip,
        test_enu_velocity_from_adsb,
        test_enu_velocity_cardinal_directions,
        test_norm,
        test_lla2enu_origin,
        test_bistatic_geometry_consistency
    ]

    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed.append(test.__name__)

    print(f"\n{'='*60}")
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)} tests failed")
        for name in failed:
            print(f"  - {name}")
        return 1
    else:
        print(f"SUCCESS: All {len(tests)} tests passed ✓")
        return 0


if __name__ == '__main__':
    exit(run_tests())
