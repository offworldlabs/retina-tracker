# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

**retina-tracker** is a multi-hypothesis aircraft tracking system for passive radar. It performs:
- Initial detection and track creation
- Distance gating to associate new detections
- Track-to-track association using nearest neighbor algorithms
- Filtering tracks using ADS-B truth data when available
- Processing detections from all contributing nodes
- Real-time telemetry solving and track updates
- Track state management, history, and deletion of stale tracks

## Development Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tracker
python tracker/track_detections.py --config config.yaml

# Run tests
pytest tests/ -v
```

### Docker Development
```bash
# Build container
docker build -t retina-tracker .

# Run container
docker run -d --name retina-tracker retina-tracker

# View logs
docker logs -f retina-tracker

# Execute commands in container
docker exec -it retina-tracker [command]
```

## Pre-Commit Verification

**CRITICAL: All changes MUST pass verification before committing.**

Before committing any code changes to retina-tracker, run these verification commands in order:

### Required Verification Commands

```bash
# 1. Lint Python code
ruff check tracker/ tests/ --select E,F,W --line-length 120

# 2. Check code formatting
ruff format --check tracker/ tests/

# 3. Run all tests
pytest tests/ -v
```

All three commands must pass with zero errors before committing.

### Auto-Fix Commands (Optional)

If linting or formatting issues are found, use these commands to automatically fix them:

```bash
# Auto-fix linting issues
ruff check --fix tracker/ tests/ --select E,F,W --line-length 120

# Auto-format code
ruff format tracker/ tests/
```

### Workflow

1. Make code changes
2. Run all three verification commands
3. If any command fails, fix the issues (use auto-fix commands if appropriate)
4. Re-run verification commands
5. Only commit if all three commands pass

**This verification process is mandatory, not optional.**

## Visual/Behavioral Verification

**IMPORTANT: Where possible, verify changes have the expected behavior using Claude in Chrome.**

When making changes that affect user-visible functionality or system behavior:

### When to Use Browser Verification

Use Claude in Chrome (browser automation) to verify:
- **Web UI changes**: Modifications to tar1090 visualization or any web interface
- **API endpoints**: Test that HTTP endpoints return expected data
- **Data visualization**: Verify plots, charts, or tracking displays render correctly
- **Integration points**: Check that components communicate as expected
- **User workflows**: Validate end-to-end user interactions work properly

### How to Verify

1. **Start the system**: Use docker compose or local development commands
2. **Use Claude in Chrome**: Navigate to relevant URLs and verify behavior
3. **Check visual elements**: Ensure UI elements appear and function correctly
4. **Test interactions**: Click buttons, submit forms, verify responses
5. **Validate data**: Confirm correct data is displayed in expected format

### Example Verification Scenarios

```bash
# Scenario: Verify tar1090 web interface displays tracks
1. Start system: docker compose up -d
2. Navigate to: http://localhost:8080
3. Verify: Aircraft tracks appear on map
4. Verify: Track metadata displays correctly
5. Verify: Real-time updates work

# Scenario: Verify API returns detection data
1. Start tracker: python tracker/track_detections.py --config config.yaml
2. Navigate to API endpoint (if exposed)
3. Verify: JSON response structure is correct
4. Verify: Data values are in expected ranges
```

**Note**: Not all changes require browser verification (e.g., internal algorithms, utility functions). Use judgment to determine when visual/behavioral verification adds value beyond automated tests.

## Testing

Tests are located in the `tests/` directory:
- `test_geometry.py` - Coordinate conversion and geometry functions
- `test_adsb_features.py` - ADS-B tracking features
- `test_anomaly_detection.py` - Anomaly detection for supersonic targets

All tests must pass before committing.

## Key Algorithms

- **Association**: Delay-Doppler matching of radar detections to ADS-B truth
- **Tracking**: Multi-hypothesis tracking algorithms
- **Data Fusion**: Track-to-track association and merging
- **Kalman Filter**: State estimation with delay-Doppler measurements
- **Anomaly Detection**: Supersonic target detection (Mach 1+)

## Configuration

Configuration is managed via YAML files and environment variables:
- `config.yaml` - Main tracking algorithm parameters
- `.env` - Environment-specific settings (radar URLs, reference location)

## Code Style Guidelines

- **Minimal code**: Write concise, idiomatic solutions
- **No comments**: Code should be self-documenting
- **Follow patterns**: Match existing architectural patterns
- **Test coverage**: All business logic must have tests
- **Linting**: Must pass Ruff checks (E, F, W rules)
- **Formatting**: Must pass Ruff formatting
