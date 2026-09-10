"""Life-like synthetic detection data for testing retina-tracker.

Generates blah2-shaped detection frames with ground truth, calibrated against a
live capture of the Atlanta reference node. See `profile` for the measured
statistics every part of this package is tuned to reproduce.
"""

from .generate import continuity, generate, summarise
from .sensor import Sensor, in_coverage
from .world import Aircraft, Fleet, Site, bistatic, build_fleet

__all__ = [
    "Aircraft",
    "Fleet",
    "Sensor",
    "Site",
    "bistatic",
    "build_fleet",
    "continuity",
    "generate",
    "in_coverage",
    "summarise",
]
