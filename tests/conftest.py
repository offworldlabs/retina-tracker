"""Restore the global config between tests.

`retina_tracker.config` keeps the config in a module global that `set_config`
replaces and every accessor (`MIN_SNR()`, `GATE_THRESHOLD()`, ...) reads at call
time, so without this a file that sets it supplies the config for every later
file in collection order. The snapshot is deep, because the config is nested and
a test that mutates a subsection in place would otherwise write through a
shallow copy into the next test.
"""

import copy

import pytest

from retina_tracker import config as config_module


@pytest.fixture(autouse=True)
def _isolate_global_config():
    saved = copy.deepcopy(config_module._config)
    yield
    config_module.set_config(saved)
