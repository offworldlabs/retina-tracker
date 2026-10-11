"""Kalman filter over bistatic range for delay-Doppler detections."""

import sys
from typing import NamedTuple

import numpy as np

from .config import (
    MEASUREMENT_NOISE_DELAY_KM2,
    MEASUREMENT_NOISE_DELAY_SNR_EXPONENT,
    MEASUREMENT_NOISE_RANGE_RATE_KM2,
    PROCESS_NOISE_JERK,
    PROCESS_NOISE_RANGE_RATE,
    WAVELENGTH_KM,
)

# Delay and range rate. A property of the design, not of an instance: Track
# needs it before it has a filter.
MEASUREMENT_DIM = 2


def _symmetrised(covariance):
    """Round-off in (I - KH)P leaves the two off-diagonal entries differing in
    the last bit. While delay and Doppler were independent chains those entries
    were exactly zero and nothing noticed; a coupled state correlates range with
    its rate, and the association gate reads S as symmetric when it applies
    Sylvester's criterion to it."""
    return (covariance + covariance.T) / 2.0


def delay_noise_scale(snr):
    """Delay noise variance relative to its 10 dB value, for a scalar or an array of SNRs.

    The one formula behind both the update and the association gate, so the
    two cannot weight and admit a detection through different covariances.
    Doppler noise does not scale: measured, it barely depends on SNR.
    """
    if snr is None:
        return 1.0
    snr_linear = 10.0 ** (np.asarray(snr, dtype=float) / 10.0)
    return np.maximum(snr_linear / 10.0, 0.1) ** -MEASUREMENT_NOISE_DELAY_SNR_EXPONENT


def doppler_to_range_rate(doppler_hz):
    return -WAVELENGTH_KM() * doppler_hz


def range_rate_to_doppler(range_rate_km_s):
    return -range_rate_km_s / WAVELENGTH_KM()


class Residual(NamedTuple):
    """What one update revealed about the filter's own consistency.

    The innovation and its covariance are what R and Q are answerable to: R is
    the floor of S, and Q sets how fast the state's share of S grows between
    updates. Returning them rather than the NIS alone is what lets the two be
    separated from recorded data, which a single scalar cannot do because a
    too-large R and a too-small Q move it in opposite directions and cancel.
    """

    innovation: np.ndarray
    S: np.ndarray
    nis: float

    @classmethod
    def degenerate(cls):
        """A singular S skipped the measurement, so nothing was learned."""
        nan = np.full(MEASUREMENT_DIM, np.nan)
        return cls(nan, np.full((MEASUREMENT_DIM, MEASUREMENT_DIM), np.nan), float(MEASUREMENT_DIM))


class KalmanFilter:
    """Constant-acceleration filter on bistatic range.

    State vector: [range_km, range_rate_km_s, range_accel_km_s2]

    Doppler is not an independent quantity to be tracked alongside delay: it is
    the range rate, through range_rate = -wavelength * doppler. A detection
    therefore measures the first two state elements directly, and the rate
    arrives at the accuracy of the Doppler bin rather than being fitted from
    successive noisy range measurements.
    """

    def __init__(self, dt=0.5):
        self.dt = dt
        self.dim_state = 3
        self.dim_meas = MEASUREMENT_DIM

        self.F = np.eye(self.dim_state)
        self.H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.Q = np.zeros((self.dim_state, self.dim_state))

    @property
    def R(self):
        return np.diag([MEASUREMENT_NOISE_DELAY_KM2(), MEASUREMENT_NOISE_RANGE_RATE_KM2()])

    def measurement_noise(self, snr):
        R = self.R
        R[0, 0] *= float(delay_noise_scale(snr))
        return R

    def predict(self, state, covariance, q_scale=1.0):
        """Advance one frame.

        q_scale multiplies the jerk driving the acceleration state. Bistatic
        range acceleration is dominated by geometry rather than by the
        aircraft, so it differs between sites by more than it differs between
        flight phases - two nodes measured 2.9x apart. There is no single
        right jerk, which is why the caller derives this from the filter's
        own residuals rather than a constant carrying it.
        """
        dt = self.dt
        # Rebuild F and Q with the current dt so predictions match the actual
        # frame interval (which can vary from 0.5 s for real nodes to 40 s for
        # synthetic fleet nodes).  Uses in-place writes to pre-allocated
        # arrays to avoid per-call np.array() allocation overhead.
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt

        F = self.F
        F[0, 1] = dt
        F[0, 2] = 0.5 * dt2
        F[1, 2] = dt

        jerk = PROCESS_NOISE_JERK() * q_scale
        Q = self.Q
        Q[0, 0] = jerk * dt5 / 20.0
        Q[0, 1] = Q[1, 0] = jerk * dt4 / 8.0
        Q[0, 2] = Q[2, 0] = jerk * dt3 / 6.0
        Q[1, 1] = jerk * dt3 / 3.0
        Q[1, 2] = Q[2, 1] = jerk * dt2 / 2.0
        Q[2, 2] = jerk * dt
        # A range-rate random walk on top of the jerk: model error that grows
        # with the prediction interval far faster than white jerk allows for.
        rate = PROCESS_NOISE_RANGE_RATE()
        Q[0, 0] += rate * dt3 / 3.0
        Q[0, 1] += rate * dt2 / 2.0
        Q[1, 0] = Q[0, 1]
        Q[1, 1] += rate * dt

        state_pred = F @ state
        cov_pred = F @ covariance @ F.T + Q

        # Floor the range at 0; zero the rate only once the floor engages.  A
        # negative range rate with a still-positive range is a legitimately
        # approaching target and must NOT be clamped.
        if state_pred[0] < 0:
            state_pred[0] = 0.0
            if state_pred[1] < 0:
                state_pred[1] = 0.0

        return state_pred, _symmetrised(cov_pred)

    def update(self, state, covariance, measurement, snr=None):
        R = self.measurement_noise(snr)

        z_pred = self.H @ state
        innovation = measurement - z_pred
        S = self.H @ covariance @ self.H.T + R

        try:
            K = covariance @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Warning: Singular innovation covariance in Kalman update, skipping measurement", file=sys.stderr)
            return state, covariance, Residual.degenerate()

        state_upd = state + K @ innovation
        cov_upd = (np.eye(self.dim_state) - K @ self.H) @ covariance
        nis = float(innovation @ np.linalg.solve(S, innovation))

        return state_upd, _symmetrised(cov_upd), Residual(innovation, S, nis)

    def get_innovation_covariance(self, covariance, snr=None):
        S = self.H @ covariance @ self.H.T + self.measurement_noise(snr)
        return S

    def get_innovation_base(self, covariance):
        """H P Hᵀ — the state contribution to S, before measurement noise.

        The gate adds per-detection SNR-scaled R on top of this, matching
        update()'s noise model.
        """
        return self.H @ covariance @ self.H.T
