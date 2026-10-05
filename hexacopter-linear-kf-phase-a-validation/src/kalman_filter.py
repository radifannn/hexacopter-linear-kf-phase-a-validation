"""
Linear Kalman Filter for 2D position + velocity tracking.

State:
    x = [N, E, N_dot, E_dot]^T

Measurement:
    z = [N, E]^T

Model:
    Constant velocity with configurable process and measurement
    covariance matrices.

This module intentionally contains no geolocation-specific logic.
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """
    Discrete linear Kalman filter.

    Parameters
    ----------
    dt : float
        Sampling interval in seconds.
    measurement_covariance : array-like, shape (2, 2)
        Measurement noise covariance R.
    process_covariance : array-like, shape (4, 4)
        Process noise covariance Q.
    initial_state : array-like, shape (4,)
        Initial state [N, E, N_dot, E_dot].
    initial_covariance : array-like, shape (4, 4), optional
        Initial estimation covariance P.
    """

    def __init__(
        self,
        dt,
        measurement_covariance,
        process_covariance,
        initial_state,
        initial_covariance=None,
    ):
        self.dt = float(dt)

        if self.dt <= 0.0:
            raise ValueError("dt must be > 0.")

        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        self.R = np.asarray(
            measurement_covariance,
            dtype=float,
        )

        self.Q = np.asarray(
            process_covariance,
            dtype=float,
        )

        self.x = np.asarray(
            initial_state,
            dtype=float,
        ).reshape(4, 1)

        if initial_covariance is None:
            self.P = np.eye(4, dtype=float)
        else:
            self.P = np.asarray(
                initial_covariance,
                dtype=float,
            )

        self._validate_shapes()
        self._update_transition_matrix()

    def _validate_shapes(self):
        if self.R.shape != (2, 2):
            raise ValueError(
                f"measurement_covariance must have shape (2, 2), "
                f"got {self.R.shape}."
            )

        if self.Q.shape != (4, 4):
            raise ValueError(
                f"process_covariance must have shape (4, 4), "
                f"got {self.Q.shape}."
            )

        if self.P.shape != (4, 4):
            raise ValueError(
                f"initial_covariance must have shape (4, 4), "
                f"got {self.P.shape}."
            )

        # Symmetrize covariance matrices to avoid small numerical asymmetry.
        self.R = 0.5 * (self.R + self.R.T)
        self.Q = 0.5 * (self.Q + self.Q.T)
        self.P = 0.5 * (self.P + self.P.T)

    def _update_transition_matrix(self):
        dt = self.dt

        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def set_dt(self, dt):
        """Change the sampling interval and rebuild F."""
        dt = float(dt)

        if dt <= 0.0:
            raise ValueError("dt must be > 0.")

        self.dt = dt
        self._update_transition_matrix()

    @property
    def state(self):
        """Current state as a 1D array of length 4."""
        return self.x[:, 0].copy()

    @property
    def current_state(self):
        """Alias for state."""
        return self.state

    def reset(self, state, covariance=None):
        """Reset the filter state and optionally its covariance."""
        self.x = np.asarray(
            state,
            dtype=float,
        ).reshape(4, 1)

        if self.x.shape != (4, 1):
            raise ValueError("state must contain exactly 4 elements.")

        if covariance is not None:
            P = np.asarray(covariance, dtype=float)

            if P.shape != (4, 4):
                raise ValueError(
                    "covariance must have shape (4, 4)."
                )

            self.P = 0.5 * (P + P.T)

    def predict(self):
        """
        Prediction step.

        Returns
        -------
        numpy.ndarray
            Predicted state as a 1D array [N, E, N_dot, E_dot].
        """
        self.x = self.F @ self.x
        self.P = (
            self.F
            @ self.P
            @ self.F.T
            + self.Q
        )

        # Limit numerical asymmetry.
        self.P = 0.5 * (self.P + self.P.T)

        return self.state

    def update(self, measurement):
        """
        Measurement update.

        Parameters
        ----------
        measurement : array-like, shape (2,)
            [N, E] position measurement.

        Returns
        -------
        numpy.ndarray
            Updated state as a 1D array.
        """
        z = np.asarray(
            measurement,
            dtype=float,
        ).reshape(2, 1)

        # Innovation.
        y = (
            z
            - self.H @ self.x
        )

        # Innovation covariance:
        # S = HPH^T + R
        S = (
            self.H
            @ self.P
            @ self.H.T
            + self.R
        )

        # Kalman gain:
        # K = P H^T S^-1
        PHt = self.P @ self.H.T

        try:
            K = np.linalg.solve(
                S,
                PHt.T,
            ).T
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Innovation covariance S is singular."
            ) from exc

        # State update.
        self.x = (
            self.x
            + K @ y
        )

        # Joseph-form covariance update for improved numerical stability:
        # P = (I-KH) P (I-KH)^T + K R K^T
        I = np.eye(4, dtype=float)
        IKH = I - K @ self.H

        self.P = (
            IKH
            @ self.P
            @ IKH.T
            + K
            @ self.R
            @ K.T
        )

        self.P = 0.5 * (self.P + self.P.T)

        return self.state
