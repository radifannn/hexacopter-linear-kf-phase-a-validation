import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sys


# ------------------------------------------------------------
# Allow import from ../src
# ------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

SRC_DIR = ROOT_DIR / "src"

sys.path.insert(
    0,
    str(SRC_DIR),
)

from kalman_filter import LinearKalmanFilter


# ============================================================
# Configuration
# ============================================================

DATA_FILE = (
    ROOT_DIR
    / "data"
    / "static_validation"
    / "R0.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "validation"
    / "kf_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DT = 0.10


# ------------------------------------------------------------
# Measurement covariance R
# Derived from static_validation_R0.csv
# ------------------------------------------------------------

R = np.array(
    [
        [
            5.1444649133792445e-05,
            0.0,
        ],
        [
            0.0,
            4.2093957356837325e-05,
        ],
    ],
    dtype=float,
)


# ------------------------------------------------------------
# Process covariance Q
# Initial value specified for Phase A
# ------------------------------------------------------------

Q = np.diag(
    [
        1e-4,
        1e-4,
        1e-3,
        1e-3,
    ]
)


# ------------------------------------------------------------
# Initial covariance
#
# Position uncertainty starts from the measured R.
# Velocity uncertainty starts at 1 (m/s)^2.
# ------------------------------------------------------------

P0 = np.diag(
    [
        R[0, 0],
        R[1, 1],
        1.0,
        1.0,
    ]
)


# ============================================================
# Load CSV
# ============================================================

def load_dataset(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    with open(
        path,
        "r",
        newline="",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    if not rows:
        raise RuntimeError(
            "CSV contains no samples."
        )

    required_columns = {
        "sample",
        "estimated_north",
        "estimated_east",
        "true_north",
        "true_east",
    }

    missing = (
        required_columns
        -
        set(rows[0].keys())
    )

    if missing:
        raise RuntimeError(
            "Missing CSV columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    measurements = np.array(
        [
            [
                float(
                    row[
                        "estimated_north"
                    ]
                ),
                float(
                    row[
                        "estimated_east"
                    ]
                ),
            ]
            for row in rows
        ],
        dtype=float,
    )

    truth = np.array(
        [
            [
                float(
                    row[
                        "true_north"
                    ]
                ),
                float(
                    row[
                        "true_east"
                    ]
                ),
            ]
            for row in rows
        ],
        dtype=float,
    )

    return (
        rows,
        measurements,
        truth,
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    estimate,
    truth,
):

    error = (
        estimate
        -
        truth
    )

    horizontal_error = np.sqrt(
        np.sum(
            error ** 2,
            axis=1,
        )
    )

    north_rmse = math.sqrt(
        np.mean(
            error[:, 0] ** 2
        )
    )

    east_rmse = math.sqrt(
        np.mean(
            error[:, 1] ** 2
        )
    )

    horizontal_rmse = math.sqrt(
        np.mean(
            horizontal_error ** 2
        )
    )

    horizontal_mae = np.mean(
        horizontal_error
    )

    north_bias = np.mean(
        error[:, 0]
    )

    east_bias = np.mean(
        error[:, 1]
    )

    return {
        "north_rmse": north_rmse,
        "east_rmse": east_rmse,
        "horizontal_rmse":
            horizontal_rmse,
        "horizontal_mae":
            horizontal_mae,
        "horizontal_max":
            np.max(horizontal_error),
        "north_bias":
            north_bias,
        "east_bias":
            east_bias,
        "horizontal_bias":
            math.hypot(
                north_bias,
                east_bias,
            ),
        "horizontal_error":
            horizontal_error,
        "error":
            error,
    }


def step_smoothness(
    position_data,
):
    """
    RMS displacement between consecutive
    position estimates.

    Lower values indicate smoother position output.
    """

    differences = np.diff(
        position_data,
        axis=0,
    )

    if len(differences) == 0:
        return 0.0

    displacement = np.sqrt(
        np.sum(
            differences ** 2,
            axis=1,
        )
    )

    return math.sqrt(
        np.mean(
            displacement ** 2
        )
    )


# ============================================================
# Plot
# ============================================================

def save_plots(
    measurements,
    filtered,
    truth,
):

    samples = np.arange(
        1,
        len(truth) + 1,
    )

    # --------------------------------------------------------
    # North
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        samples,
        truth[:, 0],
        label="True North",
    )

    plt.plot(
        samples,
        measurements[:, 0],
        label="Raw Geometric Estimate",
    )

    plt.plot(
        samples,
        filtered[:, 0],
        label="KF Estimate",
    )

    plt.xlabel(
        "Sample"
    )

    plt.ylabel(
        "North position (m)"
    )

    plt.title(
        "Static KF Validation — North Position"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "kf_north.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # East
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        samples,
        truth[:, 1],
        label="True East",
    )

    plt.plot(
        samples,
        measurements[:, 1],
        label="Raw Geometric Estimate",
    )

    plt.plot(
        samples,
        filtered[:, 1],
        label="KF Estimate",
    )

    plt.xlabel(
        "Sample"
    )

    plt.ylabel(
        "East position (m)"
    )

    plt.title(
        "Static KF Validation — East Position"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "kf_east.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Horizontal error
    # --------------------------------------------------------

    raw_error = (
        measurements
        -
        truth
    )

    filtered_error = (
        filtered
        -
        truth
    )

    raw_horizontal = np.sqrt(
        np.sum(
            raw_error ** 2,
            axis=1,
        )
    )

    filtered_horizontal = np.sqrt(
        np.sum(
            filtered_error ** 2,
            axis=1,
        )
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        samples,
        raw_horizontal * 100,
        label="Raw Geometric Error",
    )

    plt.plot(
        samples,
        filtered_horizontal * 100,
        label="KF Error",
    )

    plt.xlabel(
        "Sample"
    )

    plt.ylabel(
        "Horizontal error (cm)"
    )

    plt.title(
        "Static KF Validation — Horizontal Error"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "kf_horizontal_error.png",
        dpi=150,
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PHASE A — OFFLINE LINEAR KALMAN FILTER VALIDATION")
    print("=" * 70)

    print()
    print(
        f"Dataset:"
        f"\n  {DATA_FILE}"
    )

    print()
    print(
        f"dt = {DT:.2f} s"
    )

    print()
    print("Measurement covariance R:")

    print(R)

    print()
    print("Process covariance Q:")

    print(Q)

    print()
    print("Initial covariance P0:")

    print(P0)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        rows,
        measurements,
        truth,
    ) = load_dataset(
        DATA_FILE
    )

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    initial_state = np.array(
        [
            measurements[0, 0],
            measurements[0, 1],
            0.0,
            0.0,
        ]
    )

    # --------------------------------------------------------
    # Create KF
    # --------------------------------------------------------

    kf = LinearKalmanFilter(
        dt=DT,
        measurement_covariance=R,
        process_covariance=Q,
        initial_state=initial_state,
        initial_covariance=P0,
    )

    # --------------------------------------------------------
    # Run filter
    # --------------------------------------------------------

    filtered_states = []

    for measurement in measurements:

        filtered_state = (
            kf.step(
                measurement
            )
        )

        filtered_states.append(
            filtered_state
        )

    filtered_states = np.asarray(
        filtered_states
    )

    filtered_position = (
    filtered_states[
        :,
        :2,
        0
    ]
)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    raw_metrics = calculate_metrics(
        measurements,
        truth,
    )

    filtered_metrics = (
        calculate_metrics(
            filtered_position,
            truth,
        )
    )

    raw_smoothness = (
        step_smoothness(
            measurements
        )
    )

    filtered_smoothness = (
        step_smoothness(
            filtered_position
        )
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RAW GEOMETRIC ESTIMATE")
    print("=" * 70)

    print(
        f"North RMSE       = "
        f"{raw_metrics['north_rmse'] * 100:.4f} cm"
    )

    print(
        f"East RMSE        = "
        f"{raw_metrics['east_rmse'] * 100:.4f} cm"
    )

    print(
        f"Horizontal RMSE  = "
        f"{raw_metrics['horizontal_rmse'] * 100:.4f} cm"
    )

    print(
        f"Horizontal MAE   = "
        f"{raw_metrics['horizontal_mae'] * 100:.4f} cm"
    )

    print(
        f"Horizontal max    = "
        f"{raw_metrics['horizontal_max'] * 100:.4f} cm"
    )

    print(
        f"North bias       = "
        f"{raw_metrics['north_bias'] * 100:+.4f} cm"
    )

    print(
        f"East bias        = "
        f"{raw_metrics['east_bias'] * 100:+.4f} cm"
    )

    print(
        f"Position step RMS = "
        f"{raw_smoothness * 100:.4f} cm"
    )

    print()
    print("=" * 70)
    print("KF FILTERED ESTIMATE")
    print("=" * 70)

    print(
        f"North RMSE       = "
        f"{filtered_metrics['north_rmse'] * 100:.4f} cm"
    )

    print(
        f"East RMSE        = "
        f"{filtered_metrics['east_rmse'] * 100:.4f} cm"
    )

    print(
        f"Horizontal RMSE  = "
        f"{filtered_metrics['horizontal_rmse'] * 100:.4f} cm"
    )

    print(
        f"Horizontal MAE   = "
        f"{filtered_metrics['horizontal_mae'] * 100:.4f} cm"
    )

    print(
        f"Horizontal max    = "
        f"{filtered_metrics['horizontal_max'] * 100:.4f} cm"
    )

    print(
        f"North bias       = "
        f"{filtered_metrics['north_bias'] * 100:+.4f} cm"
    )

    print(
        f"East bias        = "
        f"{filtered_metrics['east_bias'] * 100:+.4f} cm"
    )

    print(
        f"Position step RMS = "
        f"{filtered_smoothness * 100:.4f} cm"
    )

    # --------------------------------------------------------
    # Improvement
    # --------------------------------------------------------

    rmse_improvement = (
        (
            raw_metrics[
                "horizontal_rmse"
            ]
            -
            filtered_metrics[
                "horizontal_rmse"
            ]
        )
        /
        raw_metrics[
            "horizontal_rmse"
        ]
        *
        100.0
    )

    smoothness_improvement = (
        (
            raw_smoothness
            -
            filtered_smoothness
        )
        /
        raw_smoothness
        *
        100.0
    )

    print()
    print("=" * 70)
    print("FILTER EFFECT")
    print("=" * 70)

    print(
        f"Horizontal RMSE improvement = "
        f"{rmse_improvement:.2f}%"
    )

    print(
        f"Position step RMS reduction = "
        f"{smoothness_improvement:.2f}%"
    )

    print()
    print(
        "Final filtered state:"
    )

    print(
        f"  N    = "
        f"{kf.state[0]:+.6f} m"
    )

    print(
        f"  E    = "
        f"{kf.state[1]:+.6f} m"
    )

    print(
        f"  Ndot = "
        f"{kf.state[2]:+.6f} m/s"
    )

    print(
        f"  Edot = "
        f"{kf.state[3]:+.6f} m/s"
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    save_plots(
        measurements,
        filtered_position,
        truth,
    )

    print()
    print(
        "Plots saved to:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
