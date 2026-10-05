import csv
import math
import re
import subprocess
import threading
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import sys


# ============================================================
# Project paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

sys.path.insert(
    0,
    str(SRC_DIR),
)

from kalman_filter import LinearKalmanFilter
import static_geolocation_realtime as realtime


# ============================================================
# Configuration
# ============================================================

N_SAMPLES = 100
SAMPLE_INTERVAL = 0.10

TARGET_NAME = "moving_target"

POSE_TOPIC = (
    "/world/hexacopter_runway/dynamic_pose/info"
)

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

OUTPUT_CSV = (
    OUTPUT_DIR
    / "live_kf_validation.csv"
)


TARGET_Z = 0.01


# Same Q used in offline validation
Q = np.diag([
    1e-4,
    1e-4,
    1e-3,
    1e-3
])


# ============================================================
# Ground-truth shared state
# ============================================================

truth_lock = threading.Lock()

truth_state = {
    "target_world": None,
    "timestamp": None,
}

truth_stop_event = threading.Event()


# ============================================================
# Load measurement covariance from R0
# ============================================================

def load_measurement_covariance():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"R0 dataset not found:\n{DATA_FILE}"
        )

    estimated_north = []
    estimated_east = []

    with open(
        DATA_FILE,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "estimated_north",
            "estimated_east"
        }

        if not required_columns.issubset(
            reader.fieldnames or []
        ):

            raise RuntimeError(
                "R0.csv does not contain "
                "estimated_north / estimated_east."
            )

        for row in reader:

            estimated_north.append(
                float(
                    row["estimated_north"]
                )
            )

            estimated_east.append(
                float(
                    row["estimated_east"]
                )
            )

    estimated_north = np.asarray(
        estimated_north,
        dtype=float
    )

    estimated_east = np.asarray(
        estimated_east,
        dtype=float
    )

    R = np.diag([
        np.var(
            estimated_north,
            ddof=1
        ),
        np.var(
            estimated_east,
            ddof=1
        )
    ])

    return R


# ============================================================
# Generic pose parser
# ============================================================

def parse_pose_block(block):

    name_match = re.search(
        r'name:\s*"([^"]+)"',
        block
    )

    if name_match is None:
        return None

    name = name_match.group(1)

    position_match = re.search(
        r'position\s*\{(.*?)\}',
        block,
        re.DOTALL
    )

    if position_match is None:
        return None

    position_text = (
        position_match.group(1)
    )

    def value(
        pattern,
        source,
        default=0.0
    ):

        match = re.search(
            pattern,
            source
        )

        if match is not None:

            return float(
                match.group(1)
            )

        return default

    return {
        "name": name,

        "position": np.array([
            value(
                r'x:\s*([-+0-9.eE]+)',
                position_text
            ),

            value(
                r'y:\s*([-+0-9.eE]+)',
                position_text
            ),

            value(
                r'z:\s*([-+0-9.eE]+)',
                position_text
            )
        ])
    }


# ============================================================
# Ground-truth target worker
# ============================================================

def target_truth_worker():

    process = None

    try:

        process = subprocess.Popen(
            [
                "stdbuf",
                "-oL",
                "gz",
                "topic",
                "-e",
                "-t",
                POSE_TOPIC
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        inside_pose = False
        pose_depth = 0
        pose_lines = []

        for line in process.stdout:

            if truth_stop_event.is_set():
                break

            stripped = line.strip()

            # -----------------------------------------------
            # Start pose block
            # -----------------------------------------------

            if (
                not inside_pose
                and stripped == "pose {"
            ):

                inside_pose = True

                pose_depth = 1

                pose_lines = [
                    stripped
                ]

                continue

            # -----------------------------------------------
            # Read pose block
            # -----------------------------------------------

            if inside_pose:

                pose_lines.append(
                    stripped
                )

                pose_depth += (
                    stripped.count("{")
                    - stripped.count("}")
                )

                if pose_depth == 0:

                    inside_pose = False

                    block = "\n".join(
                        pose_lines
                    )

                    pose = parse_pose_block(
                        block
                    )

                    if (
                        pose is not None
                        and pose["name"] == TARGET_NAME
                    ):

                        with truth_lock:

                            truth_state[
                                "target_world"
                            ] = (
                                pose[
                                    "position"
                                ].copy()
                            )

                            truth_state[
                                "timestamp"
                            ] = time.monotonic()

    except Exception as exc:

        print(
            f"[Truth worker error] {exc}"
        )

    finally:

        if process is not None:

            try:

                process.terminate()

                process.wait(
                    timeout=1.0
                )

            except Exception:

                try:
                    process.kill()
                except Exception:
                    pass


# ============================================================
# Convert Gazebo world delta -> NED
# ============================================================

def gazebo_delta_to_ned(delta):

    return np.array([
        delta[1],
        delta[0],
        -delta[2]
    ])


# ============================================================
# Metrics
# ============================================================

def horizontal_rmse(
    estimate,
    truth
):

    error = (
        estimate
        - truth
    )

    horizontal = np.sqrt(
        error[:, 0] ** 2
        + error[:, 1] ** 2
    )

    return math.sqrt(
        np.mean(
            horizontal ** 2
        )
    )


def horizontal_mae(
    estimate,
    truth
):

    error = (
        estimate
        - truth
    )

    horizontal = np.sqrt(
        error[:, 0] ** 2
        + error[:, 1] ** 2
    )

    return np.mean(
        horizontal
    )


def horizontal_max(
    estimate,
    truth
):

    error = (
        estimate
        - truth
    )

    horizontal = np.sqrt(
        error[:, 0] ** 2
        + error[:, 1] ** 2
    )

    return np.max(
        horizontal
    )


def position_bias(
    estimate,
    truth
):

    error = (
        estimate
        - truth
    )

    return np.mean(
        error,
        axis=0
    )


def step_rms(
    positions
):

    if len(positions) < 2:

        return 0.0

    delta = (
        np.diff(
            positions,
            axis=0
        )
    )

    step_magnitude = np.sqrt(
        delta[:, 0] ** 2
        + delta[:, 1] ** 2
    )

    return math.sqrt(
        np.mean(
            step_magnitude ** 2
        )
    )


# ============================================================
# Plot results
# ============================================================

def save_plots(
    samples,
    raw_positions,
    filtered_positions,
    truth_positions
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # North
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        samples,
        truth_positions[:, 0],
        label="True North"
    )

    plt.plot(
        samples,
        raw_positions[:, 0],
        label="Raw Geometric Estimate"
    )

    plt.plot(
        samples,
        filtered_positions[:, 0],
        label="KF Estimate"
    )

    plt.xlabel("Sample")
    plt.ylabel("North position (m)")
    plt.title(
        "Live Static KF Validation — North Position"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "live_kf_north.png",
        dpi=150
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
        truth_positions[:, 1],
        label="True East"
    )

    plt.plot(
        samples,
        raw_positions[:, 1],
        label="Raw Geometric Estimate"
    )

    plt.plot(
        samples,
        filtered_positions[:, 1],
        label="KF Estimate"
    )

    plt.xlabel("Sample")
    plt.ylabel("East position (m)")
    plt.title(
        "Live Static KF Validation — East Position"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "live_kf_east.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Horizontal error
    # --------------------------------------------------------

    raw_error = (
        raw_positions
        - truth_positions
    )

    filtered_error = (
        filtered_positions
        - truth_positions
    )

    raw_horizontal = np.sqrt(
        raw_error[:, 0] ** 2
        + raw_error[:, 1] ** 2
    )

    filtered_horizontal = np.sqrt(
        filtered_error[:, 0] ** 2
        + filtered_error[:, 1] ** 2
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        samples,
        raw_horizontal * 100,
        label="Raw Geometric Error"
    )

    plt.plot(
        samples,
        filtered_horizontal * 100,
        label="KF Error"
    )

    plt.xlabel("Sample")
    plt.ylabel("Horizontal error (cm)")
    plt.title(
        "Live Static KF Validation — Horizontal Error"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "live_kf_horizontal_error.png",
        dpi=150
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    R = load_measurement_covariance()

    P0 = np.diag([
        R[0, 0],
        R[1, 1],
        1.0,
        1.0
    ])

    print()
    print("=" * 70)
    print(
        "PHASE A — LIVE STATIC KALMAN FILTER VALIDATION"
    )
    print("=" * 70)

    print()
    print("R0 measurement covariance:")
    print(R)

    print()
    print("Q:")
    print(Q)

    print()
    print(
        f"Target Z = {TARGET_Z:.3f} m"
    )

    print(
        f"Samples   = {N_SAMPLES}"
    )

    print(
        f"Interval  = {SAMPLE_INTERVAL:.2f} s"
    )

    print()
    print(
        "Starting realtime estimator..."
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        realtime.CAMERA_PIPELINE,
        cv2.CAP_GSTREAMER
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open camera stream."
        )

    # --------------------------------------------------------
    # Start workers
    # --------------------------------------------------------

    realtime.stop_event.clear()
    truth_stop_event.clear()

    gazebo_thread = threading.Thread(
        target=realtime.gazebo_state_worker,
        daemon=True
    )

    gps_thread = threading.Thread(
        target=realtime.gps_worker,
        daemon=True
    )

    truth_thread = threading.Thread(
        target=target_truth_worker,
        daemon=True
    )

    gazebo_thread.start()
    gps_thread.start()
    truth_thread.start()

    # --------------------------------------------------------
    # Wait for state
    # --------------------------------------------------------

    print()
    print(
        "Waiting for realtime state..."
    )

    while True:

        with realtime.state_lock:

            state_ready = (
                realtime.state[
                    "uav_world"
                ] is not None
                and
                realtime.state[
                    "camera_world"
                ] is not None
                and
                realtime.state[
                    "R_world_pitch"
                ] is not None
                and
                realtime.state[
                    "gps_lat"
                ] is not None
                and
                realtime.state[
                    "gps_lon"
                ] is not None
            )

        with truth_lock:

            truth_ready = (
                truth_state[
                    "target_world"
                ] is not None
            )

        if (
            state_ready
            and truth_ready
        ):

            break

        time.sleep(
            0.05
        )

    print(
        "Realtime state ready."
    )

    print()
    print(
        "Collecting live samples..."
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    rows = []

    raw_positions = []
    filtered_positions = []
    truth_positions = []

    samples = []

    kf = None

    valid_samples = 0

    last_sample_time = 0.0
    previous_sample_time = None

    try:

        while valid_samples < N_SAMPLES:

            ret, frame = cap.read()

            if not ret:
                continue

            now = time.monotonic()

            if (
                now - last_sample_time
                < SAMPLE_INTERVAL
            ):
                continue

            detection = (
                realtime.detect_target(
                    frame
                )
            )

            if detection is None:
                continue

            # ------------------------------------------------
            # Atomic estimator-state snapshot
            # ------------------------------------------------

            with realtime.state_lock:

                uav_world = (
                    realtime.state[
                        "uav_world"
                    ].copy()
                )

                camera_world = (
                    realtime.state[
                        "camera_world"
                    ].copy()
                )

                R_world_pitch = (
                    realtime.state[
                        "R_world_pitch"
                    ].copy()
                )

                gps_lat = (
                    realtime.state[
                        "gps_lat"
                    ]
                )

                gps_lon = (
                    realtime.state[
                        "gps_lon"
                    ]
                )

            # ------------------------------------------------
            # Atomic truth snapshot
            # ------------------------------------------------

            with truth_lock:

                target_world = (
                    truth_state[
                        "target_world"
                    ].copy()
                )

            # ------------------------------------------------
            # Geometric estimator
            # ------------------------------------------------

            try:

                estimate = (
                    realtime.estimate_target(
                        detection["u"],
                        detection["v"],
                        uav_world,
                        camera_world,
                        R_world_pitch,
                        gps_lat,
                        gps_lon
                    )
                )

            except Exception:

                continue

            raw_ned = np.asarray(
                estimate["target_ned"],
                dtype=float
            )

            # ------------------------------------------------
            # Ground truth in relative NED
            # ------------------------------------------------

            target_delta = (
                target_world
                - uav_world
            )

            true_ned = (
                gazebo_delta_to_ned(
                    target_delta
                )
            )

            # ------------------------------------------------
            # KF measurement
            # ------------------------------------------------

            measurement = np.array([
                raw_ned[0],
                raw_ned[1]
            ])

            # ------------------------------------------------
            # KF initialization
            # ------------------------------------------------

            if kf is None:

                initial_state = np.array([
                    measurement[0],
                    measurement[1],
                    0.0,
                    0.0
                ])

                kf = LinearKalmanFilter(
                dt=SAMPLE_INTERVAL,
                measurement_covariance=R,
                process_covariance=Q,
                initial_state=initial_state,
                initial_covariance=P0
            )

                filtered_state = (
                    kf.update(
                        measurement
                    )
                )

            else:

                current_dt = (
                    now
                    - previous_sample_time
                )

                if current_dt <= 0:

                    current_dt = (
                        SAMPLE_INTERVAL
                    )

                filtered_state = (
                    kf.step(
                        measurement,
                        dt=current_dt
                    )
                )

            filtered_ned = (
                np.asarray(
                    filtered_state[
                        :2,
                        0
                    ],
                    dtype=float
                )
            )

            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            valid_samples += 1

            previous_sample_time = now
            last_sample_time = now

            raw_positions.append(
                measurement.copy()
            )

            filtered_positions.append(
                filtered_ned.copy()
            )

            truth_positions.append(
                true_ned[:2].copy()
            )

            samples.append(
                valid_samples
            )

            raw_horizontal_error = math.hypot(
                measurement[0]
                - true_ned[0],

                measurement[1]
                - true_ned[1]
            )

            filtered_horizontal_error = math.hypot(
                filtered_ned[0]
                - true_ned[0],

                filtered_ned[1]
                - true_ned[1]
            )

            rows.append({
                "sample":
                    valid_samples,

                "u":
                    detection["u"],

                "v":
                    detection["v"],

                "true_north":
                    true_ned[0],

                "true_east":
                    true_ned[1],

                "raw_north":
                    measurement[0],

                "raw_east":
                    measurement[1],

                "filtered_north":
                    filtered_ned[0],

                "filtered_east":
                    filtered_ned[1],

                "raw_horizontal_error":
                    raw_horizontal_error,

                "filtered_horizontal_error":
                    filtered_horizontal_error
            })

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            display = frame.copy()

            x, y, w, h = (
                detection["bbox"]
            )

            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.drawMarker(
                display,
                (
                    int(round(
                        detection["u"]
                    )),
                    int(round(
                        detection["v"]
                    ))
                ),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                12,
                2
            )

            text_lines = [
                (
                    f"Sample: "
                    f"{valid_samples}/{N_SAMPLES}"
                ),

                (
                    f"Raw N/E: "
                    f"{measurement[0]:+.3f}, "
                    f"{measurement[1]:+.3f} m"
                ),

                (
                    f"KF  N/E: "
                    f"{filtered_ned[0]:+.3f}, "
                    f"{filtered_ned[1]:+.3f} m"
                ),

                (
                    f"True N/E: "
                    f"{true_ned[0]:+.3f}, "
                    f"{true_ned[1]:+.3f} m"
                ),

                (
                    f"Raw err: "
                    f"{raw_horizontal_error * 100:.2f} cm"
                ),

                (
                    f"KF err: "
                    f"{filtered_horizontal_error * 100:.2f} cm"
                )
            ]

            y_text = 24

            for text_line in text_lines:

                cv2.putText(
                    display,
                    text_line,
                    (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

                y_text += 21

            cv2.imshow(
                "Live Static KF Validation",
                display
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):

                print()
                print(
                    "Stopped by user."
                )

                break

            print(
                f"[{valid_samples:03d}/{N_SAMPLES}] "
                f"raw=({measurement[0]:+.3f},"
                f"{measurement[1]:+.3f}) "
                f"KF=({filtered_ned[0]:+.3f},"
                f"{filtered_ned[1]:+.3f}) "
                f"raw_err="
                f"{raw_horizontal_error * 100:.2f} cm "
                f"KF_err="
                f"{filtered_horizontal_error * 100:.2f} cm"
            )

    finally:

        realtime.stop_event.set()
        truth_stop_event.set()

        cap.release()

        cv2.destroyAllWindows()

    # --------------------------------------------------------
    # Convert arrays
    # --------------------------------------------------------

    raw_positions = np.asarray(
        raw_positions
    )

    filtered_positions = np.asarray(
        filtered_positions
    )

    truth_positions = np.asarray(
        truth_positions
    )

    samples = np.asarray(
        samples
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    raw_rmse = horizontal_rmse(
        raw_positions,
        truth_positions
    )

    filtered_rmse = horizontal_rmse(
        filtered_positions,
        truth_positions
    )

    raw_mae = horizontal_mae(
        raw_positions,
        truth_positions
    )

    filtered_mae = horizontal_mae(
        filtered_positions,
        truth_positions
    )

    raw_max = horizontal_max(
        raw_positions,
        truth_positions
    )

    filtered_max = horizontal_max(
        filtered_positions,
        truth_positions
    )

    raw_bias = position_bias(
        raw_positions,
        truth_positions
    )

    filtered_bias = position_bias(
        filtered_positions,
        truth_positions
    )

    raw_step = step_rms(
        raw_positions
    )

    filtered_step = step_rms(
        filtered_positions
    )

    rmse_improvement = (
        100.0
        * (
            raw_rmse
            - filtered_rmse
        )
        / raw_rmse
    )

    smoothness_reduction = (
        100.0
        * (
            raw_step
            - filtered_step
        )
        / raw_step
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "LIVE STATIC KF VALIDATION RESULTS"
    )
    print("=" * 70)

    print()
    print(
        f"Valid samples = {len(rows)}"
    )

    print()
    print("RAW GEOMETRIC ESTIMATE")

    print(
        f"  Horizontal RMSE = "
        f"{raw_rmse * 100:.4f} cm"
    )

    print(
        f"  Horizontal MAE  = "
        f"{raw_mae * 100:.4f} cm"
    )

    print(
        f"  Horizontal max  = "
        f"{raw_max * 100:.4f} cm"
    )

    print(
        f"  North bias      = "
        f"{raw_bias[0] * 100:+.4f} cm"
    )

    print(
        f"  East bias       = "
        f"{raw_bias[1] * 100:+.4f} cm"
    )

    print(
        f"  Position step RMS = "
        f"{raw_step * 100:.4f} cm"
    )

    print()
    print("KF FILTERED ESTIMATE")

    print(
        f"  Horizontal RMSE = "
        f"{filtered_rmse * 100:.4f} cm"
    )

    print(
        f"  Horizontal MAE  = "
        f"{filtered_mae * 100:.4f} cm"
    )

    print(
        f"  Horizontal max  = "
        f"{filtered_max * 100:.4f} cm"
    )

    print(
        f"  North bias      = "
        f"{filtered_bias[0] * 100:+.4f} cm"
    )

    print(
        f"  East bias       = "
        f"{filtered_bias[1] * 100:+.4f} cm"
    )

    print(
        f"  Position step RMS = "
        f"{filtered_step * 100:.4f} cm"
    )

    print()
    print("FILTER EFFECT")

    print(
        f"  Horizontal RMSE improvement = "
        f"{rmse_improvement:.2f}%"
    )

    print(
        f"  Position step RMS reduction = "
        f"{smoothness_reduction:.2f}%"
    )

    print()
    print(
        f"CSV saved to:\n  {OUTPUT_CSV}"
    )

    save_plots(
        samples,
        raw_positions,
        filtered_positions,
        truth_positions
    )

    print()
    print(
        f"Plots saved to:\n  {OUTPUT_DIR}"
    )

    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
