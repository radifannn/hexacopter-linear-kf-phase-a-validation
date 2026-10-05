# Hexacopter Linear Kalman Filter — Phase A Validation

Phase A validation of a linear Kalman Filter (KF) applied to static ground-target geolocation on a hexacopter.

This repository is intended as a **code and methodology review package** for validating the Kalman Filter stage before proceeding to dynamic-target validation, EKF development, or MPC development.

---

## 1. Project Context

The upstream geolocation system estimates the relative position of a ground target from:

- Camera image
- Target detection
- Camera intrinsics
- Camera/gimbal/UAV coordinate transformations
- Ray-ground intersection

The upstream geometry produces the target position in the local NED frame:

\[
z =
\begin{bmatrix}
N \\
E
\end{bmatrix}
\]

where:

- \(N\) = target North position relative to UAV
- \(E\) = target East position relative to UAV

The Kalman Filter is applied **after** this geometric estimation.

Therefore, the nonlinear camera/geometric processing is outside the Kalman Filter.

---

## 2. Phase A Objective

The objective of Phase A is to determine whether a simple linear Kalman Filter:

1. is implemented correctly,
2. is mathematically consistent with the stated model,
3. behaves stably,
4. reduces measurement noise / jitter,
5. does not introduce oscillation or divergence,
6. and provides a valid foundation for later dynamic-target validation.

Phase A uses a **static ground target**.

---

## 3. Kalman Filter Model

The filter state is:

\[
x =
\begin{bmatrix}
N \\
E \\
\dot N \\
\dot E
\end{bmatrix}
\]

with constant-velocity state transition:

\[
x_k = F x_{k-1} + w_k
\]

where

\[
F =
\begin{bmatrix}
1 & 0 & \Delta t & 0 \\
0 & 1 & 0 & \Delta t \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
\]

The measurement model is:

\[
z_k = Hx_k + v_k
\]

with:

\[
H =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0
\end{bmatrix}
\]

The measurement is therefore:

\[
z_k =
\begin{bmatrix}
N_{\text{estimated}} \\
E_{\text{estimated}}
\end{bmatrix}
\]

No velocity measurement is directly supplied to the filter.

---

## 4. Measurement Noise Covariance

The measurement covariance \(R\) is derived from the static validation dataset:

`data/static_validation_R0.csv`

The R0 dataset contains the raw geometric target-position estimates and corresponding ground-truth positions.

The exact method used to derive \(R\) is an important review point because the covariance formulation must be consistent with the intended interpretation of measurement noise.

No arbitrary \(R\) value is intended to be chosen solely to improve the final numerical result.

---

## 5. Process Noise Covariance

Phase A uses a small process-noise covariance because the target is static.

The initial implementation uses a diagonal covariance of the form:

\[
Q =
\operatorname{diag}
\left(
10^{-4},
10^{-4},
10^{-3},
10^{-3}
\right)
\]

The value is intended as a small initial process-noise assumption rather than a parameter tuned specifically to minimize a single validation result.

---

## 6. Validation Structure

### Phase A.1 — Offline Validation

The offline validation script:

`validation/validate_kf_offline.py`

loads the static R0 dataset and applies the Kalman Filter sample-by-sample.

The raw geometric estimate is compared against the filtered estimate using the static ground-truth position.

Primary metrics:

- North RMSE
- East RMSE
- Horizontal RMSE
- MAE
- Maximum horizontal error
- North/East bias
- Step-to-step RMS error

The purpose is to evaluate both:

- positional accuracy
- temporal smoothness

A reduction in error is useful, but smoothing alone must not be interpreted as improved absolute accuracy.

### Phase A.2 — Live Validation

The live validation script:

`validation/static_geolocation_kf_live.py`

runs the same Kalman Filter on the real-time static geolocation stream.

The filter must use the same underlying measurement model as the offline implementation.

---

## 7. Reference Results From Development

The following values were obtained during development and are recorded here as reference observations.

### Offline static validation

Raw geometric estimate:

- Horizontal RMSE: approximately 2.037 cm
- Horizontal MAE: approximately 1.974 cm
- Maximum horizontal error: approximately 4.623 cm

Kalman Filter output:

- Horizontal RMSE: approximately 2.001 cm
- Horizontal MAE: approximately 1.958 cm
- Maximum horizontal error: approximately 3.781 cm

Observed behavior:

- Horizontal RMSE improvement was small.
- Step-to-step RMS error decreased by approximately 26%.
- The dominant positional bias remained.

This indicates that the main observed effect of the KF in Phase A is **smoothing**, while absolute geometric bias is largely inherited from the upstream estimator.

### Live static validation

Reference development result:

- Raw horizontal RMSE: approximately 3.997 cm
- KF horizontal RMSE: approximately 3.991 cm
- Step RMS decreased from approximately 1.006 cm to 0.855 cm

Again, the main observed effect was temporal smoothing rather than large absolute-position improvement.

These numbers are included only as development references and should be independently verified from the code and dataset.

---

## 8. Scope

This repository intentionally contains only the Phase A linear KF stage.

Included:

- Linear Kalman Filter
- Static-target offline validation
- Static-target live validation
- Static R0 dataset
- Upstream geolocation code required to execute the validation

Not included:

- EKF
- Jacobian derivation
- MPC
- Gimbal stabilization
- Dynamic target control
- Path planning
- Obstacle avoidance
- Target GPS as an estimator measurement

The target GPS data is used as validation ground truth only.

---

## 9. Files

```text
hexacopter-linear-kf-phase-a-validation/
├── README.md
├── requirements.txt
├── src/
│   ├── kalman_filter.py
│   ├── geolocation_geometry.py
│   └── static_geolocation_realtime.py
├── validation/
│   ├── validate_kf_offline.py
│   └── static_geolocation_kf_live.py
└── data/
    └── static_validation_R0.csv
