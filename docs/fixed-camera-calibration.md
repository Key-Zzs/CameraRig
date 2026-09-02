# Fixed-camera calibration

CameraRig estimates one fixed camera pose from repeated observations of one fixed planar
target. The workspace frame is explicitly identified with the persisted target frame:

```text
T_workspace_from_target = Identity
```

The target origin is the board's outer bottom-left. Target `+X` points to board right,
`+Y` points up, and `+Z` leaves the printed face. OpenCV PnP returns
`T_detection_camera_from_target`; CameraRig never relabels that result as its inverse.

The target detector, planar pose estimator, and fixed-pose calibrator are separate
contracts. The pose estimator consumes only `TargetObservation` and `CameraIntrinsics`,
so another planar target plugin can reuse it without importing ChArUco code.

For every frame, the solver evaluates all IPPE candidates, records their separation and
reprojection residuals, rejects negative-depth or back-facing candidates, chooses the
valid minimum-RMSE candidate, and performs LM refinement. It does not delete individual
corners to improve the result.

Accepted frame poses are aggregated by a robust SE(3) medoid. Pose-level outliers are
rejected against persisted translation and geodesic-rotation thresholds, then all inlier
correspondences jointly refine one shared pose. Even and odd frames are independently
refined for split-half stability.

With `uncertainty_validated`, every LM-refined frame also evaluates the analytic OpenCV projection
Jacobian for `p = [rvec_rad, tvec_m]`. Translation columns are scaled by the RMS target radius
before rank, singular values, and condition number are computed, so radians are not compared
directly with metres. Component pixel residuals use `dof = max(2N - 6, 1)` and a 0.25 px release
noise floor. Full-rank covariance is computed with an SVD pseudoinverse; rank-deficient geometry
has no covariance and fails closed. The gate uses worst-axis translation and rotation standard
deviations. After joint refinement, the complete inlier correspondence set is evaluated again as
the final shared pose; frame count cannot hide a poor observable-frame ratio.

The legacy 0.5 px RMSE and 1.0 px p95 precision limits remain hard per-frame and final gates for
`legacy_strict` and `pose_validated`. Under `uncertainty_validated`, residual SSE already sets
`sigma_px = max(0.25, sqrt(SSE / (2N - 6)))` and therefore propagates into pose covariance. The
legacy precision limit is not applied a second time. A separate gross model-consistency gate uses
1.5 px RMSE and 2.0 px p95 limits at both frame and final scope. These limits are intentionally
separate persisted fields in `uncertainty_validated_v1`, not rewritten fixed-config values.

Each solved frame also persists observed/projected UV, du/dv, residual norm, normalized-radius
bins, Pearson/Spearman trends, and quadrant vector statistics. These vector-field measurements help
distinguish random subpixel localization noise from radial, tangential, warp-like, or localized
model mismatch. They are diagnostic-only and do not silently introduce a new hard gate.

This covariance is conditional local pose uncertainty with fixed K/D, fixed target geometry,
correct associations, and a local Gaussian pixel approximation. It is not absolute calibration
uncertainty and excludes target warp/print scale, factory-intrinsic uncertainty, factory
Color-to-IR extrinsic uncertainty, and mount deformation. Temporal repeatability, split-half,
native depth, and downstream physical acceptance therefore remain independent hard evidence.

The final transform chain uses frame-aware operations:

```text
T_workspace_from_detection =
    T_workspace_from_target @ inverse(T_detection_from_target)

T_workspace_from_reference =
    T_workspace_from_detection @ T_detection_from_reference
```

The reference stream is normally `ir_left`. The resulting `FixedMountCalibration`
therefore maps reference-camera points into `workspace`.

The physical print measurement is provenance, not an implicit geometry correction. The
solver continues to use the nominal persisted 30 mm square and 22 mm marker geometry;
the observed 0.997 horizontal and vertical print scales are recorded as systematic-scale
information.

Native depth is an independent gross-error diagnostic. Predicted target-plane depths are
compared with local medians from raw depth pixels, but depth never participates in PnP or
pose optimization.

Offline calibration and validation use:

```bash
camera-rig calibration fixed solve \
  --config .local/configs/fixed_calibration.yaml \
  --capture .local/artifacts/capture \
  --target .local/targets/charuco_a4_v1/target_spec.json \
  --detection-report .local/reports/target-detection.json \
  --output .local/artifacts/fixed_calibration.json \
  --overlays .local/overlays/fixed

camera-rig calibration fixed validate \
  --input .local/artifacts/fixed_calibration.json
```
