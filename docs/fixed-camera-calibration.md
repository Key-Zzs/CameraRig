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
