# Calibration quality

CameraRig separates artifact validity from numerical calibration acceptance. A valid
JSON file can still describe failed evidence; a final fixed CameraBundle is emitted only
when every required check passes.

The default fixed-camera gates are:

| Check | Limit |
| --- | ---: |
| Corners per frame | at least 12 |
| Accepted frames | at least 50 |
| Accepted ratio | at least 0.90 |
| Per-frame reprojection RMSE | at most 0.50 px |
| Per-frame reprojection p95 | at most 1.00 px |
| Translation repeatability p95 | at most 3.0 mm |
| Rotation repeatability p95 | at most 0.30 deg |
| Even/odd translation delta | at most 2.0 mm |
| Even/odd rotation delta | at most 0.20 deg |
| Native-depth median absolute error | at most 20 mm |
| Native-depth p95 absolute error | at most 40 mm |

Every accepted pose must also pass finite-SE(3), cheirality, and printed-face orientation
checks. Global reprojection statistics include RMSE, median, p95, maximum, per-frame RMSE,
and per-corner-ID distributions. Thresholds and the pose-outlier policy are persisted in
the calibration artifact.

Native depth may be `SKIPPED_WITH_WARNING` only when the configured depth camera model
cannot be projected safely. It is never fabricated as a pass. A supported projection
model with insufficient or inaccurate depth samples fails its diagnostic.

Diagnostic overlays show detected points and IDs, reprojected points, residual vectors,
the projected outer boundary, and the canonical target axes. The best, median-quality,
and worst accepted frames are retained so numerical and geometric evidence can be
reviewed together.

Thresholds are not automatically relaxed when a result fails. Investigate detection
jitter, planar ambiguity, camera or board movement, distortion semantics, target scale,
or bad frames first; any justified threshold change remains explicit and persisted.
