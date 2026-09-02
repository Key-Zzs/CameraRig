# Pose-observability gate design

This record defines the implementation boundary for `uncertainty_validated`. It is intentionally
separate from raw-stream validation and from PointCloudBuilder multi-camera calibration.

## Existing gate map

1. Detector level (`targets/charuco/quality.py`): corner count/fraction, image span, marker pixel
   perimeter, and coverage. `pose_validated` makes 5% coverage advisory but retains 1% coverage,
   span, and marker-size hard gates.
2. Sixty-frame target acceptance (`targets/validation.py`, `targets/preflight.py`): frame count,
   detector success ratio, median corners/fraction/coverage, and temporal pixel jitter. The existing
   policies retain their current coverage semantics.
3. Per-frame PnP (`calibration/pose/planar_pnp.py`, `calibration/fixed/calibrator.py`): IPPE
   physical validity, minimum reprojection candidate, LM refinement, corner count, RMSE, and p95.
4. Fixed-calibration aggregate (`calibration/fixed/quality.py`): accepted-frame count/ratio, global
   reprojection, pose repeatability, split-half stability, and native-depth sanity.
5. Provision final validation (`provision/workflow.py`, `provision/validation.py`): an independent
   raw-stream report must pass before target detection; target acceptance must pass before fixed
   calibration; the fixed calibration, bundle, exact file set, and checksums must all pass.

## New-policy routing

`uncertainty_validated` changes only target/pose calibration gates. Detector coverage and image span
remain measurements and operator warnings. Detection integrity (at least 12 ChArUco corners, the
existing corner fraction, finite/non-collinear correspondences, and marker pixel scale) remains a
hard prerequisite. Active-stream factory intrinsics are required; there is no fallback to
`pose_validated`.

Every successfully solved frame records a scaled six-parameter projection Jacobian, full-rank
decision, conditional local covariance, worst-axis translation/rotation standard deviation, and an
IPPE alternative-candidate ambiguity diagnostic. Capture acceptance uses robust solve/observable
ratios and uncertainty distributions rather than requiring every frame to pass. Fixed calibration
requires both a minimum per-frame observable ratio and observability of the final shared pose, in
addition to the existing reprojection, repeatability, split-half, and native-depth gates.

OpenCV's analytic Jacobian is with respect to additive Rodrigues-vector coordinates. After stable
SVD covariance recovery, CameraRig maps the full covariance through the SO(3) left Jacobian and
reports rotation in the left-invariant camera-frame tangent space. Translation remains an additive
camera-frame vector. The scaled condition number remains explicitly defined on
`q=[opencv_rvec_rad,tvec_m/target_scale_m]`; it is not mislabeled as a unitless raw pose condition.
An ambiguity decision requires two physically valid candidates. One physically valid candidate has
no competitor and is acceptable; zero valid candidates fails before observability is evaluated.

The covariance is conditional on fixed intrinsics/distortion, fixed target geometry, correct
correspondence association, and a local Gaussian pixel perturbation. It is not absolute system
calibration uncertainty and does not include target warp/scale, intrinsic uncertainty, factory
inter-stream extrinsic uncertainty, or mount deformation.

## Independence from raw-stream validation

`StreamValidationAccumulator` remains unchanged and runs over all 300 provision frames before any
target or pose gate. `uncertainty_validated` neither bypasses it nor changes its result. A provision
failure caused by missing streams, timeouts, frame discontinuities, FPS/timestamp/synchronization,
shape/dtype, empty depth, stream variance, or identical stereo arrays is classified as raw-stream
failure and is `NOT_APPLICABLE_TO_POSE_OBSERVABILITY`.

PointCloudBuilder N-camera bundle adjustment, holdout, physical acceptance, fusion, FFS, TSDF, and
production extrinsics remain outside this change.
