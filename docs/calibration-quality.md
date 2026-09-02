# Calibration quality

CameraRig separates artifact validity from numerical calibration acceptance. A valid
JSON file can still describe failed evidence; a final fixed CameraBundle is emitted only
when every required check passes.

The legacy fixed-camera precision gates (unchanged for `legacy_strict` and
`pose_validated`) are:

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

The current `uncertainty_validated_v1` thresholds are:

| Check | Limit |
| --- | ---: |
| Pixel-noise floor | 0.25 px |
| Gross per-frame reprojection RMSE | at most 1.5 px |
| Gross per-frame reprojection p95 | at most 2.0 px |
| Gross final reprojection RMSE | at most 1.5 px |
| Gross final reprojection p95 | at most 2.0 px |
| Per-frame worst-axis translation std | at most 5.0 mm |
| Per-frame worst-axis rotation std | at most 2.0 deg |
| Final-pose worst-axis translation std | at most 2.0 mm |
| Final-pose worst-axis rotation std | at most 0.5 deg |
| Scaled Jacobian condition number | at most 100 |
| Pose solve ratio | at least 0.95 |
| Observable-frame ratio | at least 0.90 |
| Ambiguous-frame ratio | at most 0.05 |
| Competitive alternative delta chi-square | below 9.0 is competitive |
| Material alternative separation | at least 5 mm or 5 deg |

The observability, uncertainty, and ambiguity values were frozen after the metric-only
implementation, not copied into that implementation in advance. The gross-reprojection rows
remain release candidates until the synthetic evidence is reviewed together with fresh A/B/C
camera evidence. The deterministic observability release sweep contains 2,304 fully visible
configurations over 40 mm and 100 mm square lengths, 0.8--4.5 m range, 0--80 degree tilt,
center/edge/corner placement, three corner distributions, and 0.1--1.0 px noise. The 100 mm
case represents the 5-by-7, 500-by-700 mm deployment board. The sweep brackets the three
numerical limits directly: translation has cases at 4.995/5.004 mm, rotation at
1.990/2.002 deg, and scaled condition at 99.83/100.16. It contains 1,034 passes, 1,270
fails, and 583 statistically competitive, materially distinct planar alternatives.

The independent gross-reprojection release candidate is evaluated with a 45-case sweep:
nine injected vector RMSE levels (0.1, 0.25, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0, and 3.0 px)
across Gaussian, Brown-Conrady radial and tangential, projected 3-D board-warp, and local corner
cluster corruption fields. Each row records post-fit residuals, uncertainty, observability,
ground-truth pose error, robust tail summaries, and low-order board-coordinate polynomial
structure. It also
evaluates a 7-by-6 grid of candidate RMSE/p95 threshold pairs rather than reporting only the
implemented candidate. The 1.5/2.0 candidate is not fitted to one camera maximum; it must not be
called frozen until real A/B/C residual fields and margins have been reviewed. Highly absorbable
structured fields must still pass observability, repeatability, split-half, and native-depth
evidence; the vector-field report remains diagnostic-only in this version.

The 0.25 px floor is deliberately conservative relative to the seven accepted/replayed
real captures (temporal-jitter p95 0.026--0.033 px) and is independently checked by the
0.25 px Monte Carlo cases. Those real captures have frame translation p95 0.480--1.190 mm,
rotation p95 0.135--0.377 deg, and condition p95 23.1--47.8. They therefore support the
positive side of the frozen envelope without defining it alone. A high-coverage synthetic
negative at 6.39% coverage and 1 px noise fails at 6.105 mm translation uncertainty.

For the final shared pose, 100-trial, 60-frame Monte Carlo cases predict/measure
0.381/0.415 mm and 0.0139/0.0142 deg for the good-far case, and 0.192/0.186 mm and
0.0138/0.0154 deg for the moderate-oblique case. An extreme weak case predicts/measures
13.63/12.69 mm and fails the 2 mm final limit. The 0.5 deg final rotation limit remains a
conservative cap while reprojection, repeatability, split-half stability, and ambiguity
remain independent hard gates.

The capture ratios express bounded 60-frame robustness rather than averaging away weak
geometry: at least 57 frames must solve, at least 54 must be individually observable, and
ambiguous frames may be at most `floor(0.05 * solved_count)` (three only when all 60 solve);
boundary tests at 60 solved reject 56 solves, 53 observable frames, and 4 ambiguous frames.
The final shared pose must also pass separately. Delta chi-square 9 is the squared 3-sigma
component-noise likelihood separation. The 5 mm or 5 deg material-separation floor prevents
numerically different but operationally equivalent IPPE candidates from being called
ambiguous; competitive candidates beyond either workspace-scale separation remain a hard fail.

Monte Carlo keeps local covariance and multimodal ambiguity evidence separate without hiding
either. Every representative case participates in the agreement gate. The small-target/far case
has an all-solved rotation mixture std of 6.34 deg because one of 200 trials jumps by about
89 deg; the independent ambiguity diagnostic flags that jump. After conditioning only the
covariance comparison on nonambiguous IPPE modes, empirical/predicted rotation is 0.728/0.688 deg
(ratio 1.06). All 26 ambiguity-flagged trials and the unconditioned mixture remain in the report,
and the benchmark fails if any rotation jump of at least 5 deg is not ambiguity-flagged.

Coverage and image span are not hard gates in this policy. Coverage still guides target size and
operator placement, but coverage is not pose accuracy. Low coverage can pass only when the actual
correspondences satisfy every uncertainty, observability, ambiguity, temporal, reprojection, and
physical gate; high coverage can fail when they do not.

Every accepted pose must also pass finite-SE(3), cheirality, and printed-face orientation
checks. Global reprojection statistics include RMSE, median, p95, maximum, per-frame RMSE,
and per-corner-ID distributions. Thresholds and the pose-outlier policy are persisted in
the calibration artifact.

Native depth may be `SKIPPED_WITH_WARNING` only when the configured depth camera model
cannot be projected safely and the target is generated. An existing physical target requires a
real `PASS`; unsupported projection, insufficient samples, or inaccurate depth fails its
provision quality. A skipped diagnostic is never fabricated as a pass.

Diagnostic overlays show detected points and IDs, reprojected points, residual vectors,
the projected outer boundary, and the canonical target axes. The best, median-quality,
and worst accepted frames are retained so numerical and geometric evidence can be
reviewed together.

Thresholds are not automatically relaxed when a result fails. Investigate detection
jitter, planar ambiguity, camera or board movement, distortion semantics, target scale,
or bad frames first; any justified threshold change remains explicit and persisted.
