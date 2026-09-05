# Fixed-camera bootstrap qualification

The authoritative A4 production bootstrap requires a genuine target-metrology receipt.
Repeated horizontal and vertical long-baseline measurements are compared with the
resolved target geometry. The scale tolerance is preregistered from allowed downstream
translation error divided by maximum working distance, after subtracting measurement
uncertainty. The receipt records both scale axes, anisotropy, repeatability, instrument,
timestamp, provenance, and PASS/FAIL.

Acceptance is frozen before readings are available. First run
`camera-rig target metrology-policy-create` with the downstream translation-error budget and
maximum intended working distance, then preserve that policy file. Only afterwards run
`camera-rig target metrology-create --acceptance-policy <frozen-policy.json>` with the repeated
horizontal and vertical readings. The measurement command has no threshold override, so a failed
receipt cannot be made to pass by rerunning with looser acceptance knobs.

The receipt embeds the complete validated policy and its SHA-256, and its timestamp must be later
than the policy timestamp. A v2 bootstrap provision carries both the standalone policy and the
receipt and requires their content and hashes to agree.

Native RealSense depth is an independent metric source. Bootstrap evaluation uses the
factory depth scale, depth intrinsics, and internal stream transforms to compare the
PnP-predicted target plane with measured depth. It reports support, signed/absolute
residuals, robust plane offset and normal, distance-scale ratio, board-local support,
and per-frame/aggregate statistics. Missing support, unsupported projection, non-finite
depth, or plane/scale failure is a hard FAIL.

The published v2 provision contains the target-scale policy, metrology receipt, metric-depth
receipt, and a bootstrap qualification receipt. Its CameraBundle carries
`qualification_state=BOOTSTRAP_QUALIFIED`, `qualification_scope=bootstrap_only`, and
`production_authoritative=false`. It is an initializer for downstream multi-pose work,
not production multi-camera calibration.

Factory-vs-diagnostic-refit intrinsic health is intentionally evaluated later from
multiple poses. The refit never mutates factory K/D. A factory model is `SUSPECT` only
when bounded parameter changes produce absolute, relative, and paired-consistent
improvement on untouched holdout poses; weak evidence is `INSUFFICIENT_EVIDENCE`.

Structured residual remains `diagnostic_only`. A universal planar hard gate is
`NOT_SUPPORTED_DUE_TO_PLANAR_IDENTIFIABILITY_LIMIT` and cannot affect qualification.
