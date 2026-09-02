# Fixed-camera provisioning

Fixed provisioning combines existing CameraRig contracts behind one command. One strict
YAML identifies the physical device, active profiles, capture policy, pinned target,
workspace transform, reference stream, and numerical quality thresholds.

```bash
camera-rig provision fixed \
  --config .local/configs/fixed_provision.yaml \
  --output .local/artifacts/fixed_camera
```

Use `--dry-run` to validate configuration, the complete target artifact and its pinned
SHA-256, workspace semantics, output policy, and optional dependencies. A dry run does
not open the camera and does not create a final artifact.

Before publication, run the live viability preflight:

```bash
camera-rig provision preflight \
  --config .local/configs/fixed_provision.yaml \
  --report .local/reports/fixed-provision-preflight.json \
  --overlays .local/overlays/fixed-provision-preflight
```

Unlike `--dry-run`, this opens the camera. It consumes the complete provision YAML and calls the
same `run_fixed_provision_workflow` acquisition/evaluation core used by `provision fixed`. Given
the same immutable replay/synthetic input, both paths produce identical accepted frame indices,
per-frame reasons, policy thresholds, and final decision. Live captures are separate samples, so
their frame counts need not be bit-identical. The preflight report records raw-stream, target,
fixed-frame, reprojection, observability, final quality, and complete per-frame evidence. It writes
diagnostics only and never writes a CameraBundle or provision artifact.

The live command opens the D435i once. Every acquired frame contributes to raw-stream
validation; a deterministic evenly spaced subset contributes to the replay snapshot,
target detection, and fixed-pose solve. The selected source indices are persisted.
Factory intrinsics for pose observability are read from that same active session; the camera is not
closed and reopened. Replay uses the hash-bound factory artifact. Missing or unsupported
intrinsics fail as `POSE_OBSERVABILITY_INTRINSICS_UNAVAILABLE`; there is no fallback policy.

Raw-stream validation remains a separate prerequisite. `uncertainty_validated` does not modify or
bypass `StreamValidationAccumulator`. Missing streams, timeouts, discontinuities, FPS/timestamp or
sync failures, shape/dtype changes, empty depth, zero stream variance, or identical stereo arrays
remain raw-stream failures and are not pose-observability failures.

The output is built in a sibling temporary directory. CameraBundle reload, manifest
reload, checksums, cross-file identities, safe relative paths, and the exact file set
must all validate before the directory is published. Existing validated output is
replaced by default after the new artifact fully validates and is atomically swapped
into place.

If a new fixed-provision attempt fails while an older validated output exists, that output remains
unchanged by design. The CLI prints `NEW_PROVISION_ATTEMPT=FAIL`,
`EXISTING_OUTPUT_UNCHANGED=true`, and
`DO_NOT_TREAT_EXISTING_VALIDATE_AS_THIS_ATTEMPT`. Guard validation with shell control flow:

```bash
set -euo pipefail
if camera-rig provision fixed \
    --config .local/configs/fixed_provision.yaml \
    --output .local/artifacts/fixed_camera; then
  camera-rig provision validate --artifact .local/artifacts/fixed_camera
fi
```

A target preflight PASS is not a fixed-provision viability guarantee. The recommended operator
sequence is target preflight, provision preflight, fixed provision, then provision validation.

Validate a completed artifact with:

```bash
camera-rig provision validate \
  --artifact .local/artifacts/fixed_camera
```

The consumer boundary is deliberately small:

```python
from camera_rig.provision.bundle import load_and_validate_fixed_camera_bundle

bundle = load_and_validate_fixed_camera_bundle("fixed_camera/camera_bundle.json")
fixed = bundle.fixed_mount_calibration
assert fixed is not None
p_workspace = fixed.T_parent_from_camera_reference.transform_points(p_camera_reference)
```

Loading and validating the bundle does not require OpenCV, the RealSense SDK, or Pillow.
Consumers do not need to know how ChArUco detection or OpenCV PnP produced the transform.
