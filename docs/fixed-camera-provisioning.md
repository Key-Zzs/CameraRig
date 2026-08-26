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

The live command opens the D435i once. Every acquired frame contributes to raw-stream
validation; a deterministic evenly spaced subset contributes to the replay snapshot,
target detection, and fixed-pose solve. The selected source indices are persisted.

The output is built in a sibling temporary directory. CameraBundle reload, manifest
reload, checksums, cross-file identities, safe relative paths, and the exact file set
must all validate before the directory is published. Existing output is rejected by
default. `--force` builds and validates the replacement before swapping it into place.

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
