# CameraRig

CameraRig is a Python toolkit for single-physical-camera acquisition, calibration
parameter management, validation, and replay in robotics systems.

## Scope

One `CameraSession` or `CameraDriver` instance represents exactly one physical camera.
The library boundary covers device discovery and lifecycle, streams within that device,
device-local timing, factory intrinsics, transforms between the device's optical stream
frames, fixed-mount extrinsics, quality validation, and capture/replay artifacts.

## Non-goals

CameraRig does not own multi-camera grouping or synchronization, camera-to-camera
calibration, robot forward kinematics, point-cloud construction, FFS, DepthAnything,
mapping, fusion, or TSDF reconstruction. Consumers may use CameraRig artifacts without
CameraRig depending on those consumers.

## Installation

CameraRig requires Python 3.10 or newer.

```bash
python -m pip install -e ".[dev]"
```

Install the official RealSense wheel and PNG preview support for D435i acquisition:

```bash
python -m pip install -e ".[dev,realsense,viz]"
```

Install the ChArUco target generator and detector dependencies:

```bash
python -m pip install -e ".[charuco]"
```

Install the complete fixed-camera provisioning runtime:

```bash
python -m pip install -e ".[dev,provision]"
```

Keep the physical device serial in a private, ignored configuration. The public example
uses `REPLACE_WITH_DEVICE_SERIAL` deliberately.

## Command line

```bash
camera-rig --help
camera-rig --version
python -m camera_rig --help
```

Validate the included strict, single-camera contract example:

```bash
camera-rig config validate --config configs/examples/single_camera_contract.yaml
```

Validate a versioned camera bundle:

```bash
camera-rig artifact validate --bundle path/to/camera_bundle.json
```

Discover and inspect a D435i without changing device options:

```bash
camera-rig device list --driver realsense
camera-rig device inspect --config .local/configs/d435i.yaml --show-profiles
camera-rig device smoke --config .local/configs/d435i.yaml --cycles 5 \
  --report .local/reports/device-smoke.json
```

Export calibration from the profiles actually activated by the pipeline, then capture,
validate, persist, and replay raw streams:

```bash
camera-rig calibration factory export \
  --config .local/configs/d435i.yaml \
  --output .local/artifacts/factory_calibration.json
camera-rig capture validate-streams \
  --config .local/configs/d435i.yaml --frames 300 \
  --report .local/reports/stream-validation.json
camera-rig capture snapshot \
  --config .local/configs/d435i.yaml --frames 30 \
  --output .local/artifacts/sequence
camera-rig replay validate --artifact .local/artifacts/sequence
```

Generate the standard printable ChArUco board, detect it in one image, or validate all
color frames in a snapshot artifact:

```bash
camera-rig target generate \
  --config configs/targets/charuco_a4_v1.yaml \
  --output .local/targets/charuco_a4_v1
camera-rig target detect \
  --target .local/targets/charuco_a4_v1/target_spec.json \
  --image image.png \
  --output .local/reports/detection.json \
  --overlay .local/overlays/detection.png
camera-rig target validate-artifact \
  --target .local/targets/charuco_a4_v1/target_spec.json \
  --artifact .local/artifacts/sequence \
  --stream color \
  --policy uncertainty_validated \
  --report .local/reports/target-validation.json \
  --overlays .local/overlays/target-validation
```

Provision a fixed D435i from one strict YAML contract. The non-hardware `--dry-run`
checks only inputs and dependencies; the live viability preflight opens the camera and
runs the same acquisition, frame gates, shared-pose solve, and final quality evaluator as
`provision fixed`, but never publishes a bundle or fixed provision:

```bash
camera-rig provision fixed \
  --config .local/configs/fixed_provision.yaml \
  --output .local/artifacts/fixed_camera \
  --dry-run
camera-rig provision preflight \
  --config .local/configs/fixed_provision.yaml \
  --report .local/reports/fixed-provision-preflight.json \
  --overlays .local/overlays/fixed-provision-preflight \
  --evidence-root .local/validation/structured-gate/camera_a/repeat_01
```

`uncertainty_validated_v1` is explicitly `HOLD`, not a frozen release preset. Its live
preflight can retain private capture/evaluation evidence under `--evidence-root`, but reports
`UNCERTAINTY_VALIDATED_PRESET_NOT_RELEASED` and `would_publish=false` even when the candidate
numerical checks pass. `provision fixed` refuses to build a canonical CameraBundle for this HOLD
policy. Retained captures are offline validation inputs, not provisions.

Evaluate one retained repeat under in-memory K/D/target counterfactuals without changing any
capture, calibration, or provision artifact:

```bash
camera-rig calibration evaluate-model-counterfactuals \
  --detection-report .local/validation/structured-gate/camera_a/repeat_01/target/detection_report.json \
  --factory-calibration .local/validation/structured-gate/camera_a/repeat_01/factory/factory_calibration.json \
  --output .local/validation/structured-gate/camera_a/repeat_01/model-counterfactuals.json
```

The output reports sensitivity relative to the retained-data baseline, not ground-truth pose
bias. It is analysis-only evidence and cannot release a policy.

The fixed workflow defines `workspace` as the persisted ChArUco target frame and emits
`T_workspace_from_<camera>/ir_left_optical` inside a validated `CameraBundle`. The camera
and target remain fixed throughout acquisition; repeated frames measure detection and
pose repeatability rather than recalibrating intrinsics. See
[fixed-camera calibration](docs/fixed-camera-calibration.md),
[fixed-camera provisioning](docs/fixed-camera-provisioning.md), and
[calibration quality](docs/calibration-quality.md).

For candidate validation, `target.detection_policy: uncertainty_validated` selects the historical
v1 HOLD profile; it is not currently eligible for a production provision. The separately named
`uncertainty_validated_v2` structured policy is also a HOLD candidate. This codebase has no
authenticated release loader; a future release attempt additionally requires a preregistered
manifest and an untouched holdout meeting every bound.
Coverage is still reported as operator guidance and a target-size warning, but it is not pose
accuracy and is not a hard gate in this policy. Acceptance instead requires detection integrity,
PnP, a catastrophic scalar reprojection ceiling, scaled-Jacobian observability, bounded
conditional pose uncertainty, resolved planar-pose ambiguity, temporal repeatability,
split-half stability, and native-depth sanity. Reprojection residuals already determine the
pixel-noise estimate used by covariance, so the legacy 0.5/1.0 px precision thresholds are not
applied again as primary hard gates under `uncertainty_validated`; current 1.5/2.0 px gross
RMSE/p95 limits reject catastrophic projection failures. The candidate structured diagnostic uses
held-out spatial prediction, an engineering amplitude floor, and a deterministic whole-vector
permutation null. Its primary final diagnostic averages repeats by physical corner ID; per-frame
structure remains diagnostic-only. Neither residual magnitude nor conditional covariance proves
that K, D, or target geometry is correct.
Low coverage does not guarantee a pass; it only stops coverage alone from rejecting an otherwise
well-observed pose. `legacy_strict` and `pose_validated` retain their historical behavior.

Target preflight and provision preflight answer different questions. For the uncertainty policy,
target preflight reports `NUMERICAL_PASS RELEASE_HOLD` when target detection and pose
observability pass; it does not guarantee that raw-stream,
fixed-frame count/ratio, final reprojection, repeatability, split-half, or native-depth gates will
pass. While the preset is HOLD, stop after `target preflight -> provision preflight`. Continue to
`provision fixed -> provision validate` only in a future implementation that adds an authenticated
criteria/holdout loader and then produces an explicitly hash-bound `RELEASED` preset.

Synthetic development includes the 500 x 700 mm, 5 x 7, 100 mm-square, 75 mm-marker,
`DICT_4X4_50` board when a D435i must remain outside the robot workspace or view it obliquely.
Board size never creates an automatic pass: marker pixel scale, localization, uncertainty, and all
physical checks still apply. Raw-stream validation is an independent prerequisite and is never
bypassed by pose observability.

The current real structured-gate experiment uses the fixed A4 target only.
`REAL_500X700_STRUCTURED_GATE_VALIDATION=DEFERRED`.

`TargetDetector` is a hardware-independent plugin contract. The ChArUco implementation
returns image points, stable point IDs, persisted canonical target points, and 2D quality
metrics; it does not estimate target pose or camera extrinsics. See
[docs/charuco-target.md](docs/charuco-target.md) for print and validation details.

Configuration uses a singular `camera` root. Unknown fields, a plural `cameras` root,
unknown streams, invalid dimensions or frame rates, and non-string serial numbers are
rejected instead of coerced.

## Python API

`camera_rig.api` is the stable downstream interface. Consumer code should not depend on
the package's internal directory structure.

```python
import camera_rig
from camera_rig.api import CameraSession, ReplayCameraSession, load_camera_config

print(camera_rig.__version__)

config = load_camera_config("private-d435i.yaml")
with CameraSession.from_config(config) as camera:
    frame = camera.capture()

with ReplayCameraSession.from_artifact("capture-artifact") as replay:
    restored = replay.capture()
```

Load a complete fixed-camera provision artifact without depending on its internal file
layout:

```python
from camera_rig.api import load_provisioned_camera_bundle

bundle = load_provisioned_camera_bundle("fixed-camera-artifact")
fixed = bundle.fixed_mount_calibration
if fixed is None:
    raise RuntimeError("camera is not fixed-calibrated")
T_workspace_from_camera = fixed.T_parent_from_camera_reference
```

Importing `camera_rig.api` does not require RealSense or OpenCV. See the
[public API](docs/public-api.md), [stability policy](API_STABILITY.md), and
[downstream integration guide](docs/downstream-integration.md).

## Artifacts

`CameraBundle` is the versioned top-level JSON contract for one physical camera. It can
contain device identity, stream profiles, per-stream intrinsics, internal optical-frame
transforms, depth scale, an optional fixed-mount calibration, quality results, and
provenance. Fixed provisioning always populates and validates that mount record. JSON
writing is deterministic and atomic; persisted transforms are revalidated when loaded.

A capture artifact stores raw `uint8` RGB, raw `uint16` depth, both raw `uint8` infrared
streams, per-stream timing metadata, a factory-calibration artifact, and SHA-256 hashes.
All manifest paths are artifact-relative. PNG files are derived diagnostics only; replay
loads NPZ arrays and reconstructs the original `CameraFrame` contract after validating
the complete artifact. Replay does not import or require the RealSense SDK.

```text
capture-artifact/
├── manifest.json
├── factory_calibration.json
├── checksums.sha256
├── frames/
│   ├── frame_000000.npz
│   └── frame_000000.meta.json
└── previews/
    ├── frame_000000_color.png
    ├── frame_000000_depth.png
    ├── frame_000000_ir_left.png
    ├── frame_000000_ir_right.png
    └── mosaic.png
```

## Coordinate conventions

Vectors are column vectors in right-handed coordinate systems. Length is measured in
meters and time in nanoseconds. A rigid transform is a 4 x 4 homogeneous SE(3) matrix
named `T_target_from_source`, with the meaning
`p_target = T_target_from_source @ p_source`. Persistent transform records must name
both `source_frame` and `target_frame`.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Acknowledgements and citation

See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for acknowledgements and
[CITATION.cff](CITATION.cff) for citation metadata.

## 中文文档

中文说明见 [README_zh-CN.md](README_zh-CN.md)。
