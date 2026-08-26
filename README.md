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
  --report .local/reports/target-validation.json \
  --overlays .local/overlays/target-validation
```

Provision a fixed D435i from one strict YAML contract, or inspect the complete plan
without opening the camera:

```bash
camera-rig provision fixed \
  --config .local/configs/fixed_provision.yaml \
  --output .local/artifacts/fixed_camera \
  --dry-run
camera-rig provision fixed \
  --config .local/configs/fixed_provision.yaml \
  --output .local/artifacts/fixed_camera
camera-rig provision validate \
  --artifact .local/artifacts/fixed_camera
```

The fixed workflow defines `workspace` as the persisted ChArUco target frame and emits
`T_workspace_from_<camera>/ir_left_optical` inside a validated `CameraBundle`. The camera
and target remain fixed throughout acquisition; repeated frames measure detection and
pose repeatability rather than recalibrating intrinsics. See
[fixed-camera calibration](docs/fixed-camera-calibration.md),
[fixed-camera provisioning](docs/fixed-camera-provisioning.md), and
[calibration quality](docs/calibration-quality.md).

`TargetDetector` is a hardware-independent plugin contract. The ChArUco implementation
returns image points, stable point IDs, persisted canonical target points, and 2D quality
metrics; it does not estimate target pose or camera extrinsics. See
[docs/charuco-target.md](docs/charuco-target.md) for print and validation details.

Configuration uses a singular `camera` root. Unknown fields, a plural `cameras` root,
unknown streams, invalid dimensions or frame rates, and non-string serial numbers are
rejected instead of coerced.

## Python API

```python
import camera_rig
from camera_rig.capture import CameraSession, ReplayCameraSession
from camera_rig.config import load_config

print(camera_rig.__version__)

config = load_config("private-d435i.yaml")
with CameraSession.from_config(config) as camera:
    frame = camera.capture()

with ReplayCameraSession.from_artifact("capture-artifact") as replay:
    restored = replay.capture()
```

Hardware-independent contracts are available from `camera_rig.core`, target plugin
interfaces from `camera_rig.targets`, and deterministic artifact helpers from
`camera_rig.artifacts`. Importing these modules does not require RealSense or OpenCV.

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
