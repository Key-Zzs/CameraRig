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

## Command line

```bash
camera-rig --help
camera-rig --version
python -m camera_rig --help
```

## Python API

```python
import camera_rig

print(camera_rig.__version__)
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
