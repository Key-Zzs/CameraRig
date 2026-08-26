# Downstream Integration

CameraRig produces camera-frame data and calibration. Downstream software owns
point-cloud generation and multi-camera orchestration. Consumer code should import only
from `camera_rig.api`.

## Live frame

```python
from camera_rig.api import CameraSession, load_camera_config

config = load_camera_config("camera.yaml")
with CameraSession.from_config(config) as session:
    frame = session.capture()

rgb = frame.rgb
depth = frame.depth
ir_left = frame.ir_left
ir_right = frame.ir_right
```

One `CameraSession` represents exactly one physical camera. Opening a RealSense session
requires the `realsense` optional dependency; importing the API does not.

## Recorded frame

```python
from camera_rig.api import ReplayCameraSession

with ReplayCameraSession.from_artifact("capture-artifact") as session:
    frame = session.capture()

depth = frame.depth
```

Replay validates the capture artifact and reconstructs the same `CameraFrame` contract
without RealSense.

## Fixed geometry

```python
import numpy as np

from camera_rig.api import load_provisioned_camera_bundle

bundle = load_provisioned_camera_bundle("fixed-camera-artifact")
fixed = bundle.fixed_mount_calibration
if fixed is None:
    raise RuntimeError("camera is not fixed-calibrated")

transform = fixed.T_parent_from_camera_reference
print(transform.source_frame)
print(transform.target_frame)

points_camera = np.asarray([[0.0, 0.0, 1.0]])
points_workspace = transform.transform_points(points_camera)
```

The loader validates the complete artifact directory so consumers do not need to know
the internal roles of `manifest.json`, `camera_bundle.json`, or `checksums.sha256`.
Consumers depend on passed quality, frame semantics, SE(3), and fixed-mount semantics;
they do not need to depend on the calibration target, solver, or vision-library
provenance. A runnable version is available at
[`examples/consumer_fixed_camera.py`](../examples/consumer_fixed_camera.py).
