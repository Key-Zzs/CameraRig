# Target detector plugins

`TargetDetector` is the consumer-facing protocol:

```python
class TargetDetector(Protocol):
    plugin_name: str

    def detect(self, image) -> TargetObservation: ...
```

Consumers create a detector through the built-in registry and do not import a concrete
detector class:

```python
from camera_rig.targets import load_target, registry

spec = load_target("target_spec.json")
detector = registry.create(plugin_name=spec.plugin, target_spec=spec)
observation = detector.detect(image_rgb)
```

The built-in registry deliberately stays small. Adding another target family requires a
new plugin factory and resolved specification loader, but does not change
`TargetObservation`, the caller, or later calibration consumers.

Target detection accepts an image and produces matched 2D image points and canonical 3D
target points. It does not own camera capture, camera intrinsics, pose estimation, or
extrinsic calibration. Snapshot capture and replay remain separate operations.

Concrete target dependencies are optional and lazy. Importing `camera_rig`,
`camera_rig.core`, or `camera_rig.targets` does not import OpenCV. Invoking the ChArUco
plugin without its extra raises `MissingOptionalDependencyError` and points to:

```bash
pip install "camera-rig[charuco]"
```
