# CameraRig API Stability Policy

CameraRig follows Semantic Versioning for its consumer-facing interface.

## Stable v1 contracts

The following interfaces are stable throughout the 1.x release line:

- `camera_rig.api` and its explicitly declared `__all__` surface.
- The `camera-rig.bundle.v1` persisted artifact schema and field meanings.
- `CameraFrame` and `StreamFrame` core semantics.
- `RigidTransform` with the `T_target_from_source` convention.
- `FixedMountCalibration`, including `parent_frame`, `camera_reference_frame`, and
  `T_parent_from_camera_reference` semantics.

Backward-compatible additions may be made in 1.x. An intentional incompatible change to
the stable Python API requires 2.0. An incompatible persisted artifact requires a new
schema version; the meaning of `camera-rig.bundle.v1` will not be changed silently.

In particular, the 1.x line will not reverse the mathematical direction of
`T_target_from_source`, change CameraBundle v1 field meanings, or redefine the source and
target frames of fixed-mount extrinsics.

## Internal interfaces

Implementation modules are not compatibility-guaranteed, including:

- `camera_rig.drivers.*`
- `camera_rig.provision.*`
- `camera_rig.calibration.*`
- `camera_rig.targets.charuco.*`
- `camera_rig.cli.commands.*`

Downstream software should import stable contracts from `camera_rig.api` instead of these
paths. Internal modules may be refactored in a 1.x release without being treated as a
consumer API break.
