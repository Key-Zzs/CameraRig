# Architecture

## Responsibility boundary

CameraRig is a single-physical-camera acquisition, calibration-parameter management,
validation, and replay toolkit for robotics. One `CameraSession` or `CameraDriver`
instance corresponds to exactly one physical camera.

The eventual library boundary includes discovery and lifecycle for that device; RGB,
depth, IR, and other streams within the device; device-local timing; SDK factory
intrinsics; transforms between the device's optical or stream frames; a single
camera's fixed-mount extrinsic; calibration quality validation; and capture/replay
artifacts.

CameraRig never owns multi-camera arrays, groups, or synchronization; calibration
between physical cameras; robot forward kinematics; point-cloud construction; FFS;
DepthAnything; mapping; fusion; or TSDF reconstruction.

The package currently provides only installable project infrastructure. Hardware
adapters are intentionally absent.

## Dependency direction

Downstream consumers may read CameraRig artifacts. CameraRig must not import or depend
on such consumers. Hardware SDKs, when introduced behind driver boundaries, must not
be required to import the core package.

## Coordinate contract

Vectors are column vectors in right-handed frames. Length is in meters and time is in
nanoseconds. Rigid transforms use homogeneous 4 x 4 SE(3) matrices named
`T_target_from_source`, where `p_target = T_target_from_source @ p_source`. Records
must explicitly name `source_frame` and `target_frame`; ambiguous names such as `T_ab`,
`extrinsics`, `pose_matrix`, and `camera_transform` are not standalone transform
contracts.
