# Architecture

## Responsibility boundary

CameraRig is a single-physical-camera acquisition, calibration-parameter management,
validation, and replay toolkit for robotics. One `CameraSession` or `CameraDriver`
instance corresponds to exactly one physical camera.

The library boundary includes discovery and lifecycle for that device; RGB,
depth, IR, and other streams within the device; device-local timing; SDK factory
intrinsics; transforms between the device's optical or stream frames; a single
camera's fixed-mount extrinsic; calibration quality validation; and capture/replay
artifacts.

CameraRig never owns multi-camera arrays, groups, or synchronization; calibration
between physical cameras; robot forward kinematics; point-cloud construction; FFS;
DepthAnything; mapping; fusion; or TSDF reconstruction.

The optional RealSense adapter is isolated behind an injectable SDK boundary. Core data,
configuration, artifact, and replay imports remain hardware-independent.

## Contract flow

```text
RealSense SDK adapter
      ↓
CameraDriver
      ↓
CameraSession → CameraFrame + FactoryCalibration
      ↓
snapshot artifact → SDK-independent replay
      ↓
CameraBundle
      ↓
consumer such as PointCloudBuilder
```

The final arrow is a consumer relationship only. CameraRig does not implement or import
the consumer.

`TargetDetector` is a plugin protocol. No target detector is registered or implemented
in the core package. Moving-camera calibration modules are reserved interfaces whose
calls raise `FeatureNotAvailableError`; they do not return placeholders.

## Dependency direction

Downstream consumers may read CameraRig artifacts. CameraRig must not import or depend
on such consumers. The optional hardware SDK is loaded lazily and is not required to
import the core package or replay artifacts.

## Coordinate contract

Vectors are column vectors in right-handed frames. Length is in meters and time is in
nanoseconds. Rigid transforms use homogeneous 4 x 4 SE(3) matrices named
`T_target_from_source`, where `p_target = T_target_from_source @ p_source`. Records
must explicitly name `source_frame` and `target_frame`; ambiguous names such as `T_ab`,
`extrinsics`, `pose_matrix`, and `camera_transform` are not standalone transform
contracts.

Strict configuration uses a singular `camera` root. The schema rejects `cameras`,
unknown fields, unsupported mount types, and any configuration for unavailable
calibration operations.
