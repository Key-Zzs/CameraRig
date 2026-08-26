# CameraRig v1 Consumer API

`camera_rig.api` is the stable downstream interface. Its public surface is explicit and
regression-tested through `camera_rig.api.__all__`.

| Symbol | Purpose | Optional dependency | Stability |
| --- | --- | --- | --- |
| `CameraConfig` | Strict one-physical-camera configuration | None | Stable v1 |
| `CameraFrame` | One synchronized device-local frameset | None | Stable v1 |
| `StreamFrame` | One raw stream array and timing metadata | None | Stable v1 |
| `CameraIntrinsics` | Stream-specific pinhole and distortion parameters | None | Stable v1 |
| `FactoryCalibration` | Intrinsics, internal transforms, and depth scale | None | Stable v1 |
| `RigidTransform` | Frame-explicit `T_target_from_source` SE(3) transform | None | Stable v1 |
| `FixedMountCalibration` | Fixed camera-reference-to-parent transform | None | Stable v1 |
| `CameraBundle` | Typed `camera-rig.bundle.v1` artifact | None | Stable v1 |
| `CameraSession` | Live session for exactly one physical camera | `realsense` when opened | Stable v1 |
| `ReplayCameraSession` | SDK-independent validated capture replay | None | Stable v1 |
| `load_camera_config` | Schema and typed-contract configuration loader | None | Stable v1 |
| `load_camera_bundle` | Schema, typed, SE(3), and status/quality CameraBundle loader | None | Stable v1 |
| `load_provisioned_camera_bundle` | Complete fixed-provision validator and bundle loader | None | Stable v1 |

Importing `camera_rig.api`, including `CameraSession`, does not import the RealSense C
extension. A missing `realsense` extra is reported only when a live RealSense operation
needs it. Provisioned artifact consumption validates its manifest, exact file set,
checksums, nested artifacts, quality decisions, transform semantics, and cross-file
identities before returning the `CameraBundle`.

See [API_STABILITY.md](../API_STABILITY.md) for the compatibility and Semantic Versioning
policy.
