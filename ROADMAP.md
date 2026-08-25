# CameraRig Development Roadmap

## R0 Repository Bootstrap

Create the installable, testable Python repository, documentation baseline, command-line
entry point, and continuous-integration checks without camera hardware functionality.

## R1 Core Contracts & Strict Configuration

Freeze the single-camera data models, SE(3) semantics, strict configuration, error model,
target plugin interface, and JSON artifact foundation without connecting to hardware.

## R2 D435i Device Discovery & Lifecycle

Add explicit RealSense device selection, model and serial verification, and single-device
lifecycle management.

Status: implemented and validated on a physical D435i.

## R3 RealSense Factory Calibration Export

Read and export factory intrinsics, depth scale, and transforms between streams inside a
single device.

Status: implemented and validated against active profiles on a physical D435i.

## R4 Single-Camera Raw Capture & Internal Stream Validation

Capture raw RGB, depth, and IR streams from one physical camera and validate device-local
timestamp and frame relationships.

Status: implemented and validated with repeated 60-frame and 300-frame physical capture.

## R5 Snapshot / Artifact / Replay

Persist validated single-camera snapshots and replay them without camera hardware.

Status: implemented; physical snapshots, checksum validation, and SDK-independent replay
have been validated.

## R6 TargetDetector + ChArUco

Implement the first target-detection plugin using ChArUco. AprilGrid remains planned and
is not implemented in this item.

## R7 Fixed-Camera Extrinsic Calibration

Estimate, validate, and persist a fixed camera's transform to its parent frame.

## R8 One-Command Fixed Camera Provisioning

Combine device validation, capture, calibration, quality gates, and bundle generation in
a reproducible fixed-camera workflow.

## R9 PCB Integration Contract & v1.0

Stabilize the consumer-facing CameraBundle contract and integration boundary for a 1.0
release.

## Post-v1 candidates

- AprilGrid target-detection plugin (planned, not implemented).
- D405 and D455 drivers.
- Eye-in-hand, robot-world/hand-eye, and fixed eye-to-hand calibration. Wrist-mounted
  camera calibration is reserved and not implemented.
- Additional camera SDK drivers.
