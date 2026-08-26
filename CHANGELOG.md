# Changelog

All notable changes to CameraRig are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-08-25

### Added

- Generic planar IPPE pose estimation with explicit candidate, cheirality, printed-face,
  distortion, LM-refinement, reprojection, and frame-direction contracts.
- Robust multi-frame fixed-camera calibration with medoid aggregation, pose outlier
  rejection, joint refinement, split-half stability, native-depth sanity, diagnostic
  overlays, and validated `FixedMountCalibration` output.
- One-YAML, one-command fixed-camera provisioning with a single live acquisition,
  deterministic 60-of-300 frame selection, strict preflight, a populated CameraBundle,
  checksums, exact-file-set validation, and atomic replacement semantics.

### Changed

- Package version is now 0.4.0.
- Stream-validation and target-detection reports are typed, schema-validated artifacts
  that are cross-bound to capture, target, calibration, and bundle evidence.

## [0.3.0] - 2026-08-25

### Added

- A hardware-independent `TargetDetector` plugin contract and a strict ChArUco target
  specification using persisted canonical corner geometry.
- Deterministic A4 target generation with independent print-scale rulers, resolved
  provenance, single-image detection, capture-artifact validation, diagnostic overlays,
  and temporal jitter statistics.
- Synthetic distortion, partial-visibility, wrong-dictionary, geometry, packaging, and
  physical D435i validation coverage.

### Changed

- Package version is now 0.3.0.
- Capture reports record their acceptance thresholds. The aggregate median coverage
  floor is 5%, matching the per-frame detector floor and the validated fixed-workspace
  deployment while retaining the corner-hull coverage definition.

## [0.2.0] - 2026-08-25

### Added

- Exact D435i discovery by serial, canonical model and product-ID validation, strict
  profile resolution, and repeatable single-device lifecycle management.
- Active-profile RealSense factory intrinsics, directed internal transforms, depth scale,
  portable provenance, JSON Schema validation, and atomic export.
- Raw RGB, depth, left-IR, and right-IR capture with owned NumPy buffers, SDK timestamp
  metadata, internal frameset synchronization reports, and bounded-memory stream metrics.
- Atomic capture artifacts with SHA-256 validation, diagnostic PNG previews, corruption
  and path-traversal defenses, and SDK-independent replay with explicit EOF and rewind.

### Changed

- Package version is now 0.2.0.
- The public configuration example uses an explicit serial placeholder.
