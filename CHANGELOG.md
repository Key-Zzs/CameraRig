# Changelog

All notable changes to CameraRig are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
