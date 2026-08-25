# Artifact Contract

`CameraBundle` is the stable top-level JSON artifact for exactly one physical camera.
Its schema version is `camera-rig.bundle.v1`.

The bundle contains:

- status, bundle ID, and ISO-8601 creation time;
- the frozen coordinate convention;
- physical device identity with the serial represented as a string;
- stream profiles and per-stream intrinsics;
- SE(3) transforms among optical or stream frames inside that device;
- depth scale in meters per raw unit;
- optional precomputed fixed-mount calibration;
- general quality metrics, thresholds, warnings, and failure reasons;
- provenance.

A synthetic bundle used in tests must identify its synthetic provenance and must not
claim to have been read from real camera hardware.

## Persistence

JSON is UTF-8, sorted by key, indented deterministically, terminated by a newline, and
written through a temporary sibling followed by `os.replace`. NaN, infinity, non-string
object keys, pickle, and unsupported Python objects are rejected. Matrices become nested
JSON arrays and are reconstructed through `RigidTransform`, which re-runs SE(3)
validation.

`sha256_bytes` and `sha256_file` produce lowercase SHA-256 hex digests. Deterministic
formatting means equivalent bundle dictionaries produce identical bytes and hashes.

## Factory calibration artifact

`camera-rig.factory-calibration.v1` records the actual active stream profiles, one
intrinsic model per active video stream, directed `T_target_from_source` transforms from
the configured reference stream, the device-reported depth scale, quality results, and
portable provenance. It contains no runtime path.

## Capture artifact

`camera-rig.capture.v1` is a directory artifact. Raw RGB, depth, left-IR, and right-IR
arrays retain their dtype and shape in per-frame NPZ files. Separate JSON files preserve
host receive time, sensor timestamps, timestamp domains, frame numbers, supported SDK
metadata, and the single-device synchronization report.

Manifest paths are POSIX-style and artifact-relative. `checksums.sha256` covers the
manifest, factory calibration, every NPZ and metadata file, and every derived preview;
the checksum file itself is excluded to avoid recursion. Validation rejects missing or
unexpected files, schema changes, checksum mismatches, absolute paths, traversal, and
symlinks before replay exposes data.

The writer builds and validates a sibling temporary directory before committing it.
PNG previews never supply replay arrays. `ReplayCameraSession` reads NPZ data only after
full validation and has no hardware SDK dependency.
