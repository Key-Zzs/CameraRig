"""Versioned CameraRig artifact contracts and I/O."""

from camera_rig.artifacts.hashing import sha256_bytes, sha256_file
from camera_rig.artifacts.io import (
    atomic_write_json,
    deterministic_json_bytes,
    json_safe,
    load_json,
)
from camera_rig.artifacts.models import BUNDLE_SCHEMA_VERSION, CameraBundle

__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "CameraBundle",
    "atomic_write_json",
    "deterministic_json_bytes",
    "json_safe",
    "load_json",
    "sha256_bytes",
    "sha256_file",
]
