"""Stable CameraBundle loading façade."""

from __future__ import annotations

from pathlib import Path

from camera_rig.artifacts.models import CameraBundle
from camera_rig.artifacts.validation import load_and_validate_bundle
from camera_rig.core.errors import ArtifactError

__all__ = ["CameraBundle", "load_camera_bundle", "load_provisioned_camera_bundle"]


def load_camera_bundle(path: str | Path) -> CameraBundle:
    """Load a CameraBundle with schema, typed, SE(3), and decision validation."""
    bundle = load_and_validate_bundle(path)
    if bundle.status in {"passed", "failed"} and (
        (bundle.status == "passed") != bundle.quality.passed
    ):
        raise ArtifactError("camera bundle status and quality decision differ")
    return bundle


def load_provisioned_camera_bundle(root: str | Path) -> CameraBundle:
    """Validate a complete fixed-provision artifact and return its CameraBundle.

    The provision validator fails closed over its manifest, checksums, exact file set,
    nested artifacts, quality decisions, transform semantics, and cross-file identities.
    Imports remain local so importing the core consumer API never loads camera or vision
    optional dependencies.
    """
    from camera_rig.provision.bundle import load_and_validate_fixed_camera_bundle
    from camera_rig.provision.validation import load_and_validate_fixed_provision

    artifact_root = Path(root)
    manifest = load_and_validate_fixed_provision(artifact_root)
    bundle_reference = manifest.artifacts["camera_bundle"]
    return load_and_validate_fixed_camera_bundle(artifact_root / bundle_reference.path)
