"""Atomic writer for complete fixed-camera provision artifacts."""

from __future__ import annotations

import ctypes
import os
import shutil
import tempfile
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final

from camera_rig.artifacts.hashing import sha256_file
from camera_rig.artifacts.io import atomic_write_json, json_safe
from camera_rig.core._validation import require_non_empty, string_keyed_copy
from camera_rig.core.errors import ArtifactError, ContractError
from camera_rig.core.quality import QualityReport
from camera_rig.version import __version__

FIXED_PROVISION_ARTIFACT_SCHEMA_VERSION: Final = "camera-rig.fixed-provision-artifact.v1"

ARTIFACT_PATHS: Final = {
    "camera_bundle": "camera_bundle.json",
    "factory_calibration": "factory/factory_calibration.json",
    "capture_manifest": "capture/calibration_snapshot/manifest.json",
    "target_spec": "target/target_spec.json",
    "target_detection": "target/detection_report.json",
    "fixed_calibration": "calibration/fixed_calibration.json",
    "stream_validation": "reports/stream_validation.json",
}
CAPTURE_ROOT: Final = "capture/calibration_snapshot"
OVERLAY_LABELS: Final = ("best", "median_quality", "worst_accepted")
DIAGNOSTIC_OVERLAY_ROOTS: Final = {
    "target_detection": "diagnostics/overlays/target_detection",
    "fixed_calibration": "diagnostics/overlays/fixed_calibration",
}
QUALITY_CHECKS: Final = (
    "artifact_integrity_passed",
    "bundle_self_validation_passed",
    "bundle_quality_passed",
    "capture_validation_passed",
    "device_identity_match",
    "factory_quality_passed",
    "fixed_calibration_passed",
    "internal_calibration_match",
    "target_detection_passed",
    "target_identity_match",
    "stream_validation_passed",
)


@dataclass(frozen=True)
class ArtifactReference:
    """One canonical artifact-relative file and its digest."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        _safe_relative_path(self.path)
        _require_digest(self.sha256, "sha256")

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ArtifactReference:
        if set(data) != {"path", "sha256"}:
            raise ContractError("artifact reference has missing or unknown fields")
        return cls(path=_string(data["path"], "path"), sha256=_string(data["sha256"], "sha256"))


@dataclass(frozen=True)
class FixedProvisionManifest:
    """Top-level manifest for a validated, passed fixed-camera provision."""

    artifact_id: str
    created_at: str
    camera_rig_version: str
    artifacts: dict[str, ArtifactReference]
    diagnostics: dict[str, dict[str, ArtifactReference]]
    quality: QualityReport
    provenance: dict[str, object]
    status: str = field(default="passed", init=False)
    schema_version: str = field(default=FIXED_PROVISION_ARTIFACT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        try:
            uuid.UUID(self.artifact_id)
        except (ValueError, AttributeError) as error:
            raise ContractError("artifact_id must be a UUID") from error
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ContractError("created_at must be an ISO-8601 date-time") from error
        require_non_empty(self.camera_rig_version, "camera_rig_version")
        artifacts = dict(self.artifacts)
        if set(artifacts) != set(ARTIFACT_PATHS):
            raise ContractError("manifest artifacts have missing or unknown entries")
        for name, expected_path in ARTIFACT_PATHS.items():
            if artifacts[name].path != expected_path:
                raise ContractError(f"manifest {name} path must be {expected_path!r}")
        diagnostics = {group: dict(references) for group, references in self.diagnostics.items()}
        if set(diagnostics) != set(DIAGNOSTIC_OVERLAY_ROOTS):
            raise ContractError("manifest diagnostics have missing or unknown groups")
        for group, root in DIAGNOSTIC_OVERLAY_ROOTS.items():
            references = diagnostics[group]
            if set(references) != set(OVERLAY_LABELS):
                raise ContractError(f"manifest diagnostics.{group} must contain three overlays")
            for label in OVERLAY_LABELS:
                _validate_diagnostic_path(references[label].path, group, root, label)
        expected_quality = passed_provision_quality()
        if self.quality.to_dict() != expected_quality.to_dict():
            raise ContractError("manifest quality must record every required passed check")
        provenance = string_keyed_copy(self.provenance, "provenance")
        _reject_nonportable_values(provenance, "provenance")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "provenance", provenance)
        json_safe(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "created_at": self.created_at,
            "camera_rig_version": self.camera_rig_version,
            "status": self.status,
            "artifacts": {
                name: reference.to_dict() for name, reference in sorted(self.artifacts.items())
            },
            "diagnostics": {
                group: {
                    label: reference.to_dict() for label, reference in sorted(references.items())
                }
                for group, references in sorted(self.diagnostics.items())
            },
            "quality": self.quality.to_dict(),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> FixedProvisionManifest:
        if data.get("schema_version") != FIXED_PROVISION_ARTIFACT_SCHEMA_VERSION:
            raise ContractError(
                f"schema_version must be {FIXED_PROVISION_ARTIFACT_SCHEMA_VERSION!r}"
            )
        if data.get("status") != "passed":
            raise ContractError("fixed provision manifest status must be passed")
        artifacts = _object(data["artifacts"], "artifacts")
        diagnostics = _object(data["diagnostics"], "diagnostics")
        return cls(
            artifact_id=_string(data["artifact_id"], "artifact_id"),
            created_at=_string(data["created_at"], "created_at"),
            camera_rig_version=_string(data["camera_rig_version"], "camera_rig_version"),
            artifacts={
                name: ArtifactReference.from_dict(_object(value, f"artifacts.{name}"))
                for name, value in artifacts.items()
            },
            diagnostics={
                group: {
                    label: ArtifactReference.from_dict(
                        _object(value, f"diagnostics.{group}.{label}")
                    )
                    for label, value in _object(group_value, f"diagnostics.{group}").items()
                }
                for group, group_value in diagnostics.items()
            },
            quality=QualityReport.from_dict(_object(data["quality"], "quality")),
            provenance=_object(data["provenance"], "provenance"),
        )


@dataclass(frozen=True)
class FixedProvisionArtifactInputs:
    """Validated source paths consumed by the outer artifact writer."""

    camera_bundle: Path
    factory_calibration: Path
    capture_artifact: Path
    target_spec: Path
    target_detection: Path
    fixed_calibration: Path
    stream_validation: Path
    target_detection_overlays: ProvisionOverlayInputs
    fixed_calibration_overlays: ProvisionOverlayInputs


@dataclass(frozen=True)
class ProvisionOverlayInputs:
    """Explicit diagnostic overlay files for the three retained quality views."""

    best: Path
    median_quality: Path
    worst_accepted: Path

    def by_label(self) -> dict[str, Path]:
        return {
            "best": self.best,
            "median_quality": self.median_quality,
            "worst_accepted": self.worst_accepted,
        }


def passed_provision_quality() -> QualityReport:
    """Return the canonical all-gates-passed outer quality decision."""
    return QualityReport(
        passed=True,
        metrics={name: True for name in QUALITY_CHECKS},
        thresholds={"all_checks_required": True},
    )


def write_fixed_provision_artifact(
    output: str | Path,
    inputs: FixedProvisionArtifactInputs,
    *,
    provenance: dict[str, object],
    artifact_id: str | None = None,
    created_at: str | None = None,
    camera_rig_version: str = __version__,
    force: bool = True,
) -> FixedProvisionManifest:
    """Copy all inputs, self-validate, then publish one complete directory."""
    target = Path(output)
    if target.exists() and not force:
        raise ArtifactError(f"fixed provision artifact already exists: {target}")
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ArtifactError("fixed provision output must be a real directory path")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.tmp-"))
    try:
        diagnostic_paths = _copy_inputs(temporary, inputs)
        references = {
            name: ArtifactReference(path=relative, sha256=sha256_file(temporary / relative))
            for name, relative in ARTIFACT_PATHS.items()
        }
        diagnostic_references = {
            group: {
                label: ArtifactReference(
                    path=diagnostic_paths[group][label],
                    sha256=sha256_file(temporary / diagnostic_paths[group][label]),
                )
                for label in OVERLAY_LABELS
            }
            for group in DIAGNOSTIC_OVERLAY_ROOTS
        }
        manifest = FixedProvisionManifest(
            artifact_id=artifact_id or str(uuid.uuid4()),
            created_at=created_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            camera_rig_version=camera_rig_version,
            artifacts=references,
            diagnostics=diagnostic_references,
            quality=passed_provision_quality(),
            provenance=provenance,
        )
        atomic_write_json(temporary / "manifest.json", manifest.to_dict())
        _write_checksums(temporary)
        from camera_rig.provision.validation import load_and_validate_fixed_provision

        validated = load_and_validate_fixed_provision(temporary)
        _commit_directory(temporary, target, force=force)
        return validated
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _copy_inputs(root: Path, inputs: FixedProvisionArtifactInputs) -> dict[str, dict[str, str]]:
    from camera_rig.artifacts.capture_validation import validate_capture_artifact

    capture = Path(inputs.capture_artifact)
    validate_capture_artifact(capture)
    _copy_file(inputs.camera_bundle, root / ARTIFACT_PATHS["camera_bundle"])
    _copy_file(inputs.factory_calibration, root / ARTIFACT_PATHS["factory_calibration"])
    _copy_file(inputs.target_spec, root / ARTIFACT_PATHS["target_spec"])
    _copy_file(inputs.target_detection, root / ARTIFACT_PATHS["target_detection"])
    _copy_file(inputs.fixed_calibration, root / ARTIFACT_PATHS["fixed_calibration"])
    _copy_file(inputs.stream_validation, root / ARTIFACT_PATHS["stream_validation"])
    capture_destination = root / CAPTURE_ROOT
    capture_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(capture, capture_destination)
    overlay_inputs = {
        "target_detection": inputs.target_detection_overlays,
        "fixed_calibration": inputs.fixed_calibration_overlays,
    }
    diagnostic_paths: dict[str, dict[str, str]] = {}
    for group, sources in overlay_inputs.items():
        relative_root = DIAGNOSTIC_OVERLAY_ROOTS[group]
        diagnostic_paths[group] = {}
        for label, source in sources.by_label().items():
            relative = _diagnostic_destination(source, group, relative_root, label)
            diagnostic_paths[group][label] = relative
            _copy_file(source, root / relative)
    return diagnostic_paths


def _copy_file(source: str | Path, destination: Path) -> None:
    path = Path(source)
    if path.is_symlink() or not path.is_file():
        raise ArtifactError(f"provision input must be a real file: {path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def _write_checksums(root: Path) -> None:
    paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != "checksums.sha256"
    )
    lines = [f"{sha256_file(root / relative)}  {relative}" for relative in paths]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _commit_directory(temporary: Path, target: Path, *, force: bool) -> None:
    if not target.exists():
        _replace_directory(temporary, target)
        return
    if not force:
        raise ArtifactError(f"fixed provision artifact already exists: {target}")
    if target.is_symlink() or not target.is_dir():
        raise ArtifactError("fixed provision output must be a real artifact directory")
    from camera_rig.provision.validation import load_and_validate_fixed_provision

    load_and_validate_fixed_provision(target)
    _exchange_directories(temporary, target)
    # Publication has committed atomically. A stale private temporary directory is
    # preferable to reporting failure after making the new validated output live.
    with suppress(OSError):
        shutil.rmtree(temporary)


def _replace_directory(source: Path, target: Path) -> None:
    os.replace(source, target)


def _exchange_directories(source: Path, target: Path) -> None:
    """Atomically exchange two sibling directories on Linux, or fail without mutation."""
    if os.name != "posix":
        raise ArtifactError("atomic output replacement requires Linux renameat2 support")
    library = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = library.renameat2
    except AttributeError as error:
        raise ArtifactError("atomic output replacement requires renameat2 support") from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_exchange = 2
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(target),
        rename_exchange,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(target))


def _safe_relative_path(value: str) -> str:
    if not value or "\\" in value or value.casefold().startswith("file://"):
        raise ContractError(f"unsafe artifact-relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ContractError(f"unsafe artifact-relative path: {value!r}")
    return value


def _diagnostic_destination(source: str | Path, group: str, root: str, label: str) -> str:
    filename = Path(source).name
    relative = f"{root}/{filename}"
    _validate_diagnostic_path(relative, group, root, label)
    return relative


def _validate_diagnostic_path(value: str, group: str, root: str, label: str) -> None:
    _safe_relative_path(value)
    path = PurePosixPath(value)
    if (
        path.parent.as_posix() != root
        or not path.name.startswith(f"{label}_")
        or path.suffix != ".png"
        or path.name in {".", ".."}
    ):
        raise ContractError(
            f"manifest diagnostics.{group}.{label} must be a labeled PNG under {root!r}"
        )


def _reject_nonportable_values(value: object, path: str) -> None:
    if isinstance(value, str):
        windows = PureWindowsPath(value)
        if Path(value).is_absolute() or windows.is_absolute() or windows.drive:
            raise ContractError(f"{path} must not contain absolute paths")
        if value.casefold().startswith("file://"):
            raise ContractError(f"{path} must not contain file URIs")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_nonportable_values(item, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _reject_nonportable_values(item, f"{path}[{index}]")


def _require_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be an object with string keys")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")
    return value
