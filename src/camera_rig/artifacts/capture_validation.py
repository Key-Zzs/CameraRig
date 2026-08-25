"""Fail-closed validation for portable capture artifact directories."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import numpy as np
import numpy.typing as npt

from camera_rig.artifacts.capture_io import restore_camera_frame
from camera_rig.artifacts.factory_calibration import load_and_validate_factory_calibration
from camera_rig.artifacts.hashing import sha256_file
from camera_rig.artifacts.io import load_json
from camera_rig.config.validation import validate_against_named_schema
from camera_rig.core.errors import (
    ArtifactError,
    ContractError,
    SchemaValidationError,
    TransformError,
)


def validate_capture_artifact(root: str | Path) -> dict[str, object]:
    """Validate schema, path safety, exact files, checksums, and NPZ readability."""
    artifact_root = Path(root)
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ArtifactError("capture artifact root must be a real directory")
    manifest_path = artifact_root / "manifest.json"
    value = load_json(manifest_path)
    if not isinstance(value, dict):
        raise ArtifactError("capture manifest must be a JSON object")
    try:
        validate_against_named_schema(value, "capture_manifest.v1.schema.json")
    except SchemaValidationError as error:
        raise ArtifactError(str(error)) from error
    manifest: dict[str, object] = dict(value)
    _reject_absolute_values(manifest["capture_configuration"], "capture_configuration")
    _reject_absolute_values(manifest["provenance"], "provenance")
    asset_hashes = _string_mapping(manifest["checksums"], "checksums")
    referenced = {_safe_path(name) for name in asset_hashes}
    expected_references = {_safe_path(_string(manifest["factory_calibration"], "factory"))}
    frames = manifest["frames"]
    if not isinstance(frames, list):
        raise ArtifactError("frames must be an array")
    frame_count = manifest["frame_count"]
    if (
        not isinstance(frame_count, int)
        or isinstance(frame_count, bool)
        or frame_count != len(frames)
    ):
        raise ArtifactError("frame_count must equal the number of frame entries")
    for expected_index, entry_value in enumerate(frames):
        entry = _object(entry_value, "frames[]")
        if entry["index"] != expected_index:
            raise ArtifactError("frame indices must be contiguous and ordered")
        expected_references.add(_safe_path(_string(entry["data_path"], "data_path")))
        expected_references.add(_safe_path(_string(entry["metadata_path"], "metadata_path")))
        previews = _string_mapping(entry["preview_paths"], "preview_paths")
        expected_references.update(_safe_path(path) for path in previews.values())
    global_previews = manifest["global_preview_paths"]
    if not isinstance(global_previews, list):
        raise ArtifactError("global_preview_paths must be an array")
    expected_references.update(
        _safe_path(_string(path, "global preview")) for path in global_previews
    )
    if referenced != expected_references:
        raise ArtifactError("manifest checksum paths differ from referenced artifact files")
    for relative, expected_hash in asset_hashes.items():
        path = _resolved_file(artifact_root, relative)
        if sha256_file(path) != expected_hash:
            raise ArtifactError(f"checksum mismatch: {relative}")
    checksum_entries = _load_checksum_file(artifact_root / "checksums.sha256")
    expected_checksum_paths = set(asset_hashes) | {"manifest.json"}
    if set(checksum_entries) != expected_checksum_paths:
        raise ArtifactError("checksums.sha256 contains missing or unexpected paths")
    for relative, expected_hash in checksum_entries.items():
        if sha256_file(_resolved_file(artifact_root, relative)) != expected_hash:
            raise ArtifactError(f"checksum mismatch: {relative}")
    allowed_files = expected_checksum_paths | {"checksums.sha256"}
    actual_files: set[str] = set()
    for path in artifact_root.rglob("*"):
        if path.is_symlink():
            raise ArtifactError(f"capture artifact must not contain symlinks: {path.name}")
        if path.is_file():
            actual_files.add(path.relative_to(artifact_root).as_posix())
    if actual_files != allowed_files:
        raise ArtifactError("capture artifact contains missing or unexpected files")
    factory_artifact = load_and_validate_factory_calibration(
        _resolved_file(artifact_root, _string(manifest["factory_calibration"], "factory"))
    )
    factory_device = factory_artifact.calibration.device
    if (
        factory_device.camera_name != manifest["camera"]
        or factory_device.serial != manifest["serial"]
    ):
        raise ArtifactError("factory calibration camera identity differs from manifest")
    for entry_value in frames:
        entry = _object(entry_value, "frames[]")
        data_path = _resolved_file(artifact_root, _string(entry["data_path"], "data_path"))
        arrays: dict[str, npt.NDArray[np.generic]] = {}
        try:
            with np.load(data_path, allow_pickle=False) as archive:
                if set(archive.files) != {"color", "depth", "ir_left", "ir_right"}:
                    raise ArtifactError("frame NPZ must contain exactly four raw streams")
                for name in archive.files:
                    arrays[name] = np.asarray(archive[name]).copy()
        except (OSError, ValueError) as error:
            raise ArtifactError(f"could not load frame NPZ: {error}") from error
        metadata_path = _resolved_file(
            artifact_root, _string(entry["metadata_path"], "metadata_path")
        )
        metadata = load_json(metadata_path)
        if not isinstance(metadata, dict):
            raise ArtifactError("frame metadata must be a JSON object")
        try:
            restored = restore_camera_frame(dict(metadata), arrays)
        except (ContractError, TransformError) as error:
            raise ArtifactError(f"invalid frame contract: {error}") from error
        if restored.camera_name != manifest["camera"] or restored.serial != manifest["serial"]:
            raise ArtifactError("frame camera identity differs from manifest")
    return manifest


def _safe_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ArtifactError(f"unsafe relative path: {value!r}")
    if path.as_posix() != value or value.startswith("file://"):
        raise ArtifactError(f"unsafe relative path: {value!r}")
    return value


def _reject_absolute_values(value: object, path: str) -> None:
    if isinstance(value, str):
        if value.startswith("/") or value.casefold().startswith("file://"):
            raise ArtifactError(f"{path} must not contain absolute paths")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_absolute_values(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_absolute_values(item, f"{path}[{index}]")


def _resolved_file(root: Path, relative: str) -> Path:
    safe = _safe_path(relative)
    path = root / safe
    current = root
    for part in PurePosixPath(safe).parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactError(f"symlink escape is not allowed: {relative}")
    if not path.is_file():
        raise ArtifactError(f"missing artifact file: {relative}")
    return path


def _load_checksum_file(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactError("missing checksums.sha256")
    entries: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ArtifactError(f"could not read checksums.sha256: {error}") from error
    for line in lines:
        parts = line.split("  ", maxsplit=1)
        digest = parts[0]
        if (
            len(parts) != 2
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ArtifactError("invalid checksums.sha256 line")
        relative = _safe_path(parts[1])
        if relative in entries:
            raise ArtifactError("duplicate checksums.sha256 path")
        entries[relative] = digest
    return entries


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ArtifactError(f"{name} must be an object with string keys")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ArtifactError(f"{name} must be a string")
    return value


def _string_mapping(value: object, name: str) -> dict[str, str]:
    mapping = _object(value, name)
    return {
        _string(key, f"{name} key"): _string(item, f"{name} value") for key, item in mapping.items()
    }
