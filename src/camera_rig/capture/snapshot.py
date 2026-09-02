"""Atomic raw CameraFrame snapshot writer."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from camera_rig.artifacts.capture_io import camera_frame_metadata
from camera_rig.artifacts.capture_validation import validate_capture_artifact
from camera_rig.artifacts.factory_calibration import (
    FactoryCalibrationArtifact,
    write_factory_calibration,
)
from camera_rig.artifacts.hashing import sha256_file
from camera_rig.artifacts.io import atomic_write_json
from camera_rig.core.errors import ArtifactError, MissingOptionalDependencyError
from camera_rig.core.frame import CameraFrame


def write_snapshot(
    output: str | Path,
    frames: Sequence[CameraFrame],
    factory_calibration: FactoryCalibrationArtifact,
    capture_summary: dict[str, object],
    provenance: dict[str, object],
    *,
    include_previews: bool = True,
    force: bool = True,
) -> dict[str, object]:
    """Write, self-validate, then atomically commit a capture directory."""
    if not frames:
        raise ArtifactError("snapshot requires at least one CameraFrame")
    if any(frame.sync_report is None or not frame.sync_report.valid for frame in frames):
        raise ArtifactError("snapshot requires frames with valid single-device sync reports")
    target = Path(output)
    if target.exists() and not force:
        raise ArtifactError(f"capture artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.tmp-"))
    try:
        manifest = _write_contents(
            temporary,
            frames,
            factory_calibration,
            capture_summary,
            provenance,
            include_previews,
        )
        validate_capture_artifact(temporary)
        _commit_directory(temporary, target, force)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _write_contents(
    root: Path,
    frames: Sequence[CameraFrame],
    factory_calibration: FactoryCalibrationArtifact,
    capture_summary: dict[str, object],
    provenance: dict[str, object],
    include_previews: bool,
) -> dict[str, object]:
    (root / "frames").mkdir()
    if include_previews:
        (root / "previews").mkdir()
    factory_path = "factory_calibration.json"
    write_factory_calibration(root / factory_path, factory_calibration)
    hashes: dict[str, str] = {factory_path: sha256_file(root / factory_path)}
    entries: list[dict[str, object]] = []
    first_preview_images: dict[str, Any] = {}
    for index, frame in enumerate(frames):
        required = {"color", "depth", "ir_left", "ir_right"}
        if set(frame.streams) != required:
            raise ArtifactError("snapshot frames must contain exactly four raw streams")
        stem = f"frame_{index:06d}"
        data_path = f"frames/{stem}.npz"
        metadata_path = f"frames/{stem}.meta.json"
        np.savez_compressed(
            root / data_path,
            color=frame.streams["color"].data,
            depth=frame.streams["depth"].data,
            ir_left=frame.streams["ir_left"].data,
            ir_right=frame.streams["ir_right"].data,
        )
        atomic_write_json(root / metadata_path, camera_frame_metadata(frame))
        preview_paths: dict[str, str] = {}
        if include_previews:
            images = _write_frame_previews(root, stem, frame)
            for name, image in images.items():
                relative = f"previews/{stem}_{name}.png"
                preview_paths[name] = relative
                hashes[relative] = sha256_file(root / relative)
                if index == 0:
                    first_preview_images[name] = image
        hashes[data_path] = sha256_file(root / data_path)
        hashes[metadata_path] = sha256_file(root / metadata_path)
        entries.append(
            {
                "index": index,
                "data_path": data_path,
                "metadata_path": metadata_path,
                "preview_paths": preview_paths,
            }
        )
    global_previews: list[str] = []
    if include_previews:
        mosaic_path = "previews/mosaic.png"
        _write_mosaic(root / mosaic_path, first_preview_images)
        hashes[mosaic_path] = sha256_file(root / mosaic_path)
        global_previews.append(mosaic_path)
    manifest: dict[str, object] = {
        "schema_version": "camera-rig.capture.v1",
        "artifact_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "camera": frames[0].camera_name,
        "serial": frames[0].serial,
        "capture_configuration": capture_summary,
        "factory_calibration": factory_path,
        "frame_count": len(frames),
        "frames": entries,
        "global_preview_paths": global_previews,
        "quality": {
            "passed": all(
                frame.sync_report is not None and frame.sync_report.valid for frame in frames
            ),
            "validated_frame_count": len(frames),
        },
        "provenance": provenance,
        "checksums": dict(sorted(hashes.items())),
        "preview_notice": "PNG files are derived diagnostics and are never replay data",
    }
    atomic_write_json(root / "manifest.json", manifest)
    checksum_hashes = dict(hashes)
    checksum_hashes["manifest.json"] = sha256_file(root / "manifest.json")
    lines = [f"{digest}  {relative}" for relative, digest in sorted(checksum_hashes.items())]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _write_frame_previews(root: Path, stem: str, frame: CameraFrame) -> dict[str, Any]:
    image_module = _pillow_image()
    color = image_module.fromarray(frame.streams["color"].data, mode="RGB")
    depth = image_module.fromarray(_depth_preview(frame.streams["depth"].data), mode="L")
    ir_left = image_module.fromarray(frame.streams["ir_left"].data, mode="L")
    ir_right = image_module.fromarray(frame.streams["ir_right"].data, mode="L")
    images = {"color": color, "depth": depth, "ir_left": ir_left, "ir_right": ir_right}
    for name, image in images.items():
        image.save(root / f"previews/{stem}_{name}.png", format="PNG")
    return images


def _depth_preview(depth: npt.NDArray[np.generic]) -> npt.NDArray[np.uint8]:
    numeric = np.asarray(depth, dtype=np.uint16)
    valid = numeric[numeric > 0]
    result = np.zeros(numeric.shape, dtype=np.uint8)
    if valid.size == 0:
        return result
    lower, upper = np.percentile(valid, [2.0, 98.0])
    if upper <= lower:
        result[numeric > 0] = 255
        return result
    scaled = np.clip((numeric.astype(np.float64) - lower) / (upper - lower), 0.0, 1.0)
    result[numeric > 0] = np.asarray(1 + scaled[numeric > 0] * 254, dtype=np.uint8)
    return result


def _write_mosaic(path: Path, images: dict[str, Any]) -> None:
    image_module = _pillow_image()
    ordered = [images[name].convert("RGB") for name in ("color", "depth", "ir_left", "ir_right")]
    width, height = ordered[0].size
    mosaic = image_module.new("RGB", (width * 2, height * 2))
    for image, position in zip(
        ordered, ((0, 0), (width, 0), (0, height), (width, height)), strict=True
    ):
        mosaic.paste(image, position)
    mosaic.save(path, format="PNG")


def _pillow_image() -> Any:
    try:
        from PIL import Image
    except ImportError as error:
        raise MissingOptionalDependencyError(
            'snapshot previews require: pip install "camera-rig[viz]"'
        ) from error
    return Image


def _commit_directory(temporary: Path, target: Path, force: bool) -> None:
    if not target.exists():
        os.replace(temporary, target)
        return
    if not force:
        raise ArtifactError(f"capture artifact already exists: {target}")
    if target.is_symlink() or not target.is_dir():
        raise ArtifactError("capture output must be a real artifact directory")
    backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
    os.replace(target, backup)
    try:
        os.replace(temporary, target)
    except Exception:
        os.replace(backup, target)
        raise
    shutil.rmtree(backup)
