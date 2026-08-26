"""Strict fixed-camera calibration artifact contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Final

import numpy as np

from camera_rig.artifacts.io import JsonValue, atomic_write_json, json_safe, load_json
from camera_rig.config.validation import validate_against_named_schema
from camera_rig.core.errors import (
    ArtifactError,
    ContractError,
    SchemaValidationError,
    TransformError,
)
from camera_rig.core.fixed_mount import FixedMountCalibration
from camera_rig.core.quality import QualityReport
from camera_rig.core.transforms import RigidTransform

FIXED_CALIBRATION_SCHEMA_VERSION: Final = "camera-rig.fixed-calibration.v1"


@dataclass(frozen=True)
class FixedCalibrationArtifact:
    """Portable, frame-explicit result of a robust fixed-pose solve."""

    created_at: str
    workspace: dict[str, object]
    camera: dict[str, object]
    target: dict[str, object]
    inputs: dict[str, object]
    solver: dict[str, object]
    per_frame_pose_summary: tuple[dict[str, object], ...]
    aggregate: dict[str, object]
    T_detection_from_target: RigidTransform
    T_workspace_from_detection: RigidTransform
    T_detection_from_reference: RigidTransform
    T_workspace_from_reference: RigidTransform
    fixed_mount_calibration: FixedMountCalibration
    quality: QualityReport
    provenance: dict[str, object]
    schema_version: str = field(default=FIXED_CALIBRATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ContractError("created_at must be an ISO-8601 date-time") from error
        mappings = (
            "workspace",
            "camera",
            "target",
            "inputs",
            "solver",
            "aggregate",
            "provenance",
        )
        for name in mappings:
            value = dict(getattr(self, name))
            _reject_unsafe_paths(value, name)
            object.__setattr__(self, name, value)
        frames = tuple(dict(item) for item in self.per_frame_pose_summary)
        object.__setattr__(self, "per_frame_pose_summary", frames)

        workspace_frame = _string(self.workspace.get("frame"), "workspace.frame")
        target_frame = _string(self.workspace.get("target_frame"), "workspace.target_frame")
        detection_frame = _string(self.camera.get("detection_frame"), "camera.detection_frame")
        reference_frame = _string(self.camera.get("reference_frame"), "camera.reference_frame")
        T_workspace_from_target = RigidTransform.from_dict(
            _object(self.workspace.get("T_workspace_from_target"), "T_workspace_from_target")
        )
        _require_frames(T_workspace_from_target, target_frame, workspace_frame)
        _require_frames(self.T_detection_from_target, target_frame, detection_frame)
        _require_frames(self.T_workspace_from_detection, detection_frame, workspace_frame)
        _require_frames(self.T_detection_from_reference, reference_frame, detection_frame)
        _require_frames(self.T_workspace_from_reference, reference_frame, workspace_frame)

        expected_detection = T_workspace_from_target.compose(self.T_detection_from_target.inverse())
        expected_reference = self.T_workspace_from_detection.compose(
            self.T_detection_from_reference
        )
        if not np.allclose(
            expected_detection.matrix,
            self.T_workspace_from_detection.matrix,
            rtol=0.0,
            atol=1e-8,
        ):
            raise ContractError("workspace/detection transform chain is inconsistent")
        if not np.allclose(
            expected_reference.matrix,
            self.T_workspace_from_reference.matrix,
            rtol=0.0,
            atol=1e-8,
        ):
            raise ContractError("workspace/reference transform chain is inconsistent")
        fixed = self.fixed_mount_calibration
        if fixed.parent_frame != workspace_frame or fixed.camera_reference_frame != reference_frame:
            raise ContractError("fixed mount frames do not match artifact camera/workspace")
        if not np.allclose(
            fixed.T_parent_from_camera_reference.matrix,
            self.T_workspace_from_reference.matrix,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ContractError("fixed mount transform differs from artifact result")
        if self.quality.passed != fixed.quality.passed:
            raise ContractError("artifact and fixed-mount quality decisions must match")
        json_safe(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "workspace": dict(self.workspace),
            "camera": dict(self.camera),
            "target": dict(self.target),
            "inputs": dict(self.inputs),
            "solver": dict(self.solver),
            "per_frame_pose_summary": [dict(item) for item in self.per_frame_pose_summary],
            "aggregate": dict(self.aggregate),
            "T_detection_from_target": self.T_detection_from_target.to_dict(),
            "T_workspace_from_detection": self.T_workspace_from_detection.to_dict(),
            "T_detection_from_reference": self.T_detection_from_reference.to_dict(),
            "T_workspace_from_reference": self.T_workspace_from_reference.to_dict(),
            "fixed_mount_calibration": self.fixed_mount_calibration.to_dict(),
            "quality": self.quality.to_dict(),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> FixedCalibrationArtifact:
        if data.get("schema_version") != FIXED_CALIBRATION_SCHEMA_VERSION:
            raise ContractError(f"schema_version must be {FIXED_CALIBRATION_SCHEMA_VERSION!r}")
        per_frame = _array(data["per_frame_pose_summary"], "per_frame_pose_summary")
        return cls(
            created_at=_string(data["created_at"], "created_at"),
            workspace=_object(data["workspace"], "workspace"),
            camera=_object(data["camera"], "camera"),
            target=_object(data["target"], "target"),
            inputs=_object(data["inputs"], "inputs"),
            solver=_object(data["solver"], "solver"),
            per_frame_pose_summary=tuple(
                _object(item, "per_frame_pose_summary[]") for item in per_frame
            ),
            aggregate=_object(data["aggregate"], "aggregate"),
            T_detection_from_target=RigidTransform.from_dict(
                _object(data["T_detection_from_target"], "T_detection_from_target")
            ),
            T_workspace_from_detection=RigidTransform.from_dict(
                _object(data["T_workspace_from_detection"], "T_workspace_from_detection")
            ),
            T_detection_from_reference=RigidTransform.from_dict(
                _object(data["T_detection_from_reference"], "T_detection_from_reference")
            ),
            T_workspace_from_reference=RigidTransform.from_dict(
                _object(data["T_workspace_from_reference"], "T_workspace_from_reference")
            ),
            fixed_mount_calibration=FixedMountCalibration.from_dict(
                _object(data["fixed_mount_calibration"], "fixed_mount_calibration")
            ),
            quality=QualityReport.from_dict(_object(data["quality"], "quality")),
            provenance=_object(data["provenance"], "provenance"),
        )


def write_fixed_calibration(path: str | Path, artifact: FixedCalibrationArtifact) -> None:
    """Atomically write and immediately reload a fixed-calibration artifact."""
    atomic_write_json(path, artifact.to_dict())
    load_and_validate_fixed_calibration(path)


def validate_fixed_calibration_data(value: JsonValue) -> FixedCalibrationArtifact:
    if not isinstance(value, dict):
        raise ArtifactError("fixed calibration root must be a JSON object")
    try:
        validate_against_named_schema(value, "fixed_calibration.v1.schema.json")
        artifact = FixedCalibrationArtifact.from_dict(dict(value))
        if not artifact.quality.passed:
            raise ArtifactError("fixed calibration quality is not passed")
        return artifact
    except SchemaValidationError as error:
        raise ArtifactError(str(error)) from error
    except ArtifactError:
        raise
    except (KeyError, TypeError, ValueError, ContractError, TransformError) as error:
        raise ArtifactError(f"fixed calibration contract is invalid: {error}") from error


def load_and_validate_fixed_calibration(path: str | Path) -> FixedCalibrationArtifact:
    return validate_fixed_calibration_data(load_json(path))


def _require_frames(transform: RigidTransform, source: str, target: str) -> None:
    if (transform.source_frame, transform.target_frame) != (source, target):
        raise ContractError(
            f"transform frames must be {source!r} -> {target!r}, got "
            f"{transform.source_frame!r} -> {transform.target_frame!r}"
        )


def _reject_unsafe_paths(value: object, path: str) -> None:
    if isinstance(value, str):
        if value.casefold().startswith("file://") or value.startswith("/"):
            raise ContractError(f"{path} must not contain absolute paths")
        if ("path" in path.casefold() or "file" in path.casefold()) and (
            "\\" in value or ".." in PurePosixPath(value).parts
        ):
            raise ContractError(f"{path} must contain only safe relative POSIX paths")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_unsafe_paths(item, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _reject_unsafe_paths(item, f"{path}[{index}]")


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be an object with string keys")
    return value


def _array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ContractError(f"{name} must be an array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a non-empty string")
    return value
