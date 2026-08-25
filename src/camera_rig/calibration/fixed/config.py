"""Strict fixed-camera calibration configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from camera_rig.artifacts.io import json_safe
from camera_rig.config.validation import validate_against_named_schema
from camera_rig.core.errors import ArtifactError, ConfigurationError, ContractError
from camera_rig.core.transforms import RigidTransform

FIXED_CALIBRATION_CONFIG_SCHEMA_VERSION = "camera-rig.fixed-calibration-config.v1"


@dataclass(frozen=True)
class FixedSolverThresholds:
    """Numerical solver and quality thresholds persisted with each result."""

    method: str
    refinement: str
    minimum_corners_per_frame: int
    minimum_accepted_frames: int
    minimum_accepted_ratio: float
    maximum_frame_rmse_px: float
    maximum_frame_p95_px: float
    maximum_pose_translation_p95_mm: float
    maximum_pose_rotation_p95_deg: float
    maximum_split_translation_delta_mm: float
    maximum_split_rotation_delta_deg: float
    pose_outlier_translation_mm: float = 5.0
    pose_outlier_rotation_deg: float = 0.5

    def __post_init__(self) -> None:
        if self.method != "ippe" or self.refinement != "lm":
            raise ContractError("fixed calibration requires IPPE with LM refinement")
        if self.minimum_corners_per_frame < 4 or self.minimum_accepted_frames < 2:
            raise ContractError("fixed calibration frame thresholds are too small")
        if not 0 < self.minimum_accepted_ratio <= 1:
            raise ContractError("minimum_accepted_ratio must lie in (0, 1]")
        for name in (
            "maximum_frame_rmse_px",
            "maximum_frame_p95_px",
            "maximum_pose_translation_p95_mm",
            "maximum_pose_rotation_p95_deg",
            "maximum_split_translation_delta_mm",
            "maximum_split_rotation_delta_deg",
            "pose_outlier_translation_mm",
            "pose_outlier_rotation_deg",
        ):
            if float(getattr(self, name)) <= 0:
                raise ContractError(f"{name} must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "refinement": self.refinement,
            "minimum_corners_per_frame": self.minimum_corners_per_frame,
            "minimum_accepted_frames": self.minimum_accepted_frames,
            "minimum_accepted_ratio": self.minimum_accepted_ratio,
            "maximum_frame_rmse_px": self.maximum_frame_rmse_px,
            "maximum_frame_p95_px": self.maximum_frame_p95_px,
            "maximum_pose_translation_p95_mm": self.maximum_pose_translation_p95_mm,
            "maximum_pose_rotation_p95_deg": self.maximum_pose_rotation_p95_deg,
            "maximum_split_translation_delta_mm": self.maximum_split_translation_delta_mm,
            "maximum_split_rotation_delta_deg": self.maximum_split_rotation_delta_deg,
            "pose_outlier_translation_mm": self.pose_outlier_translation_mm,
            "pose_outlier_rotation_deg": self.pose_outlier_rotation_deg,
        }


@dataclass(frozen=True)
class FixedCalibrationConfig:
    """Complete offline contract for one fixed camera/target pair."""

    workspace_frame: str
    target_frame: str
    T_workspace_from_target: RigidTransform
    detection_stream: str
    reference_stream: str
    solver: FixedSolverThresholds
    native_depth_check: bool
    schema_version: str = FIXED_CALIBRATION_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FIXED_CALIBRATION_CONFIG_SCHEMA_VERSION:
            raise ContractError("unsupported fixed calibration config schema")
        if not self.workspace_frame or not self.target_frame:
            raise ContractError("workspace and target frame names must be non-empty")
        if self.T_workspace_from_target.source_frame != self.target_frame:
            raise ContractError("workspace transform source must equal target_frame")
        if self.T_workspace_from_target.target_frame != self.workspace_frame:
            raise ContractError("workspace transform target must equal workspace frame")
        if self.workspace_frame == self.target_frame:
            raise ContractError("workspace and target use distinct semantic frame names")
        if self.detection_stream not in {"color", "depth", "ir_left", "ir_right"}:
            raise ContractError("unsupported detection stream")
        if self.reference_stream not in {"color", "depth", "ir_left", "ir_right"}:
            raise ContractError("unsupported reference stream")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> FixedCalibrationConfig:
        workspace = _object(data["workspace"], "workspace")
        camera = _object(data["camera"], "camera")
        solver = _object(data["solver"], "solver")
        diagnostics = _object(data["diagnostics"], "diagnostics")
        transform_data = _object(workspace["T_workspace_from_target"], "workspace transform")
        matrix = transform_data["matrix"]
        target_frame = _string(workspace["target_frame"], "workspace.target_frame")
        workspace_frame = _string(workspace["frame"], "workspace.frame")
        return cls(
            schema_version=_string(data["schema_version"], "schema_version"),
            workspace_frame=workspace_frame,
            target_frame=target_frame,
            T_workspace_from_target=RigidTransform(
                source_frame=target_frame,
                target_frame=workspace_frame,
                matrix=matrix,  # type: ignore[arg-type]
            ),
            detection_stream=_string(camera["detection_stream"], "camera.detection_stream"),
            reference_stream=_string(camera["reference_stream"], "camera.reference_stream"),
            solver=FixedSolverThresholds(
                method=_string(solver["method"], "solver.method"),
                refinement=_string(solver["refinement"], "solver.refinement"),
                minimum_corners_per_frame=_int(
                    solver["minimum_corners_per_frame"], "solver.minimum_corners_per_frame"
                ),
                minimum_accepted_frames=_int(
                    solver["minimum_accepted_frames"], "solver.minimum_accepted_frames"
                ),
                minimum_accepted_ratio=_float(
                    solver["minimum_accepted_ratio"], "solver.minimum_accepted_ratio"
                ),
                maximum_frame_rmse_px=_float(
                    solver["maximum_frame_rmse_px"], "solver.maximum_frame_rmse_px"
                ),
                maximum_frame_p95_px=_float(
                    solver["maximum_frame_p95_px"], "solver.maximum_frame_p95_px"
                ),
                maximum_pose_translation_p95_mm=_float(
                    solver["maximum_pose_translation_p95_mm"],
                    "solver.maximum_pose_translation_p95_mm",
                ),
                maximum_pose_rotation_p95_deg=_float(
                    solver["maximum_pose_rotation_p95_deg"],
                    "solver.maximum_pose_rotation_p95_deg",
                ),
                maximum_split_translation_delta_mm=_float(
                    solver["maximum_split_translation_delta_mm"],
                    "solver.maximum_split_translation_delta_mm",
                ),
                maximum_split_rotation_delta_deg=_float(
                    solver["maximum_split_rotation_delta_deg"],
                    "solver.maximum_split_rotation_delta_deg",
                ),
                pose_outlier_translation_mm=_float(
                    solver.get("pose_outlier_translation_mm", 5.0),
                    "solver.pose_outlier_translation_mm",
                ),
                pose_outlier_rotation_deg=_float(
                    solver.get("pose_outlier_rotation_deg", 0.5),
                    "solver.pose_outlier_rotation_deg",
                ),
            ),
            native_depth_check=_bool(
                diagnostics["native_depth_check"], "diagnostics.native_depth_check"
            ),
        )


def load_fixed_config(path: str | Path) -> FixedCalibrationConfig:
    """Load a strict fixed-calibration YAML contract."""
    source = Path(path)
    try:
        decoded: object = yaml.safe_load(source.read_text(encoding="utf-8"))
        value = json_safe(decoded)
        validate_against_named_schema(value, "fixed_calibration_config.v1.schema.json")
        if not isinstance(value, dict):
            raise ConfigurationError("fixed calibration config root must be a mapping")
        return FixedCalibrationConfig.from_dict(dict(value))
    except (OSError, UnicodeError, yaml.YAMLError, ArtifactError) as error:
        raise ConfigurationError(f"could not load fixed calibration config: {error}") from error
    except (KeyError, TypeError, ValueError, ContractError) as error:
        raise ConfigurationError(f"fixed calibration config is invalid: {error}") from error


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")
    return value


def _int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an integer")
    return value


def _float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"{name} must be a number")
    return float(value)


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be a boolean")
    return value
