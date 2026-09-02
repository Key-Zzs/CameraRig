"""Strict target-detection artifact contract and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Final

from camera_rig.artifacts.io import JsonValue, atomic_write_json, json_safe, load_json
from camera_rig.config.validation import validate_against_named_schema
from camera_rig.core._validation import require_non_empty, string_keyed_copy
from camera_rig.core.errors import ArtifactError, ContractError, SchemaValidationError
from camera_rig.targets.observation import TargetObservation

TARGET_DETECTION_SCHEMA_VERSION: Final = "camera-rig.target-detection.v1"
_STREAM_NAMES: Final = frozenset({"color", "depth", "ir_left", "ir_right"})


@dataclass(frozen=True)
class TargetDetectionFrame:
    """One persisted detector result and its source-frame identity."""

    frame_index: int
    success: bool
    observation: TargetObservation
    overlay: str | None = None
    pose_diagnostic: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise ContractError("frame_index must be an integer")
        if self.frame_index < 0:
            raise ContractError("frame_index must be non-negative")
        if not isinstance(self.success, bool):
            raise ContractError("success must be a boolean")
        if self.success != self.observation.quality.passed:
            raise ContractError("frame success must match observation quality")
        if self.overlay is not None:
            _require_relative_path(self.overlay, "overlay")
        diagnostic = (
            string_keyed_copy(self.pose_diagnostic, "pose_diagnostic")
            if self.pose_diagnostic is not None
            else None
        )
        object.__setattr__(self, "pose_diagnostic", diagnostic)

    def to_dict(self, *, include_overlay: bool) -> dict[str, object]:
        result: dict[str, object] = {
            "frame_index": self.frame_index,
            "success": self.success,
            "observation": self.observation.to_dict(),
        }
        if include_overlay:
            result["overlay"] = self.overlay
        if self.pose_diagnostic is not None:
            result["pose_diagnostic"] = dict(self.pose_diagnostic)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TargetDetectionFrame:
        expected = {"frame_index", "success", "observation"}
        allowed = expected | {"overlay", "pose_diagnostic"}
        if not expected <= set(data) or not set(data) <= allowed:
            raise ContractError("target detection frame has missing or unknown fields")
        overlay = data.get("overlay")
        if overlay is not None and not isinstance(overlay, str):
            raise ContractError("overlay must be a string or null")
        return cls(
            frame_index=_integer(data["frame_index"], "frame_index"),
            success=_boolean(data["success"], "success"),
            observation=TargetObservation.from_dict(_object(data["observation"], "observation")),
            overlay=overlay,
            pose_diagnostic=(
                _object(data["pose_diagnostic"], "pose_diagnostic")
                if "pose_diagnostic" in data
                else None
            ),
        )


@dataclass(frozen=True)
class TargetDetectionArtifact:
    """Portable R6 observations bound to their target and input source hashes."""

    target_spec_sha256: str
    frame_count: int
    per_frame: tuple[TargetDetectionFrame, ...]
    aggregate: dict[str, object]
    software: dict[str, str]
    capture_manifest_sha256: str | None = None
    stream: str | None = None
    input_image_sha256: str | None = None
    acceptance: dict[str, object] | None = None
    selected_overlays: dict[str, int] | None = None
    schema_version: str = field(default=TARGET_DETECTION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        _require_digest(self.target_spec_sha256, "target_spec_sha256")
        if isinstance(self.frame_count, bool) or not isinstance(self.frame_count, int):
            raise ContractError("frame_count must be an integer")
        frames = tuple(self.per_frame)
        if self.frame_count < 1 or self.frame_count != len(frames):
            raise ContractError("frame_count must equal the non-empty per_frame length")
        if tuple(frame.frame_index for frame in frames) != tuple(range(self.frame_count)):
            raise ContractError("per_frame indices must be contiguous and start at zero")
        target_frames = {frame.observation.target_frame for frame in frames}
        if len(target_frames) != 1:
            raise ContractError("all observations must use one target frame")
        for frame in frames:
            observed_sha = frame.observation.metadata.get("target_spec_sha256")
            if observed_sha != self.target_spec_sha256:
                raise ContractError(
                    "observation target_spec_sha256 must match the report target identity"
                )

        is_capture = self.capture_manifest_sha256 is not None or self.stream is not None
        is_image = self.input_image_sha256 is not None
        if is_capture == is_image:
            raise ContractError("target detection must have exactly one input source")
        if is_capture:
            if self.capture_manifest_sha256 is None or self.stream is None:
                raise ContractError("capture target detection requires manifest SHA and stream")
            _require_digest(self.capture_manifest_sha256, "capture_manifest_sha256")
            require_non_empty(self.stream, "stream")
            if self.stream not in _STREAM_NAMES:
                raise ContractError(f"stream must be one of {sorted(_STREAM_NAMES)}")
            if self.acceptance is None or self.selected_overlays is None:
                raise ContractError("capture target detection requires acceptance and overlays")
        else:
            if self.input_image_sha256 is None:
                raise ContractError("image target detection requires input image SHA")
            _require_digest(self.input_image_sha256, "input_image_sha256")
            if self.acceptance is not None or self.selected_overlays is not None:
                raise ContractError("image target detection cannot contain capture acceptance")

        aggregate = string_keyed_copy(self.aggregate, "aggregate")
        software = dict(self.software)
        if set(software) != {"camera_rig_version", "opencv_version"}:
            raise ContractError("software provenance has missing or unknown fields")
        for name, value in software.items():
            require_non_empty(value, f"software.{name}")
        expected_success_ratio = sum(frame.success for frame in frames) / self.frame_count
        success_ratio = _number(aggregate.get("success_ratio"), "aggregate.success_ratio")
        if not math.isclose(success_ratio, expected_success_ratio, rel_tol=0.0, abs_tol=1e-12):
            raise ContractError("aggregate success_ratio does not match per_frame results")

        acceptance = None
        if self.acceptance is not None:
            acceptance = string_keyed_copy(self.acceptance, "acceptance")
            checks = _object(acceptance.get("checks"), "acceptance.checks")
            passed = _boolean(acceptance.get("passed"), "acceptance.passed")
            if passed != all(_boolean(value, "acceptance.checks[]") for value in checks.values()):
                raise ContractError("acceptance passed must equal all acceptance checks")

        selected = None
        if self.selected_overlays is not None:
            selected = {}
            for label, index in self.selected_overlays.items():
                require_non_empty(label, "selected_overlays key")
                validated_index = _integer(index, f"selected_overlays.{label}")
                if validated_index < 0 or validated_index >= self.frame_count:
                    raise ContractError(f"selected_overlays.{label} is outside per_frame")
                overlay = frames[validated_index].overlay
                if overlay is None:
                    raise ContractError(
                        f"selected_overlays.{label} points to a frame without overlay"
                    )
                selected[label] = validated_index

        object.__setattr__(self, "per_frame", frames)
        object.__setattr__(self, "aggregate", aggregate)
        object.__setattr__(self, "software", software)
        object.__setattr__(self, "acceptance", acceptance)
        object.__setattr__(self, "selected_overlays", selected)
        json_safe(self.to_dict())

    @property
    def is_capture(self) -> bool:
        """Whether this report was generated from a capture manifest."""
        return self.capture_manifest_sha256 is not None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "target_spec_sha256": self.target_spec_sha256,
            "frame_count": self.frame_count,
            "per_frame": [
                frame.to_dict(include_overlay=self.is_capture) for frame in self.per_frame
            ],
            "aggregate": dict(self.aggregate),
            "software": dict(self.software),
        }
        if self.is_capture:
            result["input_artifact"] = {
                "manifest_sha256": self.capture_manifest_sha256,
                "stream": self.stream,
            }
            result["acceptance"] = dict(self.acceptance or {})
            result["selected_overlays"] = dict(self.selected_overlays or {})
        else:
            result["input_image_sha256"] = self.input_image_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TargetDetectionArtifact:
        if data.get("schema_version") != TARGET_DETECTION_SCHEMA_VERSION:
            raise ContractError(f"schema_version must be {TARGET_DETECTION_SCHEMA_VERSION!r}")
        base = {
            "schema_version",
            "target_spec_sha256",
            "frame_count",
            "per_frame",
            "aggregate",
            "software",
        }
        capture_fields = {"input_artifact", "acceptance", "selected_overlays"}
        image_fields = {"input_image_sha256"}
        fields = set(data)
        if fields == base | capture_fields:
            input_artifact = _object(data["input_artifact"], "input_artifact")
            if set(input_artifact) != {"manifest_sha256", "stream"}:
                raise ContractError("input_artifact has missing or unknown fields")
            capture_manifest_sha256 = _string(
                input_artifact["manifest_sha256"], "input_artifact.manifest_sha256"
            )
            stream = _string(input_artifact["stream"], "input_artifact.stream")
            input_image_sha256 = None
            acceptance = _object(data["acceptance"], "acceptance")
            selected_value = _object(data["selected_overlays"], "selected_overlays")
            selected_overlays = {
                key: _integer(value, f"selected_overlays.{key}")
                for key, value in selected_value.items()
            }
        elif fields == base | image_fields:
            capture_manifest_sha256 = None
            stream = None
            input_image_sha256 = _string(data["input_image_sha256"], "input_image_sha256")
            acceptance = None
            selected_overlays = None
        else:
            raise ContractError("target detection artifact has missing or unknown fields")

        per_frame_value = _array(data["per_frame"], "per_frame")
        software_value = _object(data["software"], "software")
        return cls(
            target_spec_sha256=_string(data["target_spec_sha256"], "target_spec_sha256"),
            capture_manifest_sha256=capture_manifest_sha256,
            stream=stream,
            input_image_sha256=input_image_sha256,
            frame_count=_integer(data["frame_count"], "frame_count"),
            per_frame=tuple(
                TargetDetectionFrame.from_dict(_object(value, "per_frame[]"))
                for value in per_frame_value
            ),
            aggregate=_object(data["aggregate"], "aggregate"),
            acceptance=acceptance,
            selected_overlays=selected_overlays,
            software={
                key: _string(value, f"software.{key}") for key, value in software_value.items()
            },
        )


def write_target_detection(path: str | Path, artifact: TargetDetectionArtifact) -> None:
    """Atomically persist and immediately revalidate a target-detection artifact."""
    value = json_safe(artifact.to_dict())
    validate_target_detection_data(value)
    atomic_write_json(path, value)
    load_and_validate_target_detection(path)


def validate_target_detection_data(value: JsonValue) -> TargetDetectionArtifact:
    """Schema-validate decoded JSON and reconstruct every target observation."""
    if not isinstance(value, dict):
        raise ArtifactError("target detection root must be a JSON object")
    try:
        validate_against_named_schema(value, "target_detection.v1.schema.json")
        return TargetDetectionArtifact.from_dict(dict(value))
    except SchemaValidationError as error:
        raise ArtifactError(str(error)) from error
    except (KeyError, TypeError, ValueError, ContractError) as error:
        raise ArtifactError(f"target detection contract is invalid: {error}") from error


def load_and_validate_target_detection(path: str | Path) -> TargetDetectionArtifact:
    """Load and strictly validate a target-detection artifact."""
    return validate_target_detection_data(load_json(path))


def _require_digest(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")


def _require_relative_path(value: str, name: str) -> None:
    require_non_empty(value, name)
    if "\\" in value:
        raise ContractError(f"{name} must use a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ContractError(f"{name} must be a safe relative path")


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be an object with string keys")
    return value


def _array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ContractError(f"{name} must be an array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be a boolean")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"{name} must be a number")
    return float(value)
