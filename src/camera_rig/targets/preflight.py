"""Live pose-free target deployment preflight."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from camera_rig.artifacts.hashing import sha256_file
from camera_rig.artifacts.io import atomic_write_json
from camera_rig.capture.session import CameraSession
from camera_rig.config.models import CameraConfig
from camera_rig.core.errors import ArtifactError, ContractError
from camera_rig.core.factory_calibration import FactoryCalibration
from camera_rig.core.frame import CameraFrame
from camera_rig.targets.charuco.detector import CharucoDetector
from camera_rig.targets.charuco.overlay import write_overlay
from camera_rig.targets.charuco.quality import CharucoQualityThresholds
from camera_rig.targets.io import validate_target_artifact
from camera_rig.targets.observation import TargetObservation
from camera_rig.targets.pose_acceptance import (
    aggregate_pose_diagnostics,
    pose_frame_diagnostic,
    uncertainty_capture_acceptance,
)
from camera_rig.version import __version__


class _Session(Protocol):
    def __enter__(self) -> _Session: ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...

    def capture(self) -> CameraFrame: ...

    def get_factory_calibration(self) -> FactoryCalibration: ...


def run_target_preflight(
    *,
    camera_config: CameraConfig,
    target_path: str | Path,
    frames: int,
    stream: str,
    policy: str,
    report_path: str | Path,
    overlays_path: str | Path,
    session_factory: Callable[[CameraConfig], _Session] | None = None,
) -> dict[str, object]:
    """Capture a bounded frame set and assess target deployment without solving pose."""
    if frames < 2:
        raise ContractError("target preflight requires at least two frames")
    if session_factory is None and frames != 60:
        raise ContractError("release target preflight requires exactly 60 frames")
    if stream not in camera_config.streams or not camera_config.streams[stream].enabled:
        raise ContractError(f"target preflight stream {stream!r} must be enabled")
    target = validate_target_artifact(target_path)
    if policy == "pose_validated":
        thresholds = CharucoQualityThresholds.pose_validated()
    elif policy == "uncertainty_validated":
        thresholds = CharucoQualityThresholds.uncertainty_validated()
    else:
        thresholds = CharucoQualityThresholds()
    if policy not in {"legacy_strict", "pose_validated", "uncertainty_validated"}:
        raise ContractError(f"unsupported target preflight policy: {policy!r}")
    detector = CharucoDetector(target, thresholds=thresholds)
    make_session = session_factory or CameraSession.from_config
    observations: list[TargetObservation] = []
    images: list[npt.NDArray[np.uint8]] = []
    intrinsics = None
    with make_session(camera_config) as session:
        if policy == "uncertainty_validated":
            try:
                intrinsics = session.get_factory_calibration().intrinsics[stream]
            except (AttributeError, KeyError, ContractError) as error:
                raise ContractError("POSE_OBSERVABILITY_INTRINSICS_UNAVAILABLE") from error
        for _index in range(frames):
            frame = session.capture()
            if stream not in frame.streams:
                raise ArtifactError(f"captured frame lacks target stream {stream!r}")
            image = np.asarray(frame.streams[stream].data)
            if image.dtype != np.uint8:
                raise ArtifactError("target preflight stream must be uint8")
            observations.append(detector.detect(image))
            images.append(np.asarray(image, dtype=np.uint8).copy())
    pose_diagnostics = (
        [pose_frame_diagnostic(item, intrinsics) for item in observations]
        if intrinsics is not None
        else None
    )
    overlay_root = Path(overlays_path)
    overlay_root.mkdir(parents=True, exist_ok=True)
    selected = {
        "first": 0,
        "middle": len(observations) // 2,
        "last": len(observations) - 1,
    }
    overlay_files: dict[str, str] = {}
    for label, index in selected.items():
        name = f"{label}_frame_{index:06d}.png"
        write_overlay(overlay_root / name, images[index], observations[index])
        overlay_files[label] = name
    metrics = _aggregate(observations)
    acceptance = None
    if pose_diagnostics is not None:
        metrics["pose_observability"] = aggregate_pose_diagnostics(pose_diagnostics)
        acceptance = uncertainty_capture_acceptance(
            aggregate=metrics,
            frame_count=frames,
            minimum_frames=60,
        )
        recommendation = (
            "ADEQUATE_WITH_LOW_COVERAGE_WARNING"
            if acceptance["passed"] is True
            and any(item.quality.warnings for item in observations)
            else "ADEQUATE"
            if acceptance["passed"] is True
            else "POSE_OBSERVABILITY_FAILED"
        )
    else:
        recommendation = _recommendation(observations)
    report: dict[str, object] = {
        "schema_version": "camera-rig.target-preflight.v1",
        "status": "PASS" if recommendation.startswith("ADEQUATE") else "FAIL",
        "camera_name": camera_config.camera.name,
        "stream": stream,
        "frame_count": frames,
        "policy": policy,
        "pose_observability_used": pose_diagnostics is not None,
        "target_spec_sha256": target.artifact_sha256 or sha256_file(target_path),
        "thresholds": observations[0].quality.thresholds,
        "metrics": metrics,
        **({"acceptance": acceptance} if acceptance is not None else {}),
        "recommendation": recommendation,
        "selected_overlays": overlay_files,
        "per_frame": [
            {
                "frame_index": index,
                "passed": observation.quality.passed,
                "detected_charuco_corners": len(observation.point_ids),
                "metrics": observation.quality.metrics,
                "warnings": list(observation.quality.warnings),
                "failure_reasons": list(observation.quality.failure_reasons),
                **(
                    {"pose_diagnostic": pose_diagnostics[index]}
                    if pose_diagnostics is not None
                    else {}
                ),
            }
            for index, observation in enumerate(observations)
        ],
        "software": {"camera_rig_version": __version__, "opencv_version": target.opencv_version},
        "notice": (
            "pose-observability deployment preflight; no fixed extrinsic was persisted"
            if pose_diagnostics is not None
            else "pose-free deployment preflight; no fixed extrinsic was estimated"
        ),
    }
    atomic_write_json(report_path, report)
    return report


def _aggregate(observations: list[TargetObservation]) -> dict[str, object]:
    names = (
        "detected_charuco_corner_count",
        "corner_fraction",
        "coverage_ratio",
        "image_span_x_ratio",
        "image_span_y_ratio",
        "bounding_box_area_ratio",
        "convex_hull_coverage_ratio",
        "mean_marker_perimeter_px",
        "minimum_marker_perimeter_px",
        "corner_distribution_condition",
    )
    metrics: dict[str, object] = {
        "success_ratio": float(np.mean([item.quality.passed for item in observations]))
    }
    for name in names:
        values = np.asarray(
            [_metric_number(item.quality.metrics[name], name) for item in observations]
        )
        metrics[name] = {
            "minimum": float(np.min(values)),
            "median": float(np.median(values)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
        }
    metrics["temporal_jitter"] = _temporal_jitter(observations)
    return metrics


def _temporal_jitter(observations: list[TargetObservation]) -> dict[str, object]:
    minimum_occurrences = max(2, math.ceil(len(observations) * 0.8))
    tracks: dict[int, list[tuple[float, float]]] = {}
    for observation in observations:
        for point_id, point in zip(observation.point_ids, observation.image_points_px, strict=True):
            tracks.setdefault(point_id, []).append((float(point[0]), float(point[1])))
    radial = []
    for values in tracks.values():
        if len(values) >= minimum_occurrences:
            standard_deviation = np.std(np.asarray(values, dtype=np.float64), axis=0)
            radial.append(float(np.hypot(*standard_deviation)))
    return {
        "minimum_occurrences": minimum_occurrences,
        "eligible_corner_count": len(radial),
        "median_radial_std_px": float(np.median(radial)) if radial else 0.0,
        "p95_radial_std_px": float(np.percentile(radial, 95)) if radial else 0.0,
    }


def _recommendation(observations: list[TargetObservation]) -> str:
    if all(item.quality.passed for item in observations):
        if any(item.quality.warnings for item in observations):
            return "ADEQUATE_WITH_LOW_COVERAGE_WARNING"
        return "ADEQUATE"
    metrics = [item.quality.metrics for item in observations]
    if all(
        _metric_number(item["coverage_ratio"], "coverage_ratio") < 0.05
        and _metric_number(item["detected_charuco_corner_count"], "corner_count") >= 12
        for item in metrics
    ):
        return "LARGER_TARGET_RECOMMENDED"
    return "UNUSABLE"


def _metric_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise ArtifactError(f"target preflight metric {name!r} must be numeric")
    return float(value)
