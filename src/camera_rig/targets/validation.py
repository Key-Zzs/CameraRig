"""Single-image and capture-artifact target detection reports."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import numpy.typing as npt

from camera_rig.artifacts.hashing import sha256_file
from camera_rig.artifacts.target_detection import (
    TargetDetectionArtifact,
    write_target_detection,
)
from camera_rig.capture.replay import ReplayCameraSession
from camera_rig.core.errors import ArtifactError
from camera_rig.targets.charuco.dependencies import cv2_module
from camera_rig.targets.charuco.overlay import write_overlay
from camera_rig.targets.io import load_target
from camera_rig.targets.observation import TargetObservation
from camera_rig.targets.registry import registry
from camera_rig.version import __version__

_MINIMUM_CAPTURE_FRAMES = 60
_MINIMUM_SUCCESS_RATIO = 0.95
_MINIMUM_MEDIAN_CORNERS = 20.0
_MINIMUM_MEDIAN_CORNER_FRACTION = 0.80
_MINIMUM_MEDIAN_COVERAGE_RATIO = 0.05
_MAXIMUM_MEDIAN_JITTER_PX = 0.5
_MAXIMUM_P95_JITTER_PX = 1.0


def detect_image(
    *,
    target_path: str | Path,
    image_path: str | Path,
    report_path: str | Path,
    overlay_path: str | Path | None = None,
) -> dict[str, object]:
    """Detect one offline image and persist a portable observation."""
    cv2 = cv2_module()
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ArtifactError(f"could not read detection image: {image_path}")
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    target = load_target(target_path)
    detector = registry.create(plugin_name=target.plugin, target_spec=target)
    observation = detector.detect(image_rgb)
    result: dict[str, object] = {
        "schema_version": "camera-rig.target-detection.v1",
        "target_spec_sha256": target.artifact_sha256,
        "input_image_sha256": sha256_file(image_path),
        "frame_count": 1,
        "per_frame": [
            {
                "frame_index": 0,
                "success": observation.quality.passed,
                "observation": observation.to_dict(),
            }
        ],
        "aggregate": _aggregate([observation]),
        "software": {"camera_rig_version": __version__, "opencv_version": target.opencv_version},
    }
    artifact = TargetDetectionArtifact.from_dict(result)
    write_target_detection(report_path, artifact)
    if overlay_path is not None:
        write_overlay(overlay_path, image_rgb, observation)
    return artifact.to_dict()


def validate_capture_artifact_target(
    *,
    target_path: str | Path,
    artifact_path: str | Path,
    stream: str,
    report_path: str | Path,
    overlays_path: str | Path,
) -> dict[str, object]:
    """Replay a validated capture and report deterministic per-frame observations."""
    target = load_target(target_path)
    detector = registry.create(plugin_name=target.plugin, target_spec=target)
    session = ReplayCameraSession.from_artifact(artifact_path)
    observations: list[TargetObservation] = []
    images: list[npt.NDArray[np.uint8]] = []
    with session:
        while True:
            frame = session.poll_frame()
            if frame is None:
                break
            if stream not in frame.streams:
                raise ArtifactError(f"capture frame does not contain stream {stream!r}")
            raw_image = np.asarray(frame.streams[stream].data)
            if raw_image.dtype != np.uint8:
                raise ArtifactError(f"target detection stream {stream!r} must be uint8")
            image = np.asarray(raw_image, dtype=np.uint8)
            observation = detector.detect(image)
            observations.append(observation)
            images.append(image)
    if not observations:
        raise ArtifactError("capture artifact contains no frames")
    overlay_root = Path(overlays_path)
    overlay_root.mkdir(parents=True, exist_ok=True)
    selected = _selected_frames(observations)
    overlay_files: dict[int, str] = {}
    for label, index in selected.items():
        filename = f"{label}_frame_{index:06d}.png"
        write_overlay(overlay_root / filename, images[index], observations[index])
        overlay_files[index] = filename
    aggregate = _aggregate(observations)
    manifest_path = Path(artifact_path) / "manifest.json"
    report: dict[str, object] = {
        "schema_version": "camera-rig.target-detection.v1",
        "target_spec_sha256": target.artifact_sha256,
        "input_artifact": {
            "manifest_sha256": sha256_file(manifest_path),
            "stream": stream,
        },
        "frame_count": len(observations),
        "per_frame": [
            {
                "frame_index": index,
                "success": observation.quality.passed,
                "observation": observation.to_dict(),
                "overlay": overlay_files.get(index),
            }
            for index, observation in enumerate(observations)
        ],
        "aggregate": aggregate,
        "acceptance": _acceptance(aggregate, len(observations)),
        "selected_overlays": selected,
        "software": {"camera_rig_version": __version__, "opencv_version": target.opencv_version},
    }
    artifact = TargetDetectionArtifact.from_dict(report)
    write_target_detection(report_path, artifact)
    return artifact.to_dict()


def _aggregate(observations: list[TargetObservation]) -> dict[str, object]:
    successes = np.asarray([item.quality.passed for item in observations], dtype=np.float64)
    corners = np.asarray([len(item.point_ids) for item in observations], dtype=np.float64)
    markers = np.asarray(
        [item.quality.metrics["detected_marker_count"] for item in observations],
        dtype=np.float64,
    )
    fractions = np.asarray(
        [item.quality.metrics["corner_fraction"] for item in observations], dtype=np.float64
    )
    coverages = np.asarray(
        [item.quality.metrics["coverage_ratio"] for item in observations], dtype=np.float64
    )
    jitter = _temporal_jitter(observations)
    return {
        "success_ratio": float(np.mean(successes)),
        "detected_marker_count": _statistics(markers),
        "detected_charuco_corner_count": _statistics(corners),
        "corner_fraction": _statistics(fractions),
        "coverage_ratio": _statistics(coverages),
        "temporal_jitter": jitter,
    }


def _statistics(values: npt.NDArray[np.float64]) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _temporal_jitter(observations: list[TargetObservation]) -> dict[str, object]:
    minimum_occurrences = max(2, math.ceil(len(observations) * 0.8))
    tracks: dict[int, list[tuple[float, float]]] = {}
    for observation in observations:
        for point_id, point in zip(observation.point_ids, observation.image_points_px, strict=True):
            tracks.setdefault(point_id, []).append((float(point[0]), float(point[1])))
    per_corner: list[dict[str, object]] = []
    for point_id, values in sorted(tracks.items()):
        if len(values) < minimum_occurrences:
            continue
        points = np.asarray(values, dtype=np.float64)
        std_u, std_v = np.std(points, axis=0)
        radial = float(np.hypot(std_u, std_v))
        per_corner.append(
            {
                "point_id": point_id,
                "occurrences": len(values),
                "std_u_px": float(std_u),
                "std_v_px": float(std_v),
                "radial_std_px": radial,
            }
        )
    radial_values = np.asarray(
        [_number(item["radial_std_px"]) for item in per_corner], dtype=np.float64
    )
    return {
        "minimum_occurrences": minimum_occurrences,
        "eligible_corner_count": len(per_corner),
        "median_radial_std_px": float(np.median(radial_values)) if len(radial_values) else 0.0,
        "p95_radial_std_px": float(np.percentile(radial_values, 95)) if len(radial_values) else 0.0,
        "per_corner": per_corner,
    }


def _acceptance(aggregate: dict[str, object], frame_count: int) -> dict[str, object]:
    corners = _mapping(aggregate["detected_charuco_corner_count"])
    fractions = _mapping(aggregate["corner_fraction"])
    coverage = _mapping(aggregate["coverage_ratio"])
    jitter = _mapping(aggregate["temporal_jitter"])
    checks = {
        "frame_count_is_60": frame_count == _MINIMUM_CAPTURE_FRAMES,
        "success_ratio_at_least_0_95": (
            _number(aggregate["success_ratio"]) >= _MINIMUM_SUCCESS_RATIO
        ),
        "median_corners_at_least_20": (_number(corners["median"]) >= _MINIMUM_MEDIAN_CORNERS),
        "median_corner_fraction_at_least_0_80": (
            _number(fractions["median"]) >= _MINIMUM_MEDIAN_CORNER_FRACTION
        ),
        "median_coverage_at_least_0_05": (
            _number(coverage["median"]) >= _MINIMUM_MEDIAN_COVERAGE_RATIO
        ),
        "median_jitter_at_most_0_5_px": (
            _number(jitter["median_radial_std_px"]) <= _MAXIMUM_MEDIAN_JITTER_PX
        ),
        "p95_jitter_at_most_1_0_px": (
            _number(jitter["p95_radial_std_px"]) <= _MAXIMUM_P95_JITTER_PX
        ),
    }
    return {
        "passed": all(checks.values()),
        "thresholds": {
            "frame_count": _MINIMUM_CAPTURE_FRAMES,
            "success_ratio": _MINIMUM_SUCCESS_RATIO,
            "median_charuco_corners": _MINIMUM_MEDIAN_CORNERS,
            "median_corner_fraction": _MINIMUM_MEDIAN_CORNER_FRACTION,
            "median_coverage_ratio": _MINIMUM_MEDIAN_COVERAGE_RATIO,
            "median_jitter_px": _MAXIMUM_MEDIAN_JITTER_PX,
            "p95_jitter_px": _MAXIMUM_P95_JITTER_PX,
        },
        "checks": checks,
    }


def _selected_frames(observations: list[TargetObservation]) -> dict[str, int]:
    accepted = [index for index, item in enumerate(observations) if item.quality.passed]
    if not accepted:
        return {}
    ranked = sorted(
        accepted,
        key=lambda index: (
            len(observations[index].point_ids),
            _number(observations[index].quality.metrics["coverage_ratio"]),
            -index,
        ),
    )
    return {
        "worst_accepted": ranked[0],
        "median_quality": ranked[len(ranked) // 2],
        "best": ranked[-1],
    }


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ArtifactError("internal aggregate value must be an object")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise ArtifactError("internal aggregate value must be numeric")
    return float(value)
