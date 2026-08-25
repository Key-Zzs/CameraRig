from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from camera_rig.artifacts.io import json_safe
from camera_rig.artifacts.target_detection import (
    TargetDetectionArtifact,
    TargetDetectionFrame,
    load_and_validate_target_detection,
    validate_target_detection_data,
    write_target_detection,
)
from camera_rig.core.errors import ArtifactError
from camera_rig.core.quality import QualityReport
from camera_rig.targets.observation import TargetObservation


def _observation() -> TargetObservation:
    return TargetObservation(
        plugin_name="synthetic",
        target_frame="target",
        point_ids=(0, 1, 2, 3),
        image_points_px=np.asarray([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]),
        object_points_m=np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.0], [0.0, 0.1, 0.0]]
        ),
        image_size=(640, 480),
        quality=QualityReport(True),
        metadata={"target_spec_sha256": "1" * 64},
    )


def _statistics(value: float) -> dict[str, float]:
    return {"minimum": value, "median": value, "maximum": value, "mean": value}


def _artifact() -> TargetDetectionArtifact:
    return TargetDetectionArtifact(
        target_spec_sha256="1" * 64,
        capture_manifest_sha256="2" * 64,
        stream="color",
        frame_count=1,
        per_frame=(TargetDetectionFrame(0, True, _observation(), "best.png"),),
        aggregate={
            "success_ratio": 1.0,
            "detected_marker_count": _statistics(4.0),
            "detected_charuco_corner_count": _statistics(4.0),
            "corner_fraction": _statistics(1.0),
            "coverage_ratio": _statistics(0.1),
            "temporal_jitter": {
                "minimum_occurrences": 2,
                "eligible_corner_count": 0,
                "median_radial_std_px": 0.0,
                "p95_radial_std_px": 0.0,
                "per_corner": [],
            },
        },
        acceptance={
            "passed": True,
            "thresholds": {
                "frame_count": 1,
                "success_ratio": 0.95,
                "median_charuco_corners": 4,
                "median_corner_fraction": 0.8,
                "median_coverage_ratio": 0.05,
                "median_jitter_px": 0.5,
                "p95_jitter_px": 1.0,
            },
            "checks": {
                "frame_count_is_60": True,
                "success_ratio_at_least_0_95": True,
                "median_corners_at_least_20": True,
                "median_corner_fraction_at_least_0_80": True,
                "median_coverage_at_least_0_05": True,
                "median_jitter_at_most_0_5_px": True,
                "p95_jitter_at_most_1_0_px": True,
            },
        },
        selected_overlays={"best": 0},
        software={"camera_rig_version": "0.3.0", "opencv_version": "4.14.0"},
    )


def test_target_detection_schema_and_typed_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "detection.json"
    artifact = _artifact()
    write_target_detection(path, artifact)

    restored = load_and_validate_target_detection(path)

    assert restored.to_dict() == artifact.to_dict()
    assert restored.capture_manifest_sha256 == "2" * 64
    assert restored.stream == "color"
    assert restored.per_frame[0].observation.point_ids == (0, 1, 2, 3)


def test_target_detection_writer_rejects_before_emitting_invalid_schema(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    invalid = replace(
        _artifact(),
        acceptance={"passed": True, "thresholds": {}, "checks": {}},
        selected_overlays={"best": 0},
    )

    with pytest.raises(ArtifactError, match="acceptance"):
        write_target_detection(path, invalid)

    assert not path.exists()


@pytest.mark.parametrize("field", ["target_spec_sha256", "capture_manifest_sha256"])
def test_target_detection_rejects_invalid_bound_sha(field: str) -> None:
    value = _artifact().to_dict()
    if field == "capture_manifest_sha256":
        input_artifact = value["input_artifact"]
        assert isinstance(input_artifact, dict)
        input_artifact["manifest_sha256"] = "not-a-digest"
    else:
        value[field] = "not-a-digest"

    with pytest.raises(ArtifactError, match="does not match"):
        validate_target_detection_data(json_safe(value))


def test_target_detection_rejects_frame_count_mismatch() -> None:
    value = _artifact().to_dict()
    value["frame_count"] = 2

    with pytest.raises(ArtifactError, match="frame_count"):
        validate_target_detection_data(json_safe(value))


def test_target_detection_rejects_unknown_capture_stream() -> None:
    value = _artifact().to_dict()
    input_artifact = value["input_artifact"]
    assert isinstance(input_artifact, dict)
    input_artifact["stream"] = "unknown"

    with pytest.raises(ArtifactError, match="is not one of"):
        validate_target_detection_data(json_safe(value))


def test_target_detection_rejects_per_frame_target_identity_mismatch() -> None:
    value = _artifact().to_dict()
    per_frame = value["per_frame"]
    assert isinstance(per_frame, list)
    frame = per_frame[0]
    assert isinstance(frame, dict)
    observation = frame["observation"]
    assert isinstance(observation, dict)
    metadata = observation["metadata"]
    assert isinstance(metadata, dict)
    metadata["target_spec_sha256"] = "f" * 64

    with pytest.raises(ArtifactError, match="observation target_spec_sha256"):
        validate_target_detection_data(json_safe(value))


def test_target_detection_reconstructs_each_observation() -> None:
    value = _artifact().to_dict()
    per_frame = value["per_frame"]
    assert isinstance(per_frame, list)
    frame = per_frame[0]
    assert isinstance(frame, dict)
    observation = frame["observation"]
    assert isinstance(observation, dict)
    observation["object_points_m"] = [[0.0, 0.0, 0.0]]

    with pytest.raises(ArtifactError, match="counts must match"):
        validate_target_detection_data(json_safe(value))


def test_target_detection_rejects_incomplete_software_provenance() -> None:
    value = _artifact().to_dict()
    software = value["software"]
    assert isinstance(software, dict)
    software.pop("opencv_version")

    with pytest.raises(ArtifactError, match="opencv_version"):
        validate_target_detection_data(json_safe(value))
