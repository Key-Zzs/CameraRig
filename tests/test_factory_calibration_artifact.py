from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from camera_rig.artifacts.factory_calibration import (
    FactoryCalibrationArtifact,
    load_and_validate_factory_calibration,
    validate_factory_calibration_data,
    write_factory_calibration,
)
from camera_rig.artifacts.io import json_safe
from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.errors import ArtifactError, ContractError
from camera_rig.core.factory_calibration import FactoryCalibration
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.stream import StreamProfile
from camera_rig.core.transforms import RigidTransform


def _artifact() -> FactoryCalibrationArtifact:
    profile = StreamProfile("ir_left", 640, 480, 30, "y8", 1, "Stereo Module")
    other = StreamProfile("ir_right", 640, 480, 30, "y8", 2, "Stereo Module")
    intrinsics = {
        name: CameraIntrinsics(f"head/{name}_optical", 640, 480, 600.0, 601.0, 319.5, 239.5, "none")
        for name in ("ir_left", "ir_right")
    }
    matrix = np.eye(4)
    matrix[0, 3] = 0.05
    calibration = FactoryCalibration(
        device=CameraDeviceInfo(
            "realsense",
            "head",
            "D435i",
            "RealSense D435I",
            "placeholder",
            canonical_model="D435i",
            product_id="0B3A",
        ),
        stream_profiles={"ir_left": profile, "ir_right": other},
        intrinsics=intrinsics,
        internal_transforms=(
            RigidTransform("head/ir_left_optical", "head/ir_right_optical", matrix),
        ),
        depth_scale_m_per_unit=0.001,
    )
    return FactoryCalibrationArtifact(
        created_at=datetime.now(timezone.utc).isoformat(),
        calibration=calibration,
        quality=QualityReport(True, metrics={"intrinsics_count": 2}),
        provenance={"git_commit": "abc", "config_sha256": "0" * 64},
    )


def test_factory_artifact_schema_and_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "factory.json"
    artifact = _artifact()
    write_factory_calibration(path, artifact)
    restored = load_and_validate_factory_calibration(path)
    assert restored.to_dict() == artifact.to_dict()


def test_factory_artifact_rejects_absolute_provenance() -> None:
    artifact = _artifact()
    with pytest.raises(ContractError, match="absolute paths"):
        FactoryCalibrationArtifact(
            created_at=artifact.created_at,
            calibration=artifact.calibration,
            quality=artifact.quality,
            provenance={"config": "/private/config.yaml"},
        )


def test_factory_artifact_rejects_schema_mismatch() -> None:
    value = _artifact().to_dict()
    value["schema_version"] = "camera-rig.factory-calibration.v2"
    with pytest.raises(ArtifactError, match="schema_version"):
        validate_factory_calibration_data(json_safe(value))


def test_factory_artifact_revalidates_transform() -> None:
    value = _artifact().to_dict()
    transforms = value["internal_transforms"]
    assert isinstance(transforms, list)
    transform = transforms[0]
    assert isinstance(transform, dict)
    transform["matrix"] = np.diag([2.0, 1.0, 1.0, 1.0]).tolist()
    with pytest.raises(ArtifactError, match="orthonormal"):
        validate_factory_calibration_data(json_safe(value))
