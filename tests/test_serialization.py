from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from camera_rig.artifacts.hashing import sha256_bytes, sha256_file
from camera_rig.artifacts.io import (
    atomic_write_json,
    deterministic_json_bytes,
    json_safe,
    load_json,
)
from camera_rig.artifacts.models import CameraBundle
from camera_rig.artifacts.validation import load_and_validate_bundle, validate_bundle_data
from camera_rig.cli.main import main
from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.errors import ArtifactError, CameraRigError, ContractError
from camera_rig.core.factory_calibration import FactoryCalibration
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.stream import StreamProfile
from camera_rig.core.transforms import RigidTransform


def test_intrinsics_round_trip() -> None:
    original = CameraIntrinsics(
        "color_optical", 640, 480, 600.0, 601.0, 319.5, 239.5, "none", (0.0,)
    )
    restored = CameraIntrinsics.from_dict(original.to_dict())
    assert restored == original


def test_rigid_transform_round_trip(
    make_transform: Callable[[str, str, tuple[float, float, float]], RigidTransform],
) -> None:
    original = make_transform("depth", "color", (0.025, 0.0, 0.0))
    restored = RigidTransform.from_dict(original.to_dict())
    assert (restored.source_frame, restored.target_frame) == ("depth", "color")
    np.testing.assert_allclose(restored.matrix, original.matrix)


def test_factory_calibration_round_trip(
    make_transform: Callable[[str, str, tuple[float, float, float]], RigidTransform],
) -> None:
    device = CameraDeviceInfo("synthetic", "head", "test", "test", "001")
    profile = StreamProfile("color", 640, 480, 30, "rgb8")
    intrinsics = CameraIntrinsics("color_optical", 640, 480, 600.0, 600.0, 319.5, 239.5, "none")
    original = FactoryCalibration(
        device,
        {"color": profile},
        {"color": intrinsics},
        (make_transform("depth_optical", "color_optical", (0.02, 0.0, 0.0)),),
        0.001,
    )
    restored = FactoryCalibration.from_dict(original.to_dict())
    assert restored.device.serial == "001"
    assert restored.stream_profiles == original.stream_profiles
    np.testing.assert_allclose(
        restored.internal_transforms[0].matrix, original.internal_transforms[0].matrix
    )


def test_camera_bundle_round_trip_and_schema(sample_bundle: CameraBundle) -> None:
    data = sample_bundle.to_dict()
    restored = validate_bundle_data(json_safe(data))
    assert restored.bundle_id == sample_bundle.bundle_id
    assert restored.device.serial == "000123"
    assert restored.fixed_mount_calibration is not None
    np.testing.assert_allclose(
        restored.internal_transforms[0].matrix, sample_bundle.internal_transforms[0].matrix
    )


def test_numpy_and_nested_conversion() -> None:
    converted = json_safe(
        {
            "float": np.float32(1.25),
            "int": np.int64(7),
            "array": np.array([[1, 2], [3, 4]]),
            "tuple": (np.bool_(True), None),
        }
    )
    assert converted == {
        "float": 1.25,
        "int": 7,
        "array": [[1, 2], [3, 4]],
        "tuple": [True, None],
    }


def test_non_finite_json_is_rejected() -> None:
    with pytest.raises(ArtifactError, match="NaN"):
        deterministic_json_bytes({"value": np.nan})


def test_quality_metrics_reject_non_json_safe_values() -> None:
    from camera_rig.core.quality import QualityReport

    with pytest.raises(ContractError, match="non-JSON-safe"):
        QualityReport(True, metrics={"bad": object()})


def test_deterministic_json_and_hash(sample_bundle: CameraBundle) -> None:
    first = deterministic_json_bytes(sample_bundle.to_dict())
    second = deterministic_json_bytes(dict(reversed(list(sample_bundle.to_dict().items()))))
    assert first == second
    assert first.endswith(b"\n")
    assert sha256_bytes(first) == hashlib.sha256(first).hexdigest()


def test_atomic_write_and_file_hash(tmp_path: Path, sample_bundle: CameraBundle) -> None:
    path = tmp_path / "bundle.json"
    atomic_write_json(path, sample_bundle.to_dict())
    assert path.read_bytes() == deterministic_json_bytes(sample_bundle.to_dict())
    assert sha256_file(path) == sha256_bytes(path.read_bytes())
    restored = load_and_validate_bundle(path)
    assert restored.bundle_id == sample_bundle.bundle_id


def test_atomic_replace_failure_preserves_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(_source: os.PathLike[str] | str, _target: os.PathLike[str] | str) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(ArtifactError, match="injected replace failure"):
        atomic_write_json(path, {"new": True})
    assert path.read_text(encoding="utf-8") == '{"old": true}\n'
    assert list(tmp_path.glob(".*.tmp")) == []


def test_corrupted_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactError, match="could not load JSON"):
        load_json(path)


def test_bundle_schema_mismatch_is_rejected(sample_bundle: CameraBundle) -> None:
    data = sample_bundle.to_dict()
    data["schema_version"] = "camera-rig.bundle.v2"
    with pytest.raises(CameraRigError, match="schema_version"):
        validate_bundle_data(json_safe(data))


def test_bundle_numeric_serial_is_rejected(sample_bundle: CameraBundle) -> None:
    data = sample_bundle.to_dict()
    device = data["device"]
    assert isinstance(device, dict)
    device["serial"] = 123
    with pytest.raises(CameraRigError, match="serial"):
        validate_bundle_data(json_safe(data))


def test_bundle_load_revalidates_se3(sample_bundle: CameraBundle) -> None:
    data = sample_bundle.to_dict()
    transforms = data["internal_transforms"]
    assert isinstance(transforms, list)
    transform = transforms[0]
    assert isinstance(transform, dict)
    matrix = transform["matrix"]
    assert isinstance(matrix, list)
    first_row = matrix[0]
    assert isinstance(first_row, list)
    first_row[0] = 2.0
    with pytest.raises(CameraRigError, match="orthonormal"):
        validate_bundle_data(json_safe(data))


def test_device_from_dict_does_not_coerce_numeric_serial() -> None:
    data: dict[str, object] = {
        "driver": "synthetic",
        "camera_name": "head",
        "expected_model": "test",
        "reported_model": "test",
        "serial": 123,
        "metadata": {},
    }
    with pytest.raises(ContractError, match="serial must be a string"):
        CameraDeviceInfo.from_dict(data)


def test_written_json_is_standard_json(tmp_path: Path, sample_bundle: CameraBundle) -> None:
    path = tmp_path / "bundle.json"
    atomic_write_json(path, sample_bundle.to_dict())
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == "camera-rig.bundle.v1"


def test_artifact_cli_success(
    tmp_path: Path,
    sample_bundle: CameraBundle,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "bundle.json"
    atomic_write_json(path, sample_bundle.to_dict())
    assert main(["artifact", "validate", "--bundle", str(path)]) == 0
    output = capsys.readouterr()
    assert "valid camera-rig.bundle.v1" in output.out
    assert output.err == ""
