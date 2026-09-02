from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from camera_rig.artifacts.hashing import sha256_file
from camera_rig.calibration.fixed.config import FixedCalibrationConfig
from camera_rig.config.models import CameraConfig
from camera_rig.core.errors import ConfigurationError, SchemaValidationError
from camera_rig.provision.config import (
    FIXED_PROVISION_CALIBRATION_FRAMES,
    FIXED_PROVISION_STREAM_VALIDATION_FRAMES,
    load_provision_config,
    load_provision_config_with_sha256,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
EXAMPLE = REPOSITORY_ROOT / "configs/examples/fixed_provision_contract.yaml"
ACCEPTED_TARGET_SHA256 = "56fbd157f9553e7e78c6868e86841c37d1799af1ca0be5a8cc69efa64b845ce1"


def _example_data() -> dict[str, object]:
    value = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_yaml(tmp_path: Path, value: object) -> Path:
    path = tmp_path / "nested" / "fixed_provision.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def test_example_composes_existing_contracts_and_resolves_target_relative_to_yaml() -> None:
    config = load_provision_config(EXAMPLE)
    assert isinstance(config.extract_camera_config(), CameraConfig)
    assert isinstance(config.extract_fixed_calibration_config(), FixedCalibrationConfig)
    assert config.camera_config.camera.serial == "REPLACE_WITH_DEVICE_SERIAL"
    assert config.fixed_calibration_config.detection_stream == "color"
    assert config.fixed_calibration_config.reference_stream == "ir_left"
    assert config.acquisition.calibration_frames == FIXED_PROVISION_CALIBRATION_FRAMES
    assert config.acquisition.stream_validation_frames == FIXED_PROVISION_STREAM_VALIDATION_FRAMES
    assert config.target.expected_sha256 == ACCEPTED_TARGET_SHA256
    assert config.target.artifact_reference == "../targets/charuco_a4_v1/target_spec.json"
    assert config.target.detection_policy == "uncertainty_validated"
    assert (
        config.target.artifact_path
        == (EXAMPLE.parent / "../targets/charuco_a4_v1/target_spec.json").resolve()
    )


def test_loaded_config_digest_is_bound_to_the_parsed_byte_snapshot() -> None:
    config, digest = load_provision_config_with_sha256(EXAMPLE)
    assert config.source_path == EXAMPLE.resolve()
    assert digest == sha256_file(EXAMPLE)


def test_loading_is_dry_run_safe_when_resolved_target_does_not_exist(tmp_path: Path) -> None:
    data = _example_data()
    target = _mapping(data["target"])
    target["artifact"] = "../missing/target_spec.json"
    source = _write_yaml(tmp_path, data)
    config = load_provision_config(source)
    assert config.target.artifact_path == (source.parent / "../missing/target_spec.json").resolve()
    assert not config.target.artifact_path.exists()


def test_external_schema_reference_preserves_strict_camera_fields(tmp_path: Path) -> None:
    data = _example_data()
    _mapping(data["camera"])["device_index"] = 0
    with pytest.raises(SchemaValidationError, match=r"\$\.camera"):
        load_provision_config(_write_yaml(tmp_path, data))


def test_unknown_provision_field_is_rejected(tmp_path: Path) -> None:
    data = _example_data()
    _mapping(data["provision"])["capture_twice"] = True
    with pytest.raises(SchemaValidationError, match=r"\$\.provision"):
        load_provision_config(_write_yaml(tmp_path, data))


@pytest.mark.parametrize(
    "reference",
    [
        "/tmp/target_spec.json",
        "C:/targets/target_spec.json",
        "file://target_spec.json",
        "..\\targets\\target_spec.json",
    ],
)
def test_target_path_must_be_portable_and_yaml_relative(tmp_path: Path, reference: str) -> None:
    data = _example_data()
    _mapping(data["target"])["artifact"] = reference
    with pytest.raises(ConfigurationError, match=r"target\.artifact"):
        load_provision_config(_write_yaml(tmp_path, data))


@pytest.mark.parametrize("digest", ["0" * 63, "A" * 64, 123])
def test_target_expected_sha_is_strict(tmp_path: Path, digest: object) -> None:
    data = _example_data()
    _mapping(data["target"])["expected_sha256"] = digest
    with pytest.raises(SchemaValidationError, match=r"\$\.target\.expected_sha256"):
        load_provision_config(_write_yaml(tmp_path, data))


def test_target_artifact_must_name_resolved_spec(tmp_path: Path) -> None:
    data = _example_data()
    _mapping(data["target"])["artifact"] = "../targets/checksums.sha256"
    with pytest.raises(ConfigurationError, match=r"target_spec\.json"):
        load_provision_config(_write_yaml(tmp_path, data))


def test_target_detection_policy_is_explicit_and_strict(tmp_path: Path) -> None:
    data = _example_data()
    _mapping(data["target"])["detection_policy"] = "pose_validated"
    assert load_provision_config(_write_yaml(tmp_path, data)).target.detection_policy == (
        "pose_validated"
    )
    _mapping(data["target"])["detection_policy"] = "uncertainty_validated"
    uncertainty_root = tmp_path / "uncertainty"
    uncertainty_root.mkdir()
    assert (
        load_provision_config(_write_yaml(uncertainty_root, data)).target.detection_policy
        == "uncertainty_validated"
    )
    _mapping(data["target"])["detection_policy"] = "guess"
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    with pytest.raises(SchemaValidationError, match=r"\$\.target\.detection_policy"):
        load_provision_config(_write_yaml(invalid, data))


def test_calibration_frame_policy_is_frozen_to_detection_contract(tmp_path: Path) -> None:
    data = _example_data()
    _mapping(data["provision"])["calibration_frames"] = 59
    with pytest.raises(SchemaValidationError, match=r"\$\.provision\.calibration_frames"):
        load_provision_config(_write_yaml(tmp_path, data))


def test_stream_validation_frame_policy_is_frozen(tmp_path: Path) -> None:
    data = _example_data()
    _mapping(data["provision"])["stream_validation_frames"] = 299
    with pytest.raises(SchemaValidationError, match=r"\$\.provision\.stream_validation_frames"):
        load_provision_config(_write_yaml(tmp_path, data))


def test_detection_stream_must_be_enabled(tmp_path: Path) -> None:
    data = _example_data()
    color = _mapping(_mapping(data["streams"])["color"])
    color["enabled"] = False
    with pytest.raises(ConfigurationError, match="target detection stream must be enabled"):
        load_provision_config(_write_yaml(tmp_path, data))


def test_native_depth_check_requires_enabled_depth(tmp_path: Path) -> None:
    data = _example_data()
    depth = _mapping(_mapping(data["streams"])["depth"])
    depth["enabled"] = False
    with pytest.raises(ConfigurationError, match="native depth sanity requires"):
        load_provision_config(_write_yaml(tmp_path, data))


def test_all_four_raw_streams_are_required(tmp_path: Path) -> None:
    data = _example_data()
    ir_right = _mapping(_mapping(data["streams"])["ir_right"])
    ir_right["enabled"] = False
    with pytest.raises(ConfigurationError, match="all four raw streams"):
        load_provision_config(_write_yaml(tmp_path, data))


def test_explicit_required_streams_must_include_all_four(tmp_path: Path) -> None:
    data = _example_data()
    _mapping(data["capture"])["required_streams"] = ["color", "ir_left"]
    with pytest.raises(ConfigurationError, match="must capture all four raw streams"):
        load_provision_config(_write_yaml(tmp_path, data))


def test_provisioning_requires_owned_frame_copies(tmp_path: Path) -> None:
    data = _example_data()
    _mapping(data["capture"])["copy_frames"] = False
    with pytest.raises(ConfigurationError, match="copy_frames to be true"):
        load_provision_config(_write_yaml(tmp_path, data))


def test_camera_and_fixed_reference_streams_must_match(tmp_path: Path) -> None:
    data = _example_data()
    _mapping(data["fixed_calibration"])["reference_stream"] = "color"
    with pytest.raises(ConfigurationError, match="output_reference_stream must equal"):
        load_provision_config(_write_yaml(tmp_path, data))


def test_minimum_accepted_frames_cannot_exceed_acquisition(tmp_path: Path) -> None:
    data = _example_data()
    _mapping(data["fixed_calibration"])["minimum_accepted_frames"] = 61
    with pytest.raises(ConfigurationError, match="minimum_accepted_frames exceeds"):
        load_provision_config(_write_yaml(tmp_path, data))


def test_workspace_transform_is_identity_for_fixed_provision(tmp_path: Path) -> None:
    data = copy.deepcopy(_example_data())
    workspace = _mapping(data["workspace"])
    transform = _mapping(workspace["T_workspace_from_target"])
    matrix = transform["matrix"]
    assert isinstance(matrix, list)
    assert isinstance(matrix[0], list)
    matrix[0][3] = 0.01
    with pytest.raises(ConfigurationError, match="T_workspace_from_target to be identity"):
        load_provision_config(_write_yaml(tmp_path, data))
