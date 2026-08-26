from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import camera_rig.api
from camera_rig.api import (
    CameraBundle,
    CameraConfig,
    CameraFrame,
    CameraIntrinsics,
    CameraSession,
    FactoryCalibration,
    FixedMountCalibration,
    ReplayCameraSession,
    RigidTransform,
    StreamFrame,
    load_camera_bundle,
    load_camera_config,
    load_provisioned_camera_bundle,
)

EXPECTED_V1_API = {
    "CameraBundle",
    "CameraConfig",
    "CameraFrame",
    "CameraIntrinsics",
    "CameraSession",
    "FactoryCalibration",
    "FixedMountCalibration",
    "ReplayCameraSession",
    "RigidTransform",
    "StreamFrame",
    "load_camera_bundle",
    "load_camera_config",
    "load_provisioned_camera_bundle",
}
REPOSITORY_ROOT = Path(__file__).parents[2]
FIXTURE = REPOSITORY_ROOT / "tests/fixtures/consumer/fixed_camera_bundle_v1.json"


def test_public_api_surface_is_frozen() -> None:
    assert set(camera_rig.api.__all__) == EXPECTED_V1_API
    namespace: dict[str, object] = {}
    exec("from camera_rig.api import *", namespace)
    assert set(namespace) >= EXPECTED_V1_API


def test_public_symbols_are_the_existing_contracts() -> None:
    assert CameraBundle.__module__ == "camera_rig.artifacts.models"
    assert CameraConfig.__module__ == "camera_rig.config.models"
    assert CameraFrame.__module__ == "camera_rig.core.frame"
    assert StreamFrame.__module__ == "camera_rig.core.frame"
    assert CameraIntrinsics.__module__ == "camera_rig.core.intrinsics"
    assert FactoryCalibration.__module__ == "camera_rig.core.factory_calibration"
    assert FixedMountCalibration.__module__ == "camera_rig.core.fixed_mount"
    assert RigidTransform.__module__ == "camera_rig.core.transforms"
    assert CameraSession.__module__ == "camera_rig.capture.session"
    assert ReplayCameraSession.__module__ == "camera_rig.capture.replay"


def test_load_camera_config_uses_strict_existing_contract() -> None:
    config = load_camera_config(REPOSITORY_ROOT / "configs/examples/single_camera_contract.yaml")
    assert isinstance(config, CameraConfig)
    assert config.camera.output_reference_stream == "ir_left"


def test_bundle_and_fixed_transform_direction() -> None:
    bundle = load_camera_bundle(FIXTURE)
    assert isinstance(bundle, CameraBundle)
    assert bundle.status == "passed"
    fixed = bundle.fixed_mount_calibration
    assert fixed is not None
    assert fixed.parent_frame == "workspace"
    assert fixed.camera_reference_frame == "synthetic_camera/ir_left_optical"
    transform = fixed.T_parent_from_camera_reference
    assert transform.source_frame == "synthetic_camera/ir_left_optical"
    assert transform.target_frame == "workspace"
    point_camera = np.asarray([1.0, 2.0, 3.0])
    np.testing.assert_allclose(transform.transform_points(point_camera), [-1.5, 0.75, 4.0])
    np.testing.assert_allclose(
        transform.inverse().transform_points(transform.transform_points(point_camera)),
        point_camera,
    )


def test_bundle_loader_rejects_contradictory_passed_status(tmp_path: Path) -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data["quality"]["passed"] = False
    data["quality"]["failure_reasons"] = ["synthetic failure"]
    path = tmp_path / "contradictory_bundle.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    try:
        load_camera_bundle(path)
    except Exception as error:
        assert type(error).__name__ == "ArtifactError"
        assert "status and quality decision differ" in str(error)
    else:
        raise AssertionError("contradictory bundle was accepted")


def test_runtime_facade_import_does_not_open_or_require_a_camera() -> None:
    assert CameraSession is camera_rig.api.CameraSession
    assert ReplayCameraSession is camera_rig.api.ReplayCameraSession
    assert load_provisioned_camera_bundle is camera_rig.api.load_provisioned_camera_bundle


def test_consumer_examples_use_only_the_stable_package_boundary() -> None:
    consumer_files = (
        REPOSITORY_ROOT / "examples/consumer_fixed_camera.py",
        REPOSITORY_ROOT / "tests/typecheck/consumer_api.py",
        REPOSITORY_ROOT / "docs/downstream-integration.md",
    )
    forbidden = (
        "camera_rig.core",
        "camera_rig.provision",
        "camera_rig.drivers",
        "camera_rig.calibration",
        "camera_rig.targets.charuco",
    )
    for path in consumer_files:
        content = path.read_text(encoding="utf-8")
        assert "camera_rig.api" in content
        assert all(name not in content for name in forbidden)
