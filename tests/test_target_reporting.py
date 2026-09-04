from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import camera_rig.cli.commands.target as target_cli
import camera_rig.targets.validation as target_validation
from camera_rig.artifacts.factory_calibration import FactoryCalibrationArtifact
from camera_rig.artifacts.io import atomic_write_json
from camera_rig.artifacts.target_detection import load_and_validate_target_detection
from camera_rig.capture.snapshot import write_snapshot
from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.errors import ArtifactError
from camera_rig.core.factory_calibration import FactoryCalibration
from camera_rig.core.frame import CameraFrame, StreamFrame
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.stream import StreamProfile
from camera_rig.core.timestamps import SingleDeviceSyncReport
from camera_rig.core.transforms import RigidTransform
from camera_rig.targets.observation import TargetObservation
from camera_rig.targets.pose_acceptance import aggregate_pose_diagnostics
from camera_rig.targets.validation import detect_image, validate_capture_artifact_target

cv2 = pytest.importorskip("cv2")
pytestmark = pytest.mark.charuco


def test_uncertainty_target_cli_labels_numerical_pass_as_release_hold(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        target_cli,
        "validate_capture_artifact_target",
        lambda **_kwargs: {"acceptance": {"passed": True}, "frame_count": 60},
    )
    validate_arguments = SimpleNamespace(
        target=Path("target.json"),
        artifact=Path("capture"),
        stream="color",
        report=Path("report.json"),
        overlays=Path("overlays"),
        policy="uncertainty_validated",
    )
    assert target_cli._validate_artifact(validate_arguments) == 0
    assert "NUMERICAL_PASS RELEASE_HOLD candidate_only=true" in capsys.readouterr().out

    monkeypatch.setattr(target_cli, "load_config", lambda _path: object())
    monkeypatch.setattr(
        target_cli,
        "run_target_preflight",
        lambda **_kwargs: {
            "status": "PASS",
            "recommendation": "ADEQUATE",
            "operator_recommendation": "CANDIDATE_ONLY_ADEQUATE_RELEASE_HOLD",
        },
    )
    preflight_arguments = SimpleNamespace(
        camera_config=Path("camera.yaml"),
        target=Path("target.json"),
        frames=60,
        stream="color",
        policy="uncertainty_validated",
        report=Path("preflight.json"),
        overlays=Path("overlays"),
    )
    assert target_cli._preflight(preflight_arguments) == 0
    output = capsys.readouterr().out
    assert "NUMERICAL_PASS RELEASE_HOLD" in output
    assert "CANDIDATE_ONLY_ADEQUATE_RELEASE_HOLD" in output


def test_rank_deficient_pose_diagnostic_is_aggregated_fail_closed() -> None:
    aggregate = aggregate_pose_diagnostics(
        [
            {
                "solve_success": True,
                "observable": False,
                "observability": {
                    "translation_worst_axis_std_mm": None,
                    "rotation_worst_axis_std_deg": None,
                    "scaled_condition_number": None,
                    "candidate_ambiguity": {"ambiguous": False},
                    "failure_reasons": ["POSE_OBSERVABILITY_RANK_DEFICIENT"],
                },
            }
        ]
    )
    assert aggregate["solve_success_ratio"] == 1.0
    assert aggregate["observable_frame_ratio"] == 0.0
    assert aggregate["translation_worst_axis_std_mm"]["count"] == 0


def _canvas(target_root: Path) -> np.ndarray:
    board = cv2.imread(str(target_root / "charuco_a4_v1_board.png"), cv2.IMREAD_GRAYSCALE)
    assert board is not None
    resized = cv2.resize(board, (900, 643), interpolation=cv2.INTER_NEAREST)
    gray = np.full((960, 1280), 255, dtype=np.uint8)
    gray[150:793, 190:1090] = resized
    return np.repeat(gray[:, :, None], 3, axis=2)


def _factory() -> FactoryCalibrationArtifact:
    names = ("color", "depth", "ir_left", "ir_right")
    profiles = {
        name: StreamProfile(
            name,
            1280,
            960,
            30,
            {"color": "rgb8", "depth": "z16"}.get(name, "y8"),
            {"color": 0, "depth": 0, "ir_left": 1, "ir_right": 2}[name],
        )
        for name in names
    }
    intrinsics = {
        name: CameraIntrinsics(
            f"synthetic/{name}_optical", 1280, 960, 900.0, 900.0, 639.5, 479.5, "none"
        )
        for name in names
    }
    transforms = []
    for index, target in enumerate(("color", "depth", "ir_right")):
        matrix = np.eye(4)
        matrix[0, 3] = index * 0.01
        transforms.append(
            RigidTransform("synthetic/ir_left_optical", f"synthetic/{target}_optical", matrix)
        )
    calibration = FactoryCalibration(
        CameraDeviceInfo("synthetic", "synthetic", "synthetic", "synthetic", "test-device"),
        profiles,
        intrinsics,
        tuple(transforms),
        0.001,
    )
    return FactoryCalibrationArtifact(
        datetime.now(timezone.utc).isoformat(),
        calibration,
        QualityReport(True),
        {"source": "unit-test"},
    )


def _frames(image: np.ndarray, count: int) -> list[CameraFrame]:
    height, width, _channels = image.shape
    result = []
    for index in range(count):
        streams = {
            "color": StreamFrame("color", image, index, index * 1_000_000, "synthetic"),
            "depth": StreamFrame(
                "depth",
                np.zeros((height, width), dtype=np.uint16),
                index,
                index * 1_000_000,
                "synthetic",
            ),
            "ir_left": StreamFrame(
                "ir_left",
                np.zeros((height, width), dtype=np.uint8),
                index,
                index * 1_000_000,
                "synthetic",
            ),
            "ir_right": StreamFrame(
                "ir_right",
                np.zeros((height, width), dtype=np.uint8),
                index,
                index * 1_000_000,
                "synthetic",
            ),
        }
        result.append(
            CameraFrame(
                "synthetic",
                "test-device",
                streams,
                index * 1_000_000,
                SingleDeviceSyncReport(
                    True,
                    ("color", "depth", "ir_left", "ir_right"),
                    0,
                    {"color": 0, "depth": 0, "ir_left": 0, "ir_right": 0},
                    True,
                ),
            )
        )
    return result


def test_single_image_report_and_overlay_are_portable(
    generated_charuco_target: Path, tmp_path: Path
) -> None:
    image_path = tmp_path / "input.png"
    assert cv2.imwrite(
        str(image_path), cv2.cvtColor(_canvas(generated_charuco_target), cv2.COLOR_RGB2BGR)
    )
    report_path = tmp_path / "detection.json"
    overlay_path = tmp_path / "overlay.png"
    report = detect_image(
        target_path=generated_charuco_target / "target_spec.json",
        image_path=image_path,
        report_path=report_path,
        overlay_path=overlay_path,
    )
    assert report["schema_version"] == "camera-rig.target-detection.v1"
    assert report["aggregate"]["success_ratio"] == 1.0  # type: ignore[index]
    assert report_path.is_file() and overlay_path.is_file()
    assert load_and_validate_target_detection(report_path).input_image_sha256 is not None
    persisted = report["per_frame"][0]["observation"]  # type: ignore[index]
    observation = TargetObservation.from_dict(persisted)
    assert observation.point_ids == tuple(range(24))


def test_capture_report_persists_observations_and_temporal_jitter(
    generated_charuco_target: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "capture"
    image = _canvas(generated_charuco_target)
    write_snapshot(
        artifact,
        _frames(image, 3),
        _factory(),
        {"copy_frames": True},
        {"source": "unit-test"},
        include_previews=False,
    )
    report_path = tmp_path / "report.json"
    overlays = tmp_path / "overlays"
    report = validate_capture_artifact_target(
        target_path=generated_charuco_target / "target_spec.json",
        artifact_path=artifact,
        stream="color",
        report_path=report_path,
        overlays_path=overlays,
    )
    assert report["frame_count"] == 3
    assert report["aggregate"]["success_ratio"] == 1.0  # type: ignore[index]
    jitter = report["aggregate"]["temporal_jitter"]  # type: ignore[index]
    assert jitter["eligible_corner_count"] == 24
    assert jitter["median_radial_std_px"] == pytest.approx(0.0)
    assert jitter["p95_radial_std_px"] == pytest.approx(0.0)
    assert report["acceptance"]["passed"] is False  # type: ignore[index]
    assert report["acceptance"]["thresholds"]["median_coverage_ratio"] == 0.05  # type: ignore[index]
    assert report["acceptance"]["checks"]["median_coverage_at_least_threshold"] is True  # type: ignore[index]
    restored = load_and_validate_target_detection(report_path)
    assert restored.capture_manifest_sha256 is not None
    assert restored.stream == "color"
    assert restored.frame_count == 3
    assert len(list(overlays.glob("*.png"))) == 3
    for item in report["per_frame"]:  # type: ignore[union-attr]
        observation = TargetObservation.from_dict(item["observation"])
        assert observation.point_ids == tuple(range(24))


def test_uncertainty_capture_report_persists_pose_observability(
    generated_charuco_target: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "capture"
    image = _canvas(generated_charuco_target)
    write_snapshot(
        artifact,
        _frames(image, 3),
        _factory(),
        {"copy_frames": True},
        {"source": "unit-test"},
        include_previews=False,
    )
    report_path = tmp_path / "report.json"
    report = validate_capture_artifact_target(
        target_path=generated_charuco_target / "target_spec.json",
        artifact_path=artifact,
        stream="color",
        report_path=report_path,
        overlays_path=tmp_path / "overlays",
        policy="uncertainty_validated",
    )
    assert report["acceptance"]["policy"] == "uncertainty_validated"  # type: ignore[index]
    assert report["acceptance"]["coverage"]["hard_gate"] is False  # type: ignore[index]
    pose = report["aggregate"]["pose_observability"]  # type: ignore[index]
    assert pose["solve_success_ratio"] == 1.0
    assert pose["observable_frame_ratio"] == 1.0
    assert all("pose_diagnostic" in item for item in report["per_frame"])  # type: ignore[union-attr]
    restored = load_and_validate_target_detection(report_path)
    assert all(frame.pose_diagnostic is not None for frame in restored.per_frame)
    historical = dict(report)
    acceptance = historical["acceptance"]
    assert isinstance(acceptance, dict)
    thresholds = acceptance["thresholds"]
    assert isinstance(thresholds, dict)
    for key in (
        "release_state",
        "release_criteria_version",
        "release_manifest_sha256",
        "structured_gate_version",
    ):
        thresholds.pop(key)
    historical_path = tmp_path / "historical-v1-report.json"
    atomic_write_json(historical_path, historical)
    historical_restored = load_and_validate_target_detection(historical_path)
    assert historical_restored.frame_count == restored.frame_count


def test_uncertainty_capture_fails_closed_when_intrinsics_are_unavailable(
    generated_charuco_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "capture"
    write_snapshot(
        artifact,
        _frames(_canvas(generated_charuco_target), 3),
        _factory(),
        {"copy_frames": True},
        {"source": "unit-test"},
        include_previews=False,
    )

    def unavailable(_path: object) -> None:
        raise ArtifactError("unavailable")

    monkeypatch.setattr(target_validation, "load_and_validate_factory_calibration", unavailable)
    with pytest.raises(ArtifactError, match="POSE_OBSERVABILITY_INTRINSICS_UNAVAILABLE"):
        validate_capture_artifact_target(
            target_path=generated_charuco_target / "target_spec.json",
            artifact_path=artifact,
            stream="color",
            report_path=tmp_path / "report.json",
            overlays_path=tmp_path / "overlays",
            policy="uncertainty_validated",
        )
