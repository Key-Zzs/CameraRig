from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from camera_rig.artifacts.factory_calibration import FactoryCalibrationArtifact
from camera_rig.capture.snapshot import write_snapshot
from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.factory_calibration import FactoryCalibration
from camera_rig.core.frame import CameraFrame, StreamFrame
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.stream import StreamProfile
from camera_rig.core.timestamps import SingleDeviceSyncReport
from camera_rig.core.transforms import RigidTransform
from camera_rig.targets.observation import TargetObservation
from camera_rig.targets.validation import detect_image, validate_capture_artifact_target

cv2 = pytest.importorskip("cv2")
pytestmark = pytest.mark.charuco


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
    assert len(list(overlays.glob("*.png"))) == 3
    for item in report["per_frame"]:  # type: ignore[union-attr]
        observation = TargetObservation.from_dict(item["observation"])
        assert observation.point_ids == tuple(range(24))
