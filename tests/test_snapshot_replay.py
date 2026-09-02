from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from camera_rig.api import ReplayCameraSession
from camera_rig.artifacts.capture_validation import validate_capture_artifact
from camera_rig.artifacts.factory_calibration import FactoryCalibrationArtifact
from camera_rig.capture.snapshot import write_snapshot
from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.errors import ArtifactError, LifecycleError, ReplayEOFError
from camera_rig.core.factory_calibration import FactoryCalibration
from camera_rig.core.frame import CameraFrame, StreamFrame
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.stream import StreamProfile
from camera_rig.core.timestamps import SingleDeviceSyncReport
from camera_rig.core.transforms import RigidTransform


def _frames(count: int = 2) -> list[CameraFrame]:
    result: list[CameraFrame] = []
    for index in range(count):
        streams = {
            "color": StreamFrame(
                "color",
                np.arange(18, dtype=np.uint8).reshape(2, 3, 3) + index,
                10 + index,
                1_000_000 + index * 33_333_333,
                "hardware_clock",
                1.0 + index / 30,
                {"frame_counter": 10 + index},
            ),
            "depth": StreamFrame(
                "depth",
                np.arange(6, dtype=np.uint16).reshape(2, 3) + index,
                30 + index,
                1_000_100 + index * 33_333_333,
                "hardware_clock",
                1.0001 + index / 30,
            ),
            "ir_left": StreamFrame(
                "ir_left",
                np.arange(6, dtype=np.uint8).reshape(2, 3) + index,
                20 + index,
                1_000_000 + index * 33_333_333,
                "hardware_clock",
                1.0 + index / 30,
            ),
            "ir_right": StreamFrame(
                "ir_right",
                np.fliplr(np.arange(6, dtype=np.uint8).reshape(2, 3)) + index,
                20 + index,
                1_000_200 + index * 33_333_333,
                "hardware_clock",
                1.0002 + index / 30,
            ),
        }
        result.append(
            CameraFrame(
                "head",
                "placeholder",
                streams,
                9_000_000 + index,
                SingleDeviceSyncReport(
                    True,
                    ("color", "depth", "ir_left", "ir_right"),
                    200,
                    {"color": 0, "depth": 100, "ir_left": 0, "ir_right": 200},
                    True,
                ),
            )
        )
    return result


def _factory() -> FactoryCalibrationArtifact:
    names = ("color", "depth", "ir_left", "ir_right")
    profiles = {
        name: StreamProfile(
            name,
            3,
            2,
            30,
            {"color": "rgb8", "depth": "z16"}.get(name, "y8"),
            {"color": 0, "depth": 0, "ir_left": 1, "ir_right": 2}[name],
        )
        for name in names
    }
    intrinsics = {
        name: CameraIntrinsics(f"head/{name}_optical", 3, 2, 2.0, 2.0, 1.0, 0.5, "none")
        for name in names
    }
    transforms = []
    for index, target in enumerate(("color", "depth", "ir_right")):
        matrix = np.eye(4)
        matrix[0, 3] = index * 0.01
        transforms.append(RigidTransform("head/ir_left_optical", f"head/{target}_optical", matrix))
    calibration = FactoryCalibration(
        CameraDeviceInfo("synthetic", "head", "synthetic", "synthetic", "placeholder"),
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


def _write(root: Path, count: int = 2) -> list[CameraFrame]:
    frames = _frames(count)
    write_snapshot(
        root,
        frames,
        _factory(),
        {"copy_frames": True},
        {"source": "unit-test"},
        include_previews=False,
    )
    return frames


def test_snapshot_replay_is_bitwise_equal_and_rewindable(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    original = _write(root)
    manifest = validate_capture_artifact(root)
    assert manifest["frame_count"] == 2
    with ReplayCameraSession.from_artifact(root) as replay:
        restored = [replay.capture(), replay.capture()]
        with pytest.raises(ReplayEOFError, match="end"):
            replay.capture()
        replay.rewind()
        assert replay.capture().streams["color"].frame_number == 10
    for expected, actual in zip(original, restored, strict=True):
        assert expected.camera_name == actual.camera_name
        assert expected.serial == actual.serial
        assert expected.host_receive_timestamp_ns == actual.host_receive_timestamp_ns
        assert expected.sync_report == actual.sync_report
        for name in expected.streams:
            left = expected.streams[name]
            right = actual.streams[name]
            np.testing.assert_array_equal(left.data, right.data)
            assert left.data.dtype == right.data.dtype
            assert left.data.shape == right.data.shape
            assert left.frame_number == right.frame_number
            assert left.sensor_timestamp_ns == right.sensor_timestamp_ns
            assert left.timestamp_domain == right.timestamp_domain
            assert left.original_timestamp == right.original_timestamp
            assert left.metadata == right.metadata


def test_replay_poll_requires_open_and_has_stable_eof(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write(root, 1)
    replay = ReplayCameraSession.from_artifact(root)
    with pytest.raises(LifecycleError, match="not open"):
        replay.poll_frame()
    replay.open()
    assert replay.poll_frame() is not None
    assert replay.poll_frame() is None
    assert replay.poll_frame() is None
    replay.close()
    with pytest.raises(LifecycleError, match="not open"):
        replay.poll_frame()


@pytest.mark.parametrize("mutation", ["array", "metadata", "manifest", "checksum"])
def test_capture_corruption_is_rejected(tmp_path: Path, mutation: str) -> None:
    root = tmp_path / "artifact"
    _write(root, 1)
    if mutation == "array":
        path = root / "frames/frame_000000.npz"
        payload = bytearray(path.read_bytes())
        payload[-8] ^= 1
        path.write_bytes(payload)
    elif mutation == "metadata":
        (root / "frames/frame_000000.meta.json").unlink()
    elif mutation == "manifest":
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["preview_notice"] = "changed"
        (root / "manifest.json").write_text(json.dumps(manifest))
    else:
        path = root / "checksums.sha256"
        text = path.read_text()
        path.write_text(("0" if text[0] != "0" else "1") + text[1:])
    with pytest.raises(ArtifactError):
        validate_capture_artifact(root)


@pytest.mark.parametrize("unsafe", ["/absolute/frame.npz", "../escape.npz"])
def test_capture_unsafe_paths_are_rejected(tmp_path: Path, unsafe: str) -> None:
    root = tmp_path / "artifact"
    _write(root, 1)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["frames"][0]["data_path"] = unsafe
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match="unsafe relative path"):
        validate_capture_artifact(root)


def test_capture_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write(root, 1)
    data = root / "frames/frame_000000.npz"
    outside = tmp_path / "outside.npz"
    shutil.copyfile(data, outside)
    data.unlink()
    data.symlink_to(outside)
    with pytest.raises(ArtifactError, match="symlink"):
        validate_capture_artifact(root)


def test_capture_schema_version_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write(root, 1)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["schema_version"] = "camera-rig.capture.v2"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match="schema_version"):
        validate_capture_artifact(root)


def test_factory_camera_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write(root, 1)
    factory_path = root / "factory_calibration.json"
    factory = json.loads(factory_path.read_text())
    factory["device"]["serial"] = "different-device"
    factory_path.write_text(json.dumps(factory))
    factory_digest = hashlib.sha256(factory_path.read_bytes()).hexdigest()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checksums"]["factory_calibration.json"] = factory_digest
    manifest_path.write_text(json.dumps(manifest))
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    checksum_path = root / "checksums.sha256"
    checksum_lines = []
    for line in checksum_path.read_text().splitlines():
        _, relative = line.split("  ", maxsplit=1)
        digest = {
            "factory_calibration.json": factory_digest,
            "manifest.json": manifest_digest,
        }.get(relative, line.split("  ", maxsplit=1)[0])
        checksum_lines.append(f"{digest}  {relative}")
    checksum_path.write_text("\n".join(checksum_lines) + "\n")
    with pytest.raises(ArtifactError, match="factory calibration camera identity"):
        validate_capture_artifact(root)


def test_snapshot_overwrites_existing_artifact_by_default(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write(root, 1)
    _write(root, 2)
    assert validate_capture_artifact(root)["frame_count"] == 2


def test_snapshot_replaces_valid_artifact_without_an_override_flag(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write(root, 1)
    write_snapshot(
        root,
        _frames(2),
        _factory(),
        {"copy_frames": True},
        {"source": "replacement"},
        include_previews=False,
    )
    assert validate_capture_artifact(root)["frame_count"] == 2
