from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from camera_rig.artifacts.io import atomic_write_json, json_safe
from camera_rig.artifacts.stream_validation import (
    StreamValidationArtifact,
    load_and_validate_stream_validation,
    validate_stream_validation_data,
    write_stream_validation,
)
from camera_rig.cli.commands import capture as capture_commands
from camera_rig.cli.main import main
from camera_rig.core.errors import ArtifactError, ContractError

REPOSITORY_ROOT = Path(__file__).parents[1]
CAMERA_CONFIG = REPOSITORY_ROOT / "configs/examples/single_camera_contract.yaml"
STREAMS = ("color", "depth", "ir_left", "ir_right")


def _report() -> dict[str, object]:
    return {
        "schema_version": "camera-rig.stream-validation.v1",
        "status": "PASS",
        "requested_frames": 3,
        "received_frames": 3,
        "duration_s": 0.1,
        "per_stream_observed_fps": {name: 30.0 for name in STREAMS},
        "per_stream_frame_number_discontinuities": {name: 0 for name in STREAMS},
        "per_stream_discontinuity_ratio": {name: 0.0 for name in STREAMS},
        "per_stream_timestamp_monotonicity": {name: True for name in STREAMS},
        "per_stream_timestamp_domain_counts": {name: {"hardware_clock": 3} for name in STREAMS},
        "ir_stereo_frame_match_ratio": 1.0,
        "comparable_timestamp_skew_ns": {"p50": 0.0, "p95": 0.0, "max": 0},
        "sync_valid_ratio": 1.0,
        "timeouts": 0,
        "missing_streams": {},
        "shape_consistency": {
            "color": [[480, 640, 3]],
            "depth": [[480, 640]],
            "ir_left": [[480, 640]],
            "ir_right": [[480, 640]],
        },
        "dtype_consistency": {
            "color": ["uint8"],
            "depth": ["uint16"],
            "ir_left": ["uint8"],
            "ir_right": ["uint8"],
        },
        "depth_valid_ratio": 0.9,
        "rgb_variance": 100.0,
        "rgb_channel_variance": [90.0, 100.0, 110.0],
        "ir_variance": {"ir_left": 80.0, "ir_right": 85.0},
        "ir_distinct_ratio": 1.0,
        "failure_reasons": [],
    }


def _artifact_data() -> dict[str, object]:
    artifact = StreamValidationArtifact.from_accumulator_report(
        _report(), provenance={"camera_rig_version": "0.3.0", "config_sha256": "a" * 64}
    )
    return artifact.to_dict()


def test_writer_round_trip_preserves_measurements_and_derives_quality(tmp_path: Path) -> None:
    path = tmp_path / "stream_validation.json"
    restored = write_stream_validation(
        path,
        _report(),
        provenance={"camera_rig_version": "0.3.0", "config_sha256": "a" * 64},
    )
    assert restored == load_and_validate_stream_validation(path)
    assert restored.status == "PASS"
    assert restored.quality.passed
    assert restored.quality.metrics == {
        "requested_frames": 3,
        "received_frames": 3,
        "sync_valid_ratio": 1.0,
        "timeouts": 0,
    }
    assert restored.statistics["rgb_channel_variance"] == [90.0, 100.0, 110.0]


def test_failed_report_is_valid_and_preserves_failure_reasons(tmp_path: Path) -> None:
    report = _report()
    report["status"] = "FAIL"
    report["received_frames"] = 2
    report["timeouts"] = 1
    report["failure_reasons"] = ["received frame count differs from request"]
    restored = write_stream_validation(
        tmp_path / "failed.json", report, provenance={"source": "unit-test"}
    )
    assert not restored.quality.passed
    assert restored.failure_reasons == ("received frame count differs from request",)


def test_status_quality_corruption_fails_closed() -> None:
    data = _artifact_data()
    quality = data["quality"]
    assert isinstance(quality, dict)
    quality["passed"] = False
    with pytest.raises(ArtifactError, match=r"\$\.quality\.passed"):
        validate_stream_validation_data(json_safe(data))


def test_quality_metric_corruption_fails_typed_consistency() -> None:
    data = _artifact_data()
    quality = data["quality"]
    assert isinstance(quality, dict)
    metrics = quality["metrics"]
    assert isinstance(metrics, dict)
    metrics["received_frames"] = 2
    with pytest.raises(ArtifactError, match="quality is inconsistent"):
        validate_stream_validation_data(json_safe(data))


def test_failure_reason_corruption_fails_consistency() -> None:
    data = _artifact_data()
    data["status"] = "FAIL"
    data["failure_reasons"] = ["injected failure"]
    quality = data["quality"]
    assert isinstance(quality, dict)
    quality["passed"] = False
    quality["failure_reasons"] = ["different failure"]
    with pytest.raises(ArtifactError, match="quality is inconsistent"):
        validate_stream_validation_data(json_safe(data))


def test_unknown_or_invalid_statistic_fails_schema() -> None:
    data = _artifact_data()
    data["unexpected"] = True
    with pytest.raises(ArtifactError, match="Additional properties"):
        validate_stream_validation_data(json_safe(data))

    data = _artifact_data()
    data["sync_valid_ratio"] = 1.1
    with pytest.raises(ArtifactError, match=r"\$\.sync_valid_ratio"):
        validate_stream_validation_data(json_safe(data))


@pytest.mark.parametrize("unsafe", ["/home/user/config.yaml", "C:/rig/config.yaml", "file://x"])
def test_nonportable_provenance_is_rejected(unsafe: str) -> None:
    with pytest.raises(ContractError, match="provenance"):
        StreamValidationArtifact.from_accumulator_report(
            _report(), provenance={"nested": {"config": unsafe}}
        )


def test_corrupted_persisted_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "stream_validation.json"
    data = copy.deepcopy(_artifact_data())
    data["schema_version"] = "camera-rig.stream-validation.v2"
    atomic_write_json(path, data)
    with pytest.raises(ArtifactError, match="schema_version"):
        load_and_validate_stream_validation(path)


def test_capture_validate_streams_cli_writes_strict_artifact_without_behavior_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeAccumulator:
        def __init__(self, _config: object, requested_frames: int) -> None:
            self.requested_frames = requested_frames

        def add(self, _frame: object) -> None:
            return None

        def report(self, _timeout_count: int) -> dict[str, object]:
            report = _report()
            report["requested_frames"] = self.requested_frames
            report["received_frames"] = self.requested_frames
            return report

    class FakeSession:
        @classmethod
        def from_config(cls, _config: object) -> FakeSession:
            return cls()

        def __enter__(self) -> FakeSession:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def capture(self) -> object:
            return object()

    monkeypatch.setattr(capture_commands, "StreamValidationAccumulator", FakeAccumulator)
    monkeypatch.setattr(capture_commands, "CameraSession", FakeSession)
    report_path = tmp_path / "stream_validation.json"
    assert (
        main(
            [
                "capture",
                "validate-streams",
                "--config",
                str(CAMERA_CONFIG),
                "--frames",
                "3",
                "--report",
                str(report_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    assert output.out == "stream validation: PASS (3/3 frames)\n"
    assert output.err == ""
    restored = load_and_validate_stream_validation(report_path)
    assert restored.quality.passed
    assert restored.provenance["config_sha256"]


def test_stream_artifact_import_does_not_request_optional_runtime_packages() -> None:
    script = """
import importlib.abc
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pyrealsense2', 'cv2', 'PIL'}:
            raise RuntimeError(f'forbidden optional import: {fullname}')
        return None

sys.meta_path.insert(0, Blocker())
from camera_rig.artifacts.stream_validation import StreamValidationArtifact
print(StreamValidationArtifact.__name__)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "StreamValidationArtifact"
