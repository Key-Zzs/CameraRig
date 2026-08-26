from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from camera_rig.cli.main import main
from camera_rig.core.errors import ArtifactError
from camera_rig.provision.config import load_provision_config
from camera_rig.provision.preflight import preflight_fixed_provision

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "configs/examples/fixed_provision_contract.yaml"


def _patch_target(monkeypatch: pytest.MonkeyPatch, expected_sha256: str) -> None:
    monkeypatch.setattr(
        "camera_rig.provision.preflight.validate_target_artifact",
        lambda _path: SimpleNamespace(
            artifact_sha256=expected_sha256,
            target_frame="charuco_target",
        ),
    )
    monkeypatch.setattr(
        "camera_rig.provision.preflight._require_runtime_dependencies", lambda: None
    )


def test_fixed_preflight_is_read_only_and_reports_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_provision_config(EXAMPLE)
    _patch_target(monkeypatch, config.target.expected_sha256)
    output = tmp_path / "result"
    result = preflight_fixed_provision(config, output=output)
    assert result["camera_will_open"] is False
    assert result["final_artifact_will_be_created"] is False
    assert result["calibration_frames"] == 60
    assert not output.exists()


def test_fixed_preflight_rejects_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_provision_config(EXAMPLE)
    _patch_target(monkeypatch, config.target.expected_sha256)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ArtifactError, match="already exists"):
        preflight_fixed_provision(config, output=output)


def test_fixed_preflight_force_rejects_unowned_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_provision_config(EXAMPLE)
    _patch_target(monkeypatch, config.target.expected_sha256)
    output = tmp_path / "unowned"
    output.mkdir()
    (output / "user-file.txt").write_text("preserve")
    with pytest.raises(ArtifactError, match="only an existing validated"):
        preflight_fixed_provision(config, output=output, force=True)
    assert (output / "user-file.txt").read_text() == "preserve"


def test_fixed_preflight_rejects_wrong_target_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_provision_config(EXAMPLE)
    _patch_target(monkeypatch, "0" * 64)
    with pytest.raises(ArtifactError, match="target SHA"):
        preflight_fixed_provision(config, output=tmp_path / "result")


def test_fixed_preflight_rejects_wrong_target_coordinate_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_provision_config(EXAMPLE)
    monkeypatch.setattr(
        "camera_rig.provision.preflight.validate_target_artifact",
        lambda _path: SimpleNamespace(
            artifact_sha256=config.target.expected_sha256,
            target_frame="different_target",
        ),
    )
    monkeypatch.setattr(
        "camera_rig.provision.preflight._require_runtime_dependencies", lambda: None
    )
    with pytest.raises(ArtifactError, match="coordinate frame"):
        preflight_fixed_provision(config, output=tmp_path / "result")


def test_fixed_provision_cli_dry_run_never_enters_live_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "artifact"
    plan: dict[str, object] = {
        "enabled_streams": ["color", "depth", "ir_left", "ir_right"],
        "stream_validation_frames": 300,
        "calibration_frames": 60,
    }
    monkeypatch.setattr(
        "camera_rig.cli.commands.provision.preflight_fixed_provision",
        lambda *_args, **_kwargs: plan,
    )

    def forbidden_workflow(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run entered the live workflow")

    monkeypatch.setattr(
        "camera_rig.cli.commands.provision.run_fixed_provision_workflow",
        forbidden_workflow,
    )
    assert (
        main(
            [
                "provision",
                "fixed",
                "--config",
                str(EXAMPLE),
                "--output",
                str(output),
                "--dry-run",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "camera_opened=no" in captured.out
    assert "output_written=no" in captured.out
    assert captured.err == ""
    assert not output.exists()
