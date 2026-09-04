from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from camera_rig.cli.main import main
from camera_rig.config.validation import validate_against_named_schema
from camera_rig.core.errors import ArtifactError, ContractError, SchemaValidationError
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


def test_fixed_preflight_accepts_existing_valid_output_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_provision_config(EXAMPLE)
    _patch_target(monkeypatch, config.target.expected_sha256)
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(
        "camera_rig.provision.validation.load_and_validate_fixed_provision",
        lambda _output: object(),
    )
    result = preflight_fixed_provision(config, output=output)
    assert result["overwrite_existing"] is True


def test_fixed_preflight_default_replacement_rejects_unowned_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_provision_config(EXAMPLE)
    _patch_target(monkeypatch, config.target.expected_sha256)
    output = tmp_path / "unowned"
    output.mkdir()
    (output / "user-file.txt").write_text("preserve")
    with pytest.raises(ArtifactError, match="default replacement requires"):
        preflight_fixed_provision(config, output=output)
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
        "target_detection_policy": "uncertainty_validated",
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
    assert "INPUTS_PASS RELEASE_HOLD" in captured.out
    assert "canonical_publication_blocked=true" in captured.out
    assert "camera_opened=no" in captured.out
    assert "output_written=no" in captured.out
    assert captured.err == ""
    assert not output.exists()


def test_live_provision_preflight_cli_reports_would_pass_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "camera_rig.cli.commands.provision.run_fixed_provision_preflight",
        lambda *_args, **_kwargs: {
            "status": "PASS",
            "would_publish_fixed_provision": True,
            "fixed_pose_frames": {"frame_gate_accepted": 58, "required_frames": 50},
        },
    )
    report = tmp_path / "report.json"
    overlays = tmp_path / "overlays"
    assert (
        main(
            [
                "provision",
                "preflight",
                "--config",
                str(EXAMPLE),
                "--report",
                str(report),
                "--overlays",
                str(overlays),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "fixed provision preflight: PASS" in captured.out
    assert "frame_gate_accepted=58" in captured.out
    assert captured.err == ""


def test_failed_new_provision_warns_that_existing_output_is_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(
        "camera_rig.cli.commands.provision.preflight_fixed_provision",
        lambda *_args, **_kwargs: {"enabled_streams": []},
    )
    monkeypatch.setattr(
        "camera_rig.cli.commands.provision.run_fixed_provision_workflow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ContractError("synthetic failure")),
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
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "NEW_PROVISION_ATTEMPT=FAIL" in captured.err
    assert "EXISTING_OUTPUT_UNCHANGED=true" in captured.err
    assert "DO_NOT_TREAT_EXISTING_VALIDATE_AS_THIS_ATTEMPT" in captured.err


def test_failed_new_provision_preserves_staged_diagnostic_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "new-artifact"
    monkeypatch.setattr(
        "camera_rig.cli.commands.provision.preflight_fixed_provision",
        lambda *_args, **_kwargs: {"enabled_streams": []},
    )

    def fail_after_evidence(_config: object, staging: Path) -> None:
        evidence = staging / "calibration/fixed_calibration.failed.json"
        evidence.parent.mkdir(parents=True)
        evidence.write_text('{"status":"failed"}\n')
        raise ContractError("synthetic quality failure")

    monkeypatch.setattr(
        "camera_rig.cli.commands.provision.run_fixed_provision_workflow",
        fail_after_evidence,
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
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    marker = next(line for line in captured.err.splitlines() if line.startswith("FAILED_ATTEMPT"))
    evidence_root = tmp_path / marker.split("=", 1)[1]
    assert (evidence_root / "calibration/fixed_calibration.failed.json").is_file()
    assert not output.exists()


def test_bare_filesystem_failure_reports_attempt_and_preserves_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "new-artifact"
    monkeypatch.setattr(
        "camera_rig.cli.commands.provision.preflight_fixed_provision",
        lambda *_args, **_kwargs: {"enabled_streams": []},
    )

    def fail_with_oserror(_config: object, staging: Path) -> None:
        evidence = staging / "reports/io-failure.json"
        evidence.parent.mkdir(parents=True)
        evidence.write_text("{}\n")
        raise OSError("synthetic rename failure")

    monkeypatch.setattr(
        "camera_rig.cli.commands.provision.run_fixed_provision_workflow",
        fail_with_oserror,
    )
    with pytest.raises(OSError, match="synthetic rename failure"):
        main(["provision", "fixed", "--config", str(EXAMPLE), "--output", str(output)])
    captured = capsys.readouterr()
    assert "NEW_PROVISION_ATTEMPT=FAIL" in captured.err
    assert "FAILED_ATTEMPT_EVIDENCE=" in captured.err


def test_failed_new_provision_reports_stale_output_when_static_preflight_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(
        "camera_rig.cli.commands.provision.preflight_fixed_provision",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ArtifactError("bad target")),
    )
    assert main(["provision", "fixed", "--config", str(EXAMPLE), "--output", str(output)]) == 2
    captured = capsys.readouterr()
    assert "NEW_PROVISION_ATTEMPT=FAIL" in captured.err
    assert "EXISTING_OUTPUT_UNCHANGED=true" in captured.err


def test_fixed_preflight_schema_rejects_empty_nested_sections() -> None:
    invalid = {
        "schema_version": "camera-rig.fixed-provision-preflight.v1",
        "attempt_id": "63338aaa-ad79-4f9a-98cb-e86c742d7abc",
        "status": "PASS",
        "would_publish_fixed_provision": True,
        "camera": {"logical_name": "head"},
        "target_fingerprint": "a" * 64,
        "pose_policy": "uncertainty_validated",
        "evaluation_core": "run_fixed_provision_workflow",
        "raw_stream": {"status": "PASS", "metrics": {}, "failure_reasons": []},
        "target": {"status": "PASS"},
        "fixed_pose_frames": {"status": "EVALUATED"},
        "reprojection": {"status": "EVALUATED"},
        "observability": {"status": "EVALUATED"},
        "final": {"status": "EVALUATED", "decision": "WOULD_PASS", "failure_reasons": []},
        "per_frame": [],
        "failure_reasons": [],
        "publication": {
            "camera_bundle_written": False,
            "fixed_provision_written": False,
            "canonical_output_modified": False,
        },
    }
    with pytest.raises(SchemaValidationError):
        validate_against_named_schema(invalid, "fixed_provision_preflight.v1.schema.json")
