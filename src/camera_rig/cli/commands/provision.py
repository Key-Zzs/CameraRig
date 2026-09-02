"""One-command fixed-camera provisioning and artifact validation."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from camera_rig.core.errors import ArtifactError
from camera_rig.provision.artifact import (
    FixedProvisionArtifactInputs,
    ProvisionOverlayInputs,
    write_fixed_provision_artifact,
)
from camera_rig.provision.bundle import build_fixed_camera_bundle, write_fixed_camera_bundle
from camera_rig.provision.config import load_provision_config_with_sha256
from camera_rig.provision.preflight import preflight_fixed_provision
from camera_rig.provision.validation import load_and_validate_fixed_provision
from camera_rig.provision.workflow import run_fixed_provision_workflow
from camera_rig.version import __version__


def add_provision_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register provisioning commands on the root parser."""
    parser = commands.add_parser("provision", help="fixed-camera provisioning operations")
    groups = parser.add_subparsers(dest="provision_command", required=True)

    fixed = groups.add_parser("fixed", help="build a complete fixed-camera artifact")
    fixed.add_argument("--config", type=Path, required=True, help="one-YAML provision config")
    fixed.add_argument("--output", type=Path, required=True, help="final artifact directory")
    fixed.add_argument(
        "--dry-run",
        action="store_true",
        help="validate all non-hardware inputs without opening the camera or writing output",
    )
    fixed.set_defaults(handler=_provision_fixed)

    validate = groups.add_parser("validate", help="validate a complete provision artifact")
    validate.add_argument("--artifact", type=Path, required=True)
    validate.set_defaults(handler=_validate_provision)


def _provision_fixed(arguments: argparse.Namespace) -> int:
    config, config_sha256 = load_provision_config_with_sha256(arguments.config)
    plan = preflight_fixed_provision(config, output=arguments.output)
    if arguments.dry_run:
        enabled_streams = plan["enabled_streams"]
        if not isinstance(enabled_streams, list) or not all(
            isinstance(value, str) for value in enabled_streams
        ):
            raise ArtifactError("preflight enabled_streams result is invalid")
        dependencies = plan.get("optional_dependencies")
        dependency_summary = (
            ",".join(
                f"{name}={status}"
                for name, status in sorted(dependencies.items())
                if isinstance(name, str) and isinstance(status, str)
            )
            if isinstance(dependencies, dict)
            else "validated"
        )
        print("fixed provision dry-run: PASS")
        print(f"  streams: {','.join(enabled_streams)}")
        print(
            f"  acquisition: validation_frames={plan['stream_validation_frames']}, "
            f"calibration_frames={plan['calibration_frames']}, "
            f"selection={plan.get('selected_frame_policy', 'deterministic_evenly_spaced')}"
        )
        print(
            f"  target: artifact={plan.get('target_artifact', 'validated-relative-artifact')}, "
            f"sha256={plan.get('target_sha256', 'validated')}, "
            f"frame={plan.get('target_frame', 'validated')}"
        )
        print(
            f"  frames: workspace={plan.get('workspace_frame', 'validated')}, "
            f"detection_stream={plan.get('detection_stream', 'validated')}, "
            f"reference_stream={plan.get('reference_stream', 'validated')}"
        )
        print(
            f"  output_policy: exists={plan.get('output_exists', False)}, "
            "overwrite_existing=yes, camera_opened=no, output_written=no"
        )
        print(f"  optional_dependencies: {dependency_summary}")
        return 0

    output = arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.workflow-"))
    artifact_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        result = run_fixed_provision_workflow(config, staging)
        bundle = build_fixed_camera_bundle(
            bundle_id=artifact_id,
            created_at=created_at,
            factory=result.factory_calibration,
            stream_validation=result.stream_validation,
            target_detection=result.target_detection,
            fixed_calibration=result.fixed_calibration,
            provenance={
                "camera_rig_version": __version__,
                "config_sha256": config_sha256,
                "git_commit": _git_commit(),
                "workflow": "fixed-provision",
            },
        )
        bundle_path = staging / "camera_bundle.json"
        write_fixed_camera_bundle(bundle_path, bundle)
        manifest = write_fixed_provision_artifact(
            output,
            FixedProvisionArtifactInputs(
                camera_bundle=bundle_path,
                factory_calibration=staging / result.files["factory_calibration"],
                capture_artifact=(staging / result.files["capture_manifest"]).parent,
                target_spec=staging / result.files["target_spec"],
                target_detection=staging / result.files["target_detection"],
                fixed_calibration=staging / result.files["fixed_calibration"],
                stream_validation=staging / result.files["stream_validation"],
                target_detection_overlays=_overlay_inputs(staging, result.detection_overlays),
                fixed_calibration_overlays=_overlay_inputs(staging, result.fixed_overlays),
            ),
            artifact_id=artifact_id,
            created_at=created_at,
            provenance={
                "camera_rig_version": __version__,
                "config_sha256": config_sha256,
                "git_commit": _git_commit(),
                "selected_source_indices": list(result.selected_source_indices),
                "selection_method": "deterministic_evenly_spaced_inclusive",
                "workflow": "fixed-provision",
            },
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(
        "fixed provision: PASS "
        f"(artifact_id={manifest.artifact_id}, validation_frames="
        f"{config.acquisition.stream_validation_frames}, calibration_frames="
        f"{config.acquisition.calibration_frames})"
    )
    return 0


def _validate_provision(arguments: argparse.Namespace) -> int:
    manifest = load_and_validate_fixed_provision(arguments.artifact)
    print(
        f"valid {manifest.schema_version}: artifact_id={manifest.artifact_id!r}, "
        "status='passed', checksums=passed, bundle=passed"
    )
    return 0


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    value = " ".join(result.stdout.split()) if result.returncode == 0 else ""
    return value or "unknown"


def _overlay_inputs(root: Path, values: tuple[str, ...]) -> ProvisionOverlayInputs:
    by_label: dict[str, Path] = {}
    for value in values:
        name = Path(value).name
        for label in ("best", "median_quality", "worst_accepted"):
            if name.startswith(f"{label}_"):
                by_label[label] = root / value
                break
    if set(by_label) != {"best", "median_quality", "worst_accepted"}:
        raise ArtifactError("workflow must produce best, median-quality, and worst overlays")
    return ProvisionOverlayInputs(
        best=by_label["best"],
        median_quality=by_label["median_quality"],
        worst_accepted=by_label["worst_accepted"],
    )
