"""Printable target generation and offline detector commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from camera_rig.core.errors import ContractError
from camera_rig.targets.charuco.generator import generate_target_artifact
from camera_rig.targets.charuco.spec import load_charuco_target_spec
from camera_rig.targets.io import validate_target_artifact
from camera_rig.targets.validation import detect_image, validate_capture_artifact_target


def add_target_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = commands.add_parser("target", help="calibration target operations")
    subcommands = parser.add_subparsers(dest="target_command", required=True)

    generate = subcommands.add_parser("generate", help="generate a printable target artifact")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.set_defaults(handler=_generate)

    inspect = subcommands.add_parser("inspect", help="inspect a resolved target artifact")
    inspect.add_argument("--target", type=Path, required=True)
    inspect.set_defaults(handler=_inspect)

    detect = subcommands.add_parser("detect", help="detect a target in one offline image")
    detect.add_argument("--target", type=Path, required=True)
    detect.add_argument("--image", type=Path, required=True)
    detect.add_argument("--output", type=Path, required=True)
    detect.add_argument("--overlay", type=Path)
    detect.set_defaults(handler=_detect)

    validate = subcommands.add_parser(
        "validate-artifact", help="detect a target in every frame of a capture artifact"
    )
    validate.add_argument("--target", type=Path, required=True)
    validate.add_argument("--artifact", type=Path, required=True)
    validate.add_argument("--stream", default="color")
    validate.add_argument("--report", type=Path, required=True)
    validate.add_argument("--overlays", type=Path, required=True)
    validate.set_defaults(handler=_validate_artifact)


def _generate(arguments: argparse.Namespace) -> int:
    spec = load_charuco_target_spec(arguments.config)
    report = generate_target_artifact(spec, arguments.output)
    self_check = report["self_check"]
    assert isinstance(self_check, dict)
    expected = report["expected_charuco_corner_count"]
    print(
        f"target generation: {report['status']} "
        f"({self_check['detected_charuco_corner_count']}/{expected} corners)"
    )
    return 0


def _inspect(arguments: argparse.Namespace) -> int:
    target = validate_target_artifact(arguments.target)
    print(
        f"valid {target.SCHEMA_VERSION}: target={target.target_name!r}, "
        f"dictionary={target.dictionary}, squares={target.squares_x}x{target.squares_y}, "
        f"corners={len(target.corner_points)}, canonical_frame={target.target_frame!r}, "
        f"sha256={target.artifact_sha256}"
    )
    return 0


def _detect(arguments: argparse.Namespace) -> int:
    report = detect_image(
        target_path=arguments.target,
        image_path=arguments.image,
        report_path=arguments.output,
        overlay_path=arguments.overlay,
    )
    aggregate = report["aggregate"]
    assert isinstance(aggregate, dict)
    passed = float(aggregate["success_ratio"]) == 1.0
    print(f"target detection: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise ContractError("target detection quality gate failed; report was preserved")
    return 0


def _validate_artifact(arguments: argparse.Namespace) -> int:
    report = validate_capture_artifact_target(
        target_path=arguments.target,
        artifact_path=arguments.artifact,
        stream=arguments.stream,
        report_path=arguments.report,
        overlays_path=arguments.overlays,
    )
    acceptance = report["acceptance"]
    assert isinstance(acceptance, dict)
    passed = acceptance["passed"] is True
    print(
        f"target artifact validation: {'PASS' if passed else 'FAIL'} "
        f"({report['frame_count']} frames)"
    )
    if not passed:
        raise ContractError("target artifact acceptance failed; report and overlays were preserved")
    return 0
