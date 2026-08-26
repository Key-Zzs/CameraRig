"""Printable target generation and offline detector commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from camera_rig.config.loader import load_config
from camera_rig.core.errors import ContractError
from camera_rig.targets.charuco.generator import generate_target_artifact
from camera_rig.targets.charuco.identification import (
    ExistingBoardDimensions,
    identify_existing_board,
    register_existing_board,
)
from camera_rig.targets.charuco.spec import load_charuco_target_spec
from camera_rig.targets.io import validate_target_artifact
from camera_rig.targets.preflight import run_target_preflight
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

    identify = subcommands.add_parser(
        "identify-existing", help="rank ChArUco candidates for an existing physical board"
    )
    identify.add_argument("--image", type=Path, action="append", default=[])
    identify.add_argument("--artifact", type=Path, action="append", default=[])
    identify.add_argument("--stream", default="color")
    identify.add_argument("--board-width-mm", type=float, required=True)
    identify.add_argument("--board-height-mm", type=float, required=True)
    identify.add_argument("--square-length-mm", type=float, required=True)
    identify.add_argument("--marker-length-mm", type=float, required=True)
    identify.add_argument("--maximum-artifact-frames", type=int, default=8)
    identify.add_argument("--authoritative-source", type=Path)
    identify.add_argument("--authoritative-dictionary")
    identify.add_argument("--authoritative-legacy-pattern", choices=("true", "false"))
    identify.add_argument("--authoritative-border-bits", type=int, choices=(1, 2))
    identify.add_argument("--authoritative-orientation", choices=("portrait", "landscape"))
    identify.add_argument("--output", type=Path, required=True)
    identify.set_defaults(handler=_identify_existing)

    register = subcommands.add_parser(
        "register-existing", help="register a uniquely identified existing physical board"
    )
    register.add_argument("--identification", type=Path, required=True)
    register.add_argument("--target-name", required=True)
    register.add_argument("--target-frame", required=True)
    register.add_argument("--output", type=Path, required=True)
    register.set_defaults(handler=_register_existing)

    preflight = subcommands.add_parser(
        "preflight", help="capture pose-free target deployment quality evidence"
    )
    preflight.add_argument("--camera-config", type=Path, required=True)
    preflight.add_argument("--target", type=Path, required=True)
    preflight.add_argument("--frames", type=int, default=60)
    preflight.add_argument("--stream", default="color")
    preflight.add_argument(
        "--policy", choices=("legacy_strict", "pose_validated"), default="pose_validated"
    )
    preflight.add_argument("--report", type=Path, required=True)
    preflight.add_argument("--overlays", type=Path, required=True)
    preflight.set_defaults(handler=_preflight)


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


def _identify_existing(arguments: argparse.Namespace) -> int:
    report = identify_existing_board(
        image_paths=tuple(arguments.image),
        artifact_paths=tuple(arguments.artifact),
        stream=arguments.stream,
        dimensions=ExistingBoardDimensions(
            board_width_mm=arguments.board_width_mm,
            board_height_mm=arguments.board_height_mm,
            square_length_mm=arguments.square_length_mm,
            marker_length_mm=arguments.marker_length_mm,
        ),
        output=arguments.output,
        maximum_artifact_frames=arguments.maximum_artifact_frames,
        authoritative_source_path=arguments.authoritative_source,
        authoritative_dictionary=arguments.authoritative_dictionary,
        authoritative_legacy_pattern=(
            None
            if arguments.authoritative_legacy_pattern is None
            else arguments.authoritative_legacy_pattern == "true"
        ),
        authoritative_border_bits=arguments.authoritative_border_bits,
        authoritative_orientation=arguments.authoritative_orientation,
    )
    print(
        f"existing target identification: {report['status']} "
        f"classification={report['classification']}"
    )
    if report["status"] != "PASS":
        raise ContractError(str(report["ambiguity_reason"]))
    return 0


def _register_existing(arguments: argparse.Namespace) -> int:
    report = register_existing_board(
        identification_path=arguments.identification,
        target_name=arguments.target_name,
        target_frame=arguments.target_frame,
        output=arguments.output,
    )
    print(f"existing target registration: {report['status']} target={report['target_name']!r}")
    return 0


def _preflight(arguments: argparse.Namespace) -> int:
    report = run_target_preflight(
        camera_config=load_config(arguments.camera_config),
        target_path=arguments.target,
        frames=arguments.frames,
        stream=arguments.stream,
        policy=arguments.policy,
        report_path=arguments.report,
        overlays_path=arguments.overlays,
    )
    print(
        f"target deployment preflight: {report['status']} recommendation={report['recommendation']}"
    )
    if report["status"] != "PASS":
        raise ContractError(f"target deployment recommendation: {report['recommendation']}")
    return 0
