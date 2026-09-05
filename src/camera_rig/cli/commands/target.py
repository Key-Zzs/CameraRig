"""Printable target generation and offline detector commands."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from camera_rig.targets.metrology import (
    TargetScaleAcceptance,
    build_manual_target_metrology_waiver,
    build_target_scale_acceptance_policy,
    evaluate_target_metrology,
    load_target_metrology,
    load_target_scale_acceptance_policy,
    write_target_metrology,
    write_target_scale_acceptance_policy,
)
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

    policy = subcommands.add_parser(
        "metrology-policy-create",
        help="freeze target-scale acceptance before physical measurements",
    )
    policy.add_argument("--target", type=Path, required=True)
    policy.add_argument("--allowed-translation-error-mm", type=float, required=True)
    policy.add_argument("--maximum-working-distance-mm", type=float, required=True)
    policy.add_argument("--authority", required=True)
    policy.add_argument("--output", type=Path, required=True)
    policy.set_defaults(handler=_create_metrology_policy)

    metrology = subcommands.add_parser(
        "metrology-create", help="create a measured physical-target metrology receipt"
    )
    metrology.add_argument("--target", type=Path, required=True)
    metrology.add_argument("--horizontal-square-count", type=int, required=True)
    metrology.add_argument("--vertical-square-count", type=int, required=True)
    metrology.add_argument("--horizontal-mm", type=float, action="append", required=True)
    metrology.add_argument("--vertical-mm", type=float, action="append", required=True)
    metrology.add_argument("--measurement-method", required=True)
    metrology.add_argument("--instrument", required=True)
    metrology.add_argument("--instrument-resolution-mm", type=float, required=True)
    metrology.add_argument("--measurement-uncertainty-mm", type=float, required=True)
    metrology.add_argument("--acceptance-policy", type=Path, required=True)
    metrology.add_argument("--operator", required=True)
    metrology.add_argument("--output", type=Path, required=True)
    metrology.set_defaults(handler=_create_metrology)

    waiver = subcommands.add_parser(
        "metrology-waiver-create",
        help="record an explicit user-authorized waiver of machine metrology",
    )
    waiver.add_argument("--target", type=Path, required=True)
    waiver.add_argument("--horizontal-square-count", type=int, required=True)
    waiver.add_argument("--vertical-square-count", type=int, required=True)
    waiver.add_argument("--reported-horizontal-mm", type=float, required=True)
    waiver.add_argument("--reported-vertical-mm", type=float, required=True)
    waiver.add_argument("--acceptance-policy", type=Path, required=True)
    waiver.add_argument("--authority", required=True)
    waiver.add_argument("--authorization-statement", required=True)
    waiver.add_argument("--output", type=Path, required=True)
    waiver.set_defaults(handler=_create_metrology_waiver)

    metrology_validate = subcommands.add_parser(
        "metrology-validate", help="validate a target metrology receipt and target binding"
    )
    metrology_validate.add_argument("--target", type=Path, required=True)
    metrology_validate.add_argument("--receipt", type=Path, required=True)
    metrology_validate.set_defaults(handler=_validate_metrology)

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
    validate.add_argument(
        "--policy",
        choices=("legacy_strict", "pose_validated", "uncertainty_validated"),
        default="legacy_strict",
    )
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
        "--policy",
        choices=("legacy_strict", "pose_validated", "uncertainty_validated"),
        default="pose_validated",
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


def _create_metrology(arguments: argparse.Namespace) -> int:
    target = validate_target_artifact(arguments.target)
    policy = load_target_scale_acceptance_policy(
        arguments.acceptance_policy, expected_target_sha256=target.artifact_sha256
    )
    receipt = evaluate_target_metrology(
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        target=target,
        horizontal_square_count=arguments.horizontal_square_count,
        vertical_square_count=arguments.vertical_square_count,
        horizontal_measurements_mm=tuple(arguments.horizontal_mm),
        vertical_measurements_mm=tuple(arguments.vertical_mm),
        measurement_method=arguments.measurement_method,
        instrument=arguments.instrument,
        instrument_resolution_mm=arguments.instrument_resolution_mm,
        measurement_uncertainty_mm=arguments.measurement_uncertainty_mm,
        acceptance_policy=policy,
        provenance={"operator": arguments.operator, "source": "physical_measurement"},
    )
    write_target_metrology(arguments.output, receipt)
    print(
        f"target metrology: {receipt.status} "
        f"(horizontal_scale={receipt.results['horizontal_scale']:.8f}, "
        f"vertical_scale={receipt.results['vertical_scale']:.8f})"
    )
    return 0 if receipt.status == "PASS" else 2


def _create_metrology_waiver(arguments: argparse.Namespace) -> int:
    target = validate_target_artifact(arguments.target)
    policy = load_target_scale_acceptance_policy(
        arguments.acceptance_policy, expected_target_sha256=target.artifact_sha256
    )
    receipt = build_manual_target_metrology_waiver(
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        target=target,
        horizontal_square_count=arguments.horizontal_square_count,
        vertical_square_count=arguments.vertical_square_count,
        reported_horizontal_mm=arguments.reported_horizontal_mm,
        reported_vertical_mm=arguments.reported_vertical_mm,
        acceptance_policy=policy,
        authority=arguments.authority,
        authorization_statement=arguments.authorization_statement,
    )
    write_target_metrology(arguments.output, receipt)
    print(
        f"target metrology manual waiver: {receipt.status} "
        f"(machine_gate={receipt.acceptance['machine_gate_status']})"
    )
    return 0 if receipt.status == "PASS" else 2


def _create_metrology_policy(arguments: argparse.Namespace) -> int:
    target = validate_target_artifact(arguments.target)
    policy = build_target_scale_acceptance_policy(
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        target=target,
        acceptance=TargetScaleAcceptance(
            allowed_translation_error_mm=arguments.allowed_translation_error_mm,
            maximum_working_distance_mm=arguments.maximum_working_distance_mm,
        ),
        provenance={
            "authority": arguments.authority,
            "measurement_values_available": False,
        },
    )
    write_target_scale_acceptance_policy(arguments.output, policy)
    print(f"target metrology acceptance policy: FROZEN ({policy['policy_fingerprint']})")
    return 0


def _validate_metrology(arguments: argparse.Namespace) -> int:
    target = validate_target_artifact(arguments.target)
    receipt = load_target_metrology(
        arguments.receipt,
        expected_target=target,
    )
    print(f"valid {receipt.schema_version}: status={receipt.status}")
    return 0 if receipt.status == "PASS" else 2


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
        policy=arguments.policy,
    )
    acceptance = report["acceptance"]
    assert isinstance(acceptance, dict)
    passed = acceptance["passed"] is True
    if arguments.policy == "uncertainty_validated":
        print(
            "target artifact validation: "
            f"NUMERICAL_{'PASS' if passed else 'FAIL'} RELEASE_HOLD "
            f"candidate_only=true ({report['frame_count']} frames)"
        )
    else:
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
    if arguments.policy == "uncertainty_validated":
        print(
            "target deployment preflight: "
            f"NUMERICAL_{report['status']} RELEASE_HOLD "
            f"recommendation={report['operator_recommendation']}"
        )
    else:
        print(
            f"target deployment preflight: {report['status']} "
            f"recommendation={report['recommendation']}"
        )
    if report["status"] != "PASS":
        raise ContractError(f"target deployment recommendation: {report['recommendation']}")
    return 0
