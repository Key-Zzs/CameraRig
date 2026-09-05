from __future__ import annotations

from pathlib import Path

import pytest

from camera_rig.artifacts.io import atomic_write_json, load_json
from camera_rig.core.errors import ArtifactError, ContractError
from camera_rig.targets.io import validate_target_artifact
from camera_rig.targets.metrology import (
    TargetScaleAcceptance,
    build_target_scale_acceptance_policy,
    evaluate_target_metrology,
    load_target_metrology,
    load_target_scale_acceptance_policy,
    validate_target_scale_acceptance_policy,
    write_target_metrology,
    write_target_scale_acceptance_policy,
)


def _receipt(target_path: Path, *, horizontal_scale: float = 1.0, vertical_scale: float = 1.0):
    target = validate_target_artifact(target_path)
    policy = build_target_scale_acceptance_policy(
        created_at="2026-09-05T07:00:00Z",
        target=target,
        acceptance=TargetScaleAcceptance(3.0, 1500.0),
        provenance={"authority": "test contract", "measurement_values_available": False},
    )
    h_count = min(5, target.squares_x)
    v_count = min(4, target.squares_y)
    nominal_h = h_count * target.square_length_m * 1000.0
    nominal_v = v_count * target.square_length_m * 1000.0
    return evaluate_target_metrology(
        created_at="2026-09-05T08:00:00Z",
        target=target,
        horizontal_square_count=h_count,
        vertical_square_count=v_count,
        horizontal_measurements_mm=tuple(nominal_h * horizontal_scale for _ in range(3)),
        vertical_measurements_mm=tuple(nominal_v * vertical_scale for _ in range(3)),
        measurement_method="long-baseline repeated caliper measurement",
        instrument="digital caliper",
        instrument_resolution_mm=0.01,
        measurement_uncertainty_mm=0.05,
        acceptance_policy=policy,
        provenance={"operator": "fixture"},
    )


def test_correct_target_scale_passes_and_roundtrips(
    generated_charuco_target: Path, tmp_path: Path
) -> None:
    receipt = _receipt(generated_charuco_target / "target_spec.json")
    output = tmp_path / "target_metrology.json"
    write_target_metrology(output, receipt)
    loaded = load_target_metrology(
        output,
        expected_target_sha256=validate_target_artifact(
            generated_charuco_target / "target_spec.json"
        ).artifact_sha256,
        require_pass=True,
    )
    assert loaded.status == "PASS"
    assert loaded.results["horizontal_scale"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("horizontal_scale", "vertical_scale"),
    [(1.01, 1.01), (1.0, 1.01)],
)
def test_uniform_or_anisotropic_scale_error_fails(
    generated_charuco_target: Path,
    horizontal_scale: float,
    vertical_scale: float,
) -> None:
    receipt = _receipt(
        generated_charuco_target / "target_spec.json",
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
    )
    assert receipt.status == "FAIL"


def test_insufficient_measurements_are_rejected(generated_charuco_target: Path) -> None:
    target = validate_target_artifact(generated_charuco_target / "target_spec.json")
    with pytest.raises(ContractError, match="at least three"):
        policy = build_target_scale_acceptance_policy(
            created_at="2026-09-05T07:00:00Z",
            target=target,
            acceptance=TargetScaleAcceptance(3.0, 1500.0),
            provenance={"authority": "test contract", "measurement_values_available": False},
        )
        evaluate_target_metrology(
            created_at="2026-09-05T08:00:00Z",
            target=target,
            horizontal_square_count=2,
            vertical_square_count=2,
            horizontal_measurements_mm=(60.0, 60.0),
            vertical_measurements_mm=(60.0, 60.0),
            measurement_method="caliper",
            instrument="caliper",
            instrument_resolution_mm=0.01,
            measurement_uncertainty_mm=0.05,
            acceptance_policy=policy,
            provenance={},
        )


def test_units_and_target_fingerprint_tampering_are_rejected(
    generated_charuco_target: Path, tmp_path: Path
) -> None:
    target_path = generated_charuco_target / "target_spec.json"
    receipt = _receipt(target_path)
    value = receipt.to_dict()
    measurement = value["measurement"]
    assert isinstance(measurement, dict)
    measurement["units"] = "inch"
    output = tmp_path / "target_metrology.json"
    atomic_write_json(output, value)
    with pytest.raises(ArtifactError, match="units"):
        load_target_metrology(output)
    atomic_write_json(output, receipt.to_dict())
    with pytest.raises(ArtifactError, match="fingerprint"):
        load_target_metrology(output, expected_target_sha256="f" * 64)


def test_uncertainty_consumes_scale_budget(generated_charuco_target: Path) -> None:
    target = validate_target_artifact(generated_charuco_target / "target_spec.json")
    nominal_h = 5 * target.square_length_m * 1000.0
    nominal_v = 4 * target.square_length_m * 1000.0
    receipt = evaluate_target_metrology(
        created_at="2026-09-05T08:00:00Z",
        target=target,
        horizontal_square_count=5,
        vertical_square_count=4,
        horizontal_measurements_mm=(nominal_h, nominal_h, nominal_h),
        vertical_measurements_mm=(nominal_v, nominal_v, nominal_v),
        measurement_method="ruler",
        instrument="ruler",
        instrument_resolution_mm=1.0,
        measurement_uncertainty_mm=4.0,
        acceptance_policy=build_target_scale_acceptance_policy(
            created_at="2026-09-05T07:00:00Z",
            target=target,
            acceptance=TargetScaleAcceptance(3.0, 1500.0),
            provenance={"authority": "test contract", "measurement_values_available": False},
        ),
        provenance={},
    )
    assert receipt.status == "FAIL"
    assert receipt.results["checks"]["positive_scale_budget_after_uncertainty"] is False  # type: ignore[index]


def test_deleted_required_measurement_field_is_rejected(
    generated_charuco_target: Path, tmp_path: Path
) -> None:
    value = _receipt(generated_charuco_target / "target_spec.json").to_dict()
    measurement = value["measurement"]
    assert isinstance(measurement, dict)
    measurement.pop("instrument")
    output = tmp_path / "target_metrology.json"
    atomic_write_json(output, value)
    assert isinstance(load_json(output), dict)
    with pytest.raises(ArtifactError, match="measurement fields"):
        load_target_metrology(output)


def test_acceptance_policy_is_frozen_and_tamper_evident(
    generated_charuco_target: Path, tmp_path: Path
) -> None:
    target = validate_target_artifact(generated_charuco_target / "target_spec.json")
    policy = build_target_scale_acceptance_policy(
        created_at="2026-09-05T07:00:00Z",
        target=target,
        acceptance=TargetScaleAcceptance(3.0, 1500.0),
        provenance={"authority": "test contract", "measurement_values_available": False},
    )
    output = tmp_path / "target_scale_acceptance_policy.json"
    write_target_scale_acceptance_policy(output, policy)
    assert (
        load_target_scale_acceptance_policy(output, expected_target_sha256=target.artifact_sha256)
        == policy
    )
    policy["maximum_working_distance_mm"] = 1000.0
    atomic_write_json(output, policy)
    with pytest.raises(ArtifactError, match=r"derivation|fingerprint"):
        load_target_scale_acceptance_policy(output)


def test_receipt_is_bound_to_embedded_and_expected_frozen_policy(
    generated_charuco_target: Path, tmp_path: Path
) -> None:
    target_path = generated_charuco_target / "target_spec.json"
    receipt = _receipt(target_path)
    output = tmp_path / "target_metrology.json"
    write_target_metrology(output, receipt)
    forged_policy = dict(receipt.acceptance_policy)
    forged_policy["allowed_translation_error_mm"] = 100.0
    with pytest.raises(ArtifactError):
        load_target_metrology(
            output,
            expected_target=validate_target_artifact(target_path),
            expected_acceptance_policy=forged_policy,
            require_pass=True,
        )


def test_policy_must_declare_no_measurements_were_available(
    generated_charuco_target: Path,
) -> None:
    target = validate_target_artifact(generated_charuco_target / "target_spec.json")
    policy = build_target_scale_acceptance_policy(
        created_at="2026-09-05T07:00:00Z",
        target=target,
        acceptance=TargetScaleAcceptance(3.0, 1500.0),
        provenance={"authority": "test contract", "measurement_values_available": True},
    )
    with pytest.raises(ArtifactError, match="predate readings"):
        validate_target_scale_acceptance_policy(policy)


@pytest.mark.parametrize(
    ("container", "field", "value"),
    [
        ("results", "checks", {"forged": True}),
        ("nominal", "horizontal_baseline_mm", 100.0),
        ("measurement", "horizontal_type_a_expanded_uncertainty_mm", 1.0),
    ],
)
def test_rehashed_semantic_metrology_forgery_is_rejected(
    generated_charuco_target: Path,
    tmp_path: Path,
    container: str,
    field: str,
    value: object,
) -> None:
    receipt = _receipt(generated_charuco_target / "target_spec.json")
    raw = receipt.to_dict()
    selected = raw[container]
    assert isinstance(selected, dict)
    selected[field] = value
    output = tmp_path / "forged.json"
    atomic_write_json(output, raw)
    with pytest.raises(ArtifactError):
        load_target_metrology(output)
