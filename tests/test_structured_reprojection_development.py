from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from camera_rig.artifacts.hashing import sha256_file
from camera_rig.artifacts.io import atomic_write_json
from camera_rig.core.errors import ContractError

pytest.importorskip("cv2")


def _namespace() -> dict[str, object]:
    return runpy.run_path(
        str(Path(__file__).parents[1] / "benchmarks" / "structured_reprojection_gate.py")
    )


def test_split_manifest_is_family_bound_and_holdout_is_not_evaluated_by_design() -> None:
    namespace = _namespace()
    manifest = namespace["build_split_manifest"]()  # type: ignore[operator]
    assert manifest["schema_version"] == "camera-rig.structured-reprojection-split.v1"
    assert manifest["counts"] == {"development": 2895, "holdout": 705}
    assert len(manifest["families"]) == 3600
    assert manifest["split_unit"] == "family_id"
    assert "descendants remain together" in manifest["family_binding"]
    assert "no metrics" in manifest["holdout_access_policy"]


def test_tracked_split_receipt_binds_the_deterministic_generated_manifest(tmp_path: Path) -> None:
    namespace = _namespace()
    generated = tmp_path / "split.json"
    manifest = namespace["build_split_manifest"]()  # type: ignore[operator]
    atomic_write_json(generated, manifest)
    receipt_path = (
        Path(__file__).parents[1] / "release_manifests" / "structured_reprojection_split_v1.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "FROZEN_UNOPENED"
    assert receipt["development_family_count"] == 2895
    assert receipt["holdout_family_count"] == 705
    assert receipt["generated_split_manifest_sha256"] == sha256_file(generated)
    assert receipt["assignment_salt"] == manifest["split_salt"]
    assert receipt["generator_version"] == manifest["generator_version"]
    assert receipt["planned_holdout_final_family_count"] == 55
    assert receipt["holdout_zero_error_wilson_upper_95"] > 0.05
    assert receipt["release_eligible"] is False


def test_development_receipt_records_hold_without_opening_holdout() -> None:
    receipt_path = (
        Path(__file__).parents[1]
        / "release_manifests"
        / "structured_reprojection_development_hold_v1.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    final = receipt["final_shared_pose_development"]
    scan = receipt["threshold_scan"]
    assert receipt["status"] == "DEVELOPMENT_FAILED_HOLD"
    assert receipt["holdout_metrics_opened"] is False
    assert receipt["preregistered_holdout_evaluation"] is False
    assert receipt["release_eligible"] is False
    assert receipt["release_decision"] == "HOLD"
    assert receipt["final_shared_pose_variants_per_family"] == 10
    assert "not a release suite" in receipt["final_shared_pose_variant_scope"]
    assert final["positive_false_reject"] == 16
    assert final["positive_n"] == 223
    assert final["negative_false_accept"] == 200
    assert final["negative_n"] == 228
    assert final["challenging_negative_false_accept"] == 200
    assert final["challenging_negative_n"] == 207
    assert scan == {
        "candidate_count": 4500,
        "development_target_met_count": 0,
        "recommendation": "HOLD",
    }


def test_development_rejects_a_tampered_split_before_evaluation(tmp_path: Path) -> None:
    namespace = _namespace()
    manifest = namespace["build_split_manifest"]()  # type: ignore[operator]
    original = manifest["families"][0]["assignment"]
    manifest["families"][0]["assignment"] = (
        "holdout" if original == "development" else "development"
    )
    path = tmp_path / "tampered.json"
    atomic_write_json(path, manifest)
    with pytest.raises(ContractError, match="differs from deterministic generator"):
        namespace["run_development"](path)  # type: ignore[operator]


def test_family_worst_case_aggregation_does_not_count_descendants_as_independent() -> None:
    namespace = _namespace()
    rows = [
        {"family": {"family_id": "same"}, "error": False},
        {"family": {"family_id": "same"}, "error": True},
        {"family": {"family_id": "other"}, "error": False},
    ]
    errors, total = namespace["_family_error_count"](  # type: ignore[operator]
        rows,
        eligible=lambda _row: True,
        error=lambda row: row["error"],
    )
    assert (errors, total) == (1, 2)
    assert namespace["_wilson_upper"](90, 90) == 1.0  # type: ignore[operator]


def test_principal_xy_name_matches_same_sign_axis_perturbation() -> None:
    namespace = _namespace()
    points, rows, columns = namespace["_geometry"]("a4_30mm")  # type: ignore[operator]
    intrinsics = namespace["_intrinsics"]("d435i_wide")  # type: ignore[operator]
    variant = namespace["Variant"]("principal_xy_+5.0px", "principal", 5.0)  # type: ignore[operator]
    case = namespace["_model_case"](points, rows, columns, intrinsics, variant)  # type: ignore[operator]
    assert case.assumed_intrinsics.cx == pytest.approx(intrinsics.cx + 5.0)
    assert case.assumed_intrinsics.cy == pytest.approx(intrinsics.cy + 5.0)


def test_development_cases_include_noisy_correct_and_confidently_wrong() -> None:
    namespace = _namespace()
    family_type = namespace["Family"]
    variant_type = namespace["Variant"]
    evaluate = namespace["_evaluate"]

    noisy = family_type.from_dict(  # type: ignore[attr-defined]
        {
            "assignment": "development",
            "distance": "near",
            "family_id": "7b00831ecc9ab47291c34f2793baaab9e807c21fe41e09d2b798c1ca38c78a04",
            "geometry": "a4_30mm",
            "intrinsics_profile": "d435i_wide",
            "noise_px": 0.75,
            "placement": "edge",
            "seed_index": 0,
            "tilt_deg": 0,
            "visibility": "full",
        }
    )
    noisy_result = evaluate(  # type: ignore[operator]
        noisy,
        variant_type("correct_model", "positive"),  # type: ignore[operator]
    )
    assert noisy_result["label"] == "POSITIVE_ENGINEERING_GOOD"
    assert noisy_result["scalar_reprojection"]["rmse_px"] > 1.0
    assert noisy_result["models"]["minimal_image_board_union"]["candidate_combined_pass"]

    biased = family_type.from_dict(  # type: ignore[attr-defined]
        {
            "assignment": "development",
            "distance": "near",
            "family_id": "6ab454344dce495ae77b962e6b943f9fc1cc3198338478c9cf97ab4019cc673f",
            "geometry": "a4_30mm",
            "intrinsics_profile": "d435i_wide",
            "noise_px": 0.1,
            "placement": "center",
            "seed_index": 0,
            "tilt_deg": 15,
            "visibility": "full",
        }
    )
    biased_result = evaluate(  # type: ignore[operator]
        biased,
        variant_type("principal_xy_-10.0px", "principal", 10.0, sign=-1),  # type: ignore[operator]
    )
    assert biased_result["label"] == "NEGATIVE_POSE_BIASED"
    assert biased_result["observability_passed"] is True
    assert biased_result["scalar_reprojection"]["passed"] is True
    metrics = biased_result["models"]["minimal_image_board_union"]["metrics"]
    assert metrics["failure_reasons"] == ["STRUCTURED_RESIDUAL_MODEL_MISMATCH"]
    assert metrics["permutation_p_value"] == pytest.approx(0.001)


def test_final_shared_pose_monte_carlo_uses_sixty_repeats() -> None:
    namespace = _namespace()
    manifest = namespace["build_split_manifest"]()  # type: ignore[operator]
    family_type = namespace["Family"]
    final_family = namespace["_final_family"]
    family = next(
        family_type.from_dict(value)  # type: ignore[attr-defined]
        for value in manifest["families"]
        if value["assignment"] == "development" and final_family(family_type.from_dict(value))  # type: ignore[attr-defined,operator]
    )
    result = namespace["_evaluate_final"](  # type: ignore[operator]
        family,
        namespace["Variant"]("correct_model", "positive"),  # type: ignore[operator]
    )
    assert result["evaluation_status"] == "EVALUATED"
    assert result["frame_count"] == 60
    assert result["label"] == "POSITIVE_ENGINEERING_GOOD"
    metrics = result["models"]["minimal_image_board_union"]["metrics"]
    assert metrics["scope"] == "final"
    assert metrics["corner_count"] in {24, 35}
