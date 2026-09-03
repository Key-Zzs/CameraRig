from __future__ import annotations

import runpy
from pathlib import Path

import pytest

pytest.importorskip("cv2")


def test_stress_design_covers_synthetic_geometries_and_keeps_structure_diagnostic() -> None:
    namespace = runpy.run_path(
        str(Path(__file__).parents[1] / "benchmarks" / "reprojection_policy_holdout.py")
    )
    report = namespace["run_holdout"](seeds=1, final_trials=1, negative_final_trials=1)

    assert report["candidate_thresholds"] == {
        "maximum_gross_frame_rmse_px": 0.75,
        "maximum_gross_frame_p95_px": 1.5,
        "maximum_gross_final_rmse_px": 0.75,
        "maximum_gross_final_p95_px": 1.5,
    }
    design = report["design"]
    assert design["corner_counts"] == [35, 24, 20, 12]
    assert len(design["profiles"]) == 3
    assert all(name.startswith("synthetic_") for name in design["profiles"])
    assert design["frames_per_capture"] == 60
    summary = report["summary"]
    assert summary["release_recommendation"] == "HOLD"
    assert not summary["release_checks"]["positive_frame_false_reject_wilson_upper_at_most_0_01"]
    assert not summary["release_checks"]["positive_final_false_reject_wilson_upper_at_most_0_01"]
    assert summary["known_limit"].startswith("single-planar-pose")
    assert len(summary["positive_frame_at_0_75px_boundary_by_cell"]) == 12
    assert len(summary["positive_final_by_cell"]) == 12

    expected = {"maximum_frame_rmse_px": 0.75, "maximum_frame_p95_px": 1.5}
    assert all(
        row["policy_applied_thresholds"] == expected
        for row in report["mixed_structured_negative_trials"]
    )


def test_stress_statistics_are_fail_closed_and_cellwise() -> None:
    namespace = runpy.run_path(
        str(Path(__file__).parents[1] / "benchmarks" / "reprojection_policy_holdout.py")
    )
    wilson_upper = namespace["_wilson_upper"]
    binomial_upper_tail = namespace["_binomial_upper_tail"]

    assert wilson_upper(0, 128) == pytest.approx(0.0291369563)
    assert wilson_upper(2, 128) > wilson_upper(1, 128) > wilson_upper(0, 128)
    capture_risk = binomial_upper_tail(60, wilson_upper(0, 128), 6)
    assert capture_risk == pytest.approx(0.0017801, rel=1e-3)
