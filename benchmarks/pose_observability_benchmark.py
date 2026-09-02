"""Deterministic synthetic sweep and Monte-Carlo check for pose observability."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from camera_rig.artifacts.io import atomic_write_json
from camera_rig.calibration.pose import (
    PlanarPoseEstimator,
    PoseAmbiguityCandidate,
    UncertaintyValidatedThresholds,
    evaluate_pose_observability,
    to_opencv_camera_model,
)
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.transforms import RigidTransform
from camera_rig.targets.observation import TargetObservation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--monte-carlo-trials", type=int, default=200)
    arguments = parser.parse_args()
    if arguments.monte_carlo_trials < 50:
        parser.error("--monte-carlo-trials must be at least 50")
    started = time.perf_counter()
    sweep = _synthetic_sweep()
    monte_carlo = _monte_carlo(arguments.monte_carlo_trials)
    final_shared_monte_carlo = _final_shared_pose_monte_carlo(
        max(50, arguments.monte_carlo_trials // 2)
    )
    elapsed = time.perf_counter() - started
    successful_timings = [
        float(item["observability_elapsed_ms"])
        for item in sweep
        if item["observability_elapsed_ms"] is not None
    ]
    monte_carlo_agreement = _monte_carlo_agreement(monte_carlo)
    final_shared_agreement = _monte_carlo_agreement(final_shared_monte_carlo)
    sweep_checks = _sweep_acceptance_checks(sweep)
    overall_passed = (
        all(sweep_checks.values())
        and monte_carlo_agreement["conclusion"] == "PASS"
        and final_shared_agreement["conclusion"] == "PASS"
    )
    report = {
        "schema_version": "camera-rig.pose-observability-benchmark.v1",
        "seed": 20260902,
        "release_thresholds": UncertaintyValidatedThresholds().to_dict(),
        "sweep_case_count": len(sweep),
        "sweep": sweep,
        "monte_carlo": monte_carlo,
        "monte_carlo_agreement": monte_carlo_agreement,
        "final_shared_pose_monte_carlo": final_shared_monte_carlo,
        "final_shared_pose_monte_carlo_agreement": final_shared_agreement,
        "sweep_acceptance_checks": sweep_checks,
        "conclusion": "PASS" if overall_passed else "FAIL",
        "performance": {
            "total_elapsed_s": elapsed,
            "observability_per_frame_ms": _distribution(successful_timings),
        },
    }
    atomic_write_json(arguments.output, report)
    print(
        f"pose observability benchmark: {'PASS' if overall_passed else 'FAIL'} "
        f"({len(sweep)} sweep cases, {len(monte_carlo)} Monte-Carlo cases)"
    )
    return 0 if overall_passed else 1


def _synthetic_sweep() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    case_index = 0
    for square_m in (0.04, 0.10):
        points = _points(square_m)
        subsets = {
            "full": np.arange(24),
            "partial_distributed": np.asarray([0, 2, 3, 5, 8, 10, 13, 15, 16, 18, 21, 23]),
            "partial_clustered": np.asarray([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]),
        }
        for distance_m in (0.8, 1.5, 3.0, 4.5):
            for tilt_deg in (0.0, 15.0, 30.0, 45.0, 60.0, 70.0, 75.0, 80.0):
                for location in ("center", "edge", "corner"):
                    pose = _pose(points, distance_m, tilt_deg, location)
                    for noise_px in (0.1, 0.25, 0.5, 1.0):
                        for subset_name, selected in subsets.items():
                            selected_points = points[selected]
                            clean_pixels = _project(selected_points, pose)
                            noisy_pixels = clean_pixels + np.random.default_rng(
                                20260902 + case_index
                            ).normal(0.0, noise_px, clean_pixels.shape)
                            case_index += 1
                            cases.append(
                                _sweep_case(
                                    square_m=square_m,
                                    distance_m=distance_m,
                                    tilt_deg=tilt_deg,
                                    location=location,
                                    noise_px=noise_px,
                                    subset_name=subset_name,
                                    points=selected_points,
                                    pixels=noisy_pixels,
                                    pose=pose,
                                )
                            )
    return cases


def _sweep_case(
    *,
    square_m: float,
    distance_m: float,
    tilt_deg: float,
    location: str,
    noise_px: float,
    subset_name: str,
    points: np.ndarray,
    pixels: np.ndarray,
    pose: RigidTransform,
) -> dict[str, object]:
    coverage = _coverage(pixels)
    span_x = float(np.ptp(pixels[:, 0]) / 1280.0)
    span_y = float(np.ptp(pixels[:, 1]) / 720.0)
    minimum_projected_spacing = _minimum_projected_spacing(pixels)
    old_pose_validated = (
        len(points) >= 12
        and coverage >= 0.01
        and span_x >= 0.10
        and span_y >= 0.10
        and minimum_projected_spacing * 4.0 >= 20.0
    )
    observation = TargetObservation(
        plugin_name="synthetic-charuco-grid",
        target_frame="target",
        point_ids=tuple(range(len(points))),
        image_points_px=pixels,
        object_points_m=points,
        image_size=(1280, 720),
        quality=QualityReport(True),
    )
    try:
        estimate = PlanarPoseEstimator().estimate(observation, _intrinsics())
        ambiguity_candidates = tuple(
            PoseAmbiguityCandidate(
                index=item.index,
                T_camera_from_target=item.T_camera_from_target,
                valid=item.validity.valid,
                reprojection_sse_px2=float(np.sum(np.square(item.reprojection.residuals_px))),
            )
            for item in estimate.candidates
        )
        before = time.perf_counter()
        metrics = evaluate_pose_observability(
            object_points_m=points,
            image_points_px=pixels,
            T_camera_from_target=estimate.T_camera_from_target,
            intrinsics=_intrinsics(),
            ambiguity_candidates=ambiguity_candidates,
        )
        elapsed_ms: float | None = (time.perf_counter() - before) * 1000.0
        translation_error_mm = float(
            1000.0
            * np.linalg.norm(estimate.T_camera_from_target.matrix[:3, 3] - pose.matrix[:3, 3])
        )
        rotation_error_deg = _rotation_error_deg(estimate.T_camera_from_target, pose)
        result: dict[str, object] = {
            "solve_success": True,
            "translation_worst_axis_std_mm": metrics.translation_worst_axis_std_mm,
            "rotation_worst_axis_std_deg": metrics.rotation_worst_axis_std_deg,
            "scaled_condition_number": metrics.scaled_condition_number,
            "ambiguous": metrics.candidate_ambiguity.ambiguous,
            "uncertainty_validated_result": "PASS" if metrics.passed else "FAIL",
            "failure_reasons": list(metrics.failure_reasons),
            "actual_translation_error_mm": translation_error_mm,
            "actual_rotation_error_deg": rotation_error_deg,
            "observability_elapsed_ms": elapsed_ms,
        }
    except Exception as error:
        result = {
            "solve_success": False,
            "translation_worst_axis_std_mm": None,
            "rotation_worst_axis_std_deg": None,
            "scaled_condition_number": None,
            "ambiguous": None,
            "uncertainty_validated_result": "FAIL",
            "failure_reasons": [f"POSE_SOLVE_FAILED: {error}"],
            "actual_translation_error_mm": None,
            "actual_rotation_error_deg": None,
            "observability_elapsed_ms": None,
        }
    return {
        "square_length_m": square_m,
        "distance_m": distance_m,
        "tilt_deg": tilt_deg,
        "image_location": location,
        "pixel_noise_px": noise_px,
        "corner_subset": subset_name,
        "corner_count": len(points),
        "coverage_ratio": coverage,
        "image_span_x_ratio": span_x,
        "image_span_y_ratio": span_y,
        "minimum_projected_corner_spacing_px": minimum_projected_spacing,
        "all_correspondences_in_image": bool(
            np.all((pixels[:, 0] >= 0.0) & (pixels[:, 0] < 1280.0))
            and np.all((pixels[:, 1] >= 0.0) & (pixels[:, 1] < 720.0))
        ),
        "pose_validated_result": "PASS" if old_pose_validated else "FAIL",
        **result,
    }


def _monte_carlo(trials: int) -> list[dict[str, object]]:
    definitions = (
        ("good_near", 0.10, 1.0, 30.0, 0.25),
        ("good_far", 0.10, 3.0, 60.0, 0.25),
        ("moderate_oblique", 0.10, 1.5, 60.0, 0.5),
        ("small_target_far", 0.04, 3.0, 45.0, 0.5),
        ("noisy", 0.10, 1.0, 15.0, 1.0),
    )
    result: list[dict[str, object]] = []
    for case_index, (name, square_m, distance_m, tilt_deg, noise_px) in enumerate(definitions):
        points = _points(square_m)
        pose = _pose(points, distance_m, tilt_deg, "center")
        pixels = _project(points, pose)
        prediction_thresholds = replace(
            UncertaintyValidatedThresholds(),
            pixel_noise_floor_px=max(0.25, noise_px),
        )
        nominal_pixels = pixels + np.random.default_rng(20261902 + case_index).normal(
            0.0, max(noise_px * 0.05, 1e-4), pixels.shape
        )
        nominal_observation = TargetObservation(
            plugin_name="synthetic-charuco-grid",
            target_frame="target",
            point_ids=tuple(range(len(points))),
            image_points_px=nominal_pixels,
            object_points_m=points,
            image_size=(1280, 720),
            quality=QualityReport(True),
        )
        nominal_estimate = PlanarPoseEstimator(prediction_thresholds).estimate(
            nominal_observation, _intrinsics()
        )
        nominal_candidates = tuple(
            PoseAmbiguityCandidate(
                index=item.index,
                T_camera_from_target=item.T_camera_from_target,
                valid=item.validity.valid,
                reprojection_sse_px2=float(np.sum(np.square(item.reprojection.residuals_px))),
            )
            for item in nominal_estimate.candidates
        )
        predicted = evaluate_pose_observability(
            object_points_m=points,
            image_points_px=pixels,
            T_camera_from_target=pose,
            intrinsics=_intrinsics(),
            ambiguity_candidates=nominal_candidates,
            thresholds=prediction_thresholds,
        )
        translation_errors: list[np.ndarray] = []
        rotation_errors: list[np.ndarray] = []
        nonambiguous_translation_errors: list[np.ndarray] = []
        nonambiguous_rotation_errors: list[np.ndarray] = []
        ambiguous_trials = 0
        mode_jump_trials = 0
        flagged_mode_jump_trials = 0
        failures = 0
        generator = np.random.default_rng(20260902 + case_index)
        for _trial in range(trials):
            noisy = pixels + generator.normal(0.0, noise_px, pixels.shape)
            observation = TargetObservation(
                plugin_name="synthetic-charuco-grid",
                target_frame="target",
                point_ids=tuple(range(len(points))),
                image_points_px=noisy,
                object_points_m=points,
                image_size=(1280, 720),
                quality=QualityReport(True),
            )
            try:
                estimate = PlanarPoseEstimator(prediction_thresholds).estimate(
                    observation, _intrinsics()
                )
            except Exception:
                failures += 1
                continue
            translation_error = estimate.T_camera_from_target.matrix[:3, 3] - pose.matrix[:3, 3]
            relative = estimate.T_camera_from_target.matrix[:3, :3] @ pose.matrix[:3, :3].T
            rotation_vector, _jacobian = _cv2().Rodrigues(relative)
            rotation_error = np.asarray(rotation_vector).reshape(3)
            translation_errors.append(translation_error)
            rotation_errors.append(rotation_error)
            is_ambiguous = estimate.observability.candidate_ambiguity.ambiguous
            ambiguous_trials += int(is_ambiguous)
            mode_jump = math.degrees(float(np.linalg.norm(rotation_error))) >= 5.0
            mode_jump_trials += int(mode_jump)
            flagged_mode_jump_trials += int(mode_jump and is_ambiguous)
            if not is_ambiguous:
                nonambiguous_translation_errors.append(translation_error)
                nonambiguous_rotation_errors.append(rotation_error)
        translation_array = np.asarray(translation_errors, dtype=np.float64)
        rotation_array = np.asarray(rotation_errors, dtype=np.float64)
        empirical_translation = _worst_axis_std(translation_array) * 1000.0
        empirical_rotation = math.degrees(_worst_axis_std(rotation_array))
        local_empirical_translation = (
            _worst_axis_std(np.asarray(nonambiguous_translation_errors, dtype=np.float64)) * 1000.0
        )
        local_empirical_rotation = math.degrees(
            _worst_axis_std(np.asarray(nonambiguous_rotation_errors, dtype=np.float64))
        )
        predicted_translation = float(predicted.translation_worst_axis_std_mm or 0.0)
        predicted_rotation = float(predicted.rotation_worst_axis_std_deg or 0.0)
        result.append(
            {
                "name": name,
                "trial_count": trials,
                "solve_failures": failures,
                "ambiguous_trial_count": ambiguous_trials,
                "mode_jump_trial_count": mode_jump_trials,
                "flagged_mode_jump_trial_count": flagged_mode_jump_trials,
                "all_mode_jumps_flagged_ambiguous": (mode_jump_trials == flagged_mode_jump_trials),
                "pixel_noise_px": noise_px,
                "release_prediction_passed": predicted.passed,
                "prediction_failure_reasons": list(predicted.failure_reasons),
                "prediction_candidate_ambiguity": (predicted.candidate_ambiguity.to_dict()),
                "coverage_ratio": _coverage(pixels),
                "predicted_translation_worst_std_mm": predicted_translation,
                "empirical_all_solved_translation_worst_std_mm": empirical_translation,
                "empirical_nonambiguous_translation_worst_std_mm": (local_empirical_translation),
                "translation_ratio_nonambiguous_empirical_over_predicted": (
                    local_empirical_translation / predicted_translation
                    if predicted_translation > 0
                    else None
                ),
                "predicted_rotation_worst_std_deg": predicted_rotation,
                "empirical_all_solved_rotation_worst_std_deg": empirical_rotation,
                "empirical_nonambiguous_rotation_worst_std_deg": local_empirical_rotation,
                "rotation_ratio_nonambiguous_empirical_over_predicted": (
                    local_empirical_rotation / predicted_rotation
                    if predicted_rotation > 0
                    else None
                ),
            }
        )
    return result


def _final_shared_pose_monte_carlo(trials: int) -> list[dict[str, object]]:
    definitions = (
        ("final_good_far", 0.10, 3.0, 60.0, 0.25),
        ("final_moderate_oblique", 0.10, 1.5, 60.0, 0.5),
        ("final_extreme_weak", 0.04, 8.0, 70.0, 0.5),
    )
    frame_count = 60
    result: list[dict[str, object]] = []
    for case_index, (name, square_m, distance_m, tilt_deg, noise_px) in enumerate(definitions):
        points = _points(square_m)
        pose = _pose(points, distance_m, tilt_deg, "center")
        pixels = _project(points, pose)
        stacked_points = np.tile(points, (frame_count, 1))
        stacked_pixels = np.tile(pixels, (frame_count, 1))
        thresholds = replace(
            UncertaintyValidatedThresholds(), pixel_noise_floor_px=max(0.25, noise_px)
        )
        nominal_pixels = stacked_pixels + np.random.default_rng(20262902 + case_index).normal(
            0.0, max(noise_px * 0.05, 1e-4), stacked_pixels.shape
        )
        nominal_observation = TargetObservation(
            plugin_name="synthetic-charuco-grid",
            target_frame="target",
            point_ids=tuple(range(len(stacked_points))),
            image_points_px=nominal_pixels,
            object_points_m=stacked_points,
            image_size=(1280, 720),
            quality=QualityReport(True),
        )
        nominal_estimate = PlanarPoseEstimator(thresholds).estimate(
            nominal_observation, _intrinsics()
        )
        nominal_candidates = tuple(
            PoseAmbiguityCandidate(
                index=item.index,
                T_camera_from_target=item.T_camera_from_target,
                valid=item.validity.valid,
                reprojection_sse_px2=float(np.sum(np.square(item.reprojection.residuals_px))),
            )
            for item in nominal_estimate.candidates
        )
        predicted = evaluate_pose_observability(
            object_points_m=stacked_points,
            image_points_px=stacked_pixels,
            T_camera_from_target=pose,
            intrinsics=_intrinsics(),
            ambiguity_candidates=nominal_candidates,
            thresholds=thresholds,
            scope="final",
        )
        translation_errors: list[np.ndarray] = []
        rotation_errors: list[np.ndarray] = []
        failures = 0
        generator = np.random.default_rng(20263902 + case_index)
        for _trial in range(trials):
            noisy = stacked_pixels + generator.normal(0.0, noise_px, stacked_pixels.shape)
            observation = TargetObservation(
                plugin_name="synthetic-charuco-grid",
                target_frame="target",
                point_ids=tuple(range(len(stacked_points))),
                image_points_px=noisy,
                object_points_m=stacked_points,
                image_size=(1280, 720),
                quality=QualityReport(True),
            )
            try:
                estimate = PlanarPoseEstimator(thresholds).estimate(observation, _intrinsics())
            except Exception:
                failures += 1
                continue
            translation_errors.append(
                estimate.T_camera_from_target.matrix[:3, 3] - pose.matrix[:3, 3]
            )
            relative = estimate.T_camera_from_target.matrix[:3, :3] @ pose.matrix[:3, :3].T
            rotation_vector, _jacobian = _cv2().Rodrigues(relative)
            rotation_errors.append(np.asarray(rotation_vector).reshape(3))
        empirical_translation = (
            _worst_axis_std(np.asarray(translation_errors, dtype=np.float64)) * 1000.0
        )
        empirical_rotation = math.degrees(
            _worst_axis_std(np.asarray(rotation_errors, dtype=np.float64))
        )
        predicted_translation = float(predicted.translation_worst_axis_std_mm or 0.0)
        predicted_rotation = float(predicted.rotation_worst_axis_std_deg or 0.0)
        result.append(
            {
                "name": name,
                "frame_count": frame_count,
                "trial_count": trials,
                "solve_failures": failures,
                "pixel_noise_px": noise_px,
                "release_prediction_passed": predicted.passed,
                "prediction_failure_reasons": list(predicted.failure_reasons),
                "coverage_ratio": _coverage(pixels),
                "predicted_translation_worst_std_mm": predicted_translation,
                "empirical_translation_worst_std_mm": empirical_translation,
                "translation_ratio_empirical_over_predicted": (
                    empirical_translation / predicted_translation
                    if predicted_translation > 0
                    else None
                ),
                "predicted_rotation_worst_std_deg": predicted_rotation,
                "empirical_rotation_worst_std_deg": empirical_rotation,
                "rotation_ratio_empirical_over_predicted": (
                    empirical_rotation / predicted_rotation if predicted_rotation > 0 else None
                ),
            }
        )
    return result


def _points(square_m: float) -> np.ndarray:
    return np.asarray(
        [[square_m * column, square_m * row, 0.0] for row in range(6) for column in range(4)],
        dtype=np.float64,
    )


def _pose(points: np.ndarray, distance_m: float, tilt_deg: float, location: str) -> RigidTransform:
    angle = math.radians(tilt_deg)
    rotation_y = np.asarray(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    rotation = rotation_y @ np.diag([1.0, -1.0, -1.0])
    centroid = np.mean(points, axis=0)
    translation = -(rotation @ centroid)
    translation[2] += distance_m
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    pose = RigidTransform("target", "synthetic/color_optical", matrix)
    if location == "center":
        return pose

    # Move the complete target toward the right or lower-right margin. Synthetic
    # correspondences outside the sensor image are not admissible threshold evidence.
    for _iteration in range(4):
        projected = _project(points, pose)
        delta_u = 1260.0 - float(np.max(projected[:, 0]))
        delta_v = 0.0 if location == "edge" else 700.0 - float(np.max(projected[:, 1]))
        matrix = pose.matrix.copy()
        matrix[0, 3] += delta_u * distance_m / _intrinsics().fx
        matrix[1, 3] += delta_v * distance_m / _intrinsics().fy
        pose = RigidTransform("target", "synthetic/color_optical", matrix)
    projected = _project(points, pose)
    if not (
        np.all((projected[:, 0] >= 0.0) & (projected[:, 0] < 1280.0))
        and np.all((projected[:, 1] >= 0.0) & (projected[:, 1] < 720.0))
    ):
        raise RuntimeError("synthetic edge/corner placement left the image bounds")
    return pose


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        "synthetic/color_optical",
        1280,
        720,
        900.0,
        905.0,
        639.5,
        359.5,
        "none",
    )


def _project(points: np.ndarray, pose: RigidTransform) -> np.ndarray:
    cv2 = _cv2()
    model = to_opencv_camera_model(_intrinsics())
    rvec, _jacobian = cv2.Rodrigues(pose.matrix[:3, :3])
    pixels, _jacobian = cv2.projectPoints(
        points,
        rvec,
        pose.matrix[:3, 3],
        model.camera_matrix,
        model.distortion_coeffs,
    )
    return np.asarray(pixels, dtype=np.float64).reshape(-1, 2)


def _coverage(pixels: np.ndarray) -> float:
    cv2 = _cv2()
    hull = cv2.convexHull(np.asarray(pixels, dtype=np.float32))
    return float(cv2.contourArea(hull)) / (1280.0 * 720.0)


def _minimum_projected_spacing(pixels: np.ndarray) -> float:
    deltas = pixels[:, None, :] - pixels[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    distances[distances == 0] = np.inf
    return float(np.min(distances))


def _rotation_error_deg(actual: RigidTransform, expected: RigidTransform) -> float:
    relative = expected.matrix[:3, :3].T @ actual.matrix[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _worst_axis_std(values: np.ndarray) -> float:
    if len(values) < 2:
        return math.inf
    covariance = np.cov(values, rowvar=False, ddof=1)
    return math.sqrt(max(float(np.max(np.linalg.eigvalsh(covariance))), 0.0))


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _monte_carlo_agreement(cases: list[dict[str, object]]) -> dict[str, object]:
    checks: dict[str, bool] = {}
    for case in cases:
        name = str(case["name"])
        translation_ratio_value = case.get(
            "translation_ratio_nonambiguous_empirical_over_predicted",
            case.get("translation_ratio_empirical_over_predicted"),
        )
        rotation_ratio_value = case.get(
            "rotation_ratio_nonambiguous_empirical_over_predicted",
            case.get("rotation_ratio_empirical_over_predicted"),
        )
        translation_ratio = float(translation_ratio_value)  # type: ignore[arg-type]
        rotation_ratio = float(rotation_ratio_value)  # type: ignore[arg-type]
        checks[f"{name}_translation_ratio_in_0_5_to_2"] = 0.5 <= translation_ratio <= 2.0
        checks[f"{name}_rotation_ratio_in_0_5_to_2"] = 0.5 <= rotation_ratio <= 2.0
        if "all_mode_jumps_flagged_ambiguous" in case:
            checks[f"{name}_all_mode_jumps_flagged_ambiguous"] = (
                case["all_mode_jumps_flagged_ambiguous"] is True
            )
    return {
        "case_count": len(cases),
        "checks": checks,
        "conclusion": "PASS" if checks and all(checks.values()) else "FAIL",
        "note": (
            "local covariance is compared with nonambiguous single-mode trials for every case; "
            "all solved-trial mixture error remains reported, and every >=5 deg mode jump "
            "must be flagged by the independent IPPE ambiguity gate"
        ),
    }


def _sweep_acceptance_checks(cases: list[dict[str, object]]) -> dict[str, bool]:
    return {
        "all_correspondences_in_image": all(
            case["all_correspondences_in_image"] is True for case in cases
        ),
        "all_pose_solves_completed": all(case["solve_success"] is True for case in cases),
        "contains_low_coverage_good_pass": any(
            float(case["coverage_ratio"]) < 0.01 and case["uncertainty_validated_result"] == "PASS"
            for case in cases
        ),
        "contains_high_coverage_poor_fail": any(
            float(case["coverage_ratio"]) > 0.05 and case["uncertainty_validated_result"] == "FAIL"
            for case in cases
        ),
        "contains_low_coverage_poor_fail": any(
            float(case["coverage_ratio"]) < 0.01 and case["uncertainty_validated_result"] == "FAIL"
            for case in cases
        ),
        "contains_ambiguous_fail": any(
            case["ambiguous"] is True and case["uncertainty_validated_result"] == "FAIL"
            for case in cases
        ),
    }


def _cv2():  # type: ignore[no-untyped-def]
    from camera_rig.calibration.pose.dependencies import cv2_module

    return cv2_module()


if __name__ == "__main__":
    raise SystemExit(main())
