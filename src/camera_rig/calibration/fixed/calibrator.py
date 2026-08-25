"""Offline fixed-camera calibration service over typed, hash-bound inputs."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from datetime import datetime, timezone

import numpy as np
import numpy.typing as npt

from camera_rig.artifacts.factory_calibration import FactoryCalibrationArtifact
from camera_rig.artifacts.hashing import sha256_bytes
from camera_rig.artifacts.io import deterministic_json_bytes
from camera_rig.artifacts.target_detection import TargetDetectionArtifact
from camera_rig.calibration.fixed.aggregation import (
    distribution,
    even_odd_partition,
    pose_delta,
    pose_inlier_indices,
    pose_medoid_index,
)
from camera_rig.calibration.fixed.artifact import FixedCalibrationArtifact
from camera_rig.calibration.fixed.config import FixedCalibrationConfig
from camera_rig.calibration.fixed.quality import evaluate_fixed_calibration_quality
from camera_rig.calibration.pose import (
    PlanarPoseEstimate,
    PlanarPoseEstimator,
    RefinedPlanarPose,
    project_points_px,
    refine_planar_pose_lm,
)
from camera_rig.core.errors import ContractError, TransformError
from camera_rig.core.fixed_mount import FixedMountCalibration
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.transform_graph import TransformGraph
from camera_rig.core.transforms import RigidTransform
from camera_rig.targets.observation import TargetObservation

_TRANSFORM_ATOL = 1e-7


class FixedCameraCalibrator:
    """Calibrate one fixed camera from generic persisted target observations."""

    def __init__(self, pose_estimator: PlanarPoseEstimator | None = None) -> None:
        self._pose_estimator = pose_estimator or PlanarPoseEstimator()

    def calibrate(
        self,
        config: FixedCalibrationConfig,
        target_detection: TargetDetectionArtifact,
        factory_calibration: FactoryCalibrationArtifact,
        *,
        target_spec_sha256: str,
        capture_manifest_sha256: str,
        factory_calibration_sha256: str,
        target_detection_sha256: str,
        print_provenance: Mapping[str, object],
        native_depth_evaluator: Callable[[RigidTransform, tuple[int, ...]], Mapping[str, object]],
        provenance: Mapping[str, object] | None = None,
        created_at: str | None = None,
    ) -> FixedCalibrationArtifact:
        """Solve, aggregate, gate, and compose a fixed-camera calibration artifact."""
        inputs = _validate_inputs(
            config=config,
            target_detection=target_detection,
            factory_calibration=factory_calibration,
            target_spec_sha256=target_spec_sha256,
            capture_manifest_sha256=capture_manifest_sha256,
            factory_calibration_sha256=factory_calibration_sha256,
            target_detection_sha256=target_detection_sha256,
        )
        detection_intrinsics, reference_frame, detection_from_reference = inputs
        print_evidence = _validated_print_provenance(print_provenance)

        estimates: dict[int, PlanarPoseEstimate] = {}
        summaries: list[dict[str, object]] = []
        frame_gate_indices: list[int] = []
        for frame in target_detection.per_frame:
            summary, estimate = self._estimate_frame(
                frame.frame_index,
                frame.success,
                frame.observation,
                detection_intrinsics,
                config,
            )
            summaries.append(summary)
            if estimate is not None:
                estimates[frame.frame_index] = estimate
                if summary["accepted"] is True:
                    frame_gate_indices.append(frame.frame_index)
        if not frame_gate_indices:
            raise ContractError("fixed calibration has no frame passing the pose frame gates")

        frame_gate_poses = [
            estimates[frame_index].T_camera_from_target for frame_index in frame_gate_indices
        ]
        medoid_position = pose_medoid_index(frame_gate_poses)
        inlier_positions = pose_inlier_indices(
            frame_gate_poses,
            medoid_position,
            maximum_translation_mm=config.solver.pose_outlier_translation_mm,
            maximum_rotation_deg=config.solver.pose_outlier_rotation_deg,
        )
        inlier_indices = [frame_gate_indices[position] for position in inlier_positions]
        inlier_set = set(inlier_indices)
        medoid_pose = frame_gate_poses[medoid_position]
        for position, frame_index in enumerate(frame_gate_indices):
            medoid_delta = pose_delta(frame_gate_poses[position], medoid_pose)
            summaries[frame_index]["medoid_translation_delta_mm"] = medoid_delta.translation_mm
            summaries[frame_index]["medoid_rotation_delta_deg"] = medoid_delta.rotation_deg
            summaries[frame_index]["pose_inlier"] = frame_index in inlier_set
            if frame_index not in inlier_set:
                summaries[frame_index]["accepted"] = False
                failures = summaries[frame_index]["failure_reasons"]
                assert isinstance(failures, list)
                failures.append("pose_outlier_threshold_exceeded")
        if not inlier_indices:
            raise ContractError("fixed calibration pose outlier policy rejected every frame")

        medoid_frame_index = frame_gate_indices[medoid_position]
        final_refinement = _refine_shared_pose(
            inlier_indices,
            target_detection,
            estimates[medoid_frame_index].T_camera_from_target,
            detection_intrinsics,
        )
        if not final_refinement.validity.valid:
            raise ContractError(
                "joint fixed-pose refinement failed physical validation: "
                f"{list(final_refinement.validity.failure_reasons)}"
            )
        final_pose = final_refinement.T_camera_from_target
        depth_diagnostic = _validated_depth_diagnostic(
            native_depth_evaluator(final_pose, tuple(inlier_indices))
        )
        reprojection = _aggregate_reprojection(
            inlier_indices,
            target_detection,
            final_pose,
            detection_intrinsics,
        )
        per_frame_final = reprojection["per_frame"]
        assert isinstance(per_frame_final, list)
        for item in per_frame_final:
            assert isinstance(item, dict)
            frame_index = item["frame_index"]
            assert isinstance(frame_index, int)
            summaries[frame_index].update(
                {
                    "final_pose_reprojection_rmse_px": item["rmse_px"],
                    "final_pose_reprojection_median_px": item["median_px"],
                    "final_pose_reprojection_p95_px": item["p95_px"],
                    "final_pose_reprojection_max_px": item["maximum_px"],
                }
            )
        repeatability = _pose_repeatability(inlier_indices, estimates, final_pose)
        split_half = _split_half_stability(
            inlier_indices,
            target_detection,
            estimates,
            detection_intrinsics,
        )
        quality = evaluate_fixed_calibration_quality(
            thresholds=config.solver,
            frame_count=target_detection.frame_count,
            accepted_frames=len(inlier_indices),
            global_reprojection=_mapping(reprojection["global"]),
            pose_repeatability=repeatability,
            split_half=split_half,
            native_depth_sanity=depth_diagnostic,
        )

        workspace_from_detection = config.T_workspace_from_target.compose(final_pose.inverse())
        workspace_from_reference = workspace_from_detection.compose(detection_from_reference)
        _validate_transform_chain(
            config.T_workspace_from_target,
            final_pose,
            detection_from_reference,
            workspace_from_detection,
            workspace_from_reference,
        )
        mount_provenance: dict[str, object] = {
            "target_spec_sha256": target_spec_sha256,
            "capture_manifest_sha256": capture_manifest_sha256,
            "factory_calibration_sha256": factory_calibration_sha256,
            "target_detection_sha256": target_detection_sha256,
            "detection_stream": config.detection_stream,
            "reference_stream": config.reference_stream,
        }
        fixed_mount = FixedMountCalibration(
            parent_frame=config.workspace_frame,
            camera_reference_frame=reference_frame,
            T_parent_from_camera_reference=workspace_from_reference,
            quality=quality,
            provenance=mount_provenance,
        )
        artifact_provenance = {
            "service": "FixedCameraCalibrator",
            "target_detection_software": dict(target_detection.software),
            **dict(provenance or {}),
        }
        return FixedCalibrationArtifact(
            created_at=created_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            workspace={
                "frame": config.workspace_frame,
                "target_frame": config.target_frame,
                "T_workspace_from_target": config.T_workspace_from_target.to_dict(),
            },
            camera={
                "detection_stream": config.detection_stream,
                "detection_frame": detection_intrinsics.frame,
                "reference_stream": config.reference_stream,
                "reference_frame": reference_frame,
            },
            target={
                "target_spec_sha256": target_spec_sha256,
                "print_provenance": print_evidence,
            },
            inputs={
                "capture_manifest_sha256": capture_manifest_sha256,
                "factory_calibration_sha256": factory_calibration_sha256,
                "target_detection_sha256": target_detection_sha256,
            },
            solver={
                "method": config.solver.method,
                "refinement": config.solver.refinement,
                "thresholds": config.solver.to_dict(),
                "camera_model": dict(final_refinement.camera_model_diagnostics),
                "pose_medoid_frame_index": medoid_frame_index,
                "pose_inlier_policy": {
                    "maximum_translation_mm": config.solver.pose_outlier_translation_mm,
                    "maximum_rotation_deg": config.solver.pose_outlier_rotation_deg,
                },
            },
            per_frame_pose_summary=tuple(summaries),
            aggregate={
                "accepted_frames": len(inlier_indices),
                "accepted_ratio": len(inlier_indices) / target_detection.frame_count,
                "reprojection": reprojection,
                "pose_repeatability": repeatability,
                "split_half": split_half,
                "native_depth_sanity": depth_diagnostic,
            },
            T_detection_from_target=final_pose,
            T_workspace_from_detection=workspace_from_detection,
            T_detection_from_reference=detection_from_reference,
            T_workspace_from_reference=workspace_from_reference,
            fixed_mount_calibration=fixed_mount,
            quality=quality,
            provenance=artifact_provenance,
        )

    def _estimate_frame(
        self,
        frame_index: int,
        detection_success: bool,
        observation: TargetObservation,
        intrinsics: CameraIntrinsics,
        config: FixedCalibrationConfig,
    ) -> tuple[dict[str, object], PlanarPoseEstimate | None]:
        reasons: list[str] = []
        corner_count = len(observation.point_ids)
        if not detection_success:
            reasons.append("target_detection_failed")
        if corner_count < config.solver.minimum_corners_per_frame:
            reasons.append("insufficient_corners")
        summary: dict[str, object] = {
            "frame_index": frame_index,
            "corner_count": corner_count,
            "candidate_count": 0,
            "selected_candidate": None,
            "candidates": [],
            "candidate_separations": [],
            "T_camera_from_target": None,
            "cheirality": False,
            "face_orientation": False,
            "reprojection_rmse_px": None,
            "reprojection_median_px": None,
            "reprojection_p95_px": None,
            "reprojection_max_px": None,
            "final_pose_reprojection_rmse_px": None,
            "final_pose_reprojection_median_px": None,
            "final_pose_reprojection_p95_px": None,
            "final_pose_reprojection_max_px": None,
            "medoid_translation_delta_mm": None,
            "medoid_rotation_delta_deg": None,
            "frame_gate_accepted": False,
            "pose_inlier": False,
            "accepted": False,
            "failure_reasons": reasons,
        }
        if reasons:
            return summary, None
        try:
            estimate = self._pose_estimator.estimate(observation, intrinsics)
        except ContractError as error:
            reasons.append(f"pose_solve_failed: {error}")
            return summary, None
        reprojection = estimate.reprojection
        if reprojection.rmse_px > config.solver.maximum_frame_rmse_px:
            reasons.append("frame_reprojection_rmse_exceeded")
        if reprojection.p95_px > config.solver.maximum_frame_p95_px:
            reasons.append("frame_reprojection_p95_exceeded")
        accepted = not reasons
        summary.update(
            {
                "candidate_count": estimate.candidate_count,
                "selected_candidate": estimate.selected_candidate_index,
                "candidates": [item.to_dict() for item in estimate.candidates],
                "candidate_separations": [
                    item.to_dict() for item in estimate.candidate_separations
                ],
                "T_camera_from_target": estimate.T_camera_from_target.to_dict(),
                "cheirality": estimate.refined_validity.cheirality,
                "face_orientation": estimate.refined_validity.printed_face_orientation,
                "reprojection_rmse_px": reprojection.rmse_px,
                "reprojection_median_px": reprojection.median_px,
                "reprojection_p95_px": reprojection.p95_px,
                "reprojection_max_px": reprojection.maximum_px,
                "frame_gate_accepted": accepted,
                "pose_inlier": accepted,
                "accepted": accepted,
            }
        )
        return summary, estimate


def calibrate_fixed_camera(
    config: FixedCalibrationConfig,
    target_detection: TargetDetectionArtifact,
    factory_calibration: FactoryCalibrationArtifact,
    *,
    target_spec_sha256: str,
    capture_manifest_sha256: str,
    factory_calibration_sha256: str,
    target_detection_sha256: str,
    print_provenance: Mapping[str, object],
    native_depth_evaluator: Callable[[RigidTransform, tuple[int, ...]], Mapping[str, object]],
    provenance: Mapping[str, object] | None = None,
    created_at: str | None = None,
) -> FixedCalibrationArtifact:
    """Convenience entry point using the production planar pose estimator."""
    return FixedCameraCalibrator().calibrate(
        config,
        target_detection,
        factory_calibration,
        target_spec_sha256=target_spec_sha256,
        capture_manifest_sha256=capture_manifest_sha256,
        factory_calibration_sha256=factory_calibration_sha256,
        target_detection_sha256=target_detection_sha256,
        print_provenance=print_provenance,
        native_depth_evaluator=native_depth_evaluator,
        provenance=provenance,
        created_at=created_at,
    )


def _validate_inputs(
    *,
    config: FixedCalibrationConfig,
    target_detection: TargetDetectionArtifact,
    factory_calibration: FactoryCalibrationArtifact,
    target_spec_sha256: str,
    capture_manifest_sha256: str,
    factory_calibration_sha256: str,
    target_detection_sha256: str,
) -> tuple[CameraIntrinsics, str, RigidTransform]:
    for name, digest in (
        ("target_spec_sha256", target_spec_sha256),
        ("capture_manifest_sha256", capture_manifest_sha256),
        ("factory_calibration_sha256", factory_calibration_sha256),
        ("target_detection_sha256", target_detection_sha256),
    ):
        _require_digest(digest, name)
    if not target_detection.is_capture:
        raise ContractError("fixed calibration requires a capture target-detection artifact")
    if target_detection.target_spec_sha256 != target_spec_sha256:
        raise ContractError("target detection target SHA does not match the requested target")
    if target_detection.capture_manifest_sha256 != capture_manifest_sha256:
        raise ContractError("target detection capture SHA does not match the requested capture")
    if target_detection.stream != config.detection_stream:
        raise ContractError("target detection stream does not match fixed calibration config")
    if (
        sha256_bytes(deterministic_json_bytes(target_detection.to_dict()))
        != target_detection_sha256
    ):
        raise ContractError("target detection SHA does not match the typed detection artifact")
    if (
        sha256_bytes(deterministic_json_bytes(factory_calibration.to_dict()))
        != factory_calibration_sha256
    ):
        raise ContractError("factory calibration SHA does not match the typed factory artifact")
    acceptance = target_detection.acceptance
    if acceptance is None or acceptance.get("passed") is not True:
        raise ContractError("R6 target-detection acceptance must pass before fixed calibration")
    if not factory_calibration.quality.passed:
        raise ContractError("factory calibration quality must pass before fixed calibration")
    calibration = factory_calibration.calibration
    try:
        detection_intrinsics = calibration.intrinsics[config.detection_stream]
        reference_intrinsics = calibration.intrinsics[config.reference_stream]
    except KeyError as error:
        raise ContractError(
            f"factory calibration is missing stream intrinsics: {error.args[0]}"
        ) from error
    for frame in target_detection.per_frame:
        observation = frame.observation
        if observation.target_frame != config.target_frame:
            raise ContractError("target observation frame does not match fixed calibration target")
        if observation.image_size != (
            detection_intrinsics.width,
            detection_intrinsics.height,
        ):
            raise ContractError("target observation image size does not match detection intrinsics")
    _validate_observation_geometry(target_detection)
    graph = TransformGraph()
    try:
        for transform in calibration.internal_transforms:
            graph.add(transform)
        detection_from_reference = graph.resolve(
            reference_intrinsics.frame, detection_intrinsics.frame
        )
    except TransformError as error:
        raise ContractError(f"factory internal transform graph is invalid: {error}") from error
    return detection_intrinsics, reference_intrinsics.frame, detection_from_reference


def _refine_shared_pose(
    frame_indices: list[int],
    target_detection: TargetDetectionArtifact,
    initial_pose: RigidTransform,
    intrinsics: CameraIntrinsics,
) -> RefinedPlanarPose:
    objects = np.vstack(
        [target_detection.per_frame[index].observation.object_points_m for index in frame_indices]
    )
    images = np.vstack(
        [target_detection.per_frame[index].observation.image_points_px for index in frame_indices]
    )
    return refine_planar_pose_lm(initial_pose, objects, images, intrinsics)


def _aggregate_reprojection(
    frame_indices: list[int],
    target_detection: TargetDetectionArtifact,
    pose: RigidTransform,
    intrinsics: CameraIntrinsics,
) -> dict[str, object]:
    all_residuals: list[float] = []
    frame_metrics: list[dict[str, object]] = []
    per_corner: dict[int, list[float]] = {}
    frame_rmse: list[float] = []
    for frame_index in frame_indices:
        observation = target_detection.per_frame[frame_index].observation
        projected = project_points_px(observation.object_points_m, pose, intrinsics)
        residuals = np.linalg.norm(projected - observation.image_points_px, axis=1)
        residual_values = [float(value) for value in residuals]
        metrics = _residual_metrics(residual_values)
        frame_metrics.append({"frame_index": frame_index, **metrics})
        frame_rmse.append(_number(metrics["rmse_px"], "rmse_px"))
        all_residuals.extend(residual_values)
        for point_id, residual in zip(observation.point_ids, residual_values, strict=True):
            per_corner.setdefault(point_id, []).append(residual)
    return {
        "global": _residual_metrics(all_residuals),
        "per_frame": frame_metrics,
        "per_frame_rmse_px": distribution(frame_rmse),
        "per_corner_id": {
            str(point_id): distribution(values) for point_id, values in sorted(per_corner.items())
        },
    }


def _residual_metrics(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ContractError("reprojection residuals must be non-empty and finite")
    return {
        "count": len(values),
        "minimum_px": float(np.min(array)),
        "rmse_px": float(np.sqrt(np.mean(np.square(array)))),
        "median_px": float(np.median(array)),
        "mean_px": float(np.mean(array)),
        "p95_px": float(np.percentile(array, 95)),
        "maximum_px": float(np.max(array)),
        "std_px": float(np.std(array)),
    }


def _pose_repeatability(
    frame_indices: list[int],
    estimates: dict[int, PlanarPoseEstimate],
    final_pose: RigidTransform,
) -> dict[str, object]:
    deltas = [
        pose_delta(estimates[index].T_camera_from_target, final_pose) for index in frame_indices
    ]
    return {
        "translation_mm": distribution([item.translation_mm for item in deltas]),
        "rotation_deg": distribution([item.rotation_deg for item in deltas]),
    }


def _split_half_stability(
    frame_indices: list[int],
    target_detection: TargetDetectionArtifact,
    estimates: dict[int, PlanarPoseEstimate],
    intrinsics: CameraIntrinsics,
) -> dict[str, object]:
    try:
        even, odd = even_odd_partition(frame_indices)
    except ContractError as error:
        return {
            "status": "UNAVAILABLE",
            "method": "even_odd",
            "even_frame_indices": [index for index in frame_indices if index % 2 == 0],
            "odd_frame_indices": [index for index in frame_indices if index % 2 == 1],
            "translation_delta_mm": None,
            "rotation_delta_deg": None,
            "failure_reason": str(error),
        }
    even_pose = _refine_partition(even, target_detection, estimates, intrinsics)
    odd_pose = _refine_partition(odd, target_detection, estimates, intrinsics)
    delta = pose_delta(even_pose, odd_pose)
    return {
        "status": "AVAILABLE",
        "method": "even_odd",
        "even_frame_indices": even,
        "odd_frame_indices": odd,
        "T_detection_from_target_even": even_pose.to_dict(),
        "T_detection_from_target_odd": odd_pose.to_dict(),
        "translation_delta_mm": delta.translation_mm,
        "rotation_delta_deg": delta.rotation_deg,
    }


def _refine_partition(
    indices: list[int],
    target_detection: TargetDetectionArtifact,
    estimates: dict[int, PlanarPoseEstimate],
    intrinsics: CameraIntrinsics,
) -> RigidTransform:
    poses = [estimates[index].T_camera_from_target for index in indices]
    initial = poses[pose_medoid_index(poses)]
    result = _refine_shared_pose(indices, target_detection, initial, intrinsics)
    if not result.validity.valid:
        raise ContractError("split-half joint refinement failed physical validation")
    return result.T_camera_from_target


def _validate_transform_chain(
    workspace_from_target: RigidTransform,
    detection_from_target: RigidTransform,
    detection_from_reference: RigidTransform,
    workspace_from_detection: RigidTransform,
    workspace_from_reference: RigidTransform,
) -> None:
    identity = detection_from_target.inverse().compose(detection_from_target)
    if not np.allclose(identity.matrix, np.eye(4), rtol=0.0, atol=_TRANSFORM_ATOL):
        raise ContractError("detection pose inverse round trip failed")
    expected_detection = workspace_from_target.compose(detection_from_target.inverse())
    expected_reference = expected_detection.compose(detection_from_reference)
    if not np.allclose(
        expected_detection.matrix,
        workspace_from_detection.matrix,
        rtol=0.0,
        atol=_TRANSFORM_ATOL,
    ):
        raise ContractError("workspace/detection chain validation failed")
    if not np.allclose(
        expected_reference.matrix,
        workspace_from_reference.matrix,
        rtol=0.0,
        atol=_TRANSFORM_ATOL,
    ):
        raise ContractError("workspace/reference chain validation failed")
    point = np.asarray([0.02, -0.01, 0.5], dtype=np.float64)
    round_trip = workspace_from_reference.inverse().transform_points(
        workspace_from_reference.transform_points(point)
    )
    if not np.allclose(round_trip, point, rtol=0.0, atol=_TRANSFORM_ATOL):
        raise ContractError("workspace/reference point round trip failed")


def _validated_depth_diagnostic(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    status = result.get("status")
    if not isinstance(status, str) or not status:
        raise ContractError("native depth diagnostic requires a status")
    if status == "SKIPPED_WITH_WARNING":
        warning = result.get("warning", result.get("reason"))
        if not isinstance(warning, str) or not warning.strip():
            raise ContractError("skipped native depth diagnostic requires a warning")
    return result


def _validate_observation_geometry(target_detection: TargetDetectionArtifact) -> None:
    plugins = {frame.observation.plugin_name for frame in target_detection.per_frame}
    if len(plugins) != 1:
        raise ContractError("target observations must use one detector plugin")
    points_by_id: dict[int, npt.NDArray[np.float64]] = {}
    for frame in target_detection.per_frame:
        observation = frame.observation
        for point_id, point in zip(observation.point_ids, observation.object_points_m, strict=True):
            previous = points_by_id.get(point_id)
            if previous is not None and not np.allclose(previous, point, rtol=0.0, atol=1e-9):
                raise ContractError(
                    f"target object geometry changed across frames for point ID {point_id}"
                )
            points_by_id[point_id] = point


def _validated_print_provenance(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    expected = {
        "horizontal_print_scale",
        "vertical_print_scale",
        "maximum_observed_print_scale_error",
        "geometry_policy",
    }
    if set(result) != expected:
        raise ContractError("print provenance has missing or unknown fields")
    for name in expected - {"geometry_policy"}:
        numeric = _number(result[name], f"print_provenance.{name}")
        if numeric < 0:
            raise ContractError(f"print_provenance.{name} must be non-negative")
        result[name] = numeric
    policy = result["geometry_policy"]
    if not isinstance(policy, str) or not policy.strip():
        raise ContractError("print provenance geometry_policy must be non-empty")
    return result


def _require_digest(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ContractError("internal calibration aggregate must be an object")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ContractError(f"{name} must be a finite number")
    return float(value)
