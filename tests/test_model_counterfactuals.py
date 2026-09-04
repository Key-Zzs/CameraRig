from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from camera_rig.artifacts.factory_calibration import FactoryCalibrationArtifact
from camera_rig.artifacts.target_detection import TargetDetectionArtifact
from camera_rig.calibration.fixed.counterfactuals import evaluate_model_counterfactuals
from camera_rig.calibration.pose import project_points_px
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.transforms import RigidTransform
from camera_rig.targets.observation import TargetObservation

pytest.importorskip("cv2")


def test_retained_detection_counterfactuals_are_analysis_only() -> None:
    intrinsics = CameraIntrinsics(
        frame="camera/color_optical",
        width=1280,
        height=720,
        fx=900.0,
        fy=905.0,
        cx=639.5,
        cy=359.5,
        distortion_model="none",
    )
    points = np.asarray(
        [[0.03 * column, 0.03 * row, 0.0] for row in range(5) for column in range(7)],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.diag([1.0, -1.0, -1.0])
    matrix[:3, 3] = [-0.09, 0.06, 0.72]
    pose = RigidTransform("target", intrinsics.frame, matrix)
    pixels = project_points_px(points, pose, intrinsics)
    frames = []
    for index in range(4):
        noise = np.random.default_rng(index).normal(0.0, 0.05, pixels.shape)
        observation = TargetObservation(
            plugin_name="synthetic-grid",
            target_frame="target",
            point_ids=tuple(range(len(points))),
            image_points_px=pixels + noise,
            object_points_m=points,
            image_size=(1280, 720),
            quality=QualityReport(True),
        )
        frames.append(SimpleNamespace(success=True, observation=observation))
    detection = cast(
        TargetDetectionArtifact,
        SimpleNamespace(is_capture=True, stream="color", per_frame=tuple(frames)),
    )
    factory = cast(
        FactoryCalibrationArtifact,
        SimpleNamespace(calibration=SimpleNamespace(intrinsics={"color": intrinsics})),
    )
    report = evaluate_model_counterfactuals(
        detection,
        factory,
        detection_report_sha256="a" * 64,
        factory_calibration_sha256="b" * 64,
    )
    assert report["status"] == "ANALYSIS_ONLY_NO_GROUND_TRUTH"
    assert report["production_artifact_mutation"] is False
    assert report["candidate_policy"]["release_state"] == "HOLD"
    counterfactuals = report["counterfactuals"]
    assert isinstance(counterfactuals, list) and len(counterfactuals) == 9
    assert all("_poses" not in value for value in counterfactuals)
    principal = next(value for value in counterfactuals if value["name"] == "principal_plus_5px")
    assert principal["pose_sensitivity_from_baseline"]["translation_mm"]["maximum"] > 0
