"""Target-agnostic planar pose estimation contracts."""

from camera_rig.calibration.pose.camera_model import OpenCVCameraModel, to_opencv_camera_model
from camera_rig.calibration.pose.observability import (
    PoseAmbiguityCandidate,
    PoseAmbiguityMetrics,
    PoseObservabilityMetrics,
    UncertaintyValidatedThresholds,
    evaluate_pose_ambiguity,
    evaluate_pose_observability,
    projection_jacobian_first_six,
)
from camera_rig.calibration.pose.planar_pnp import (
    CandidateSeparation,
    PlanarPoseEstimate,
    PlanarPoseEstimator,
    PoseCandidateDiagnostic,
)
from camera_rig.calibration.pose.projection import project_points_px
from camera_rig.calibration.pose.refinement import RefinedPlanarPose, refine_planar_pose_lm
from camera_rig.calibration.pose.reprojection import ReprojectionMetrics
from camera_rig.calibration.pose.validation import PoseValidity, validate_planar_pose

__all__ = [
    "CandidateSeparation",
    "OpenCVCameraModel",
    "PlanarPoseEstimate",
    "PlanarPoseEstimator",
    "PoseAmbiguityCandidate",
    "PoseAmbiguityMetrics",
    "PoseCandidateDiagnostic",
    "PoseObservabilityMetrics",
    "PoseValidity",
    "RefinedPlanarPose",
    "ReprojectionMetrics",
    "UncertaintyValidatedThresholds",
    "evaluate_pose_ambiguity",
    "evaluate_pose_observability",
    "project_points_px",
    "projection_jacobian_first_six",
    "refine_planar_pose_lm",
    "to_opencv_camera_model",
    "validate_planar_pose",
]
