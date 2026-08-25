"""Target-agnostic planar pose estimation contracts."""

from camera_rig.calibration.pose.camera_model import OpenCVCameraModel, to_opencv_camera_model
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
    "PoseCandidateDiagnostic",
    "PoseValidity",
    "RefinedPlanarPose",
    "ReprojectionMetrics",
    "project_points_px",
    "refine_planar_pose_lm",
    "to_opencv_camera_model",
    "validate_planar_pose",
]
