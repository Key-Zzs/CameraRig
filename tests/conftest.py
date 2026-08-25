from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from camera_rig.artifacts.models import CameraBundle
from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.fixed_mount import FixedMountCalibration
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.stream import StreamProfile
from camera_rig.core.transforms import RigidTransform


@pytest.fixture
def make_transform() -> Callable[[str, str, tuple[float, float, float]], RigidTransform]:
    def factory(
        source: str,
        target: str,
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> RigidTransform:
        matrix = np.eye(4)
        matrix[:3, 3] = translation
        return RigidTransform(source, target, matrix)

    return factory


@pytest.fixture
def sample_bundle(
    make_transform: Callable[[str, str, tuple[float, float, float]], RigidTransform],
) -> CameraBundle:
    device = CameraDeviceInfo(
        driver="synthetic",
        camera_name="head",
        expected_model="synthetic-test-camera",
        reported_model="synthetic-test-camera",
        serial="000123",
        metadata={"source": "unit-test"},
    )
    color_profile = StreamProfile("color", 640, 480, 30, "rgb8")
    depth_profile = StreamProfile("depth", 640, 480, 30, "z16")
    color_intrinsics = CameraIntrinsics(
        "head/color_optical",
        640,
        480,
        600.0,
        601.0,
        319.5,
        239.5,
        "none",
    )
    depth_intrinsics = CameraIntrinsics(
        "head/depth_optical",
        640,
        480,
        590.0,
        591.0,
        319.0,
        239.0,
        "brown-conrady",
        (0.1, -0.01, 0.0, 0.0, 0.0),
    )
    internal = make_transform("head/depth_optical", "head/color_optical", (0.025, 0.0, 0.0))
    fixed_transform = make_transform("head/color_optical", "workspace", (0.2, -0.1, 1.0))
    quality = QualityReport(True, metrics={"synthetic_score": np.float64(1.0)})
    fixed = FixedMountCalibration(
        parent_frame="workspace",
        camera_reference_frame="head/color_optical",
        T_parent_from_camera_reference=fixed_transform,
        quality=quality,
        provenance={"source": "synthetic-test"},
    )
    return CameraBundle(
        status="synthetic",
        bundle_id="synthetic-unit-test",
        created_at="2026-08-25T00:00:00Z",
        device=device,
        stream_profiles={"color": color_profile, "depth": depth_profile},
        intrinsics={"color": color_intrinsics, "depth": depth_intrinsics},
        internal_transforms=(internal,),
        depth_scale_m_per_unit=0.001,
        fixed_mount_calibration=fixed,
        quality=quality,
        provenance={"source": "synthetic-test", "real_hardware": False},
    )
