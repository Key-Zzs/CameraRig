"""Hardware-independent CameraRig core contracts."""

from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.errors import (
    ArtifactError,
    CameraRigError,
    ConfigurationError,
    ContractError,
    DeviceBusyError,
    DeviceDisconnectedError,
    DeviceMismatchError,
    DeviceNotFoundError,
    FeatureNotAvailableError,
    FrameTimeoutError,
    LifecycleError,
    MissingOptionalDependencyError,
    ProfileNotSupportedError,
    ReplayEOFError,
    SchemaValidationError,
    TransformError,
    UnsupportedDriverError,
)
from camera_rig.core.factory_calibration import FactoryCalibration
from camera_rig.core.fixed_mount import FixedMountCalibration
from camera_rig.core.frame import CameraFrame, StreamFrame
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.stream import StreamProfile
from camera_rig.core.timestamps import SingleDeviceSyncReport
from camera_rig.core.transform_graph import TransformGraph
from camera_rig.core.transforms import RigidTransform, compose

__all__ = [
    "ArtifactError",
    "CameraDeviceInfo",
    "CameraFrame",
    "CameraIntrinsics",
    "CameraRigError",
    "ConfigurationError",
    "ContractError",
    "DeviceBusyError",
    "DeviceDisconnectedError",
    "DeviceMismatchError",
    "DeviceNotFoundError",
    "FactoryCalibration",
    "FeatureNotAvailableError",
    "FixedMountCalibration",
    "FrameTimeoutError",
    "LifecycleError",
    "MissingOptionalDependencyError",
    "ProfileNotSupportedError",
    "QualityReport",
    "ReplayEOFError",
    "RigidTransform",
    "SchemaValidationError",
    "SingleDeviceSyncReport",
    "StreamFrame",
    "StreamProfile",
    "TransformError",
    "TransformGraph",
    "UnsupportedDriverError",
    "compose",
]
