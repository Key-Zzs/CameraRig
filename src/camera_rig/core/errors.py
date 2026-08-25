"""Stable CameraRig exception hierarchy."""


class CameraRigError(Exception):
    """Base class for expected CameraRig failures."""


class ConfigurationError(CameraRigError):
    """Configuration input could not be loaded or interpreted."""


class SchemaValidationError(ConfigurationError):
    """A versioned configuration or artifact schema was violated."""


class TransformError(CameraRigError):
    """A rigid transform is invalid or frame-incompatible."""


class ArtifactError(CameraRigError):
    """An artifact could not be serialized, loaded, or validated."""


class ContractError(CameraRigError):
    """A core data contract was violated."""


class FeatureNotAvailableError(CameraRigError):
    """A reserved feature was invoked before implementation."""


class MissingOptionalDependencyError(CameraRigError):
    """A requested driver dependency is not installed."""


class UnsupportedDriverError(CameraRigError):
    """A configuration names an unknown camera driver."""


class DeviceNotFoundError(CameraRigError):
    """The configured physical camera could not be found."""


class DeviceMismatchError(CameraRigError):
    """The selected device identity does not match the configuration."""


class ProfileNotSupportedError(CameraRigError):
    """A requested or active stream profile violates the contract."""


class DeviceBusyError(CameraRigError):
    """The physical device is already in use."""


class DeviceDisconnectedError(CameraRigError):
    """The physical device disconnected during an operation."""


class LifecycleError(CameraRigError):
    """A camera driver lifecycle transition is invalid or failed."""


class FrameTimeoutError(CameraRigError):
    """A frame did not arrive within the configured timeout."""


class ReplayEOFError(CameraRigError):
    """Replay capture was requested after the final stored frame."""
