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
