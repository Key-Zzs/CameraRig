"""One-command provisioning configuration contracts."""

from camera_rig.provision.artifact import (
    FIXED_PROVISION_ARTIFACT_SCHEMA_VERSION,
    FixedProvisionArtifactInputs,
    FixedProvisionManifest,
    ProvisionOverlayInputs,
    write_fixed_provision_artifact,
)
from camera_rig.provision.config import (
    FIXED_PROVISION_CONFIG_SCHEMA_VERSION,
    ProvisionConfig,
    load_provision_config,
    load_provision_config_with_sha256,
    validate_provision_config_data,
)
from camera_rig.provision.validation import load_and_validate_fixed_provision

__all__ = [
    "FIXED_PROVISION_ARTIFACT_SCHEMA_VERSION",
    "FIXED_PROVISION_CONFIG_SCHEMA_VERSION",
    "FixedProvisionArtifactInputs",
    "FixedProvisionManifest",
    "ProvisionConfig",
    "ProvisionOverlayInputs",
    "load_and_validate_fixed_provision",
    "load_provision_config",
    "load_provision_config_with_sha256",
    "validate_provision_config_data",
    "write_fixed_provision_artifact",
]
