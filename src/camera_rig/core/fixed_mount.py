"""Fixed single-camera mount calibration contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from camera_rig.core._validation import decoded_string, require_non_empty, string_keyed_copy
from camera_rig.core.errors import ContractError
from camera_rig.core.quality import QualityReport
from camera_rig.core.transforms import RigidTransform


@dataclass(frozen=True)
class FixedMountCalibration:
    """A precomputed fixed camera-to-parent transform; no solver is provided here."""

    parent_frame: str
    camera_reference_frame: str
    T_parent_from_camera_reference: RigidTransform
    quality: QualityReport
    provenance: dict[str, object] = field(default_factory=dict)
    mount_type: Literal["fixed"] = field(default="fixed", init=False)

    def __post_init__(self) -> None:
        require_non_empty(self.parent_frame, "parent_frame")
        require_non_empty(self.camera_reference_frame, "camera_reference_frame")
        transform = self.T_parent_from_camera_reference
        if transform.source_frame != self.camera_reference_frame:
            raise ContractError(
                "fixed mount transform source does not match camera_reference_frame"
            )
        if transform.target_frame != self.parent_frame:
            raise ContractError("fixed mount transform target does not match parent_frame")
        object.__setattr__(self, "provenance", string_keyed_copy(self.provenance, "provenance"))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe-compatible representation."""
        return {
            "mount_type": self.mount_type,
            "parent_frame": self.parent_frame,
            "camera_reference_frame": self.camera_reference_frame,
            "T_parent_from_camera_reference": self.T_parent_from_camera_reference.to_dict(),
            "quality": self.quality.to_dict(),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> FixedMountCalibration:
        """Reconstruct a fixed-mount record from decoded JSON data."""
        if data.get("mount_type") != "fixed":
            raise ContractError("mount_type must be 'fixed'")
        transform = data["T_parent_from_camera_reference"]
        quality = data["quality"]
        provenance = data.get("provenance", {})
        if not isinstance(transform, dict) or not isinstance(quality, dict):
            raise TypeError("fixed mount transform and quality must be objects")
        if not isinstance(provenance, dict):
            raise TypeError("fixed mount provenance must be an object")
        return cls(
            parent_frame=decoded_string(data["parent_frame"], "parent_frame"),
            camera_reference_frame=decoded_string(
                data["camera_reference_frame"], "camera_reference_frame"
            ),
            T_parent_from_camera_reference=RigidTransform.from_dict(
                {str(key): value for key, value in transform.items()}
            ),
            quality=QualityReport.from_dict({str(key): value for key, value in quality.items()}),
            provenance={
                decoded_string(key, "provenance key"): value for key, value in provenance.items()
            },
        )
