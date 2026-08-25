"""Single-device timing report contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from camera_rig.core._validation import (
    decoded_bool,
    decoded_int,
    decoded_string,
    require_non_negative_int,
    string_keyed_copy,
)
from camera_rig.core.errors import ContractError


@dataclass(frozen=True)
class SingleDeviceSyncReport:
    """Reported relationships among streams inside one physical camera.

    This is a data contract only; it does not perform timestamp synchronization.
    """

    valid: bool
    comparable_streams: tuple[str, ...]
    max_skew_ns: int | None
    per_stream_skew_ns: dict[str, int] = field(default_factory=dict)
    frame_number_match: bool | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "comparable_streams", tuple(self.comparable_streams))
        if len(set(self.comparable_streams)) != len(self.comparable_streams):
            raise ContractError("comparable_streams must not contain duplicates")
        if self.max_skew_ns is not None:
            require_non_negative_int(self.max_skew_ns, "max_skew_ns")
        copied = string_keyed_copy(self.per_stream_skew_ns, "per_stream_skew_ns")
        for stream_name, skew_ns in copied.items():
            require_non_negative_int(skew_ns, f"per_stream_skew_ns[{stream_name!r}]")
        object.__setattr__(self, "per_stream_skew_ns", copied)
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe-compatible representation."""
        return {
            "valid": self.valid,
            "comparable_streams": list(self.comparable_streams),
            "max_skew_ns": self.max_skew_ns,
            "per_stream_skew_ns": dict(self.per_stream_skew_ns),
            "frame_number_match": self.frame_number_match,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SingleDeviceSyncReport:
        """Reconstruct a report from decoded JSON data."""
        comparable = data.get("comparable_streams", [])
        skews = data.get("per_stream_skew_ns", {})
        warnings = data.get("warnings", [])
        if not isinstance(comparable, list) or not isinstance(skews, dict):
            raise TypeError("sync report arrays and objects have invalid types")
        if not isinstance(warnings, list):
            raise TypeError("warnings must be an array")
        max_skew = data.get("max_skew_ns")
        frame_match = data.get("frame_number_match")
        return cls(
            valid=decoded_bool(data["valid"], "valid"),
            comparable_streams=tuple(
                decoded_string(value, "comparable_streams[]") for value in comparable
            ),
            max_skew_ns=None if max_skew is None else decoded_int(max_skew, "max_skew_ns"),
            per_stream_skew_ns={
                decoded_string(key, "per_stream_skew_ns key"): decoded_int(
                    value, "per_stream_skew_ns value"
                )
                for key, value in skews.items()
            },
            frame_number_match=(
                None if frame_match is None else decoded_bool(frame_match, "frame_number_match")
            ),
            warnings=tuple(decoded_string(value, "warnings[]") for value in warnings),
        )
