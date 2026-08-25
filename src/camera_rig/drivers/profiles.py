"""SDK-independent RealSense identity and profile rules."""

from __future__ import annotations

import re

from camera_rig.config.models import CameraConfig
from camera_rig.core.errors import DeviceMismatchError, ProfileNotSupportedError
from camera_rig.core.stream import StreamProfile

STREAM_LAYOUT = {
    "color": ("color", 0, "rgb8", "RGB Camera"),
    "depth": ("depth", 0, "z16", "Stereo Module"),
    "ir_left": ("infrared", 1, "y8", "Stereo Module"),
    "ir_right": ("infrared", 2, "y8", "Stereo Module"),
}


def canonical_d435i(reported_model: str, product_id: str | None) -> str | None:
    """Return the canonical model only for explicit D435i identifiers."""
    normalized = re.sub(r"[^a-z0-9]", "", reported_model.casefold())
    model_matches = normalized in {"d435i", "realsensed435i", "intelrealsensed435i"}
    product_matches = product_id is not None and product_id.casefold() == "0b3a"
    if model_matches and (product_id is None or product_matches):
        return "D435i"
    if not reported_model.strip() and product_matches:
        return "D435i"
    return None


def validate_model(expected: str, reported: str, product_id: str | None) -> str:
    """Validate configured and reported identities against an explicit map."""
    expected_canonical = canonical_d435i(expected, "0B3A")
    reported_canonical = canonical_d435i(reported, product_id)
    if expected_canonical != "D435i" or reported_canonical != expected_canonical:
        raise DeviceMismatchError(
            f"device model mismatch: expected={expected!r}, reported={reported!r}, "
            f"product_id={product_id!r}"
        )
    return reported_canonical


def requested_profiles(config: CameraConfig) -> tuple[StreamProfile, ...]:
    """Convert enabled configuration streams into explicit SDK identities."""
    result: list[StreamProfile] = []
    for name, settings in config.streams.items():
        if not settings.enabled:
            continue
        try:
            _, index, required_format, sensor = STREAM_LAYOUT[name]
        except KeyError as error:
            raise ProfileNotSupportedError(f"unsupported RealSense stream: {name!r}") from error
        profile = settings.profile
        if profile.format.casefold() != required_format:
            raise ProfileNotSupportedError(
                f"stream {name!r} requires format {required_format!r}, got {profile.format!r}"
            )
        result.append(
            StreamProfile(
                stream_name=name,
                width=profile.width,
                height=profile.height,
                fps=profile.fps,
                format=required_format,
                index=index,
                sensor_identifier=sensor,
            )
        )
    return tuple(result)


def validate_supported(
    requested: tuple[StreamProfile, ...], supported: tuple[StreamProfile, ...]
) -> None:
    """Fail before opening unless every exact request is enumerated by its sensor."""
    for wanted in requested:
        match = next((candidate for candidate in supported if _matches(wanted, candidate)), None)
        if match is None:
            raise ProfileNotSupportedError(
                f"requested profile is not supported: {wanted.to_dict()}"
            )


def validate_active(
    requested: tuple[StreamProfile, ...], active: tuple[StreamProfile, ...]
) -> None:
    """Reject SDK substitutions after resolve/start."""
    for wanted in requested:
        candidates = [value for value in active if value.stream_name == wanted.stream_name]
        if len(candidates) != 1 or not _matches(wanted, candidates[0], require_sensor=False):
            raise ProfileNotSupportedError(
                f"active profile differs from request for {wanted.stream_name!r}: "
                f"requested={wanted.to_dict()}, active={[value.to_dict() for value in candidates]}"
            )


def _matches(
    wanted: StreamProfile, candidate: StreamProfile, *, require_sensor: bool = True
) -> bool:
    equal = (
        wanted.stream_name == candidate.stream_name
        and wanted.index == candidate.index
        and wanted.width == candidate.width
        and wanted.height == candidate.height
        and wanted.fps == candidate.fps
        and wanted.format.casefold() == candidate.format.casefold()
    )
    if not equal or not require_sensor or wanted.sensor_identifier is None:
        return equal
    return (
        candidate.sensor_identifier is not None
        and wanted.sensor_identifier.casefold() in candidate.sensor_identifier.casefold()
    )
