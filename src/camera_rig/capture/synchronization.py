"""Single-device frameset timing validation without cross-device claims."""

from __future__ import annotations

from camera_rig.config.models import CameraConfig
from camera_rig.core.frame import StreamFrame
from camera_rig.core.timestamps import SingleDeviceSyncReport


def build_sync_report(
    config: CameraConfig, streams: dict[str, StreamFrame]
) -> SingleDeviceSyncReport:
    """Compare only streams whose sensor timestamps use the same domain."""
    reference_name = config.camera.output_reference_stream
    reference = streams[reference_name]
    skews: dict[str, int] = {}
    comparable: list[str] = []
    warnings: list[str] = []
    if reference.sensor_timestamp_ns is None or reference.timestamp_domain is None:
        warnings.append(f"reference stream {reference_name!r} has no comparable timestamp")
    else:
        for name, frame in streams.items():
            if frame.sensor_timestamp_ns is None or frame.timestamp_domain is None:
                warnings.append(f"stream {name!r} has no sensor timestamp")
                continue
            if frame.timestamp_domain != reference.timestamp_domain:
                warnings.append(
                    f"stream {name!r} timestamp domain {frame.timestamp_domain!r} differs "
                    f"from reference domain {reference.timestamp_domain!r}"
                )
                continue
            comparable.append(name)
            skews[name] = abs(frame.sensor_timestamp_ns - reference.sensor_timestamp_ns)
    max_skew = max(skews.values()) if skews else None
    left = streams.get("ir_left")
    right = streams.get("ir_right")
    stereo_match = None
    if left is not None and right is not None:
        stereo_match = left.frame_number == right.frame_number
        if not stereo_match:
            warnings.append("IR left/right frame numbers differ")
    threshold_ns = round(config.capture.sync.max_comparable_stream_skew_ms * 1_000_000)
    valid = max_skew is not None and max_skew <= threshold_ns
    if config.capture.sync.require_stereo_frame_number_match and stereo_match is not None:
        valid = valid and stereo_match
    return SingleDeviceSyncReport(
        valid=valid,
        comparable_streams=tuple(comparable),
        max_skew_ns=max_skew,
        per_stream_skew_ns=skews,
        frame_number_match=stereo_match,
        warnings=tuple(warnings),
    )
