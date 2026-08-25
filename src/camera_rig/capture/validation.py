"""Streaming statistics and validity checks for one physical camera."""

from __future__ import annotations

import time
from collections import Counter
from itertools import pairwise

import numpy as np

from camera_rig.config.models import CameraConfig
from camera_rig.core.frame import CameraFrame


class StreamValidationAccumulator:
    """Bounded-memory validation over a sequence of CameraFrame values."""

    def __init__(self, config: CameraConfig, requested_frames: int) -> None:
        self.config = config
        self.requested_frames = requested_frames
        self.started_ns = time.monotonic_ns()
        self.received_frames = 0
        self.missing_streams: Counter[str] = Counter()
        self.frame_numbers: dict[str, list[int]] = {}
        self.timestamps: dict[str, list[int]] = {}
        self.domains: dict[str, Counter[str]] = {}
        self.shapes: dict[str, set[tuple[int, ...]]] = {}
        self.dtypes: dict[str, set[str]] = {}
        self.skews: list[int] = []
        self.stereo_matches = 0
        self.stereo_compared = 0
        self.sync_valid = 0
        self.depth_valid_ratios: list[float] = []
        self.color_variances: list[float] = []
        self.color_channel_variances: list[list[float]] = []
        self.ir_variances: dict[str, list[float]] = {"ir_left": [], "ir_right": []}
        self.ir_distinct = 0

    def add(self, frame: CameraFrame) -> None:
        self.received_frames += 1
        required = self.config.capture.required_streams or tuple(
            name for name, settings in self.config.streams.items() if settings.enabled
        )
        for name in required:
            stream = frame.streams.get(name)
            if stream is None:
                self.missing_streams[name] += 1
                continue
            self.frame_numbers.setdefault(name, []).append(stream.frame_number)
            if stream.sensor_timestamp_ns is not None:
                self.timestamps.setdefault(name, []).append(stream.sensor_timestamp_ns)
            if stream.timestamp_domain is not None:
                self.domains.setdefault(name, Counter())[stream.timestamp_domain] += 1
            self.shapes.setdefault(name, set()).add(tuple(stream.data.shape))
            self.dtypes.setdefault(name, set()).add(str(stream.data.dtype))
        if frame.sync_report is not None:
            if frame.sync_report.max_skew_ns is not None:
                self.skews.append(frame.sync_report.max_skew_ns)
            self.sync_valid += int(frame.sync_report.valid)
            if frame.sync_report.frame_number_match is not None:
                self.stereo_compared += 1
                self.stereo_matches += int(frame.sync_report.frame_number_match)
        color = frame.color
        if color is not None:
            data = color.data.astype(np.float64, copy=False)
            self.color_variances.append(float(np.var(data)))
            self.color_channel_variances.append(
                [float(np.var(data[..., channel])) for channel in range(3)]
            )
        depth = frame.depth
        if depth is not None:
            self.depth_valid_ratios.append(float(np.count_nonzero(depth.data) / depth.data.size))
        for name in ("ir_left", "ir_right"):
            stream = frame.streams.get(name)
            if stream is not None:
                self.ir_variances[name].append(
                    float(np.var(np.asarray(stream.data, dtype=np.float64)))
                )
        if frame.ir_left is not None and frame.ir_right is not None:
            self.ir_distinct += int(not np.array_equal(frame.ir_left.data, frame.ir_right.data))

    def report(self, timeout_count: int = 0) -> dict[str, object]:
        duration_s = (time.monotonic_ns() - self.started_ns) / 1_000_000_000
        discontinuities: dict[str, int] = {}
        monotonicity: dict[str, bool] = {}
        observed_fps: dict[str, float] = {}
        drop_ratios: dict[str, float] = {}
        for name, numbers in self.frame_numbers.items():
            discontinuities[name] = sum(
                current != previous + 1 for previous, current in pairwise(numbers)
            )
            denominator = max(len(numbers) - 1, 1)
            drop_ratios[name] = discontinuities[name] / denominator
            timestamps = self.timestamps.get(name, [])
            monotonicity[name] = all(
                current > previous for previous, current in pairwise(timestamps)
            )
            elapsed_ns = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
            observed_fps[name] = (
                (len(timestamps) - 1) * 1_000_000_000 / elapsed_ns if elapsed_ns > 0 else 0.0
            )
        stereo_ratio = self.stereo_matches / self.stereo_compared if self.stereo_compared else 0.0
        sync_valid_ratio = self.sync_valid / self.received_frames if self.received_frames else 0.0
        failures: list[str] = []
        if self.received_frames != self.requested_frames:
            failures.append("received frame count differs from request")
        if self.missing_streams:
            failures.append("required streams were missing")
        for name, settings in self.config.streams.items():
            if not settings.enabled:
                continue
            if observed_fps.get(name, 0.0) < settings.profile.fps * 0.9:
                failures.append(f"{name} observed FPS is below 90% of requested")
            if drop_ratios.get(name, 1.0) > 0.01:
                failures.append(f"{name} discontinuity ratio exceeds 1%")
            if not monotonicity.get(name, False):
                failures.append(f"{name} sensor timestamps are not strictly monotonic")
            if len(self.shapes.get(name, set())) != 1:
                failures.append(f"{name} shape changed during capture")
            if len(self.dtypes.get(name, set())) != 1:
                failures.append(f"{name} dtype changed during capture")
        if timeout_count:
            failures.append("frame timeouts occurred")
        if self.config.capture.sync.require_stereo_frame_number_match and stereo_ratio < 1.0:
            failures.append("IR left/right frame numbers did not always match")
        if sync_valid_ratio < 1.0:
            failures.append("one or more frames failed the internal sync threshold")
        if _mean(self.color_variances) <= 0:
            failures.append("color stream has no variance")
        if _mean(self.depth_valid_ratios) <= 0:
            failures.append("depth stream has no valid pixels")
        if any(_mean(values) <= 0 for values in self.ir_variances.values()):
            failures.append("one or more IR streams have no variance")
        if self.ir_distinct != self.received_frames:
            failures.append("IR left/right arrays were bitwise identical")
        skew_stats = _percentiles(self.skews)
        return {
            "schema_version": "camera-rig.stream-validation.v1",
            "status": "PASS" if not failures else "FAIL",
            "requested_frames": self.requested_frames,
            "received_frames": self.received_frames,
            "duration_s": duration_s,
            "per_stream_observed_fps": observed_fps,
            "per_stream_frame_number_discontinuities": discontinuities,
            "per_stream_discontinuity_ratio": drop_ratios,
            "per_stream_timestamp_monotonicity": monotonicity,
            "per_stream_timestamp_domain_counts": {
                name: dict(counts) for name, counts in self.domains.items()
            },
            "ir_stereo_frame_match_ratio": stereo_ratio,
            "comparable_timestamp_skew_ns": skew_stats,
            "sync_valid_ratio": sync_valid_ratio,
            "timeouts": timeout_count,
            "missing_streams": dict(self.missing_streams),
            "shape_consistency": {
                name: [list(shape) for shape in sorted(shapes)]
                for name, shapes in self.shapes.items()
            },
            "dtype_consistency": {name: sorted(dtypes) for name, dtypes in self.dtypes.items()},
            "depth_valid_ratio": _mean(self.depth_valid_ratios),
            "rgb_variance": _mean(self.color_variances),
            "rgb_channel_variance": _vector_mean(self.color_channel_variances),
            "ir_variance": {name: _mean(values) for name, values in self.ir_variances.items()},
            "ir_distinct_ratio": (
                self.ir_distinct / self.received_frames if self.received_frames else 0.0
            ),
            "failure_reasons": failures,
        }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _vector_mean(values: list[list[float]]) -> list[float]:
    return np.mean(np.asarray(values), axis=0).tolist() if values else []


def _percentiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.int64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": int(np.max(array)),
    }
