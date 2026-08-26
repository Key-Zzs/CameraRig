"""Single-open acquisition primitive for fixed-camera provisioning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from camera_rig.capture.session import CameraSession
from camera_rig.capture.validation import StreamValidationAccumulator
from camera_rig.config.models import CameraConfig
from camera_rig.core.errors import ContractError
from camera_rig.core.factory_calibration import FactoryCalibration
from camera_rig.core.frame import CameraFrame
from camera_rig.drivers.realsense.factory_calibration import extract_factory_calibration
from camera_rig.provision.config import ProvisionAcquisitionSettings


class AcquisitionSession(Protocol):
    """Minimal live-session interface used by the workflow and fake integrations."""

    def __enter__(self) -> AcquisitionSession: ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...

    def capture(self) -> CameraFrame: ...


SessionFactory = Callable[[CameraConfig], AcquisitionSession]
FactoryExtractor = Callable[[AcquisitionSession], FactoryCalibration]


@dataclass(frozen=True)
class AcquisitionResult:
    """Bounded-memory output from one camera-open interval."""

    factory_calibration: FactoryCalibration
    retained_frames: tuple[CameraFrame, ...]
    selected_source_indices: tuple[int, ...]
    stream_validation_report: dict[str, object]


def evenly_spaced_indices(total_frames: int, selected_frames: int) -> tuple[int, ...]:
    """Return deterministic, unique indices spanning the complete acquisition interval."""
    if total_frames < 1 or selected_frames < 1 or selected_frames > total_frames:
        raise ContractError("evenly spaced selection requires 1 <= selected <= total")
    indices = tuple(
        int(value) for value in np.linspace(0, total_frames - 1, selected_frames, dtype=np.int64)
    )
    if len(indices) != selected_frames or len(set(indices)) != selected_frames:
        raise ContractError("evenly spaced selection produced duplicate indices")
    if indices[0] != 0 or indices[-1] != total_frames - 1:
        raise ContractError("evenly spaced selection must span first and last frame")
    return indices


def acquire_fixed_provision_frames(
    camera_config: CameraConfig,
    settings: ProvisionAcquisitionSettings,
    *,
    session_factory: SessionFactory | None = None,
    factory_extractor: FactoryExtractor | None = None,
) -> AcquisitionResult:
    """Open once, extract factory data, and validate every post-session-warmup frame.

    The public ``CameraSession`` contract owns device warmup during ``open``; this layer never
    consumes a second warmup interval. Injected sessions must model that same boundary.
    """
    make_session = session_factory or _default_session_factory
    extract_factory = factory_extractor or _default_factory_extractor
    selected = evenly_spaced_indices(settings.stream_validation_frames, settings.calibration_frames)
    selected_set = set(selected)
    retained: list[CameraFrame] = []
    accumulator = StreamValidationAccumulator(camera_config, settings.stream_validation_frames)
    with make_session(camera_config) as session:
        factory = extract_factory(session)
        for source_index in range(settings.stream_validation_frames):
            frame = session.capture()
            accumulator.add(frame)
            if source_index in selected_set:
                retained.append(frame)
    if len(retained) != settings.calibration_frames:
        raise ContractError("acquisition did not retain the configured calibration frame count")
    return AcquisitionResult(
        factory_calibration=factory,
        retained_frames=tuple(retained),
        selected_source_indices=selected,
        stream_validation_report=accumulator.report(timeout_count=0),
    )


def _default_session_factory(config: CameraConfig) -> AcquisitionSession:
    return CameraSession.from_config(config)


def _default_factory_extractor(session: AcquisitionSession) -> FactoryCalibration:
    if not isinstance(session, CameraSession):
        raise ContractError("default factory extractor requires CameraSession")
    return extract_factory_calibration(session.driver)
