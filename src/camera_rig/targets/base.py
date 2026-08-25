"""Detector plugin interface; no concrete target detector is implemented here."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from camera_rig.targets.observation import TargetObservation


@runtime_checkable
class TargetDetector(Protocol):
    """Protocol implemented by future target-detection plugins."""

    plugin_name: str

    def detect(self, image: npt.NDArray[np.generic]) -> TargetObservation:
        """Detect a calibration target in one image."""
        ...
