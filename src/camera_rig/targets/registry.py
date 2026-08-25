"""Small built-in registry for target detector plugins."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from camera_rig.core.errors import ContractError
from camera_rig.targets.base import TargetDetector

DetectorFactory = Callable[[object], TargetDetector]


class TargetDetectorRegistry:
    """Create detectors by stable plugin name without exposing implementations."""

    def __init__(self) -> None:
        self._factories: dict[str, DetectorFactory] = {}

    def register(self, plugin_name: str, factory: DetectorFactory) -> None:
        if not plugin_name or plugin_name in self._factories:
            raise ContractError(f"target detector plugin is already registered: {plugin_name!r}")
        self._factories[plugin_name] = factory

    def create(self, *, plugin_name: str, target_spec: object) -> TargetDetector:
        self._load_builtin(plugin_name)
        try:
            factory = self._factories[plugin_name]
        except KeyError as error:
            raise ContractError(f"unknown target detector plugin: {plugin_name!r}") from error
        return factory(target_spec)

    def available_plugins(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def _load_builtin(self, plugin_name: str) -> None:
        if plugin_name == "charuco" and plugin_name not in self._factories:
            from camera_rig.targets.charuco.detector import CharucoDetector

            def create(target_spec: object) -> TargetDetector:
                return cast(TargetDetector, CharucoDetector(target_spec))

            self.register(plugin_name, create)


registry = TargetDetectorRegistry()
