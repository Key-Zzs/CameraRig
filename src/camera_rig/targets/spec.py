"""Detector-independent target specification marker protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TargetSpec(Protocol):
    """Minimum identity consumed by the target detector registry."""

    plugin: str
    target_name: str
    target_frame: str
