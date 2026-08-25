from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_source_has_no_forbidden_implementation_dependencies() -> None:
    forbidden = (
        "cv2.aruco",
        "import cv2",
        "PointCloudBuilder",
        "FFS",
        "MultiCamera",
    )
    for path in (REPOSITORY_ROOT / "src").rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in content, f"forbidden implementation token {token!r} in {path}"


def test_realsense_extension_is_isolated_from_core() -> None:
    for directory in ("core", "config", "artifacts", "targets", "calibration"):
        for path in (REPOSITORY_ROOT / "src/camera_rig" / directory).rglob("*.py"):
            assert "pyrealsense2" not in path.read_text(encoding="utf-8")


def test_configuration_uses_singular_camera_root() -> None:
    schema = (REPOSITORY_ROOT / "schemas/camera_config.v1.schema.json").read_text(encoding="utf-8")
    assert '"camera"' in schema
    assert '"cameras"' not in schema


def test_core_import_does_not_request_hardware_packages() -> None:
    script = """
import importlib.abc
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pyrealsense2', 'cv2'}:
            raise RuntimeError(f'forbidden hardware import: {fullname}')
        return None

sys.meta_path.insert(0, Blocker())
import camera_rig
import camera_rig.core
print(camera_rig.__version__)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1.0"
