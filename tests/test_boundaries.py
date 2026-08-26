from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_source_has_no_forbidden_implementation_dependencies() -> None:
    forbidden = (
        "calibrateCamera",
        "calibrateHandEye",
        "PointCloudBuilder",
        "FFS",
        "MultiCamera",
    )
    for path in (REPOSITORY_ROOT / "src").rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in content, f"forbidden implementation token {token!r} in {path}"
        relative = path.relative_to(REPOSITORY_ROOT)
        if not relative.is_relative_to(Path("src/camera_rig/calibration/pose")):
            for token in ("solvePnPGeneric", "solvePnPRefineLM", "projectPoints"):
                assert token not in content, f"pose token {token!r} escaped pose boundary: {path}"
        assert "solvePnP(" not in content, f"default solvePnP is forbidden in {path}"


def test_opencv_import_is_confined_to_lazy_charuco_dependency_boundary() -> None:
    allowed = {
        Path("src/camera_rig/calibration/pose/dependencies.py"),
        Path("src/camera_rig/targets/charuco/dependencies.py"),
    }
    for path in (REPOSITORY_ROOT / "src").rglob("*.py"):
        relative = path.relative_to(REPOSITORY_ROOT)
        if relative not in allowed:
            assert (
                re.search(r"^\s*import cv2\s*$", path.read_text(encoding="utf-8"), re.MULTILINE)
                is None
            )


def test_realsense_extension_is_isolated_from_core() -> None:
    for directory in ("core", "config", "artifacts", "targets", "calibration"):
        for path in (REPOSITORY_ROOT / "src/camera_rig" / directory).rglob("*.py"):
            assert "pyrealsense2" not in path.read_text(encoding="utf-8")


def test_configuration_uses_singular_camera_root() -> None:
    schema = (REPOSITORY_ROOT / "schemas/camera_config.v1.schema.json").read_text(encoding="utf-8")
    assert '"camera"' in schema
    assert '"cameras"' not in schema


def test_public_api_import_does_not_request_optional_packages() -> None:
    script = """
import importlib.abc
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pyrealsense2', 'cv2', 'PIL', 'reportlab'}:
            raise RuntimeError(f'forbidden hardware import: {fullname}')
        return None

sys.meta_path.insert(0, Blocker())
import camera_rig
import camera_rig.api
from camera_rig.api import (
    CameraBundle,
    CameraFrame,
    CameraSession,
    ReplayCameraSession,
    load_camera_bundle,
    load_provisioned_camera_bundle,
)
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
    assert result.stdout.strip() == "1.1.0.dev0"


def test_replay_import_does_not_request_hardware_or_preview_packages() -> None:
    script = """
import importlib.abc
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pyrealsense2', 'PIL', 'cv2'}:
            raise RuntimeError(f'forbidden replay import: {fullname}')
        return None

sys.meta_path.insert(0, Blocker())
from camera_rig.capture.replay import ReplayCameraSession
print(ReplayCameraSession.__name__)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ReplayCameraSession"


def test_charuco_missing_dependency_error_is_stable() -> None:
    script = """
import importlib.abc
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'cv2':
            raise ModuleNotFoundError('blocked cv2 for optional dependency test')
        return None

sys.meta_path.insert(0, Blocker())
from camera_rig.core.errors import MissingOptionalDependencyError
from camera_rig.targets.charuco.dependencies import cv2_module
try:
    cv2_module()
except MissingOptionalDependencyError as error:
    print(error)
else:
    raise AssertionError('expected MissingOptionalDependencyError')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert 'pip install "camera-rig[charuco]"' in result.stdout


def test_planar_pose_missing_dependency_error_is_stable() -> None:
    script = """
import importlib.abc
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'cv2':
            raise ModuleNotFoundError('blocked cv2 for optional dependency test')
        return None

sys.meta_path.insert(0, Blocker())
from camera_rig.core.errors import MissingOptionalDependencyError
from camera_rig.calibration.pose.dependencies import cv2_module
try:
    cv2_module()
except MissingOptionalDependencyError as error:
    print(error)
else:
    raise AssertionError('expected MissingOptionalDependencyError')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert 'pip install "camera-rig[charuco]"' in result.stdout
