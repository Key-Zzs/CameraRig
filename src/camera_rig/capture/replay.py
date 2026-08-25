"""SDK-independent replay of validated raw capture artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from camera_rig.artifacts.capture_io import restore_camera_frame
from camera_rig.artifacts.capture_validation import validate_capture_artifact
from camera_rig.artifacts.io import load_json
from camera_rig.core.errors import ArtifactError, LifecycleError, ReplayEOFError
from camera_rig.core.frame import CameraFrame


class ReplayCameraSession:
    """Sequential CameraFrame playback with explicit open, EOF, and rewind."""

    def __init__(self, artifact: str | Path) -> None:
        self.artifact = Path(artifact)
        self.manifest = validate_capture_artifact(self.artifact)
        entries = self.manifest["frames"]
        if not isinstance(entries, list):
            raise ArtifactError("capture manifest frames must be an array")
        self._entries = entries
        self._index = 0
        self._open = False

    @classmethod
    def from_artifact(cls, path: str | Path) -> ReplayCameraSession:
        return cls(path)

    @property
    def frame_count(self) -> int:
        return len(self._entries)

    @property
    def position(self) -> int:
        return self._index

    def open(self) -> None:
        if self._open:
            raise LifecycleError("replay session is already open")
        self._open = True

    def close(self) -> None:
        self._open = False

    def capture(self) -> CameraFrame:
        if not self._open:
            raise LifecycleError("replay session is not open")
        if self._index >= len(self._entries):
            raise ReplayEOFError("replay reached end of artifact")
        entry = self._entry(self._entries[self._index])
        metadata_value = load_json(self.artifact / self._string(entry["metadata_path"]))
        if not isinstance(metadata_value, dict):
            raise ArtifactError("frame metadata must be a JSON object")
        arrays: dict[str, npt.NDArray[np.generic]] = {}
        with np.load(
            self.artifact / self._string(entry["data_path"]), allow_pickle=False
        ) as archive:
            for name in archive.files:
                arrays[name] = np.asarray(archive[name]).copy()
        frame = restore_camera_frame(dict(metadata_value), arrays)
        self._index += 1
        return frame

    def wait_for_frame(self) -> CameraFrame:
        return self.capture()

    def poll_frame(self) -> CameraFrame | None:
        if not self._open:
            raise LifecycleError("replay session is not open")
        if self._index >= len(self._entries):
            return None
        return self.capture()

    def reset(self) -> None:
        self._index = 0

    rewind = reset

    def __enter__(self) -> ReplayCameraSession:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @staticmethod
    def _entry(value: object) -> dict[str, object]:
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise ArtifactError("capture frame entry must be an object")
        return value

    @staticmethod
    def _string(value: object) -> str:
        if not isinstance(value, str):
            raise ArtifactError("capture frame path must be a string")
        return value
