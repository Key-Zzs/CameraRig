"""Resolved, OpenCV-version-independent ChArUco target artifact contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from camera_rig.core.errors import ArtifactError


@dataclass(frozen=True)
class ResolvedCharucoTarget:
    """Persisted ID-to-canonical-point mapping used by every later observation."""

    SCHEMA_VERSION: ClassVar[str] = "camera-rig.target.charuco-resolved.v1"

    target_name: str
    target_frame: str
    dictionary: str
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    border_bits: int
    legacy_pattern: bool
    board_width_m: float
    board_height_m: float
    corner_points: tuple[tuple[int, tuple[float, float, float]], ...]
    marker_ids: tuple[int, ...]
    camera_rig_version: str
    opencv_version: str
    source_config_sha256: str
    board_png_sha256: str
    print_pdf_sha256: str
    artifact_sha256: str = ""
    plugin: str = "charuco"

    def __post_init__(self) -> None:
        if self.plugin != "charuco" or not self.target_name or not self.target_frame:
            raise ArtifactError("invalid resolved ChArUco target identity")
        if self.marker_length_m <= 0 or self.marker_length_m >= self.square_length_m:
            raise ArtifactError("invalid resolved ChArUco target dimensions")
        ids = tuple(item[0] for item in self.corner_points)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise ArtifactError("resolved ChArUco corner IDs must be unique and sorted")
        expected = (self.squares_x - 1) * (self.squares_y - 1)
        if len(ids) != expected:
            raise ArtifactError(f"resolved ChArUco target must contain {expected} corners")
        points = np.asarray([item[1] for item in self.corner_points], dtype=np.float64)
        if points.shape != (expected, 3) or not np.isfinite(points).all():
            raise ArtifactError("resolved ChArUco corner geometry is invalid")
        if not np.equal(points[:, 2], 0.0).all():
            raise ArtifactError("resolved ChArUco canonical points must lie on z=0")
        if (points[:, 0] <= 0).any() or (points[:, 0] >= self.board_width_m).any():
            raise ArtifactError("resolved ChArUco x coordinates lie outside the board")
        if (points[:, 1] <= 0).any() or (points[:, 1] >= self.board_height_m).any():
            raise ArtifactError("resolved ChArUco y coordinates lie outside the board")
        if not np.isclose(self.board_width_m, self.squares_x * self.square_length_m):
            raise ArtifactError("resolved board width differs from square geometry")
        if not np.isclose(self.board_height_m, self.squares_y * self.square_length_m):
            raise ArtifactError("resolved board height differs from square geometry")
        for point_id, point in self.corner_points:
            row, column = divmod(point_id, self.squares_x - 1)
            expected_point = (
                (column + 1) * self.square_length_m,
                self.board_height_m - (row + 1) * self.square_length_m,
                0.0,
            )
            if not np.allclose(point, expected_point, rtol=0.0, atol=1e-7):
                raise ArtifactError(
                    f"persisted canonical geometry is inconsistent for corner ID {point_id}"
                )
        if self.marker_ids != tuple(sorted(set(self.marker_ids))):
            raise ArtifactError("resolved marker IDs must be unique and sorted")

    def object_points_for(self, point_ids: tuple[int, ...]) -> npt.NDArray[np.float64]:
        if not point_ids:
            return np.empty((0, 3), dtype=np.float64)
        lookup = dict(self.corner_points)
        try:
            points = [lookup[point_id] for point_id in point_ids]
        except KeyError as error:
            raise ArtifactError(f"detected unknown ChArUco corner ID: {error.args[0]}") from error
        return np.asarray(points, dtype=np.float64)

    def with_artifact_sha256(self, digest: str) -> ResolvedCharucoTarget:
        return replace(self, artifact_sha256=digest)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "target_name": self.target_name,
            "plugin": self.plugin,
            "target_frame": self.target_frame,
            "dictionary": self.dictionary,
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_length_m": self.square_length_m,
            "marker_length_m": self.marker_length_m,
            "border_bits": self.border_bits,
            "legacy_pattern": self.legacy_pattern,
            "board_width_m": self.board_width_m,
            "board_height_m": self.board_height_m,
            "coordinate_frame": {
                "origin": "outer_bottom_left",
                "x_axis": "board_right",
                "y_axis": "board_up",
                "z_axis": "out_of_printed_face",
                "right_handed": True,
            },
            "charuco_corners": [
                {"id": point_id, "object_point_m": list(point)}
                for point_id, point in self.corner_points
            ],
            "marker_ids": list(self.marker_ids),
            "generator": {
                "camera_rig_version": self.camera_rig_version,
                "opencv_version": self.opencv_version,
            },
            "source_config_sha256": self.source_config_sha256,
            "board_png_sha256": self.board_png_sha256,
            "print_pdf_sha256": self.print_pdf_sha256,
            "files": {
                "board_png": f"{self.target_name}_board.png",
                "print_pdf": f"{self.target_name}_print.pdf",
                "preview_png": f"{self.target_name}_preview.png",
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ResolvedCharucoTarget:
        required = {
            "schema_version",
            "target_name",
            "plugin",
            "target_frame",
            "dictionary",
            "squares_x",
            "squares_y",
            "square_length_m",
            "marker_length_m",
            "border_bits",
            "legacy_pattern",
            "board_width_m",
            "board_height_m",
            "coordinate_frame",
            "charuco_corners",
            "marker_ids",
            "generator",
            "source_config_sha256",
            "board_png_sha256",
            "print_pdf_sha256",
            "files",
        }
        if set(data) != required or data.get("schema_version") != cls.SCHEMA_VERSION:
            raise ArtifactError("resolved ChArUco target has missing or unknown fields")
        coordinate = _object(data["coordinate_frame"], "coordinate_frame")
        if coordinate != {
            "origin": "outer_bottom_left",
            "x_axis": "board_right",
            "y_axis": "board_up",
            "z_axis": "out_of_printed_face",
            "right_handed": True,
        }:
            raise ArtifactError("resolved target uses an unsupported coordinate frame")
        files = _object(data["files"], "files")
        if set(files) != {"board_png", "print_pdf", "preview_png"}:
            raise ArtifactError("resolved target files mapping is invalid")
        for value in files.values():
            if not isinstance(value, str) or "/" in value or "\\" in value:
                raise ArtifactError("resolved target artifact paths must be relative filenames")
        corners_value = data["charuco_corners"]
        if not isinstance(corners_value, list):
            raise ArtifactError("charuco_corners must be an array")
        corners: list[tuple[int, tuple[float, float, float]]] = []
        for item_value in corners_value:
            item = _object(item_value, "charuco_corners[]")
            if set(item) != {"id", "object_point_m"}:
                raise ArtifactError("ChArUco corner has missing or unknown fields")
            point_id = _int(item["id"], "corner id")
            point_value = item["object_point_m"]
            if not isinstance(point_value, list) or len(point_value) != 3:
                raise ArtifactError("object_point_m must contain three numbers")
            point = tuple(_float(value, "object_point_m[]") for value in point_value)
            corners.append((point_id, (point[0], point[1], point[2])))
        generator = _object(data["generator"], "generator")
        if set(generator) != {"camera_rig_version", "opencv_version"}:
            raise ArtifactError("resolved target generator provenance is invalid")
        marker_value = data["marker_ids"]
        if not isinstance(marker_value, list):
            raise ArtifactError("marker_ids must be an array")
        return cls(
            target_name=_string(data["target_name"], "target_name"),
            plugin=_string(data["plugin"], "plugin"),
            target_frame=_string(data["target_frame"], "target_frame"),
            dictionary=_string(data["dictionary"], "dictionary"),
            squares_x=_int(data["squares_x"], "squares_x"),
            squares_y=_int(data["squares_y"], "squares_y"),
            square_length_m=_float(data["square_length_m"], "square_length_m"),
            marker_length_m=_float(data["marker_length_m"], "marker_length_m"),
            border_bits=_int(data["border_bits"], "border_bits"),
            legacy_pattern=_bool(data["legacy_pattern"], "legacy_pattern"),
            board_width_m=_float(data["board_width_m"], "board_width_m"),
            board_height_m=_float(data["board_height_m"], "board_height_m"),
            corner_points=tuple(corners),
            marker_ids=tuple(_int(value, "marker_ids[]") for value in marker_value),
            camera_rig_version=_string(generator["camera_rig_version"], "camera_rig_version"),
            opencv_version=_string(generator["opencv_version"], "opencv_version"),
            source_config_sha256=_digest(data["source_config_sha256"], "source_config_sha256"),
            board_png_sha256=_digest(data["board_png_sha256"], "board_png_sha256"),
            print_pdf_sha256=_digest(data["print_pdf_sha256"], "print_pdf_sha256"),
        )


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ArtifactError(f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{name} must be a non-empty string")
    return value


def _int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactError(f"{name} must be an integer")
    return value


def _float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(value):
        raise ArtifactError(f"{name} must be a finite number")
    return float(value)


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ArtifactError(f"{name} must be a boolean")
    return value


def _digest(value: object, name: str) -> str:
    text = _string(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ArtifactError(f"{name} must be a lowercase SHA-256 digest")
    return text
