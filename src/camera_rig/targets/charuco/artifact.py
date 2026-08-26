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
        expected_markers = (self.squares_x * self.squares_y) // 2
        if len(self.marker_ids) != expected_markers:
            raise ArtifactError(
                f"resolved ChArUco target must contain {expected_markers} marker IDs"
            )
        if len(set(self.marker_ids)) != len(self.marker_ids) or any(
            marker_id < 0 for marker_id in self.marker_ids
        ):
            raise ArtifactError("resolved marker IDs must be unique and non-negative")
        dictionary_capacity = _dictionary_capacity(self.dictionary)
        if any(marker_id >= dictionary_capacity for marker_id in self.marker_ids):
            raise ArtifactError("resolved marker ID exceeds configured dictionary capacity")

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


@dataclass(frozen=True)
class ResolvedCharucoTargetV2(ResolvedCharucoTarget):
    """Resolved target supporting generated or independently identified physical boards."""

    SCHEMA_VERSION: ClassVar[str] = "camera-rig.target.charuco-resolved.v2"

    source_type: str = "existing_physical"
    physical_measurement: dict[str, object] | None = None
    identification: dict[str, object] | None = None
    artifact_files: dict[str, str] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.source_type not in {"generated", "existing_physical"}:
            raise ArtifactError("resolved v2 source_type must be generated or existing_physical")
        if not isinstance(self.physical_measurement, dict):
            raise ArtifactError("resolved v2 physical_measurement must be an object")
        if not isinstance(self.identification, dict):
            raise ArtifactError("resolved v2 identification must be an object")
        if not isinstance(self.artifact_files, dict):
            raise ArtifactError("resolved v2 artifact_files must be an object")
        files = dict(self.artifact_files)
        if not all(
            isinstance(key, str)
            and isinstance(value, str)
            and value
            and "/" not in value
            and "\\" not in value
            for key, value in files.items()
        ):
            raise ArtifactError("resolved v2 artifact files must be relative filenames")
        if self.source_type == "existing_physical" and files:
            raise ArtifactError("existing physical targets must not claim generated print files")
        physical = dict(self.physical_measurement)
        required_measurements = {
            "nominal_width_mm",
            "nominal_height_mm",
            "square_length_mm",
            "marker_length_mm",
            "source",
        }
        if set(physical) != required_measurements:
            raise ArtifactError("resolved v2 physical measurement has missing or unknown fields")
        expected_measurements = {
            "nominal_width_mm": self.board_width_m * 1000.0,
            "nominal_height_mm": self.board_height_m * 1000.0,
            "square_length_mm": self.square_length_m * 1000.0,
            "marker_length_mm": self.marker_length_m * 1000.0,
        }
        for name, expected in expected_measurements.items():
            value = _float(physical[name], f"physical_measurement.{name}")
            if not np.isclose(value, expected, rtol=0.0, atol=1e-6):
                raise ArtifactError(f"physical measurement {name} differs from target geometry")
        expected_source = (
            "user-provided-and-vision-verified"
            if self.source_type == "existing_physical"
            else "generated"
        )
        if physical["source"] != expected_source:
            raise ArtifactError("physical measurement source is inconsistent with source_type")
        identification = dict(self.identification)
        if identification.get("candidate_uniqueness") is not True:
            raise ArtifactError("resolved v2 identification must record a unique candidate")
        evidence_hashes = identification.get("evidence_hashes")
        if not isinstance(evidence_hashes, list) or not evidence_hashes:
            raise ArtifactError("resolved v2 identification requires evidence hashes")
        for index, digest in enumerate(evidence_hashes):
            _digest(digest, f"identification.evidence_hashes[{index}]")
        if identification.get("opencv_version") != self.opencv_version:
            raise ArtifactError("identification OpenCV version differs from generator provenance")
        if self.source_type == "existing_physical":
            required_identification = {
                "candidate_uniqueness",
                "identification_report_sha256",
                "evidence_hashes",
                "opencv_version",
                "recognition_metrics",
            }
            if set(identification) != required_identification:
                raise ArtifactError("existing-target identification has missing or unknown fields")
            _digest(
                identification["identification_report_sha256"],
                "identification.identification_report_sha256",
            )
            if not isinstance(identification["recognition_metrics"], dict):
                raise ArtifactError("identification recognition_metrics must be an object")
            recognition = dict(identification["recognition_metrics"])
            if set(recognition) != {
                "minimum_charuco_corners",
                "minimum_expected_marker_fraction",
                "evidence_frame_count",
            }:
                raise ArtifactError(
                    "identification recognition_metrics has missing or unknown fields"
                )
            if _int(recognition["minimum_charuco_corners"], "minimum_charuco_corners") < 20:
                raise ArtifactError("existing-target identification has too few corners")
            marker_fraction = _float(
                recognition["minimum_expected_marker_fraction"],
                "minimum_expected_marker_fraction",
            )
            if marker_fraction < 0.80 or marker_fraction > 1.0:
                raise ArtifactError("existing-target identification marker fraction is invalid")
            if _int(recognition["evidence_frame_count"], "evidence_frame_count") < 2:
                raise ArtifactError("existing-target identification requires multiple frames")
        else:
            if (
                set(identification)
                != {
                    "candidate_uniqueness",
                    "method",
                    "evidence_hashes",
                    "opencv_version",
                }
                or identification.get("method") != "generated-self-check"
            ):
                raise ArtifactError("generated-target identification contract is invalid")
            allowed_file_sets = (
                {"board_png", "print_pdf", "preview_png"},
                {"board_png", "print_pdf", "preview_png", "scale_check_pdf"},
            )
            if set(files) not in allowed_file_sets:
                raise ArtifactError("generated v2 target files mapping is invalid")
        object.__setattr__(self, "physical_measurement", dict(self.physical_measurement))
        object.__setattr__(self, "identification", dict(self.identification))
        object.__setattr__(self, "artifact_files", files)

    def to_dict(self) -> dict[str, object]:
        value = super().to_dict()
        value["schema_version"] = self.SCHEMA_VERSION
        value["source_type"] = self.source_type
        value["physical_measurement"] = dict(self.physical_measurement or {})
        value["identification"] = dict(self.identification or {})
        value["files"] = dict(self.artifact_files or {})
        if self.source_type == "existing_physical":
            value.pop("source_config_sha256")
            value.pop("board_png_sha256")
            value.pop("print_pdf_sha256")
        return value

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ResolvedCharucoTargetV2:
        common = {
            "schema_version",
            "source_type",
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
            "physical_measurement",
            "identification",
            "generator",
            "files",
        }
        source_type = _string(data.get("source_type"), "source_type")
        generated_fields = {"source_config_sha256", "board_png_sha256", "print_pdf_sha256"}
        expected = common | (generated_fields if source_type == "generated" else set())
        if set(data) != expected or data.get("schema_version") != cls.SCHEMA_VERSION:
            raise ArtifactError("resolved ChArUco v2 target has missing or unknown fields")
        coordinate = _object(data["coordinate_frame"], "coordinate_frame")
        if coordinate != {
            "origin": "outer_bottom_left",
            "x_axis": "board_right",
            "y_axis": "board_up",
            "z_axis": "out_of_printed_face",
            "right_handed": True,
        }:
            raise ArtifactError("resolved target uses an unsupported coordinate frame")
        corners_value = data["charuco_corners"]
        if not isinstance(corners_value, list):
            raise ArtifactError("charuco_corners must be an array")
        corners: list[tuple[int, tuple[float, float, float]]] = []
        for item_value in corners_value:
            item = _object(item_value, "charuco_corners[]")
            if set(item) != {"id", "object_point_m"}:
                raise ArtifactError("ChArUco corner has missing or unknown fields")
            point_value = item["object_point_m"]
            if not isinstance(point_value, list) or len(point_value) != 3:
                raise ArtifactError("object_point_m must contain three numbers")
            point = tuple(_float(value, "object_point_m[]") for value in point_value)
            corners.append((_int(item["id"], "corner id"), (point[0], point[1], point[2])))
        generator = _object(data["generator"], "generator")
        if set(generator) != {"camera_rig_version", "opencv_version"}:
            raise ArtifactError("resolved target generator provenance is invalid")
        marker_value = data["marker_ids"]
        if not isinstance(marker_value, list):
            raise ArtifactError("marker_ids must be an array")
        files = _object(data["files"], "files")
        physical = _object(data["physical_measurement"], "physical_measurement")
        identification = _object(data["identification"], "identification")
        empty_digest = "0" * 64
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
            source_config_sha256=(
                _digest(data["source_config_sha256"], "source_config_sha256")
                if source_type == "generated"
                else empty_digest
            ),
            board_png_sha256=(
                _digest(data["board_png_sha256"], "board_png_sha256")
                if source_type == "generated"
                else empty_digest
            ),
            print_pdf_sha256=(
                _digest(data["print_pdf_sha256"], "print_pdf_sha256")
                if source_type == "generated"
                else empty_digest
            ),
            source_type=source_type,
            physical_measurement=physical,
            identification=identification,
            artifact_files={key: _string(value, f"files.{key}") for key, value in files.items()},
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


def _dictionary_capacity(name: str) -> int:
    if name == "DICT_ARUCO_ORIGINAL":
        return 1024
    parts = name.split("_")
    if (
        len(parts) == 3
        and parts[0] == "DICT"
        and parts[1]
        in {
            "4X4",
            "5X5",
            "6X6",
            "7X7",
        }
    ):
        try:
            capacity = int(parts[2])
        except ValueError as error:
            raise ArtifactError("resolved target dictionary is unsupported") from error
        if capacity in {50, 100, 250, 1000}:
            return capacity
    raise ArtifactError("resolved target dictionary is unsupported")
