"""Fail-closed identification and registration of an existing physical ChArUco board."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from camera_rig.artifacts.hashing import sha256_bytes, sha256_file
from camera_rig.artifacts.io import atomic_write_json, load_json
from camera_rig.capture.replay import ReplayCameraSession
from camera_rig.core.errors import ArtifactError, ContractError
from camera_rig.targets.charuco.artifact import ResolvedCharucoTargetV2
from camera_rig.targets.charuco.dependencies import cv2_module
from camera_rig.targets.charuco.geometry import (
    SUPPORTED_DICTIONARIES,
    canonical_corners_from_board,
    create_board,
)
from camera_rig.targets.io import validate_target_artifact
from camera_rig.version import __version__

IDENTIFICATION_SCHEMA_VERSION = "camera-rig.target-identification.v1"


@dataclass(frozen=True)
class ExistingBoardDimensions:
    """User-supplied nominal board measurements used only to enumerate candidates."""

    board_width_mm: float
    board_height_mm: float
    square_length_mm: float
    marker_length_mm: float

    def __post_init__(self) -> None:
        values = (
            self.board_width_mm,
            self.board_height_mm,
            self.square_length_mm,
            self.marker_length_mm,
        )
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ContractError("existing-board dimensions must be finite and positive")
        if self.marker_length_mm >= self.square_length_mm:
            raise ContractError("marker length must be smaller than square length")


@dataclass(frozen=True)
class _Candidate:
    dictionary: str
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    border_bits: int
    legacy_pattern: bool
    target_name: str = "identification_candidate"
    target_frame: str = "charuco_target"

    @property
    def board_width_m(self) -> float:
        return self.squares_x * self.square_length_m

    @property
    def board_height_m(self) -> float:
        return self.squares_y * self.square_length_m

    @property
    def charuco_corner_count(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def key(self) -> str:
        return (
            f"{self.squares_x}x{self.squares_y}:{self.dictionary}:"
            f"legacy={str(self.legacy_pattern).lower()}:border={self.border_bits}"
        )


def identify_existing_board(
    *,
    image_paths: tuple[str | Path, ...] = (),
    artifact_paths: tuple[str | Path, ...] = (),
    stream: str = "color",
    dimensions: ExistingBoardDimensions,
    output: str | Path,
    maximum_artifact_frames: int = 8,
    authoritative_source_path: str | Path | None = None,
    authoritative_dictionary: str | None = None,
    authoritative_legacy_pattern: bool | None = None,
    authoritative_border_bits: int | None = None,
    authoritative_orientation: str | None = None,
) -> dict[str, object]:
    """Scan all supported layouts and persist a complete deterministic ranking."""
    if maximum_artifact_frames < 2:
        raise ContractError("maximum_artifact_frames must be at least two")
    images, evidence, release_source_ids = _load_evidence(
        image_paths=image_paths,
        artifact_paths=artifact_paths,
        stream=stream,
        maximum_artifact_frames=maximum_artifact_frames,
    )
    if not images:
        raise ContractError("identify-existing requires at least one image or capture artifact")
    constraints = _authoritative_constraints(
        source_path=authoritative_source_path,
        dictionary=authoritative_dictionary,
        legacy_pattern=authoritative_legacy_pattern,
        border_bits=authoritative_border_bits,
        orientation=authoritative_orientation,
    )
    if authoritative_source_path is not None:
        evidence.append(
            {
                "kind": "authoritative_source",
                "sha256": sha256_file(authoritative_source_path),
            }
        )
    cv2 = cv2_module()
    candidates = _candidates(dimensions)
    ranking = [
        _evaluate_candidate(candidate, images, release_source_ids, cv2) for candidate in candidates
    ]
    ranking.sort(
        key=lambda item: (
            -_integer(item["accepted_frame_count"], "accepted_frame_count"),
            -_integer(item["minimum_charuco_corners"], "minimum_charuco_corners"),
            -_number(
                item["minimum_expected_marker_fraction"],
                "minimum_expected_marker_fraction",
            ),
            -_number(item["mean_charuco_corners"], "mean_charuco_corners"),
            str(item["candidate_key"]),
        )
    )
    vision_passing = [item for item in ranking if item["passes_unique_candidate_gate"] is True]
    release_passing = [
        item for item in vision_passing if item["passes_release_source_gate"] is True
    ]
    constrained_vision = [
        item for item in vision_passing if _matches_constraints(item, constraints)
    ]
    passing = [item for item in release_passing if _matches_constraints(item, constraints)]
    gridboard = (
        not vision_passing
        and any(
            _number(item["minimum_expected_marker_fraction"], "minimum_expected_marker_fraction")
            >= 0.80
            for item in ranking
        )
        and all(
            _integer(item["maximum_charuco_corners"], "maximum_charuco_corners") < 4
            for item in ranking
        )
    )
    unique = len(passing) == 1
    winner = dict(passing[0]) if unique else None
    if gridboard:
        classification = "ARUCO_GRIDBOARD_NOT_CHARUCO"
    elif unique:
        classification = "CHARUCO_EXISTING_PHYSICAL"
    else:
        classification = "UNRESOLVED_CHARUCO_CANDIDATE"
    report: dict[str, object] = {
        "schema_version": IDENTIFICATION_SCHEMA_VERSION,
        "status": "PASS" if unique else "PAUSED_FOR_USER_VALIDATION",
        "classification": classification,
        "candidate_uniqueness": unique,
        "identification_basis": (
            "vision-and-authoritative-source" if constraints else "vision-only"
        ),
        "authoritative_constraints": constraints,
        "ambiguity_reason": _ambiguity_reason(
            passing if passing else constrained_vision,
            len(images),
            gridboard,
            len(set(filter(None, release_source_ids))),
        ),
        "physical_measurement": {
            "nominal_width_mm": dimensions.board_width_mm,
            "nominal_height_mm": dimensions.board_height_mm,
            "square_length_mm": dimensions.square_length_mm,
            "marker_length_mm": dimensions.marker_length_mm,
            "source": "user-provided-and-vision-verified",
        },
        "evidence": evidence,
        "evidence_frame_count": len(images),
        "distinct_capture_source_count": len(set(filter(None, release_source_ids))),
        "winner": winner,
        "candidate_ranking": ranking,
        "acceptance": {
            "minimum_evidence_frames": 2,
            "minimum_distinct_capture_sources": 2,
            "minimum_charuco_corners_per_frame": 20,
            "minimum_expected_marker_fraction_per_frame": 0.80,
            "requires_unique_dictionary": True,
            "requires_unique_orientation": True,
            "requires_unique_legacy_pattern": True,
            "requires_unique_border_bits": True,
            "requires_marker_layout_consistency": True,
        },
        "software": {"camera_rig_version": __version__, "opencv_version": cv2.__version__},
    }
    atomic_write_json(output, report)
    return report


def register_existing_board(
    *,
    identification_path: str | Path,
    target_name: str,
    target_frame: str,
    output: str | Path,
) -> dict[str, object]:
    """Publish a three-file existing-target artifact without fabricating print assets."""
    value = load_json(identification_path)
    if not isinstance(value, dict) or value.get("schema_version") != IDENTIFICATION_SCHEMA_VERSION:
        raise ArtifactError("unsupported existing-board identification report")
    winner = _validated_unique_winner(value)
    physical = _object(value.get("physical_measurement"), "physical_measurement")
    evidence = value.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ArtifactError("identification evidence must be a non-empty array")
    candidate = _Candidate(
        dictionary=_string(winner.get("dictionary"), "winner.dictionary"),
        squares_x=_integer(winner.get("squares_x"), "winner.squares_x"),
        squares_y=_integer(winner.get("squares_y"), "winner.squares_y"),
        square_length_m=_number(physical.get("square_length_mm"), "square_length_mm") / 1000.0,
        marker_length_m=_number(physical.get("marker_length_mm"), "marker_length_mm") / 1000.0,
        border_bits=_integer(winner.get("border_bits"), "winner.border_bits"),
        legacy_pattern=_boolean(winner.get("legacy_pattern"), "winner.legacy_pattern"),
        target_name=target_name,
        target_frame=target_frame,
    )
    board, _dictionary, cv2 = create_board(candidate)
    marker_ids = tuple(int(value) for value in np.asarray(board.getIds()).reshape(-1))
    reported_marker_ids = winner.get("expected_marker_ids_in_layout_order")
    if not isinstance(reported_marker_ids, list) or marker_ids != tuple(reported_marker_ids):
        raise ArtifactError("identification winner marker layout differs from OpenCV candidate")
    identification_sha = sha256_file(identification_path)
    evidence_hashes = [
        _string(_object(item, "evidence[]").get("sha256"), "evidence[].sha256") for item in evidence
    ]
    resolved = ResolvedCharucoTargetV2(
        target_name=target_name,
        target_frame=target_frame,
        dictionary=candidate.dictionary,
        squares_x=candidate.squares_x,
        squares_y=candidate.squares_y,
        square_length_m=candidate.square_length_m,
        marker_length_m=candidate.marker_length_m,
        border_bits=candidate.border_bits,
        legacy_pattern=candidate.legacy_pattern,
        board_width_m=candidate.board_width_m,
        board_height_m=candidate.board_height_m,
        corner_points=canonical_corners_from_board(candidate, board),
        marker_ids=marker_ids,
        camera_rig_version=__version__,
        opencv_version=str(cv2.__version__),
        source_config_sha256="0" * 64,
        board_png_sha256="0" * 64,
        print_pdf_sha256="0" * 64,
        source_type="existing_physical",
        physical_measurement={
            "nominal_width_mm": _number(physical.get("nominal_width_mm"), "nominal_width_mm"),
            "nominal_height_mm": _number(physical.get("nominal_height_mm"), "nominal_height_mm"),
            "square_length_mm": _number(physical.get("square_length_mm"), "square_length_mm"),
            "marker_length_mm": _number(physical.get("marker_length_mm"), "marker_length_mm"),
            "source": "user-provided-and-vision-verified",
        },
        identification={
            "candidate_uniqueness": True,
            "identification_report_sha256": identification_sha,
            "evidence_hashes": evidence_hashes,
            "opencv_version": str(cv2.__version__),
            "recognition_metrics": {
                "minimum_charuco_corners": winner.get("minimum_charuco_corners"),
                "minimum_expected_marker_fraction": winner.get("minimum_expected_marker_fraction"),
                "evidence_frame_count": value.get("evidence_frame_count"),
            },
        },
        artifact_files={},
    )
    destination = Path(output)
    if destination.exists():
        raise ArtifactError(f"target artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        spec_path = temporary / "target_spec.json"
        atomic_write_json(spec_path, resolved.to_dict())
        report: dict[str, object] = {
            "schema_version": "camera-rig.target-registration.v1",
            "status": "PASS",
            "source_type": "existing_physical",
            "target_name": target_name,
            "target_spec_sha256": sha256_file(spec_path),
            "identification_report_sha256": identification_sha,
            "generated_print_assets": False,
        }
        atomic_write_json(temporary / "registration_report.json", report)
        payloads = ("registration_report.json", "target_spec.json")
        lines = [f"{sha256_file(temporary / name)}  {name}" for name in payloads]
        (temporary / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
        validate_target_artifact(spec_path)
        os.replace(temporary, destination)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_evidence(
    *,
    image_paths: tuple[str | Path, ...],
    artifact_paths: tuple[str | Path, ...],
    stream: str,
    maximum_artifact_frames: int,
) -> tuple[
    list[npt.NDArray[np.uint8]],
    list[dict[str, object]],
    list[str | None],
]:
    cv2 = cv2_module()
    images: list[npt.NDArray[np.uint8]] = []
    evidence: list[dict[str, object]] = []
    release_source_ids: list[str | None] = []
    for source in image_paths:
        image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ArtifactError(f"could not read existing-board image: {source}")
        images.append(np.asarray(image, dtype=np.uint8))
        evidence.append({"kind": "image", "sha256": sha256_file(source)})
        release_source_ids.append(None)
    for source in artifact_paths:
        session = ReplayCameraSession.from_artifact(source)
        if session.frame_count < 1:
            raise ArtifactError("capture artifact contains no evidence frames")
        indices = set(
            int(value)
            for value in np.linspace(
                0,
                session.frame_count - 1,
                min(maximum_artifact_frames, session.frame_count),
                dtype=np.int64,
            )
        )
        manifest = load_json(Path(source) / "manifest.json")
        manifest_object = _object(manifest, "capture manifest")
        camera = _string(manifest_object.get("camera"), "capture manifest camera")
        serial = _string(manifest_object.get("serial"), "capture manifest serial")
        source_identity = sha256_bytes(f"{camera}\0{serial}".encode())
        manifest_sha256 = sha256_file(Path(source) / "manifest.json")
        with session:
            index = 0
            while True:
                frame = session.poll_frame()
                if frame is None:
                    break
                if index in indices:
                    if stream not in frame.streams:
                        raise ArtifactError(f"capture frame lacks evidence stream {stream!r}")
                    raw = np.asarray(frame.streams[stream].data)
                    if raw.dtype != np.uint8:
                        raise ArtifactError("existing-board evidence stream must be uint8")
                    gray = raw.copy() if raw.ndim == 2 else cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
                    images.append(np.asarray(gray, dtype=np.uint8))
                    evidence.append(
                        {
                            "kind": "capture_frame",
                            "artifact_manifest_sha256": manifest_sha256,
                            "release_source_identity_sha256": source_identity,
                            "frame_index": index,
                            "stream": stream,
                            "sha256": sha256_bytes(np.ascontiguousarray(gray).tobytes()),
                        }
                    )
                    release_source_ids.append(source_identity)
                index += 1
    return images, evidence, release_source_ids


def _candidates(dimensions: ExistingBoardDimensions) -> tuple[_Candidate, ...]:
    width_squares = dimensions.board_width_mm / dimensions.square_length_mm
    height_squares = dimensions.board_height_mm / dimensions.square_length_mm
    rounded = (round(width_squares), round(height_squares))
    if not np.allclose((width_squares, height_squares), rounded, rtol=0.0, atol=1e-6):
        raise ContractError("board dimensions must be integer multiples of square length")
    orientations = {rounded, (rounded[1], rounded[0])}
    return tuple(
        _Candidate(
            dictionary=dictionary,
            squares_x=squares_x,
            squares_y=squares_y,
            square_length_m=dimensions.square_length_mm / 1000.0,
            marker_length_m=dimensions.marker_length_mm / 1000.0,
            border_bits=border_bits,
            legacy_pattern=legacy_pattern,
        )
        for squares_x, squares_y in sorted(orientations)
        for dictionary in SUPPORTED_DICTIONARIES
        for legacy_pattern in (False, True)
        for border_bits in (1, 2)
    )


def _evaluate_candidate(
    candidate: _Candidate,
    images: list[npt.NDArray[np.uint8]],
    release_source_ids: list[str | None],
    cv2: Any,
) -> dict[str, object]:
    board, _dictionary, _cv2 = create_board(candidate)
    parameters = cv2.aruco.DetectorParameters()
    parameters.markerBorderBits = candidate.border_bits
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), parameters)
    expected_marker_ids = tuple(int(value) for value in np.asarray(board.getIds()).reshape(-1))
    expected_set = set(expected_marker_ids)
    frames: list[dict[str, object]] = []
    for index, image in enumerate(images):
        _corners, corner_ids, _marker_corners, marker_ids = detector.detectBoard(image)
        detected_markers = (
            tuple(int(value) for value in np.asarray(marker_ids).reshape(-1))
            if marker_ids is not None
            else ()
        )
        detected_corners = (
            tuple(int(value) for value in np.asarray(corner_ids).reshape(-1))
            if corner_ids is not None
            else ()
        )
        marker_fraction = len(expected_set.intersection(detected_markers)) / len(expected_set)
        corner_layout_consistent = all(
            0 <= value < candidate.charuco_corner_count for value in detected_corners
        ) and len(set(detected_corners)) == len(detected_corners)
        accepted = (
            len(detected_corners) >= 20 and marker_fraction >= 0.80 and corner_layout_consistent
        )
        frames.append(
            {
                "evidence_index": index,
                "detected_marker_ids": list(detected_markers),
                "detected_charuco_corner_ids": list(detected_corners),
                "detected_charuco_corners": len(detected_corners),
                "expected_marker_fraction": marker_fraction,
                "marker_layout_consistent": corner_layout_consistent,
                "accepted": accepted,
            }
        )
    corner_counts = [
        _integer(item["detected_charuco_corners"], "detected_charuco_corners") for item in frames
    ]
    marker_fractions = [
        _number(item["expected_marker_fraction"], "expected_marker_fraction") for item in frames
    ]
    accepted_count = sum(item["accepted"] is True for item in frames)
    capture_sources = sorted(set(filter(None, release_source_ids)))
    per_source_consistent = all(
        all(
            frames[index]["accepted"] is True
            for index, source_id in enumerate(release_source_ids)
            if source_id == capture_source
        )
        for capture_source in capture_sources
    )
    return {
        "candidate_key": candidate.key,
        "dictionary": candidate.dictionary,
        "squares_x": candidate.squares_x,
        "squares_y": candidate.squares_y,
        "orientation": (
            "portrait" if candidate.board_height_m > candidate.board_width_m else "landscape"
        ),
        "legacy_pattern": candidate.legacy_pattern,
        "border_bits": candidate.border_bits,
        "expected_marker_ids_in_layout_order": list(expected_marker_ids),
        "accepted_frame_count": accepted_count,
        "minimum_charuco_corners": min(corner_counts),
        "maximum_charuco_corners": max(corner_counts),
        "mean_charuco_corners": float(np.mean(corner_counts)),
        "minimum_expected_marker_fraction": min(marker_fractions),
        "passes_unique_candidate_gate": len(images) >= 2 and accepted_count == len(images),
        "passes_release_source_gate": (len(capture_sources) >= 2 and per_source_consistent),
        "per_frame": frames,
    }


def _ambiguity_reason(
    passing: list[dict[str, object]],
    evidence_count: int,
    gridboard: bool,
    release_source_count: int,
) -> str | None:
    if gridboard:
        return "markers were detected without a ChArUco intersection/layout match"
    if evidence_count < 2:
        return "at least two evidence frames are required"
    if len(passing) > 1:
        fields = ("dictionary", "squares_x", "squares_y", "legacy_pattern", "border_bits")
        ambiguous = [field for field in fields if len({item[field] for item in passing}) > 1]
        return "multiple candidates remain indistinguishable: " + ", ".join(ambiguous)
    if len(passing) == 1 and release_source_count < 2:
        return (
            "preliminary candidate only; final registration requires consistent evidence "
            "from at least two distinct capture-camera identities"
        )
    if not passing:
        return "no candidate met all marker-layout and ChArUco-corner gates"
    return None


def _authoritative_constraints(
    *,
    source_path: str | Path | None,
    dictionary: str | None,
    legacy_pattern: bool | None,
    border_bits: int | None,
    orientation: str | None,
) -> dict[str, object]:
    values = {
        "dictionary": dictionary,
        "legacy_pattern": legacy_pattern,
        "border_bits": border_bits,
        "orientation": orientation,
    }
    constrained = any(value is not None for value in values.values())
    if constrained and source_path is None:
        raise ContractError("authoritative candidate constraints require --authoritative-source")
    if source_path is not None and not constrained:
        raise ContractError("--authoritative-source requires at least one candidate constraint")
    if dictionary is not None and dictionary not in SUPPORTED_DICTIONARIES:
        raise ContractError(f"unsupported authoritative dictionary: {dictionary!r}")
    if border_bits is not None and border_bits not in {1, 2}:
        raise ContractError("authoritative border bits must be 1 or 2")
    if orientation is not None and orientation not in {"portrait", "landscape"}:
        raise ContractError("authoritative orientation must be portrait or landscape")
    return {key: value for key, value in values.items() if value is not None}


def _matches_constraints(candidate: dict[str, object], constraints: dict[str, object]) -> bool:
    return all(candidate.get(key) == value for key, value in constraints.items())


def _validated_unique_winner(report: Mapping[str, object]) -> dict[str, object]:
    expected_report_fields = {
        "schema_version",
        "status",
        "classification",
        "candidate_uniqueness",
        "identification_basis",
        "authoritative_constraints",
        "ambiguity_reason",
        "physical_measurement",
        "evidence",
        "evidence_frame_count",
        "distinct_capture_source_count",
        "winner",
        "candidate_ranking",
        "acceptance",
        "software",
    }
    if set(report) != expected_report_fields:
        raise ArtifactError("identification report has missing or unknown fields")
    physical = _object(report.get("physical_measurement"), "physical_measurement")
    if (
        set(physical)
        != {
            "nominal_width_mm",
            "nominal_height_mm",
            "square_length_mm",
            "marker_length_mm",
            "source",
        }
        or physical.get("source") != "user-provided-and-vision-verified"
    ):
        raise ArtifactError("identification physical measurement contract is invalid")
    for name in (
        "nominal_width_mm",
        "nominal_height_mm",
        "square_length_mm",
        "marker_length_mm",
    ):
        if _number(physical.get(name), f"physical_measurement.{name}") <= 0:
            raise ArtifactError("identification physical measurements must be positive")
    if _number(physical["marker_length_mm"], "marker_length_mm") >= _number(
        physical["square_length_mm"], "square_length_mm"
    ):
        raise ArtifactError("identification marker length must be smaller than square length")
    evidence_value = report.get("evidence")
    if not isinstance(evidence_value, list) or not evidence_value:
        raise ArtifactError("identification evidence must be a non-empty array")
    frame_evidence: list[dict[str, object]] = []
    authoritative_count = 0
    source_ids: list[str | None] = []
    for index, item_value in enumerate(evidence_value):
        item = _object(item_value, f"evidence[{index}]")
        kind = item.get("kind")
        if kind == "image":
            if set(item) != {"kind", "sha256"}:
                raise ArtifactError("image evidence has missing or unknown fields")
            _digest_value(item.get("sha256"), f"evidence[{index}].sha256")
            frame_evidence.append(item)
            source_ids.append(None)
        elif kind == "capture_frame":
            expected = {
                "kind",
                "artifact_manifest_sha256",
                "release_source_identity_sha256",
                "frame_index",
                "stream",
                "sha256",
            }
            if set(item) != expected:
                raise ArtifactError("capture evidence has missing or unknown fields")
            _digest_value(
                item.get("artifact_manifest_sha256"),
                f"evidence[{index}].artifact_manifest_sha256",
            )
            source_id = _digest_value(
                item.get("release_source_identity_sha256"),
                f"evidence[{index}].release_source_identity_sha256",
            )
            if _integer(item.get("frame_index"), f"evidence[{index}].frame_index") < 0:
                raise ArtifactError("capture evidence frame_index must be non-negative")
            _string(item.get("stream"), f"evidence[{index}].stream")
            _digest_value(item.get("sha256"), f"evidence[{index}].sha256")
            frame_evidence.append(item)
            source_ids.append(source_id)
        elif kind == "authoritative_source":
            if set(item) != {"kind", "sha256"}:
                raise ArtifactError("authoritative evidence has missing or unknown fields")
            _digest_value(item.get("sha256"), f"evidence[{index}].sha256")
            authoritative_count += 1
        else:
            raise ArtifactError("identification evidence kind is unsupported")
    frame_count = _integer(report.get("evidence_frame_count"), "evidence_frame_count")
    if frame_count != len(frame_evidence):
        raise ArtifactError("identification evidence frame count is inconsistent")
    distinct_sources = len(set(filter(None, source_ids)))
    if (
        _integer(report.get("distinct_capture_source_count"), "distinct_capture_source_count")
        != distinct_sources
    ):
        raise ArtifactError("identification capture-source count is inconsistent")

    constraints = _object(report.get("authoritative_constraints"), "authoritative_constraints")
    if not set(constraints).issubset(
        {"dictionary", "legacy_pattern", "border_bits", "orientation"}
    ):
        raise ArtifactError("identification authoritative constraints are invalid")
    if "dictionary" in constraints and constraints["dictionary"] not in SUPPORTED_DICTIONARIES:
        raise ArtifactError("identification authoritative dictionary is invalid")
    if "legacy_pattern" in constraints:
        _boolean(constraints["legacy_pattern"], "authoritative_constraints.legacy_pattern")
    if "border_bits" in constraints and constraints["border_bits"] not in {1, 2}:
        raise ArtifactError("identification authoritative border bits are invalid")
    if "orientation" in constraints and constraints["orientation"] not in {
        "portrait",
        "landscape",
    }:
        raise ArtifactError("identification authoritative orientation is invalid")
    expected_basis = "vision-and-authoritative-source" if constraints else "vision-only"
    if report.get("identification_basis") != expected_basis:
        raise ArtifactError("identification basis is inconsistent")
    if authoritative_count != (1 if constraints else 0):
        raise ArtifactError("identification authoritative-source evidence is inconsistent")

    acceptance = _object(report.get("acceptance"), "acceptance")
    if acceptance != {
        "minimum_evidence_frames": 2,
        "minimum_distinct_capture_sources": 2,
        "minimum_charuco_corners_per_frame": 20,
        "minimum_expected_marker_fraction_per_frame": 0.80,
        "requires_unique_dictionary": True,
        "requires_unique_orientation": True,
        "requires_unique_legacy_pattern": True,
        "requires_unique_border_bits": True,
        "requires_marker_layout_consistency": True,
    }:
        raise ArtifactError("identification acceptance policy is inconsistent")
    software = _object(report.get("software"), "software")
    if set(software) != {"camera_rig_version", "opencv_version"}:
        raise ArtifactError("identification software provenance is invalid")
    _string(software.get("camera_rig_version"), "software.camera_rig_version")
    cv2 = cv2_module()
    if software.get("opencv_version") != cv2.__version__:
        raise ArtifactError("identification OpenCV version differs from registration runtime")

    ranking_value = report.get("candidate_ranking")
    if not isinstance(ranking_value, list) or not ranking_value:
        raise ArtifactError("identification candidate_ranking must be a non-empty array")
    ranking = [
        _validated_candidate_entry(
            _object(item, f"candidate_ranking[{index}]"),
            frame_count=frame_count,
            source_ids=source_ids,
        )
        for index, item in enumerate(ranking_value)
    ]
    square_length_mm = _number(physical["square_length_mm"], "square_length_mm")
    nominal_sides = sorted(
        (
            _number(physical["nominal_width_mm"], "nominal_width_mm"),
            _number(physical["nominal_height_mm"], "nominal_height_mm"),
        )
    )
    for item in ranking:
        candidate_sides = sorted(
            (
                _integer(item["squares_x"], "squares_x") * square_length_mm,
                _integer(item["squares_y"], "squares_y") * square_length_mm,
            )
        )
        if not np.allclose(candidate_sides, nominal_sides, rtol=0.0, atol=1e-6):
            raise ArtifactError("identification candidate geometry differs from measurements")
    keys = [item["candidate_key"] for item in ranking]
    if len(set(keys)) != len(keys):
        raise ArtifactError("identification candidate keys must be unique")
    dimensions = ExistingBoardDimensions(
        board_width_mm=_number(physical["nominal_width_mm"], "nominal_width_mm"),
        board_height_mm=_number(physical["nominal_height_mm"], "nominal_height_mm"),
        square_length_mm=_number(physical["square_length_mm"], "square_length_mm"),
        marker_length_mm=_number(physical["marker_length_mm"], "marker_length_mm"),
    )
    expected_candidates = {candidate.key: candidate for candidate in _candidates(dimensions)}
    if set(keys) != set(expected_candidates):
        raise ArtifactError("identification ranking does not cover the complete candidate universe")
    for item in ranking:
        candidate = expected_candidates[str(item["candidate_key"])]
        if any(
            item[name] != getattr(candidate, name)
            for name in (
                "dictionary",
                "squares_x",
                "squares_y",
                "legacy_pattern",
                "border_bits",
            )
        ):
            raise ArtifactError("identification candidate fields differ from enumerated candidate")
        board, _dictionary, _cv2 = create_board(candidate)
        expected_marker_ids = [int(value) for value in np.asarray(board.getIds()).reshape(-1)]
        if item["expected_marker_ids_in_layout_order"] != expected_marker_ids:
            raise ArtifactError("identification candidate marker layout is inconsistent")
    expected_order = sorted(
        ranking,
        key=lambda item: (
            -_integer(item["accepted_frame_count"], "accepted_frame_count"),
            -_integer(item["minimum_charuco_corners"], "minimum_charuco_corners"),
            -_number(item["minimum_expected_marker_fraction"], "marker_fraction"),
            -_number(item["mean_charuco_corners"], "mean_charuco_corners"),
            str(item["candidate_key"]),
        ),
    )
    if ranking != expected_order:
        raise ArtifactError("identification candidate ranking order is inconsistent")
    vision_passing = [item for item in ranking if item["passes_unique_candidate_gate"] is True]
    release_passing = [
        item for item in vision_passing if item["passes_release_source_gate"] is True
    ]
    constrained_vision = [
        item for item in vision_passing if _matches_constraints(item, constraints)
    ]
    passing = [item for item in release_passing if _matches_constraints(item, constraints)]
    gridboard = (
        not vision_passing
        and any(
            _number(item["minimum_expected_marker_fraction"], "marker_fraction") >= 0.80
            for item in ranking
        )
        and all(
            _integer(item["maximum_charuco_corners"], "maximum_charuco_corners") < 4
            for item in ranking
        )
    )
    unique = len(passing) == 1
    expected_classification = (
        "ARUCO_GRIDBOARD_NOT_CHARUCO"
        if gridboard
        else "CHARUCO_EXISTING_PHYSICAL"
        if unique
        else "UNRESOLVED_CHARUCO_CANDIDATE"
    )
    if (
        report.get("status") != ("PASS" if unique else "PAUSED_FOR_USER_VALIDATION")
        or report.get("classification") != expected_classification
        or report.get("candidate_uniqueness") is not unique
    ):
        raise ArtifactError("identification conclusion is inconsistent with candidate evidence")
    expected_winner = dict(passing[0]) if unique else None
    if report.get("winner") != expected_winner:
        raise ArtifactError("identification winner is inconsistent with candidate ranking")
    if report.get("ambiguity_reason") != _ambiguity_reason(
        passing if passing else constrained_vision,
        frame_count,
        gridboard,
        distinct_sources,
    ):
        raise ArtifactError("identification ambiguity reason is inconsistent")
    if not unique or expected_winner is None:
        raise ArtifactError("identification is not uniquely accepted; registration is forbidden")
    return expected_winner


def _validated_candidate_entry(
    item: dict[str, object], *, frame_count: int, source_ids: list[str | None]
) -> dict[str, object]:
    expected_fields = {
        "candidate_key",
        "dictionary",
        "squares_x",
        "squares_y",
        "orientation",
        "legacy_pattern",
        "border_bits",
        "expected_marker_ids_in_layout_order",
        "accepted_frame_count",
        "minimum_charuco_corners",
        "maximum_charuco_corners",
        "mean_charuco_corners",
        "minimum_expected_marker_fraction",
        "passes_unique_candidate_gate",
        "passes_release_source_gate",
        "per_frame",
    }
    if set(item) != expected_fields:
        raise ArtifactError("candidate ranking entry has missing or unknown fields")
    per_frame_value = item.get("per_frame")
    if not isinstance(per_frame_value, list) or len(per_frame_value) != frame_count:
        raise ArtifactError("candidate per_frame evidence count is inconsistent")
    accepted: list[bool] = []
    corners: list[int] = []
    fractions: list[float] = []
    for index, frame_value in enumerate(per_frame_value):
        frame = _object(frame_value, f"candidate.per_frame[{index}]")
        if set(frame) != {
            "evidence_index",
            "detected_marker_ids",
            "detected_charuco_corner_ids",
            "detected_charuco_corners",
            "expected_marker_fraction",
            "marker_layout_consistent",
            "accepted",
        }:
            raise ArtifactError("candidate frame has missing or unknown fields")
        if _integer(frame.get("evidence_index"), "evidence_index") != index:
            raise ArtifactError("candidate evidence indices are inconsistent")
        marker_ids = frame.get("detected_marker_ids")
        corner_ids = frame.get("detected_charuco_corner_ids")
        if not isinstance(marker_ids, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in marker_ids
        ):
            raise ArtifactError("candidate detected marker IDs are invalid")
        if not isinstance(corner_ids, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in corner_ids
        ):
            raise ArtifactError("candidate detected corner IDs are invalid")
        corner_count = _integer(frame.get("detected_charuco_corners"), "corner count")
        if corner_count != len(corner_ids):
            raise ArtifactError("candidate corner count is inconsistent")
        fraction = _number(frame.get("expected_marker_fraction"), "marker fraction")
        if not 0.0 <= fraction <= 1.0:
            raise ArtifactError("candidate marker fraction lies outside [0, 1]")
        layout = _boolean(frame.get("marker_layout_consistent"), "marker layout")
        expected_accepted = corner_count >= 20 and fraction >= 0.80 and layout
        if frame.get("accepted") is not expected_accepted:
            raise ArtifactError("candidate frame acceptance is inconsistent")
        accepted.append(expected_accepted)
        corners.append(corner_count)
        fractions.append(fraction)
    accepted_count = sum(accepted)
    calculated = {
        "accepted_frame_count": accepted_count,
        "minimum_charuco_corners": min(corners),
        "maximum_charuco_corners": max(corners),
        "mean_charuco_corners": float(np.mean(corners)),
        "minimum_expected_marker_fraction": min(fractions),
        "passes_unique_candidate_gate": frame_count >= 2 and accepted_count == frame_count,
        "passes_release_source_gate": len(set(filter(None, source_ids))) >= 2
        and all(
            all(accepted[index] for index, value in enumerate(source_ids) if value == source_id)
            for source_id in set(filter(None, source_ids))
        ),
    }
    for name, expected in calculated.items():
        if item.get(name) != expected:
            raise ArtifactError(f"candidate aggregate {name} is inconsistent")
    dictionary = _string(item.get("dictionary"), "candidate.dictionary")
    if dictionary not in SUPPORTED_DICTIONARIES:
        raise ArtifactError("candidate dictionary is unsupported")
    squares_x = _integer(item.get("squares_x"), "candidate.squares_x")
    squares_y = _integer(item.get("squares_y"), "candidate.squares_y")
    legacy = _boolean(item.get("legacy_pattern"), "candidate.legacy_pattern")
    border_bits = _integer(item.get("border_bits"), "candidate.border_bits")
    if border_bits not in {1, 2}:
        raise ArtifactError("candidate border bits are invalid")
    expected_key = (
        f"{squares_x}x{squares_y}:{dictionary}:legacy={str(legacy).lower()}:border={border_bits}"
    )
    if item.get("candidate_key") != expected_key:
        raise ArtifactError("candidate key is inconsistent")
    orientation = "portrait" if squares_y > squares_x else "landscape"
    if item.get("orientation") != orientation:
        raise ArtifactError("candidate orientation is inconsistent")
    expected_ids = item.get("expected_marker_ids_in_layout_order")
    if not isinstance(expected_ids, list) or len(expected_ids) != squares_x * squares_y // 2:
        raise ArtifactError("candidate expected marker layout is invalid")
    if len(set(expected_ids)) != len(expected_ids) or not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in expected_ids
    ):
        raise ArtifactError("candidate expected marker IDs are invalid")
    return item


def _digest_value(value: object, name: str) -> str:
    text = _string(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ArtifactError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ArtifactError(f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(value):
        raise ArtifactError(f"{name} must be a finite number")
    return float(value)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ArtifactError(f"{name} must be a boolean")
    return value
