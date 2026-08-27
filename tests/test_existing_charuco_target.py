# ruff: noqa: E402

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

cv2 = pytest.importorskip("cv2")
pytest.importorskip("reportlab")

from camera_rig.artifacts.factory_calibration import FactoryCalibrationArtifact
from camera_rig.capture.snapshot import write_snapshot
from camera_rig.config.loader import load_config
from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.errors import ArtifactError, ContractError
from camera_rig.core.factory_calibration import FactoryCalibration
from camera_rig.core.frame import CameraFrame, StreamFrame
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.stream import StreamProfile
from camera_rig.core.timestamps import SingleDeviceSyncReport
from camera_rig.core.transforms import RigidTransform
from camera_rig.targets.charuco.detector import CharucoDetector
from camera_rig.targets.charuco.generator import generate_target_artifact
from camera_rig.targets.charuco.geometry import create_board
from camera_rig.targets.charuco.identification import (
    ExistingBoardDimensions,
    _Candidate,
    _candidate_equivalence_proof,
    _spatial_marker_layout,
    identify_existing_board,
    register_existing_board,
)
from camera_rig.targets.charuco.quality import CharucoQualityThresholds
from camera_rig.targets.charuco.spec import load_charuco_target_spec
from camera_rig.targets.io import load_target, validate_target_artifact
from camera_rig.targets.preflight import run_target_preflight
from camera_rig.targets.validation import _acceptance

pytestmark = pytest.mark.charuco


def _write_identification_capture(
    root: Path, image: np.ndarray, *, camera_name: str, serial: str
) -> Path:
    height, width = image.shape
    names = ("color", "depth", "ir_left", "ir_right")
    profiles = {
        name: StreamProfile(
            name,
            width,
            height,
            30,
            {"color": "rgb8", "depth": "z16"}.get(name, "y8"),
            {"color": 0, "depth": 0, "ir_left": 1, "ir_right": 2}[name],
        )
        for name in names
    }
    intrinsics = {
        name: CameraIntrinsics(
            f"{camera_name}/{name}_optical",
            width,
            height,
            800.0,
            800.0,
            width / 2.0,
            height / 2.0,
            "none",
        )
        for name in names
    }
    transforms = tuple(
        RigidTransform(
            f"{camera_name}/ir_left_optical",
            f"{camera_name}/{name}_optical",
            np.eye(4, dtype=np.float64),
        )
        for name in ("color", "depth", "ir_right")
    )
    calibration = FactoryCalibration(
        CameraDeviceInfo("synthetic", camera_name, "synthetic", "synthetic", serial),
        profiles,
        intrinsics,
        transforms,
        0.001,
    )
    factory = FactoryCalibrationArtifact(
        datetime.now(timezone.utc).isoformat(),
        calibration,
        QualityReport(True),
        {"source": "unit-test"},
    )
    frame = CameraFrame(
        camera_name=camera_name,
        serial=serial,
        streams={
            "color": StreamFrame("color", np.repeat(image[:, :, None], 3, axis=2), 0),
            "depth": StreamFrame("depth", np.zeros((height, width), dtype=np.uint16), 0),
            "ir_left": StreamFrame("ir_left", np.zeros_like(image), 0),
            "ir_right": StreamFrame("ir_right", np.zeros_like(image), 0),
        },
        host_receive_timestamp_ns=0,
        sync_report=SingleDeviceSyncReport(
            valid=True,
            comparable_streams=names,
            max_skew_ns=0,
            per_stream_skew_ns={name: 0 for name in names},
            frame_number_match=True,
        ),
    )
    write_snapshot(
        root,
        [frame],
        factory,
        {"copy_frames": True},
        {"source": "unit-test"},
        include_previews=False,
    )
    return root


def _v2_config(
    path: Path,
    *,
    name: str = "custom_even_board",
    squares_x: int = 6,
    squares_y: int = 4,
    dictionary: str = "DICT_5X5_100",
) -> Path:
    width_mm = squares_x * 30.0
    height_mm = squares_y * 30.0
    value = {
        "schema_version": "camera-rig.target.charuco.v2",
        "target": {
            "name": name,
            "plugin": "charuco",
            "target_frame": "charuco_target",
            "dictionary": dictionary,
            "squares_x": squares_x,
            "squares_y": squares_y,
            "square_length_m": 0.030,
            "marker_length_m": 0.022,
            "border_bits": 1,
            "legacy_pattern": False,
        },
        "coordinate_frame": {
            "origin": "outer_bottom_left",
            "x_axis": "board_right",
            "y_axis": "board_up",
            "z_axis": "out_of_printed_face",
        },
        "print": {
            "mode": "generated",
            "page": {"type": "custom", "width_mm": width_mm, "height_mm": height_mm},
            "layout": {"board_x_mm": 0.0, "board_y_mm": 0.0},
            "dpi": 150,
            "board_only": True,
            "scale_check": {
                "enabled": True,
                "separate_page": True,
                "horizontal_length_mm": 100.0,
                "vertical_length_mm": 100.0,
            },
        },
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_v2_custom_page_generates_and_validates_without_changing_v1(tmp_path: Path) -> None:
    spec = load_charuco_target_spec(_v2_config(tmp_path / "target.yaml"))
    assert spec.schema_version == "camera-rig.target.charuco.v2"
    assert spec.page_width_mm == pytest.approx(180.0)
    assert spec.page_height_mm == pytest.approx(120.0)
    output = tmp_path / "target"
    generate_target_artifact(spec, output)
    target = validate_target_artifact(output / "target_spec.json")
    assert target.SCHEMA_VERSION == "camera-rig.target.charuco-resolved.v2"
    assert target.source_type == "generated"
    assert {path.name for path in output.iterdir()} == {
        "checksums.sha256",
        "custom_even_board_board.png",
        "custom_even_board_preview.png",
        "custom_even_board_print.pdf",
        "custom_even_board_scale_check.pdf",
        "generation_report.json",
        "target_spec.json",
    }


def test_odd_row_legacy_candidates_fold_by_proven_equivalence_and_register(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "generated"
    generate_target_artifact(
        load_charuco_target_spec(
            _v2_config(
                tmp_path / "target.yaml",
                squares_x=5,
                squares_y=7,
                dictionary="DICT_4X4_100",
            )
        ),
        generated,
    )
    board = cv2.imread(str(generated / "custom_even_board_board.png"), cv2.IMREAD_GRAYSCALE)
    assert board is not None
    captures = (
        _write_identification_capture(
            tmp_path / "capture_a", board, camera_name="camera_a", serial="source-a"
        ),
        _write_identification_capture(
            tmp_path / "capture_b", board, camera_name="camera_b", serial="source-b"
        ),
    )
    identification = tmp_path / "identification.json"
    report = identify_existing_board(
        artifact_paths=captures,
        dimensions=ExistingBoardDimensions(210.0, 150.0, 30.0, 22.0),
        output=identification,
        authoritative_dictionary="DICT_4X4_100",
    )
    assert report["status"] == "PASS"
    assert report["candidate_uniqueness"] is True
    assert report["identification_basis"] == "vision-and-authoritative-user-metadata"
    assert report["equivalent_candidate_fields"] == ["legacy_pattern"]
    equivalence_classes = report["equivalence_classes"]
    assert isinstance(equivalence_classes, list) and len(equivalence_classes) == 1
    equivalence_class = equivalence_classes[0]
    assert len(equivalence_class["candidate_keys"]) == 2
    assert equivalence_class["equivalent_candidate_fields"] == ["legacy_pattern"]
    proof = equivalence_class["pairwise_proofs"][0]
    assert proof["equivalent"] is True
    assert all(proof["checks"].values())
    resolved = report["resolved_identity"]
    assert resolved["dictionary"] == "DICT_4X4_100"
    assert resolved["orientation"] == "portrait"
    assert resolved["canonical_legacy_pattern"] is False
    registered = tmp_path / "registered"
    register_existing_board(
        identification_path=identification,
        target_name="future_existing_board",
        target_frame="charuco_target",
        output=registered,
    )
    target = validate_target_artifact(registered / "target_spec.json")
    assert target.dictionary == "DICT_4X4_100"
    assert target.legacy_pattern is False


def test_non_equivalent_legacy_candidates_do_not_fold() -> None:
    first = _Candidate("DICT_4X4_100", 6, 6, 0.1, 0.075, 1, False)
    second = _Candidate("DICT_4X4_100", 6, 6, 0.1, 0.075, 1, True)
    proof = _candidate_equivalence_proof(first, second)
    assert proof["equivalent"] is False
    assert proof["checks"]["marker_corner_geometry_equal"] is False
    assert proof["checks"]["generated_binary_board_image_equal"] is False


def test_wrong_authoritative_dictionary_reports_visual_conflict(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generate_target_artifact(
        load_charuco_target_spec(
            _v2_config(
                tmp_path / "target.yaml",
                squares_x=5,
                squares_y=7,
                dictionary="DICT_5X5_100",
            )
        ),
        generated,
    )
    board = cv2.imread(str(generated / "custom_even_board_board.png"), cv2.IMREAD_GRAYSCALE)
    assert board is not None
    report = identify_existing_board(
        artifact_paths=(
            _write_identification_capture(
                tmp_path / "capture_a", board, camera_name="camera_a", serial="source-a"
            ),
            _write_identification_capture(
                tmp_path / "capture_b", board, camera_name="camera_b", serial="source-b"
            ),
        ),
        dimensions=ExistingBoardDimensions(210.0, 150.0, 30.0, 22.0),
        output=tmp_path / "identification.json",
        authoritative_dictionary="DICT_4X4_100",
    )
    assert report["status"] == "FAIL"
    assert (
        report["classification"]
        == "USER_AUTHORITATIVE_DICTIONARY_CONFLICTS_WITH_VISUAL_EVIDENCE"
    )
    assert report["winner"] is None


def test_spatial_marker_id_layout_rejects_permuted_ids() -> None:
    candidate = _Candidate("DICT_4X4_100", 5, 7, 0.1, 0.075, 1, False)
    board, _dictionary, _cv2 = create_board(candidate)
    marker_corners = np.asarray(board.getObjPoints(), dtype=np.float64)[:, :, :2] * 500.0
    marker_corners += np.array([200.0, 100.0])
    marker_ids = np.asarray(board.getIds(), dtype=np.int32).reshape(-1, 1)
    valid = _spatial_marker_layout(
        board=board,
        marker_corners=marker_corners,
        marker_ids=marker_ids,
        image_shape=(1000, 1000),
        cv2=cv2,
    )
    assert valid["consistent"] is True
    invalid = _spatial_marker_layout(
        board=board,
        marker_corners=marker_corners,
        marker_ids=np.roll(marker_ids, 1, axis=0),
        image_shape=(1000, 1000),
        cv2=cv2,
    )
    assert invalid["consistent"] is False


def test_even_row_board_is_uniquely_identified_and_registered(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    spec = load_charuco_target_spec(
        _v2_config(
            tmp_path / "target.yaml",
            squares_x=6,
            squares_y=6,
            dictionary="DICT_ARUCO_ORIGINAL",
        )
    )
    generate_target_artifact(spec, generated)
    board = generated / "custom_even_board_board.png"
    image = cv2.imread(str(board), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    captures = (
        _write_identification_capture(
            tmp_path / "capture_a", image, camera_name="camera_a", serial="source-a"
        ),
        _write_identification_capture(
            tmp_path / "capture_b", image, camera_name="camera_b", serial="source-b"
        ),
    )
    identification = tmp_path / "identification.json"
    report = identify_existing_board(
        artifact_paths=captures,
        dimensions=ExistingBoardDimensions(180.0, 180.0, 30.0, 22.0),
        output=identification,
    )
    assert report["status"] == "PASS"
    winner = report["winner"]
    assert isinstance(winner, dict)
    assert winner["legacy_pattern"] is False
    registered = tmp_path / "registered"
    register_existing_board(
        identification_path=identification,
        target_name="registered_existing",
        target_frame="charuco_target",
        output=registered,
    )
    assert {path.name for path in registered.iterdir()} == {
        "checksums.sha256",
        "registration_report.json",
        "target_spec.json",
    }
    target = validate_target_artifact(registered / "target_spec.json")
    assert target.source_type == "existing_physical"
    assert target.artifact_files == {}
    assert (
        "print_pdf"
        not in json.loads((registered / "target_spec.json").read_text(encoding="utf-8"))["files"]
    )


def test_authoritative_source_can_resolve_a_vision_indistinguishable_family(
    tmp_path: Path,
) -> None:
    config_path = _v2_config(
        tmp_path / "target.yaml", squares_x=6, squares_y=6, dictionary="DICT_5X5_100"
    )
    generated = tmp_path / "generated"
    generate_target_artifact(load_charuco_target_spec(config_path), generated)
    board = generated / "custom_even_board_board.png"
    image = cv2.imread(str(board), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    captures = (
        _write_identification_capture(
            tmp_path / "capture_a", image, camera_name="camera_a", serial="source-a"
        ),
        _write_identification_capture(
            tmp_path / "capture_b", image, camera_name="camera_b", serial="source-b"
        ),
    )
    report = identify_existing_board(
        artifact_paths=captures,
        dimensions=ExistingBoardDimensions(180.0, 180.0, 30.0, 22.0),
        output=tmp_path / "identification.json",
        authoritative_source_path=config_path,
        authoritative_dictionary="DICT_5X5_100",
        authoritative_legacy_pattern=False,
        authoritative_border_bits=1,
        authoritative_orientation="landscape",
    )
    assert report["status"] == "PASS"
    assert report["identification_basis"] == "vision-and-authoritative-source"
    assert report["authoritative_constraints"]["dictionary"] == "DICT_5X5_100"
    assert all("target.yaml" not in str(item) for item in report["evidence"])


def test_release_identification_requires_two_distinct_camera_identities(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    spec = load_charuco_target_spec(
        _v2_config(
            tmp_path / "target.yaml",
            squares_x=6,
            squares_y=6,
            dictionary="DICT_ARUCO_ORIGINAL",
        )
    )
    generate_target_artifact(spec, generated)
    image = cv2.imread(str(generated / "custom_even_board_board.png"), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    report = identify_existing_board(
        artifact_paths=(
            _write_identification_capture(
                tmp_path / "capture_a", image, camera_name="camera", serial="same-source"
            ),
            _write_identification_capture(
                tmp_path / "capture_b", image, camera_name="camera", serial="same-source"
            ),
        ),
        dimensions=ExistingBoardDimensions(180.0, 180.0, 30.0, 22.0),
        output=tmp_path / "identification.json",
    )
    assert report["status"] == "PAUSED_FOR_USER_VALIDATION"
    assert report["distinct_capture_source_count"] == 1
    assert "two distinct" in str(report["ambiguity_reason"])


def test_registration_rejects_forged_identification_conclusion(
    generated_charuco_target: Path, tmp_path: Path
) -> None:
    board = generated_charuco_target / "charuco_a4_v1_board.png"
    identification = tmp_path / "identification.json"
    report = identify_existing_board(
        image_paths=(board, board),
        dimensions=ExistingBoardDimensions(210.0, 150.0, 30.0, 22.0),
        output=identification,
    )
    report["status"] = "PASS"
    report["classification"] = "CHARUCO_EXISTING_PHYSICAL"
    report["candidate_uniqueness"] = True
    report["winner"] = report["candidate_ranking"][0]
    identification.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ArtifactError, match="inconsistent"):
        register_existing_board(
            identification_path=identification,
            target_name="forged",
            target_frame="charuco_target",
            output=tmp_path / "registered",
        )


def test_registration_rejects_deleted_ambiguous_candidates(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generate_target_artifact(
        load_charuco_target_spec(
            _v2_config(
                tmp_path / "target.yaml",
                squares_x=6,
                squares_y=6,
                dictionary="DICT_5X5_100",
            )
        ),
        generated,
    )
    image = cv2.imread(str(generated / "custom_even_board_board.png"), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    identification = tmp_path / "identification.json"
    report = identify_existing_board(
        artifact_paths=(
            _write_identification_capture(
                tmp_path / "capture_a", image, camera_name="camera_a", serial="source-a"
            ),
            _write_identification_capture(
                tmp_path / "capture_b", image, camera_name="camera_b", serial="source-b"
            ),
        ),
        dimensions=ExistingBoardDimensions(180.0, 180.0, 30.0, 22.0),
        output=identification,
    )
    ranking = report["candidate_ranking"]
    assert isinstance(ranking, list)
    passing = [item for item in ranking if item["passes_release_source_gate"]]
    assert len(passing) > 1
    winner = passing[0]
    report["candidate_ranking"] = [
        item for item in ranking if not item["passes_release_source_gate"]
    ] + [winner]
    report["candidate_ranking"].sort(
        key=lambda item: (
            -int(item["accepted_frame_count"]),
            -int(item["minimum_charuco_corners"]),
            -float(item["minimum_expected_marker_fraction"]),
            -float(item["mean_charuco_corners"]),
            str(item["candidate_key"]),
        )
    )
    report["status"] = "PASS"
    report["classification"] = "CHARUCO_EXISTING_PHYSICAL"
    report["candidate_uniqueness"] = True
    report["winner"] = winner
    report["ambiguity_reason"] = None
    identification.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ArtifactError, match="complete candidate universe"):
        register_existing_board(
            identification_path=identification,
            target_name="forged-deletion",
            target_frame="charuco_target",
            output=tmp_path / "registered",
        )


def test_runtime_board_preserves_persisted_marker_layout_order(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generate_target_artifact(
        load_charuco_target_spec(_v2_config(tmp_path / "target.yaml", squares_x=6, squares_y=6)),
        generated,
    )
    target = load_target(generated / "target_spec.json")
    permuted = replace(target, marker_ids=tuple(reversed(target.marker_ids)))
    board, _dictionary, _cv2 = create_board(permuted)
    assert tuple(int(value) for value in np.asarray(board.getIds()).reshape(-1)) == tuple(
        reversed(target.marker_ids)
    )


def test_partial_charuco_observation_is_not_misclassified_as_gridboard(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "generated"
    generate_target_artifact(
        load_charuco_target_spec(_v2_config(tmp_path / "target.yaml", squares_x=6, squares_y=6)),
        generated,
    )
    image = cv2.imread(str(generated / "custom_even_board_board.png"), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    image[:, image.shape[1] // 2 :] = 255
    partial = tmp_path / "partial.png"
    assert cv2.imwrite(str(partial), image)
    report = identify_existing_board(
        image_paths=(partial, partial),
        dimensions=ExistingBoardDimensions(180.0, 180.0, 30.0, 22.0),
        output=tmp_path / "identification.json",
    )
    assert report["classification"] == "UNRESOLVED_CHARUCO_CANDIDATE"
    assert max(int(item["maximum_charuco_corners"]) for item in report["candidate_ranking"]) >= 4


def test_v2_semantic_tamper_is_rejected_before_companion_validation(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generate_target_artifact(
        load_charuco_target_spec(_v2_config(tmp_path / "target.yaml")), generated
    )
    value = json.loads((generated / "target_spec.json").read_text(encoding="utf-8"))
    value["physical_measurement"]["unknown"] = 1
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ArtifactError, match="physical measurement"):
        load_target(tampered)


def test_release_preflight_rejects_short_capture_before_hardware(tmp_path: Path) -> None:
    config = load_config(Path(__file__).parents[1] / "configs/examples/single_camera_contract.yaml")
    with pytest.raises(ContractError, match="exactly 60"):
        run_target_preflight(
            camera_config=config,
            target_path=tmp_path / "unused.json",
            frames=3,
            stream="color",
            policy="pose_validated",
            report_path=tmp_path / "preflight.json",
            overlays_path=tmp_path / "overlays",
        )


def test_pose_validated_retains_low_coverage_warning(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    spec = load_charuco_target_spec(_v2_config(tmp_path / "target.yaml", squares_x=6, squares_y=6))
    generate_target_artifact(spec, generated)
    target = load_target(generated / "target_spec.json")
    board = cv2.imread(str(generated / "custom_even_board_board.png"), cv2.IMREAD_GRAYSCALE)
    assert board is not None
    small = cv2.resize(board, (240, 240), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((960, 1280), 255, dtype=np.uint8)
    canvas[360:600, 520:760] = small
    legacy = CharucoDetector(target).detect(canvas)
    pose_validated = CharucoDetector(
        target, thresholds=CharucoQualityThresholds.pose_validated()
    ).detect(canvas)
    assert legacy.quality.metrics["coverage_ratio"] < 0.05
    assert not legacy.quality.passed
    assert pose_validated.quality.passed
    assert pose_validated.quality.warnings == (
        "board coverage below recommended deployment coverage",
    )
    assert pose_validated.quality.metrics["image_span_x_ratio"] >= 0.10
    assert pose_validated.quality.metrics["image_span_y_ratio"] >= 0.10


def test_pose_validated_capture_acceptance_keeps_five_percent_check_visible() -> None:
    statistics = {"minimum": 0.02, "median": 0.03, "maximum": 0.04, "mean": 0.03}
    aggregate = {
        "success_ratio": 1.0,
        "detected_charuco_corner_count": {
            "minimum": 20.0,
            "median": 24.0,
            "maximum": 25.0,
            "mean": 24.0,
        },
        "corner_fraction": {
            "minimum": 0.8,
            "median": 0.96,
            "maximum": 1.0,
            "mean": 0.95,
        },
        "coverage_ratio": statistics,
        "temporal_jitter": {
            "median_radial_std_px": 0.1,
            "p95_radial_std_px": 0.2,
        },
    }
    legacy = _acceptance(aggregate, 60)
    pose = _acceptance(aggregate, 60, policy="pose_validated")
    assert legacy["passed"] is False
    assert pose["passed"] is True
    assert pose["checks"]["median_coverage_at_least_threshold"] is True
    assert pose["recommendations"]["median_coverage_at_least_0_05"] is False
    assert pose["thresholds"]["median_coverage_ratio"] == pytest.approx(0.01)


def test_pose_validated_aggregate_uses_declared_corner_and_fraction_gates() -> None:
    aggregate = {
        "success_ratio": 1.0,
        "detected_charuco_corner_count": {
            "minimum": 12.0,
            "median": 12.0,
            "maximum": 12.0,
            "mean": 12.0,
        },
        "corner_fraction": {
            "minimum": 0.50,
            "median": 0.50,
            "maximum": 0.50,
            "mean": 0.50,
        },
        "coverage_ratio": {
            "minimum": 0.01,
            "median": 0.01,
            "maximum": 0.01,
            "mean": 0.01,
        },
        "temporal_jitter": {
            "median_radial_std_px": 0.1,
            "p95_radial_std_px": 0.2,
        },
    }
    pose = _acceptance(aggregate, 60, policy="pose_validated")
    assert pose["passed"] is True
    assert pose["thresholds"]["median_charuco_corners"] == 12.0
    assert pose["thresholds"]["median_corner_fraction"] == 0.50


class _FakeSession:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.index = 0

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def capture(self) -> CameraFrame:
        index = self.index
        self.index += 1
        return CameraFrame(
            camera_name="synthetic_camera",
            serial="synthetic",
            streams={
                "color": StreamFrame(
                    stream_name="color",
                    data=np.repeat(self.image[:, :, None], 3, axis=2),
                    frame_number=index,
                )
            },
            host_receive_timestamp_ns=index,
        )


def test_pose_free_target_preflight_writes_metrics_and_overlays(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    spec = load_charuco_target_spec(_v2_config(tmp_path / "target.yaml", squares_x=6, squares_y=6))
    generate_target_artifact(spec, generated)
    board = cv2.imread(str(generated / "custom_even_board_board.png"), cv2.IMREAD_GRAYSCALE)
    assert board is not None
    resized = cv2.resize(board, (300, 300), interpolation=cv2.INTER_NEAREST)
    image = np.full((480, 640), 255, dtype=np.uint8)
    image[90:390, 170:470] = resized
    config = load_config(Path(__file__).parents[1] / "configs/examples/single_camera_contract.yaml")
    report = run_target_preflight(
        camera_config=config,
        target_path=generated / "target_spec.json",
        frames=3,
        stream="color",
        policy="pose_validated",
        report_path=tmp_path / "preflight.json",
        overlays_path=tmp_path / "overlays",
        session_factory=lambda _config: _FakeSession(image),
    )
    assert report["status"] == "PASS"
    assert report["recommendation"] == "ADEQUATE"
    assert report["notice"].startswith("pose-free")
    assert set(path.name for path in (tmp_path / "overlays").iterdir()) == {
        "first_frame_000000.png",
        "middle_frame_000001.png",
        "last_frame_000002.png",
    }
