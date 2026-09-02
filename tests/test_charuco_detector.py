from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from camera_rig.core.errors import ContractError
from camera_rig.targets.charuco.detector import CharucoDetector
from camera_rig.targets.io import load_target
from camera_rig.targets.observation import TargetObservation
from camera_rig.targets.registry import registry

cv2 = pytest.importorskip("cv2")
pytestmark = pytest.mark.charuco


def _detector_and_canvas(target_root: Path) -> tuple[CharucoDetector, np.ndarray]:
    target = load_target(target_root / "target_spec.json")
    detector = CharucoDetector(target)
    board = cv2.imread(str(target_root / "charuco_a4_v1_board.png"), cv2.IMREAD_GRAYSCALE)
    assert board is not None
    resized = cv2.resize(board, (900, 643), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((960, 1280), 255, dtype=np.uint8)
    canvas[150:793, 190:1090] = resized
    return detector, canvas


def test_clean_board_detects_all_canonical_corners(generated_charuco_target: Path) -> None:
    detector, image = _detector_and_canvas(generated_charuco_target)
    observation = detector.detect(image)
    assert observation.point_ids == tuple(range(24))
    assert observation.image_points_px.shape == (24, 2)
    assert observation.object_points_m.shape == (24, 3)
    assert observation.quality.passed
    assert observation.quality.metrics["detected_marker_count"] == 17
    assert observation.quality.metrics["corner_fraction"] == 1.0
    assert observation.quality.metrics["coverage_ratio"] > 0.20


def test_rgb_input_is_explicitly_supported(generated_charuco_target: Path) -> None:
    detector, gray = _detector_and_canvas(generated_charuco_target)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    assert detector.detect(rgb).point_ids == detector.detect(gray).point_ids


@pytest.mark.parametrize(
    "variant",
    [
        "scaled",
        "translated",
        "rotated",
        "homography",
        "dim",
        "low_contrast",
        "blur",
        "noise",
        "combined",
    ],
)
def test_synthetic_perturbations_retain_main_corners(
    generated_charuco_target: Path, variant: str
) -> None:
    detector, image = _detector_and_canvas(generated_charuco_target)
    rng = np.random.default_rng(20260825)
    if variant == "scaled":
        smaller = cv2.resize(image[150:793, 190:1090], (720, 514), interpolation=cv2.INTER_AREA)
        image = np.full_like(image, 255)
        image[220:734, 280:1000] = smaller
    elif variant == "translated":
        matrix = np.float32([[1, 0, 75], [0, 1, -60]])
        image = cv2.warpAffine(image, matrix, (1280, 960), borderValue=255)
    elif variant == "rotated":
        matrix = cv2.getRotationMatrix2D((640, 480), 8.0, 0.95)
        image = cv2.warpAffine(image, matrix, (1280, 960), borderValue=255)
    elif variant in {"homography", "combined"}:
        source = np.float32([[190, 150], [1090, 150], [1090, 793], [190, 793]])
        destination = np.float32([[250, 185], [1030, 120], [1110, 825], [160, 760]])
        matrix = cv2.getPerspectiveTransform(source, destination)
        image = cv2.warpPerspective(image, matrix, (1280, 960), borderValue=255)
    elif variant == "dim":
        image = np.asarray(25 + image.astype(np.float64) * 0.75, dtype=np.uint8)
    elif variant == "low_contrast":
        image = np.asarray(85 + image.astype(np.float64) * (100.0 / 255.0), dtype=np.uint8)
    if variant in {"blur", "combined"}:
        image = cv2.GaussianBlur(image, (3, 3), 0.7)
    if variant in {"noise", "combined"}:
        noise = rng.normal(0.0, 3.0, image.shape)
        image = np.asarray(np.clip(image.astype(np.float64) + noise, 0, 255), dtype=np.uint8)
    observation = detector.detect(image)
    assert len(observation.point_ids) >= 18
    assert tuple(sorted(observation.point_ids)) == observation.point_ids
    assert len(observation.point_ids) == len(observation.image_points_px)
    assert len(observation.point_ids) == len(observation.object_points_m)


def test_partial_visibility_returns_a_correct_id_subset(generated_charuco_target: Path) -> None:
    detector, image = _detector_and_canvas(generated_charuco_target)
    image[:, :480] = 255
    observation = detector.detect(image)
    assert 0 < len(observation.point_ids) < 24
    assert observation.point_ids == tuple(sorted(set(observation.point_ids)))
    expected = detector.target_spec.object_points_for(observation.point_ids)
    assert np.array_equal(observation.object_points_m, expected)


def test_partial_out_of_frame_returns_consistent_subset(generated_charuco_target: Path) -> None:
    detector, image = _detector_and_canvas(generated_charuco_target)
    translated = cv2.warpAffine(
        image, np.float32([[1, 0, -430], [0, 1, 0]]), (1280, 960), borderValue=255
    )
    observation = detector.detect(translated)
    assert 0 < len(observation.point_ids) < 24
    assert observation.image_points_px.shape[0] == observation.object_points_m.shape[0]


def test_wrong_dictionary_fails_quality_gate(generated_charuco_target: Path) -> None:
    detector, image = _detector_and_canvas(generated_charuco_target)
    wrong = CharucoDetector(replace(detector.target_spec, dictionary="DICT_4X4_50"))
    observation = wrong.detect(image)
    assert not observation.quality.passed
    assert len(observation.point_ids) < 12


def test_zero_corner_observation_round_trips_with_explicit_empty_shapes(
    generated_charuco_target: Path,
) -> None:
    detector, image = _detector_and_canvas(generated_charuco_target)
    observation = detector.detect(np.full_like(image, 127))
    assert observation.point_ids == ()
    assert observation.image_points_px.shape == (0, 2)
    assert observation.object_points_m.shape == (0, 3)
    assert observation.quality.passed is False

    restored = TargetObservation.from_dict(observation.to_dict())
    assert restored.to_dict() == observation.to_dict()
    assert restored.image_points_px.shape == (0, 2)
    assert restored.object_points_m.shape == (0, 3)


@pytest.mark.parametrize(
    "image",
    [
        np.empty((0, 1), dtype=np.uint8),
        np.empty((1, 0), dtype=np.uint8),
        np.zeros((20, 20, 4), dtype=np.uint8),
        np.zeros((20, 20), dtype=np.float32),
        np.full((20, 20), np.nan, dtype=np.float64),
    ],
)
def test_invalid_inputs_are_rejected(generated_charuco_target: Path, image: np.ndarray) -> None:
    detector, _canvas = _detector_and_canvas(generated_charuco_target)
    with pytest.raises(ContractError):
        detector.detect(image)


def test_registry_preserves_generic_consumer_contract(generated_charuco_target: Path) -> None:
    target = load_target(generated_charuco_target / "target_spec.json")
    detector = registry.create(plugin_name="charuco", target_spec=target)
    assert detector.plugin_name == "charuco"
    assert (
        registry.create(plugin_name="charuco", target_spec=target).__class__ is detector.__class__
    )


def test_detector_never_reinterprets_persisted_geometry_at_detection_time() -> None:
    source = (Path(__file__).parents[1] / "src/camera_rig/targets/charuco/detector.py").read_text()
    assert "getChessboardCorners" not in source
