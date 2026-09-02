from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("reportlab")

from camera_rig.artifacts.hashing import sha256_file
from camera_rig.core.errors import ArtifactError
from camera_rig.targets.charuco.generator import generate_target_artifact
from camera_rig.targets.charuco.spec import load_charuco_target_spec
from camera_rig.targets.io import load_target, validate_target_artifact

REPOSITORY_ROOT = Path(__file__).parents[1]
STANDARD_CONFIG = REPOSITORY_ROOT / "configs/targets/charuco_a4_v1.yaml"
pytestmark = pytest.mark.charuco


def test_generator_writes_exact_portable_file_set(generated_charuco_target: Path) -> None:
    assert {path.name for path in generated_charuco_target.iterdir()} == {
        "target_spec.json",
        "charuco_a4_v1_board.png",
        "charuco_a4_v1_print.pdf",
        "charuco_a4_v1_preview.png",
        "checksums.sha256",
        "generation_report.json",
    }
    text = (generated_charuco_target / "target_spec.json").read_text(encoding="utf-8")
    assert "/home/" not in text
    assert "charuco_target" in text


def test_canonical_geometry_is_frozen_by_id(generated_charuco_target: Path) -> None:
    target = load_target(generated_charuco_target / "target_spec.json")
    assert len(target.corner_points) == 24
    points = dict(target.corner_points)
    assert points[0] == pytest.approx((0.030, 0.120, 0.0))
    assert points[5] == pytest.approx((0.180, 0.120, 0.0))
    assert points[18] == pytest.approx((0.030, 0.030, 0.0))
    assert points[23] == pytest.approx((0.180, 0.030, 0.0))
    array = np.asarray(list(points.values()))
    assert np.equal(array[:, 2], 0.0).all()
    assert np.allclose(np.diff(np.unique(array[:, 0])), 0.030)
    assert np.allclose(np.diff(np.unique(array[:, 1])), 0.030)
    assert target.board_width_m == pytest.approx(0.210)
    assert target.board_height_m == pytest.approx(0.150)


def test_generated_board_self_detects_all_ids(generated_charuco_target: Path) -> None:
    report = json.loads(
        (generated_charuco_target / "generation_report.json").read_text(encoding="utf-8")
    )
    assert report["self_check"]["detected_charuco_corner_ids"] == list(range(24))
    assert report["self_check"]["detected_charuco_corner_count"] == 24
    assert report["self_check"]["canonical_points_match"] is True


def test_checksums_cover_every_artifact_except_checksum_file(
    generated_charuco_target: Path,
) -> None:
    entries = {}
    for line in (generated_charuco_target / "checksums.sha256").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    assert set(entries) == {
        path.name for path in generated_charuco_target.iterdir() if path.name != "checksums.sha256"
    }
    assert all(
        sha256_file(generated_charuco_target / name) == digest for name, digest in entries.items()
    )
    validated = validate_target_artifact(generated_charuco_target / "target_spec.json")
    assert validated.target_name == "charuco_a4_v1"


def test_resolved_geometry_tampering_fails_closed(
    generated_charuco_target: Path, tmp_path: Path
) -> None:
    data = json.loads((generated_charuco_target / "target_spec.json").read_text())
    data["charuco_corners"][0]["object_point_m"][1] = 0.09
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ArtifactError, match="corner ID 0"):
        load_target(path)


def test_companion_file_tampering_fails_artifact_validation(
    generated_charuco_target: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "target"
    shutil.copytree(generated_charuco_target, copied)
    board = copied / "charuco_a4_v1_board.png"
    payload = bytearray(board.read_bytes())
    payload[-1] ^= 1
    board.write_bytes(payload)
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        validate_target_artifact(copied / "target_spec.json")


def test_pdf_and_resolved_artifacts_are_deterministic(tmp_path: Path) -> None:
    spec = load_charuco_target_spec(STANDARD_CONFIG)
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_target_artifact(spec, first)
    generate_target_artifact(spec, second)
    for name in (
        "charuco_a4_v1_board.png",
        "charuco_a4_v1_print.pdf",
        "charuco_a4_v1_preview.png",
        "target_spec.json",
        "generation_report.json",
        "checksums.sha256",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_generator_overwrites_existing_target_directory_by_default(tmp_path: Path) -> None:
    spec = load_charuco_target_spec(STANDARD_CONFIG)
    output = tmp_path / "target"
    generate_target_artifact(spec, output)
    (output / "stale.txt").write_text("stale", encoding="utf-8")

    generate_target_artifact(spec, output)

    assert not (output / "stale.txt").exists()
    validate_target_artifact(output / "target_spec.json")
