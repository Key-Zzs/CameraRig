"""Deterministic printable ChArUco target artifact generation."""

from __future__ import annotations

import os
import shutil
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from camera_rig.artifacts.hashing import sha256_file
from camera_rig.artifacts.io import atomic_write_json
from camera_rig.core.errors import ArtifactError, MissingOptionalDependencyError
from camera_rig.targets.charuco.artifact import (
    ResolvedCharucoTarget,
    ResolvedCharucoTargetV2,
)
from camera_rig.targets.charuco.detector import CharucoDetector
from camera_rig.targets.charuco.geometry import canonical_corners_from_board, create_board
from camera_rig.targets.charuco.spec import CharucoTargetSpec
from camera_rig.targets.io import load_target
from camera_rig.version import __version__

MM_PER_INCH = 25.4


def generate_target_artifact(spec: CharucoTargetSpec, output: str | Path) -> dict[str, object]:
    """Generate, self-detect, checksum, and atomically publish a target directory."""
    target = Path(output)
    if target.exists():
        raise ArtifactError(f"target artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        board, _dictionary, cv2 = create_board(spec)
        board_path = temporary / f"{spec.target_name}_board.png"
        pdf_path = temporary / f"{spec.target_name}_print.pdf"
        preview_path = temporary / f"{spec.target_name}_preview.png"
        scale_check_path = temporary / f"{spec.target_name}_scale_check.pdf"
        width_px = round(spec.board_width_m * 1000.0 / MM_PER_INCH * spec.dpi)
        height_px = round(spec.board_height_m * 1000.0 / MM_PER_INCH * spec.dpi)
        board_image = board.generateImage(
            (width_px, height_px), marginSize=0, borderBits=spec.border_bits
        )
        if board_image.dtype != np.uint8 or set(np.unique(board_image).tolist()) != {0, 255}:
            raise ArtifactError("generated board PNG must be binary uint8")
        if not cv2.imwrite(str(board_path), board_image):
            raise ArtifactError("OpenCV could not write generated board PNG")
        _write_print_pdf(spec, board_path, pdf_path)
        if spec.schema_version == "camera-rig.target.charuco.v2" and spec.separate_scale_check:
            _write_scale_check_pdf(spec, scale_check_path)
        _write_preview(spec, board_image, preview_path, cv2)
        common: dict[str, object] = {
            "target_name": spec.target_name,
            "target_frame": spec.target_frame,
            "dictionary": spec.dictionary,
            "squares_x": spec.squares_x,
            "squares_y": spec.squares_y,
            "square_length_m": spec.square_length_m,
            "marker_length_m": spec.marker_length_m,
            "border_bits": spec.border_bits,
            "legacy_pattern": spec.legacy_pattern,
            "board_width_m": spec.board_width_m,
            "board_height_m": spec.board_height_m,
            "corner_points": canonical_corners_from_board(spec, board),
            "marker_ids": tuple(int(value) for value in np.asarray(board.getIds()).reshape(-1)),
            "camera_rig_version": __version__,
            "opencv_version": str(cv2.__version__),
            "source_config_sha256": spec.source_config_sha256,
            "board_png_sha256": sha256_file(board_path),
            "print_pdf_sha256": sha256_file(pdf_path),
        }
        resolved: ResolvedCharucoTarget
        if spec.schema_version == "camera-rig.target.charuco.v2":
            files = {
                "board_png": board_path.name,
                "print_pdf": pdf_path.name,
                "preview_png": preview_path.name,
            }
            if spec.separate_scale_check:
                files["scale_check_pdf"] = scale_check_path.name
            resolved = ResolvedCharucoTargetV2(
                **common,  # type: ignore[arg-type]
                source_type="generated",
                physical_measurement={
                    "nominal_width_mm": spec.board_width_m * 1000.0,
                    "nominal_height_mm": spec.board_height_m * 1000.0,
                    "square_length_mm": spec.square_length_m * 1000.0,
                    "marker_length_mm": spec.marker_length_m * 1000.0,
                    "source": "generated",
                },
                identification={
                    "candidate_uniqueness": True,
                    "method": "generated-self-check",
                    "evidence_hashes": [sha256_file(board_path)],
                    "opencv_version": str(cv2.__version__),
                },
                artifact_files=files,
            )
        else:
            resolved = ResolvedCharucoTarget(**common)  # type: ignore[arg-type]
        spec_path = temporary / "target_spec.json"
        atomic_write_json(spec_path, resolved.to_dict())
        loaded = load_target(spec_path)
        decoded_board = cv2.imread(str(board_path), cv2.IMREAD_GRAYSCALE)
        if decoded_board is None:
            raise ArtifactError("could not reload generated board PNG")
        observation = CharucoDetector(loaded).detect(decoded_board)
        expected_ids = tuple(point_id for point_id, _point in loaded.corner_points)
        if (
            observation.point_ids != expected_ids
            or len(observation.point_ids) != spec.charuco_corner_count
        ):
            raise ArtifactError(
                "generated board self-check did not detect every configured corner ID"
            )
        if not np.array_equal(
            observation.object_points_m,
            loaded.object_points_for(observation.point_ids),
        ):
            raise ArtifactError("generated board self-check changed persisted canonical geometry")
        marker_metadata = observation.metadata["marker_ids"]
        if not isinstance(marker_metadata, list):
            raise ArtifactError("self-check marker IDs metadata is invalid")
        detected_markers: tuple[int, ...] = tuple(int(value) for value in marker_metadata)
        if detected_markers != loaded.marker_ids:
            raise ArtifactError("generated board self-check marker IDs differ from target spec")
        report: dict[str, object] = {
            "schema_version": "camera-rig.target-generation.v1",
            "status": "PASS",
            "target_name": spec.target_name,
            "target_spec_sha256": sha256_file(spec_path),
            "board_pixel_size": [width_px, height_px],
            "board_physical_size_mm": [
                spec.board_width_m * 1000.0,
                spec.board_height_m * 1000.0,
            ],
            "expected_charuco_corner_count": spec.charuco_corner_count,
            "self_check": {
                "detected_marker_ids": list(detected_markers),
                "detected_charuco_corner_ids": list(observation.point_ids),
                "detected_charuco_corner_count": len(observation.point_ids),
                "canonical_points_match": True,
                "quality": observation.quality.to_dict(),
            },
            "software": {"camera_rig_version": __version__, "opencv_version": cv2.__version__},
        }
        atomic_write_json(temporary / "generation_report.json", report)
        checksum_names = [
            f"{spec.target_name}_board.png",
            f"{spec.target_name}_print.pdf",
            f"{spec.target_name}_preview.png",
            "generation_report.json",
            "target_spec.json",
        ]
        if spec.schema_version == "camera-rig.target.charuco.v2" and spec.separate_scale_check:
            checksum_names.append(f"{spec.target_name}_scale_check.pdf")
        lines = [f"{sha256_file(temporary / name)}  {name}" for name in sorted(checksum_names)]
        (temporary / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_print_pdf(spec: CharucoTargetSpec, board_path: Path, path: Path) -> None:
    if spec.schema_version == "camera-rig.target.charuco.v2":
        _write_print_pdf_v2(spec, board_path, path)
        return
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen.canvas import Canvas
    except ImportError as error:
        raise MissingOptionalDependencyError(
            'ChArUco support requires: pip install "camera-rig[charuco]"'
        ) from error
    page_width, page_height = landscape(A4)
    canvas = Canvas(str(path), pagesize=(page_width, page_height), invariant=1, pageCompression=1)
    canvas.setTitle(f"CameraRig {spec.target_name}")
    canvas.setAuthor("CameraRig")
    canvas.setFillColorRGB(1, 1, 1)
    canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)
    board_x = 15.0 * mm
    board_y = 30.0 * mm
    canvas.drawImage(
        ImageReader(BytesIO(board_path.read_bytes())),
        board_x,
        board_y,
        width=spec.board_width_m * 1000.0 * mm,
        height=spec.board_height_m * 1000.0 * mm,
        preserveAspectRatio=False,
        anchor="sw",
        mask=None,
    )
    canvas.setStrokeColorRGB(0, 0, 0)
    canvas.setFillColorRGB(0, 0, 0)
    canvas.setLineWidth(0.35 * mm)
    _horizontal_ruler(
        canvas,
        20.0 * mm,
        15.0 * mm,
        spec.horizontal_check_length_mm * mm,
        spec.horizontal_check_length_mm,
        mm,
    )
    _vertical_ruler(
        canvas,
        270.0 * mm,
        55.0 * mm,
        spec.vertical_check_length_mm * mm,
        spec.vertical_check_length_mm,
        mm,
    )
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(130.0 * mm, 17.0 * mm, "PRINT AT 100% / ACTUAL SIZE")
    canvas.drawString(130.0 * mm, 12.5 * mm, "DO NOT FIT TO PAGE")
    canvas.setFont("Helvetica", 7)
    canvas.drawString(229.0 * mm, 188.0 * mm, spec.target_name)
    canvas.drawString(229.0 * mm, 184.0 * mm, "TOP  +Y up")
    canvas.drawString(229.0 * mm, 180.0 * mm, "+X right; +Z out")
    canvas.drawString(
        229.0 * mm,
        174.0 * mm,
        f"Board nominal width: {spec.board_width_m * 1000.0:.2f} mm",
    )
    canvas.drawString(
        229.0 * mm,
        170.0 * mm,
        f"Board nominal height: {spec.board_height_m * 1000.0:.2f} mm",
    )
    canvas.showPage()
    canvas.save()


def _write_print_pdf_v2(spec: CharucoTargetSpec, board_path: Path, path: Path) -> None:
    try:
        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen.canvas import Canvas
    except ImportError as error:
        raise MissingOptionalDependencyError(
            'ChArUco support requires: pip install "camera-rig[charuco]"'
        ) from error
    page_size = (spec.page_width_mm * mm, spec.page_height_mm * mm)
    canvas = Canvas(str(path), pagesize=page_size, invariant=1, pageCompression=1)
    canvas.setTitle(f"CameraRig {spec.target_name}")
    canvas.setAuthor("CameraRig")
    canvas.setFillColorRGB(1, 1, 1)
    canvas.rect(0, 0, page_size[0], page_size[1], fill=1, stroke=0)
    canvas.drawImage(
        ImageReader(BytesIO(board_path.read_bytes())),
        spec.board_x_mm * mm,
        spec.board_y_mm * mm,
        width=spec.board_width_m * 1000.0 * mm,
        height=spec.board_height_m * 1000.0 * mm,
        preserveAspectRatio=False,
        anchor="sw",
        mask=None,
    )
    if not spec.board_only and not spec.separate_scale_check:
        canvas.setStrokeColorRGB(0, 0, 0)
        _horizontal_ruler(
            canvas,
            10.0 * mm,
            10.0 * mm,
            spec.horizontal_check_length_mm * mm,
            spec.horizontal_check_length_mm,
            mm,
        )
    canvas.showPage()
    canvas.save()


def _write_scale_check_pdf(spec: CharucoTargetSpec, path: Path) -> None:
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.units import mm
        from reportlab.pdfgen.canvas import Canvas
    except ImportError as error:
        raise MissingOptionalDependencyError(
            'ChArUco support requires: pip install "camera-rig[charuco]"'
        ) from error
    page_size = landscape(A4)
    canvas = Canvas(str(path), pagesize=page_size, invariant=1, pageCompression=1)
    canvas.setTitle(f"CameraRig {spec.target_name} scale check")
    canvas.setAuthor("CameraRig")
    _horizontal_ruler(
        canvas,
        20.0 * mm,
        20.0 * mm,
        spec.horizontal_check_length_mm * mm,
        spec.horizontal_check_length_mm,
        mm,
    )
    _vertical_ruler(
        canvas,
        270.0 * mm,
        5.0 * mm,
        spec.vertical_check_length_mm * mm,
        spec.vertical_check_length_mm,
        mm,
    )
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(20.0 * mm, 190.0 * mm, "PRINT AT 100% / ACTUAL SIZE")
    canvas.showPage()
    canvas.save()


def _horizontal_ruler(
    canvas: Any, x: float, y: float, length: float, nominal_mm: float, mm: float
) -> None:
    canvas.line(x, y, x + length, y)
    for position in (x, x + length):
        canvas.line(position, y - 2 * mm, position, y + 2 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(x, y + 3 * mm, f"Horizontal check: {nominal_mm:.2f} mm")


def _vertical_ruler(
    canvas: Any, x: float, y: float, length: float, nominal_mm: float, mm: float
) -> None:
    canvas.line(x, y, x, y + length)
    for position in (y, y + length):
        canvas.line(x - 2 * mm, position, x + 2 * mm, position)
    canvas.saveState()
    canvas.translate(x + 4 * mm, y)
    canvas.rotate(90)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(0, 0, f"Vertical check: {nominal_mm:.2f} mm")
    canvas.restoreState()


def _write_preview(spec: CharucoTargetSpec, board: Any, path: Path, cv2: Any) -> None:
    if spec.schema_version == "camera-rig.target.charuco.v2":
        _write_preview_v2(spec, board, path, cv2)
        return
    dpi = 150
    page_width = round(297.0 / MM_PER_INCH * dpi)
    page_height = round(210.0 / MM_PER_INCH * dpi)
    preview = np.full((page_height, page_width), 255, dtype=np.uint8)
    board_width = round(spec.board_width_m * 1000.0 / MM_PER_INCH * dpi)
    board_height = round(spec.board_height_m * 1000.0 / MM_PER_INCH * dpi)
    board_preview = cv2.resize(board, (board_width, board_height), interpolation=cv2.INTER_NEAREST)
    x = round(15.0 / MM_PER_INCH * dpi)
    y_from_bottom = round(30.0 / MM_PER_INCH * dpi)
    y = page_height - y_from_bottom - board_height
    preview[y : y + board_height, x : x + board_width] = board_preview
    cv2.line(preview, (118, page_height - 89), (709, page_height - 89), 0, 2)
    cv2.line(preview, (1595, 325), (1595, 916), 0, 2)
    cv2.putText(
        preview,
        "PRINT AT 100% - DO NOT FIT",
        (768, page_height - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        0,
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), preview):
        raise ArtifactError("OpenCV could not write target preview PNG")


def _write_preview_v2(spec: CharucoTargetSpec, board: Any, path: Path, cv2: Any) -> None:
    dpi = min(spec.dpi, 150)
    page_width = round(spec.page_width_mm / MM_PER_INCH * dpi)
    page_height = round(spec.page_height_mm / MM_PER_INCH * dpi)
    preview = np.full((page_height, page_width), 255, dtype=np.uint8)
    board_width = round(spec.board_width_m * 1000.0 / MM_PER_INCH * dpi)
    board_height = round(spec.board_height_m * 1000.0 / MM_PER_INCH * dpi)
    board_preview = cv2.resize(board, (board_width, board_height), interpolation=cv2.INTER_NEAREST)
    x = round(spec.board_x_mm / MM_PER_INCH * dpi)
    y_from_bottom = round(spec.board_y_mm / MM_PER_INCH * dpi)
    y = page_height - y_from_bottom - board_height
    preview[y : y + board_height, x : x + board_width] = board_preview
    if not cv2.imwrite(str(path), preview):
        raise ArtifactError("OpenCV could not write target preview PNG")
