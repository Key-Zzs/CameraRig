"""Portable target artifact loading."""

from __future__ import annotations

from pathlib import Path

from camera_rig.artifacts.hashing import sha256_file
from camera_rig.artifacts.io import load_json
from camera_rig.core.errors import ArtifactError
from camera_rig.targets.charuco.artifact import (
    ResolvedCharucoTarget,
    ResolvedCharucoTargetV2,
)


def load_target(path: str | Path) -> ResolvedCharucoTarget:
    """Load a supported resolved target artifact, failing closed by schema version."""
    value = load_json(path)
    if not isinstance(value, dict):
        raise ArtifactError("resolved target spec must be a JSON object")
    schema_version = value.get("schema_version")
    if schema_version == "camera-rig.target.charuco-resolved.v1":
        return ResolvedCharucoTarget.from_dict(dict(value)).with_artifact_sha256(sha256_file(path))
    if schema_version == "camera-rig.target.charuco-resolved.v2":
        return ResolvedCharucoTargetV2.from_dict(dict(value)).with_artifact_sha256(
            sha256_file(path)
        )
    raise ArtifactError(f"unsupported resolved target schema: {schema_version!r}")


def validate_target_artifact(path: str | Path) -> ResolvedCharucoTarget:
    """Validate the exact resolved target directory and every companion checksum."""
    source = Path(path)
    target = load_target(source)
    root = source.parent
    if isinstance(target, ResolvedCharucoTargetV2):
        if target.source_type == "existing_physical":
            expected_payloads = {"registration_report.json", "target_spec.json"}
        else:
            expected_payloads = set((target.artifact_files or {}).values()) | {
                "generation_report.json",
                "target_spec.json",
            }
    else:
        expected_payloads = {
            f"{target.target_name}_board.png",
            f"{target.target_name}_print.pdf",
            f"{target.target_name}_preview.png",
            "generation_report.json",
            "target_spec.json",
        }
    expected_files = expected_payloads | {"checksums.sha256"}
    actual_files: set[str] = set()
    for candidate in root.iterdir():
        if candidate.is_symlink():
            raise ArtifactError(f"target artifact must not contain symlinks: {candidate.name}")
        if candidate.is_file():
            actual_files.add(candidate.name)
    if actual_files != expected_files:
        raise ArtifactError("target artifact contains missing or unexpected files")
    checksums = _load_checksums(root / "checksums.sha256")
    if set(checksums) != expected_payloads:
        raise ArtifactError("target checksums contain missing or unexpected paths")
    for name, digest in checksums.items():
        if sha256_file(root / name) != digest:
            raise ArtifactError(f"target artifact checksum mismatch: {name}")
    if not isinstance(target, ResolvedCharucoTargetV2) or target.source_type == "generated":
        artifact_files = (
            target.artifact_files if isinstance(target, ResolvedCharucoTargetV2) else {}
        )
        board_name = (artifact_files or {}).get("board_png", f"{target.target_name}_board.png")
        print_name = (artifact_files or {}).get("print_pdf", f"{target.target_name}_print.pdf")
        if checksums[board_name] != target.board_png_sha256:
            raise ArtifactError("resolved target board PNG checksum is inconsistent")
        if checksums[print_name] != target.print_pdf_sha256:
            raise ArtifactError("resolved target print PDF checksum is inconsistent")
    report_name = (
        "registration_report.json"
        if isinstance(target, ResolvedCharucoTargetV2) and target.source_type == "existing_physical"
        else "generation_report.json"
    )
    report = load_json(root / report_name)
    if not isinstance(report, dict):
        raise ArtifactError("target generation report must be an object")
    if report.get("status") != "PASS" or report.get("target_spec_sha256") != target.artifact_sha256:
        raise ArtifactError("target report is inconsistent with resolved target")
    if isinstance(target, ResolvedCharucoTargetV2) and target.source_type == "existing_physical":
        identification = target.identification or {}
        if (
            set(report)
            != {
                "schema_version",
                "status",
                "source_type",
                "target_name",
                "target_spec_sha256",
                "identification_report_sha256",
                "generated_print_assets",
            }
            or report.get("schema_version") != "camera-rig.target-registration.v1"
            or report.get("source_type") != "existing_physical"
            or report.get("target_name") != target.target_name
            or report.get("generated_print_assets") is not False
            or report.get("identification_report_sha256")
            != identification.get("identification_report_sha256")
        ):
            raise ArtifactError("existing-target registration provenance is inconsistent")
    if report_name == "generation_report.json" and report.get(
        "expected_charuco_corner_count"
    ) != len(target.corner_points):
        raise ArtifactError("target generation report has inconsistent corner count")
    return target


def _load_checksums(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ArtifactError(f"could not read target checksums: {error}") from error
    result: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2:
            raise ArtifactError("invalid target checksum line")
        digest, name = parts
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not name
            or "/" in name
            or "\\" in name
            or name in result
        ):
            raise ArtifactError("invalid target checksum entry")
        result[name] = digest
    return result
