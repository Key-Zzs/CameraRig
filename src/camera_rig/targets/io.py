"""Portable target artifact loading."""

from __future__ import annotations

from pathlib import Path

from camera_rig.artifacts.hashing import sha256_file
from camera_rig.artifacts.io import load_json
from camera_rig.core.errors import ArtifactError
from camera_rig.targets.charuco.artifact import ResolvedCharucoTarget


def load_target(path: str | Path) -> ResolvedCharucoTarget:
    """Load a supported resolved target artifact, failing closed by schema version."""
    value = load_json(path)
    if not isinstance(value, dict):
        raise ArtifactError("resolved target spec must be a JSON object")
    schema_version = value.get("schema_version")
    if schema_version == "camera-rig.target.charuco-resolved.v1":
        return ResolvedCharucoTarget.from_dict(dict(value)).with_artifact_sha256(sha256_file(path))
    raise ArtifactError(f"unsupported resolved target schema: {schema_version!r}")


def validate_target_artifact(path: str | Path) -> ResolvedCharucoTarget:
    """Validate the exact resolved target directory and every companion checksum."""
    source = Path(path)
    target = load_target(source)
    root = source.parent
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
    if checksums[f"{target.target_name}_board.png"] != target.board_png_sha256:
        raise ArtifactError("resolved target board PNG checksum is inconsistent")
    if checksums[f"{target.target_name}_print.pdf"] != target.print_pdf_sha256:
        raise ArtifactError("resolved target print PDF checksum is inconsistent")
    report = load_json(root / "generation_report.json")
    if not isinstance(report, dict):
        raise ArtifactError("target generation report must be an object")
    if (
        report.get("status") != "PASS"
        or report.get("target_spec_sha256") != target.artifact_sha256
        or report.get("expected_charuco_corner_count") != len(target.corner_points)
    ):
        raise ArtifactError("target generation report is inconsistent with resolved target")
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
