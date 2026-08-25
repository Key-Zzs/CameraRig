from __future__ import annotations

import re
from pathlib import Path

import pytest

import camera_rig
from camera_rig.cli.main import main


def test_package_exposes_version() -> None:
    assert camera_rig.__version__ == "0.2.0"


def test_cli_help() -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0


def test_cli_version() -> None:
    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0


def test_readmes_do_not_contain_progress_sections() -> None:
    repository_root = Path(__file__).parents[1]
    forbidden_heading = re.compile(
        r"^#{1,6}\s+(?:R\d+\b|Roadmap\b|Milestones?\b|Current Progress\b|"
        r"Future Work\b|TODO\b|阶段\b|里程碑\b|路线图\b|当前进度\b|后续计划\b)",
        re.IGNORECASE | re.MULTILINE,
    )
    for filename in ("README.md", "README_zh-CN.md"):
        content = (repository_root / filename).read_text(encoding="utf-8")
        assert forbidden_heading.search(content) is None
        for token in (
            "R2",
            "R3",
            "R4",
            "R5",
            "阶段",
            "当前进度",
            "未来计划",
            "Roadmap",
            "Milestone",
            "TODO",
        ):
            assert token.casefold() not in content.casefold()
