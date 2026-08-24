"""真实模型测试只使用临时合成个性化资料。"""

from pathlib import Path

import pytest

import src.agents as agents_module
import src.facade as facade_module
import src.workflow as workflow_module
from src.personalization import initialize_personalization


@pytest.fixture(autouse=True)
def synthetic_personalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    student_root = tmp_path / "personalization" / "student"
    templates = student_root / "templates"
    templates.mkdir(parents=True)
    soul_template = templates / "SOUL.md"
    owner_template = templates / "OWNER.md"
    soul_template.write_text(
        "# 合成测试人格\n\n保持清楚、耐心。\n",
        encoding="utf-8",
    )
    owner_template.write_text(
        "---\n"
        "schema_version: 1\n"
        "auto_memory: false\n"
        "preferred_name: null\n"
        "grade_band: null\n"
        "languages: []\n"
        "interests: []\n"
        "learning_goals: []\n"
        "strengths: []\n"
        "challenges: []\n"
        "response_preferences: []\n"
        "---\n\n"
        "# 合成测试资料\n",
        encoding="utf-8",
    )
    snapshot = initialize_personalization(
        soul_path=student_root / "SOUL.md",
        owner_path=student_root / "OWNER.md",
        soul_template_path=soul_template,
        owner_template_path=owner_template,
        student_root=student_root,
    )
    monkeypatch.setattr(
        agents_module,
        "load_personalization",
        lambda: snapshot,
    )
    monkeypatch.setattr(
        facade_module,
        "load_personalization",
        lambda: snapshot,
    )
    monkeypatch.setattr(
        workflow_module,
        "load_personalization",
        lambda: snapshot,
    )
