from pathlib import Path

import pytest

from src.artifacts import (
    ArtifactError,
    PROMPT_TEMPLATE_PATH,
    SKILLS_PATH,
    V4_PROMPT_PATH,
    discover_skills,
    read_markdown,
    read_skill,
)


def test_default_prompt_template_contains_socratic_rules() -> None:
    prompt = read_markdown(PROMPT_TEMPLATE_PATH)

    assert "一次只能问一个简短问题" in prompt
    assert "不直接给出填空答案或完整答案" in prompt


def test_default_prompt_template_routes_non_exercises_to_normal_mode() -> None:
    prompt = read_markdown(PROMPT_TEMPLATE_PATH)

    assert "最高优先级：先判断回答模式" in prompt
    assert "普通模式" in prompt
    assert "不得编造题目" in prompt
    assert "不强制提问" in prompt
    assert "不回答与学习无关的问题" not in prompt


def test_v4_prompt_is_all_subject_and_preserves_write_boundary() -> None:
    prompt = read_markdown(V4_PROMPT_PATH)

    assert "全学科" in prompt
    assert "知识解释" in prompt
    assert "苏格拉底" in prompt
    assert "不得声称已经保存错题" in prompt
    assert "不得直接透露当前个性化练习的参考答案" in prompt


def test_read_markdown_returns_trimmed_content(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("\n# 测试 Prompt\n\n保留内部空行。\n", encoding="utf-8")

    assert read_markdown(prompt_path) == "# 测试 Prompt\n\n保留内部空行。"


def test_read_markdown_reports_missing_file(tmp_path: Path) -> None:
    prompt_path = tmp_path / "missing.md"

    with pytest.raises(ArtifactError, match=r"未找到 .*missing\.md"):
        read_markdown(prompt_path)


def test_read_markdown_reports_empty_file(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text(" \n\t", encoding="utf-8")

    with pytest.raises(ArtifactError, match=r"prompt\.md 内容为空"):
        read_markdown(prompt_path)


def test_read_markdown_reports_non_utf8_file(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(ArtifactError, match="不是 UTF-8 编码"):
        read_markdown(prompt_path)


def test_default_skills_use_standard_metadata_and_steps() -> None:
    skills = discover_skills(SKILLS_PATH)

    assert [skill.name for skill in skills] == [
        "english-quest",
        "sorting-out-mistakes",
    ]
    skills_by_name = {skill.name: skill for skill in skills}

    quest = read_skill(skills_by_name["english-quest"].path)
    assert "闯关" in quest.metadata.description
    assert "普通英语问答" in quest.metadata.description
    assert "五关" in quest.instructions
    assert "每轮只能有一个" in quest.instructions
    assert "任务报告" in quest.instructions
    assert "不要声称游戏结果已经保存" in quest.instructions
    assert "scripts/quest_state.py" in quest.instructions
    assert "assets/detective-board.svg" in quest.instructions
    quest_root = skills_by_name["english-quest"].path.parent
    assert (quest_root / "scripts" / "quest_state.py").is_file()
    assert (quest_root / "assets" / "detective-board.svg").is_file()

    mistakes = read_skill(skills_by_name["sorting-out-mistakes"].path)
    assert "整理错题" in mistakes.metadata.description
    assert "原题" in mistakes.metadata.description
    assert "Step 1" in mistakes.instructions
    assert "student/mistakes/inbox/" in mistakes.instructions
    assert "student/mistakes/records/" in mistakes.instructions
    assert "save_mistake" in mistakes.instructions
    assert "schema_version" in mistakes.instructions
    assert "next_review_at" in mistakes.instructions


def test_read_skill_requires_frontmatter(tmp_path: Path) -> None:
    skill_dir = tmp_path / "sorting-out-mistakes"
    skill_dir.mkdir()
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text("# 没有 frontmatter", encoding="utf-8")

    with pytest.raises(ArtifactError, match="缺少 YAML frontmatter"):
        read_skill(skill_path)


def test_read_skill_requires_name_to_match_parent_directory(tmp_path: Path) -> None:
    skill_dir = tmp_path / "sorting-out-mistakes"
    skill_dir.mkdir()
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "---\n"
        "name: another-skill\n"
        "description: 测试 Skill\n"
        "---\n"
        "\n"
        "Step 1：测试。\n",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="必须与父目录 sorting-out-mistakes 一致"):
        read_skill(skill_path)


def test_discover_skills_reloads_metadata_after_edit(tmp_path: Path) -> None:
    skills_path = tmp_path / "skill"
    skill_dir = skills_path / "sorting-out-mistakes"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "---\n"
        "name: sorting-out-mistakes\n"
        "description: 第一版触发说明\n"
        "---\n"
        "\n"
        "Step 1：第一版。\n",
        encoding="utf-8",
    )

    first = discover_skills(skills_path)
    skill_path.write_text(
        "---\n"
        "name: sorting-out-mistakes\n"
        "description: 第二版触发说明\n"
        "---\n"
        "\n"
        "Step 1：第二版。\n",
        encoding="utf-8",
    )
    second = discover_skills(skills_path)

    assert first[0].description == "第一版触发说明"
    assert second[0].description == "第二版触发说明"
