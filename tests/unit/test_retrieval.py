from pathlib import Path

import pytest

from src.artifacts import ArtifactError
from src.retrieval import (
    KnowledgeCardError,
    citation_from_card,
    discover_knowledge_cards,
    format_knowledge_context,
    match_knowledge_card,
    read_knowledge_card,
    retrieve_knowledge,
)


def _card_markdown(
    *,
    card_id: str = "english-grammar-present-perfect",
    title: str = "现在完成时",
    subject: str = "english",
    category: str = "grammar",
    grade: str = "junior-high",
    language: str = "zh-CN",
    keywords: tuple[str, ...] = (
        "现在完成时",
        "present perfect",
        "three times",
        "already",
    ),
    aliases: tuple[str, ...] = (
        "times",
        "次数表达",
        "repeated experience",
    ),
    core_rule: str = "have/has + 过去分词，表示过去发生且与现在有关的动作或经历。",
    example: str = "I have read this book three times.",
    common_mistake: str = "看到次数时，不要误用现在进行时。",
) -> str:
    keyword_lines = "\n".join(f"  - {term}" for term in keywords)
    alias_lines = "\n".join(f"  - {term}" for term in aliases)
    return (
        "---\n"
        "schema_version: 2\n"
        f"id: {card_id}\n"
        f"title: {title}\n"
        f"subject: {subject}\n"
        f"category: {category}\n"
        f"grade: {grade}\n"
        f"language: {language}\n"
        "keywords:\n"
        f"{keyword_lines}\n"
        "aliases:\n"
        f"{alias_lines}\n"
        "---\n"
        "\n"
        "# 核心规则\n"
        "\n"
        f"{core_rule}\n"
        "\n"
        "## 例句\n"
        "\n"
        f"{example}\n"
        "\n"
        "## 易错提醒\n"
        "\n"
        f"{common_mistake}\n"
    )


CARD_MARKDOWN = _card_markdown()


def _write_card(path: Path, content: str = CARD_MARKDOWN) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_read_knowledge_card_parses_yaml_metadata_and_markdown_sections(
    tmp_path: Path,
) -> None:
    card = read_knowledge_card(_write_card(tmp_path / "present-perfect.md"))

    assert card.card_id == "english-grammar-present-perfect"
    assert card.title == "现在完成时"
    assert card.subject == "english"
    assert card.category == "grammar"
    assert card.grade == "junior-high"
    assert card.language == "zh-CN"
    assert card.keywords == (
        "现在完成时",
        "present perfect",
        "three times",
        "already",
    )
    assert card.aliases == ("times", "次数表达", "repeated experience")
    assert card.core_rule.startswith("have/has + 过去分词")
    assert card.example == "I have read this book three times."
    assert card.common_mistake == "看到次数时，不要误用现在进行时。"
    assert card.source == "student/knowledge/present-perfect.md"


def test_read_knowledge_card_preserves_nested_analysis_in_parent_section(
    tmp_path: Path,
) -> None:
    analysis = (
        "three times 表示动作发生过多次，且次数累计截止到现在，"
        "所以使用现在完成时。"
    )
    content = CARD_MARKDOWN.replace(
        "I have read this book three times.\n\n",
        f"I have read this book three times.\n\n### 句子解析\n\n{analysis}\n\n",
    )

    card = read_knowledge_card(
        _write_card(tmp_path / "present-perfect.md", content)
    )
    hit = match_knowledge_card("four times", card)

    assert card.example == (
        "I have read this book three times.\n\n"
        f"### 句子解析\n\n{analysis}"
    )
    assert hit is not None
    assert "### 句子解析" in format_knowledge_context(hit)
    assert analysis in format_knowledge_context(hit)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (CARD_MARKDOWN.removeprefix("---\n"), "缺少 YAML front matter"),
        (
            CARD_MARKDOWN.replace("schema_version: 2", "schema_version: ["),
            "YAML front matter 无法解析",
        ),
        (
            CARD_MARKDOWN.replace("schema_version: 2", "schema_version: 1").replace(
                "category: grammar\n",
                "",
            ),
            "仅支持 schema_version: 2",
        ),
        (
            CARD_MARKDOWN.replace("language: zh-CN\n", ""),
            "缺少必填元数据：language",
        ),
        (
            CARD_MARKDOWN.replace("aliases:\n", "unknown: value\naliases:\n"),
            "不支持的元数据字段：unknown",
        ),
        (
            CARD_MARKDOWN.replace("keywords:\n  - 现在完成时", "keywords: 现在完成时"),
            "元数据 keywords 必须是字符串列表",
        ),
        (
            CARD_MARKDOWN.replace("category: grammar", "category: Grammar Rules"),
            "元数据 category 必须使用小写英文 kebab-case",
        ),
        (
            CARD_MARKDOWN.replace("subject: english", "subject: English"),
            "元数据 subject 必须使用小写英文 kebab-case",
        ),
        (
            CARD_MARKDOWN.replace("# 核心规则", "不要解析这段前置文本\n\n# 核心规则"),
            "正文必须从 # 核心规则 开始",
        ),
        (
            CARD_MARKDOWN.replace("## 例句\n\nI have read this book three times.\n\n", ""),
            "缺少必填正文段落：例句",
        ),
        (
            CARD_MARKDOWN + "\n## 例句\n\n重复例句\n",
            "重复定义了正文段落 例句",
        ),
        (
            CARD_MARKDOWN.replace(
                "## 易错提醒",
                "## 同级未知段落\n\n不应接受。\n\n## 易错提醒",
            ),
            "包含不支持的正文标题：## 同级未知段落",
        ),
    ],
)
def test_read_knowledge_card_rejects_invalid_structure(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    card_path = _write_card(tmp_path / "invalid-card.md", content)

    with pytest.raises(KnowledgeCardError, match=message):
        read_knowledge_card(card_path)


def test_read_knowledge_card_reports_file_errors(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.md"
    empty_path = tmp_path / "empty.md"
    invalid_path = tmp_path / "invalid.md"
    empty_path.write_text("", encoding="utf-8")
    invalid_path.write_bytes(b"\xff\xfe")

    with pytest.raises(ArtifactError, match="missing.md"):
        read_knowledge_card(missing_path)
    with pytest.raises(ArtifactError, match="内容为空"):
        read_knowledge_card(empty_path)
    with pytest.raises(ArtifactError, match="不是 UTF-8"):
        read_knowledge_card(invalid_path)


def test_read_knowledge_card_rejects_non_kebab_case_filename(tmp_path: Path) -> None:
    card_path = _write_card(tmp_path / "Present Perfect.md")

    with pytest.raises(KnowledgeCardError, match="文件名必须使用小写英文 kebab-case"):
        read_knowledge_card(card_path)


def test_discover_knowledge_cards_reads_sorted_nested_markdown_files(
    tmp_path: Path,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    _write_card(
        knowledge_dir / "english" / "grammar" / "past-simple.md",
        _card_markdown(
            card_id="english-grammar-past-simple",
            title="一般过去时",
            keywords=("一般过去时", "past simple", "yesterday"),
            aliases=("finished past action",),
            core_rule="动词使用过去式。",
            example="I saw the movie yesterday.",
            common_mistake="明确过去时间不用现在完成时。",
        ),
    )
    _write_card(
        knowledge_dir / "english" / "reading" / "main-idea.md",
        _card_markdown(
            card_id="english-reading-main-idea",
            title="主旨大意",
            category="reading",
            keywords=("主旨大意", "main idea"),
            aliases=("central idea",),
        ),
    )
    _write_card(
        knowledge_dir / "math" / "algebra" / "linear-equation.md",
        _card_markdown(
            card_id="math-algebra-linear-equation",
            title="一元一次方程",
            subject="math",
            category="algebra",
            keywords=("一元一次方程", "linear equation"),
            aliases=("一次方程",),
        ),
    )

    cards = discover_knowledge_cards(knowledge_dir)

    assert [card.card_id for card in cards] == [
        "english-grammar-past-simple",
        "english-reading-main-idea",
        "math-algebra-linear-equation",
    ]


def test_discover_knowledge_cards_rejects_duplicate_ids(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    _write_card(knowledge_dir / "first.md")
    _write_card(knowledge_dir / "second.md")

    with pytest.raises(KnowledgeCardError, match="重复的知识卡 id"):
        discover_knowledge_cards(knowledge_dir)


def test_discover_knowledge_cards_reports_missing_or_empty_directory(
    tmp_path: Path,
) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ArtifactError, match="missing"):
        discover_knowledge_cards(tmp_path / "missing")
    with pytest.raises(KnowledgeCardError, match="没有找到 Markdown 知识卡"):
        discover_knowledge_cards(empty_dir)


def test_discover_knowledge_cards_rejects_symbolic_links(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    target = _write_card(tmp_path / "outside.md")
    nested_dir = knowledge_dir / "english" / "grammar"
    nested_dir.mkdir(parents=True)
    (nested_dir / "linked.md").symlink_to(target)

    with pytest.raises(KnowledgeCardError, match="不能是符号链接"):
        discover_knowledge_cards(knowledge_dir)


def test_discover_knowledge_cards_rejects_symbolic_link_directories(
    tmp_path: Path,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    external_dir = tmp_path / "external"
    _write_card(external_dir / "outside.md")
    knowledge_dir.mkdir()
    (knowledge_dir / "linked-category").symlink_to(
        external_dir,
        target_is_directory=True,
    )

    with pytest.raises(KnowledgeCardError, match="不能是符号链接"):
        discover_knowledge_cards(knowledge_dir)


@pytest.mark.parametrize(
    ("query", "expected_fields", "expected_terms"),
    [
        (
            "现在完成时应该怎么判断？",
            ["标题", "关键词"],
            ["现在完成时", "现在完成时"],
        ),
        ("What does PRESENT PERFECT mean?", ["关键词"], ["present perfect"]),
        ("I have seen it four times.", ["别名"], ["times"]),
        ("Is it already finished?", ["关键词"], ["already"]),
    ],
)
def test_match_knowledge_card_matches_retrieval_metadata(
    tmp_path: Path,
    query: str,
    expected_fields: list[str],
    expected_terms: list[str],
) -> None:
    card = read_knowledge_card(_write_card(tmp_path / "present-perfect.md"))

    hit = match_knowledge_card(query, card)

    assert hit is not None
    assert [match.field for match in hit.matches] == expected_fields
    assert sorted(
        term for match in hit.matches for term in match.terms
    ) == sorted(expected_terms)


def test_match_knowledge_card_uses_ascii_word_boundaries(tmp_path: Path) -> None:
    card = read_knowledge_card(_write_card(tmp_path / "present-perfect.md"))

    assert match_knowledge_card("It happened four times.", card) is not None
    assert match_knowledge_card("Read the timestamp.", card) is None


def test_category_is_context_metadata_not_a_retrieval_term(tmp_path: Path) -> None:
    card = read_knowledge_card(_write_card(tmp_path / "present-perfect.md"))

    assert card.category == "grammar"
    assert match_knowledge_card("请介绍 grammar 分类。", card) is None


def test_retrieve_knowledge_ranks_and_limits_candidates(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    for index in range(5):
        _write_card(
            knowledge_dir / f"card-{index}.md",
            _card_markdown(
                card_id=f"english-card-{index}",
                title=f"卡片 {index}",
                keywords=("shared clue",),
                aliases=("shared",),
            ),
        )
    _write_card(
        knowledge_dir / "best.md",
        _card_markdown(
            card_id="english-best",
            title="Shared clue",
            keywords=("shared clue",),
            aliases=("shared",),
        ),
    )

    hits = retrieve_knowledge("shared clue", knowledge_dir, limit=3)

    assert len(hits) == 3
    assert hits[0].card.card_id == "english-best"
    assert hits[0].score > hits[1].score


def test_retrieve_knowledge_reloads_all_cards_on_every_call(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    card_path = _write_card(knowledge_dir / "present-perfect.md")

    first = retrieve_knowledge("four times", knowledge_dir)
    card_path.write_text(
        CARD_MARKDOWN.replace("  - times\n", "  - completed experience\n"),
        encoding="utf-8",
    )
    second = retrieve_knowledge("four times", knowledge_dir)
    third = retrieve_knowledge("completed experience", knowledge_dir)

    assert first
    assert second == ()
    assert third


def test_citation_and_context_use_stable_id_and_real_evidence(tmp_path: Path) -> None:
    card = read_knowledge_card(_write_card(tmp_path / "present-perfect.md"))
    hit = match_knowledge_card("four times", card)

    assert hit is not None
    citation = citation_from_card(card, ("例句", "易错提醒"))
    context = format_knowledge_context(hit)

    assert citation == {
        "id": "english-grammar-present-perfect",
        "source": "student/knowledge/present-perfect.md",
        "title": "现在完成时",
        "matches": [
            {
                "field": "例句",
                "terms": [],
                "excerpt": "I have read this book three times.",
                "method": "semantic",
            },
            {
                "field": "易错提醒",
                "terms": [],
                "excerpt": "看到次数时，不要误用现在进行时。",
                "method": "semantic",
            },
        ],
    }
    assert '<student_knowledge_card id="english-grammar-present-perfect">' in context
    assert "- 分类：grammar" in context
    assert "- 候选命中 别名：times" in context
    assert "- 核心规则：have/has + 过去分词" in context
    assert context.endswith("</student_knowledge_card>")


def test_title_changes_do_not_change_stable_card_id(tmp_path: Path) -> None:
    original = read_knowledge_card(_write_card(tmp_path / "present-perfect.md"))
    renamed = read_knowledge_card(
        _write_card(
            tmp_path / "present-perfect.md",
            CARD_MARKDOWN.replace("title: 现在完成时", "title: 现在完成时态"),
        )
    )

    assert renamed.title != original.title
    assert renamed.card_id == original.card_id
    assert renamed.card_id == "english-grammar-present-perfect"
