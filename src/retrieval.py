"""V3 YAML Markdown knowledge-card discovery, retrieval, and citations."""

from dataclasses import dataclass
from html import escape
from pathlib import Path
import re
from typing import Literal
import unicodedata

import yaml

from src.artifacts import ArtifactError, PROJECT_ROOT, read_markdown
from src.schemas import Citation, CitationField, CitationMatch


KNOWLEDGE_DIRECTORY = PROJECT_ROOT / "student" / "knowledge"
KNOWLEDGE_CARD_PATH = (
    KNOWLEDGE_DIRECTORY / "english" / "grammar" / "present-perfect.md"
)

_METADATA_FIELDS = {
    "schema_version",
    "id",
    "title",
    "subject",
    "category",
    "grade",
    "language",
    "keywords",
    "aliases",
}
_REQUIRED_METADATA_FIELDS = _METADATA_FIELDS
_SECTION_LEVELS = {"核心规则": 1, "例句": 2, "易错提醒": 2}
_SECTION_FIELDS: dict[str, CitationField] = {
    "核心规则": "核心规则",
    "例句": "例句",
    "易错提醒": "易错提醒",
}
_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_ASCII_TERM_PATTERN = re.compile(r"^[a-z0-9]+(?:\s+[a-z0-9]+)*$")
EvidenceField = Literal["核心规则", "例句", "易错提醒"]


class KnowledgeCardError(ArtifactError):
    """The knowledge-card collection does not match the course schema."""


@dataclass(frozen=True)
class KnowledgeCard:
    """One validated YAML front matter Markdown knowledge card."""

    card_id: str
    title: str
    subject: str
    category: str
    grade: str
    language: str
    keywords: tuple[str, ...]
    aliases: tuple[str, ...]
    core_rule: str
    example: str
    common_mistake: str
    source: str


@dataclass(frozen=True)
class KnowledgeFieldMatch:
    """One deterministic metadata match used for candidate generation."""

    field: CitationField
    terms: tuple[str, ...]
    excerpt: str


@dataclass(frozen=True)
class KnowledgeHit:
    """A ranked candidate produced by deterministic metadata matching."""

    card: KnowledgeCard
    matches: tuple[KnowledgeFieldMatch, ...]
    score: int


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _knowledge_source(path: Path) -> str:
    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError:
        relative = path
    parts = relative.parts
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == ("student", "knowledge"):
            return Path(*parts[index:]).as_posix()
    return f"student/knowledge/{path.name}"


def _front_matter(content: str, display_path: str) -> tuple[dict[str, object], str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise KnowledgeCardError(f"{display_path} 缺少 YAML front matter。")

    try:
        closing_index = next(
            index for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise KnowledgeCardError(
            f"{display_path} 的 YAML front matter 缺少结束分隔线。"
        ) from exc

    yaml_text = "\n".join(lines[1:closing_index])
    try:
        metadata = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise KnowledgeCardError(
            f"{display_path} 的 YAML front matter 无法解析。"
        ) from exc
    if not isinstance(metadata, dict):
        raise KnowledgeCardError(
            f"{display_path} 的 YAML front matter 必须是键值映射。"
        )
    return metadata, "\n".join(lines[closing_index + 1 :]).strip()


def _string_metadata(
    metadata: dict[str, object],
    field: str,
    display_path: str,
) -> str:
    value = metadata[field]
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeCardError(
            f"{display_path} 元数据 {field} 必须是非空字符串。"
        )
    return value.strip()


def _string_list_metadata(
    metadata: dict[str, object],
    field: str,
    display_path: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = metadata[field]
    if (
        not isinstance(value, list)
        or (not value and not allow_empty)
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise KnowledgeCardError(
            f"{display_path} 元数据 {field} 必须是字符串列表。"
        )

    values: list[str] = []
    seen: set[str] = set()
    for item in value:
        clean_item = item.strip()
        normalized = _normalize(clean_item)
        if normalized in seen:
            continue
        seen.add(normalized)
        values.append(clean_item)
    return tuple(values)


def _markdown_sections(body: str, display_path: str) -> dict[str, str]:
    matches = list(_HEADING_PATTERN.finditer(body))
    if matches and body[: matches[0].start()].strip():
        raise KnowledgeCardError(
            f"{display_path} 的正文必须从 # 核心规则 开始。"
        )
    section_matches: list[tuple[str, re.Match[str]]] = []
    active_section_level: int | None = None
    seen_sections: set[str] = set()
    for match in matches:
        level = len(match.group(1))
        name = match.group(2).strip()
        expected_level = _SECTION_LEVELS.get(name)
        if expected_level is None:
            if active_section_level is not None and level > active_section_level:
                continue
            raise KnowledgeCardError(
                f"{display_path} 包含不支持的正文标题：{match.group(0)}。"
            )
        if level != expected_level:
            raise KnowledgeCardError(
                f"{display_path} 包含不支持的正文标题：{match.group(0)}。"
            )
        if name in seen_sections:
            raise KnowledgeCardError(f"{display_path} 重复定义了正文段落 {name}。")
        seen_sections.add(name)
        section_matches.append((name, match))
        active_section_level = level

    sections: dict[str, str] = {}
    for index, (name, match) in enumerate(section_matches):
        start = match.end()
        end = (
            section_matches[index + 1][1].start()
            if index + 1 < len(section_matches)
            else len(body)
        )
        value = body[start:end].strip()
        if not value:
            raise KnowledgeCardError(f"{display_path} 正文段落 {name} 的内容为空。")
        sections[name] = value

    missing = [name for name in _SECTION_LEVELS if name not in sections]
    if missing:
        raise KnowledgeCardError(
            f"{display_path} 缺少必填正文段落：{'、'.join(missing)}。"
        )
    if [name for name, _ in section_matches] != list(_SECTION_LEVELS):
        raise KnowledgeCardError(
            f"{display_path} 正文段落必须按核心规则、例句、易错提醒排列。"
        )
    return sections


def read_knowledge_card(path: str | Path = KNOWLEDGE_CARD_PATH) -> KnowledgeCard:
    """Read and strictly validate one YAML front matter Markdown card."""

    card_path = Path(path)
    display_path = _display_path(card_path)
    if card_path.is_symlink():
        raise KnowledgeCardError(f"{display_path} 不能是符号链接。")
    if (
        card_path.suffix != ".md"
        or _ID_PATTERN.fullmatch(card_path.stem) is None
    ):
        raise KnowledgeCardError(
            f"{display_path} 文件名必须使用小写英文 kebab-case，并以 .md 结尾。"
        )
    content = read_markdown(card_path)
    metadata, body = _front_matter(content, display_path)

    if "schema_version" not in metadata:
        raise KnowledgeCardError(
            f"{display_path} 缺少必填元数据：schema_version。"
        )
    schema_version = metadata["schema_version"]
    if type(schema_version) is not int or schema_version != 2:
        raise KnowledgeCardError(f"{display_path} 仅支持 schema_version: 2。")

    unknown = sorted(str(field) for field in set(metadata) - _METADATA_FIELDS)
    if unknown:
        raise KnowledgeCardError(
            f"{display_path} 不支持的元数据字段：{'、'.join(unknown)}。"
        )
    missing = sorted(_REQUIRED_METADATA_FIELDS - set(metadata))
    if missing:
        raise KnowledgeCardError(
            f"{display_path} 缺少必填元数据：{'、'.join(missing)}。"
        )
    card_id = _string_metadata(metadata, "id", display_path)
    if _ID_PATTERN.fullmatch(card_id) is None:
        raise KnowledgeCardError(
            f"{display_path} 元数据 id 必须使用小写英文 kebab-case。"
        )
    subject = _string_metadata(metadata, "subject", display_path)
    if _ID_PATTERN.fullmatch(subject) is None:
        raise KnowledgeCardError(
            f"{display_path} 元数据 subject 必须使用小写英文 kebab-case。"
        )
    category = _string_metadata(metadata, "category", display_path)
    if _ID_PATTERN.fullmatch(category) is None:
        raise KnowledgeCardError(
            f"{display_path} 元数据 category 必须使用小写英文 kebab-case。"
        )
    keywords = _string_list_metadata(metadata, "keywords", display_path)
    aliases = _string_list_metadata(
        metadata,
        "aliases",
        display_path,
        allow_empty=True,
    )
    sections = _markdown_sections(body, display_path)

    return KnowledgeCard(
        card_id=card_id,
        title=_string_metadata(metadata, "title", display_path),
        subject=subject,
        category=category,
        grade=_string_metadata(metadata, "grade", display_path),
        language=_string_metadata(metadata, "language", display_path),
        keywords=keywords,
        aliases=aliases,
        core_rule=sections["核心规则"],
        example=sections["例句"],
        common_mistake=sections["易错提醒"],
        source=_knowledge_source(card_path),
    )


def discover_knowledge_cards(
    path: str | Path = KNOWLEDGE_DIRECTORY,
) -> tuple[KnowledgeCard, ...]:
    """Read all nested Markdown cards, or one explicit card file."""

    knowledge_path = Path(path)
    display_path = _display_path(knowledge_path)
    if not knowledge_path.exists():
        raise ArtifactError(f"文件或目录不存在：{display_path}")
    if knowledge_path.is_symlink():
        raise KnowledgeCardError(f"{display_path} 不能是符号链接。")
    if knowledge_path.is_file():
        return (read_knowledge_card(knowledge_path),)
    if not knowledge_path.is_dir():
        raise ArtifactError(f"知识库路径不是文件或目录：{display_path}")

    descendants = sorted(knowledge_path.rglob("*"))
    for candidate in descendants:
        if candidate.is_symlink():
            raise KnowledgeCardError(
                f"{_display_path(candidate)} 不能是符号链接。"
            )
    card_paths = [
        candidate
        for candidate in descendants
        if candidate.is_file() and candidate.suffix == ".md"
    ]
    if not card_paths:
        raise KnowledgeCardError(f"{display_path} 没有找到 Markdown 知识卡。")

    cards = tuple(read_knowledge_card(card_path) for card_path in card_paths)
    seen_ids: dict[str, str] = {}
    for card in cards:
        previous_source = seen_ids.get(card.card_id)
        if previous_source is not None:
            raise KnowledgeCardError(
                f"重复的知识卡 id {card.card_id}："
                f"{previous_source}、{card.source}。"
            )
        seen_ids[card.card_id] = card.source
    return cards


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _contains_term(query: str, term: str) -> bool:
    normalized_query = _normalize(query)
    normalized_term = _normalize(term)
    if not normalized_query or not normalized_term:
        return False
    if _ASCII_TERM_PATTERN.fullmatch(normalized_term):
        phrase = r"\s+".join(re.escape(word) for word in normalized_term.split())
        return (
            re.search(
                rf"(?<![a-z0-9]){phrase}(?![a-z0-9])",
                normalized_query,
            )
            is not None
        )
    return normalized_term in normalized_query


def match_knowledge_card(query: str, card: KnowledgeCard) -> KnowledgeHit | None:
    """Generate a candidate from exact title, keyword, and alias metadata."""

    matches: list[KnowledgeFieldMatch] = []
    score = 0
    if _contains_term(query, card.title):
        matches.append(
            KnowledgeFieldMatch("标题", (card.title,), card.title)
        )
        score += 100

    matched_keywords = tuple(
        term for term in card.keywords if _contains_term(query, term)
    )
    if matched_keywords:
        matches.append(
            KnowledgeFieldMatch("关键词", matched_keywords, "、".join(card.keywords))
        )
        score += 60 * len(matched_keywords)

    matched_aliases = tuple(
        term for term in card.aliases if _contains_term(query, term)
    )
    if matched_aliases:
        matches.append(
            KnowledgeFieldMatch("别名", matched_aliases, "、".join(card.aliases))
        )
        score += 20 * len(matched_aliases)

    if not matches:
        return None
    return KnowledgeHit(card=card, matches=tuple(matches), score=score)


def retrieve_knowledge(
    query: str,
    path: str | Path = KNOWLEDGE_DIRECTORY,
    *,
    limit: int = 3,
) -> tuple[KnowledgeHit, ...]:
    """Reload, rank, and cap deterministic candidates for this turn."""

    if limit < 1:
        raise ValueError("limit 必须大于等于 1。")
    hits = [
        hit
        for card in discover_knowledge_cards(path)
        if (hit := match_knowledge_card(query, card)) is not None
    ]
    hits.sort(key=lambda hit: (-hit.score, hit.card.card_id))
    return tuple(hits[:limit])


def _single_line(value: str) -> str:
    return escape(" ".join(value.split()), quote=False)


def format_knowledge_context(hit: KnowledgeHit) -> str:
    """Inject one complete, explicitly untrusted candidate into the prompt."""

    card = hit.card
    lines = [
        f'<student_knowledge_card id="{card.card_id}">',
        f"- 来源：{_single_line(card.source)}",
        f"- 标题：{_single_line(card.title)}",
        f"- 学科：{_single_line(card.subject)}",
        f"- 分类：{_single_line(card.category)}",
        f"- 年级：{_single_line(card.grade)}",
        f"- 语言：{_single_line(card.language)}",
        f"- 关键词：{_single_line('、'.join(card.keywords))}",
        f"- 别名：{_single_line('、'.join(card.aliases))}",
    ]
    lines.extend(
        f"- 候选命中 {match.field}：{_single_line('、'.join(match.terms))}"
        for match in hit.matches
    )
    lines.extend(
        [
            f"- 核心规则：{_single_line(card.core_rule)}",
            f"- 例句：{_single_line(card.example)}",
            f"- 易错提醒：{_single_line(card.common_mistake)}",
            "</student_knowledge_card>",
        ]
    )
    return "\n".join(lines)


def citation_from_card(
    card: KnowledgeCard,
    evidence_fields: tuple[EvidenceField, ...],
) -> Citation:
    """Build a citation only from validated original card evidence."""

    evidence = {
        "核心规则": card.core_rule,
        "例句": card.example,
        "易错提醒": card.common_mistake,
    }
    matches: list[CitationMatch] = [
        {
            "field": _SECTION_FIELDS[field],
            "terms": [],
            "excerpt": evidence[field],
            "method": "semantic",
        }
        for field in evidence_fields
    ]
    return {
        "id": card.card_id,
        "source": card.source,
        "title": card.title,
        "matches": matches,
    }
