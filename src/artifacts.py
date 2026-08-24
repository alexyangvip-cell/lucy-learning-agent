"""学生可编辑 Markdown 文件的读取与校验。"""

from dataclasses import dataclass
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "student" / "prompt.md"
PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "student" / "templates" / "prompt.md"
V4_PROMPT_PATH = PROJECT_ROOT / "student" / "v4-prompt.md"
SKILLS_PATH = PROJECT_ROOT / "student" / "skill"


@dataclass(frozen=True)
class SkillMetadata:
    """Agent 启动时可见的 Skill 元数据。"""

    name: str
    description: str
    path: Path


@dataclass(frozen=True)
class SkillArtifact:
    """按需加载后的完整 Skill。"""

    metadata: SkillMetadata
    instructions: str


class ArtifactError(ValueError):
    """学生文件缺失、为空或无法读取。"""


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def read_markdown(path: str | Path) -> str:
    """使用 UTF-8 读取非空 Markdown，每次调用都重新访问磁盘。"""

    markdown_path = Path(path)
    display_path = _display_path(markdown_path)
    try:
        content = markdown_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ArtifactError(
            f"未找到 {display_path}。请从课程包恢复该文件后重试。"
        ) from exc
    except UnicodeDecodeError as exc:
        raise ArtifactError(
            f"无法读取 {display_path}：文件不是 UTF-8 编码。"
            "请将文件另存为 UTF-8 后重试。"
        ) from exc
    except OSError as exc:
        raise ArtifactError(
            f"无法读取 {display_path}。请确认它是可访问的 Markdown 文件后重试。"
        ) from exc

    clean_content = content.strip()
    if not clean_content:
        raise ArtifactError(
            f"{display_path} 内容为空。请填写内容、保存文件后重试。"
        )
    return clean_content


_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _frontmatter_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value


def read_skill(path: str | Path) -> SkillArtifact:
    """读取并校验课程支持的标准 SKILL.md。"""

    skill_path = Path(path)
    display_path = _display_path(skill_path)
    content = read_markdown(skill_path)
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ArtifactError(
            f"{display_path} 缺少 YAML frontmatter。"
            "请在文件开头用 --- 包围 name 和 description。"
        )

    try:
        closing_index = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ArtifactError(
            f"{display_path} 的 YAML frontmatter 没有结束标记。"
            "请在元数据后补充一行 ---。"
        ) from exc

    fields: dict[str, str] = {}
    for line in lines[1:closing_index]:
        if not line.strip() or line[:1].isspace():
            continue
        key, separator, raw_value = line.partition(":")
        normalized_key = key.strip()
        if separator and normalized_key in {"name", "description"}:
            if normalized_key in fields:
                raise ArtifactError(
                    f"{display_path} 重复定义了 {normalized_key}。"
                    "请只保留一个字段。"
                )
            fields[normalized_key] = _frontmatter_value(raw_value)

    missing_fields = [
        field_name
        for field_name in ("name", "description")
        if not fields.get(field_name)
    ]
    if missing_fields:
        missing = "、".join(missing_fields)
        raise ArtifactError(
            f"{display_path} 缺少必填字段：{missing}。"
            "请补全 YAML frontmatter 后重试。"
        )

    name = fields["name"]
    description = fields["description"]
    if len(name) > 64 or not _SKILL_NAME_PATTERN.fullmatch(name):
        raise ArtifactError(
            f"{display_path} 的 name 不符合规范。"
            "请使用不超过 64 个字符的小写字母、数字和单个连字符。"
        )
    if skill_path.parent.name != name:
        raise ArtifactError(
            f"{display_path} 的 name 必须与父目录 {skill_path.parent.name} 一致。"
            f"请将 name 改为 {skill_path.parent.name}，或重命名 Skill 目录。"
        )
    if len(description) > 1024:
        raise ArtifactError(
            f"{display_path} 的 description 超过 1024 个字符。"
            "请缩短用途和触发条件说明。"
        )

    instructions = "\n".join(lines[closing_index + 1 :]).strip()
    if not instructions:
        raise ArtifactError(
            f"{display_path} 缺少 Skill 操作步骤。"
            "请在 YAML frontmatter 后填写 Markdown 指令。"
        )

    metadata = SkillMetadata(
        name=name,
        description=description,
        path=skill_path,
    )
    return SkillArtifact(metadata=metadata, instructions=instructions)


def discover_skills(path: str | Path = SKILLS_PATH) -> list[SkillMetadata]:
    """发现 Skill 目录，并只向 Agent 暴露名称、描述和文件位置。"""

    skills_path = Path(path)
    display_path = _display_path(skills_path)
    if not skills_path.is_dir():
        raise ArtifactError(
            f"未找到 {display_path}。请从课程包恢复 Skill 目录后重试。"
        )

    skill_files = sorted(
        candidate / "SKILL.md"
        for candidate in skills_path.iterdir()
        if candidate.is_dir() and (candidate / "SKILL.md").is_file()
    )
    if not skill_files:
        raise ArtifactError(
            f"{display_path} 中没有可用的 SKILL.md。"
            "请创建以 Skill 名称命名的子目录，并在其中添加 SKILL.md。"
        )

    skills: list[SkillMetadata] = []
    seen_names: set[str] = set()
    for skill_file in skill_files:
        metadata = read_skill(skill_file).metadata
        if metadata.name in seen_names:
            raise ArtifactError(
                f"{display_path} 中存在重复的 Skill 名称 {metadata.name}。"
                "请为每个 Skill 使用唯一名称。"
            )
        seen_names.add(metadata.name)
        skills.append(metadata)
    return skills
