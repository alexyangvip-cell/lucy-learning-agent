"""正式错题读取与累计学习报告的安全持久化。"""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml

from src.artifacts import PROJECT_ROOT
from src.schemas import PracticeItem
from src.storage import MAX_MARKDOWN_BYTES, MISTAKES_RECORDS_PATH, STUDENT_ROOT


REPORTS_PATH = STUDENT_ROOT / "reports"
LEARNING_REPORT_PATH = REPORTS_PATH / "learning-review.md"
_REPORT_SCHEMA_VERSION = 1
_MISTAKE_FIELDS = (
    "学科",
    "题型",
    "原题",
    "我的答案",
    "正确答案",
    "正确思路",
    "错因",
    "知识点",
    "下次提醒",
)
_MISTAKE_METADATA = {
    "schema_version",
    "id",
    "subject",
    "topic",
    "status",
    "created_at",
    "review_count",
    "next_review_at",
    "source",
}


class ReportDataError(ValueError):
    """正式错题或报告文件不符合稳定数据格式。"""


class NoMistakeRecordsError(ReportDataError):
    """当前没有可用于复盘的正式错题。"""


class ReportConflictError(ValueError):
    """累计报告在本次更新期间被其他操作修改。"""


@dataclass(frozen=True)
class MistakeRecord:
    """从一条正式 Markdown 错题记录中解析出的稳定字段。"""

    mistake_id: str
    subject: str
    topic: str
    status: str
    created_at: str
    review_count: int
    next_review_at: str | None
    source: str
    subject_label: str
    problem_type: str
    original_question: str
    student_answer: str
    correct_answer: str
    correct_reasoning: str
    error_reason: str
    knowledge_point: str
    next_reminder: str
    path: Path
    content_digest: str


@dataclass(frozen=True)
class ReportSnapshot:
    """写入前用于比较并发修改的累计报告快照。"""

    content: str | None
    digest: str | None
    version: int
    request_id: str | None


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _front_matter(content: str, path: Path) -> tuple[dict[str, Any], str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ReportDataError(f"{_display_path(path)} 缺少 YAML front matter。")
    try:
        closing_index = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ReportDataError(
            f"{_display_path(path)} 的 YAML front matter 缺少结束分隔线。"
        ) from exc
    try:
        metadata = yaml.safe_load("\n".join(lines[1:closing_index]))
    except yaml.YAMLError as exc:
        raise ReportDataError(
            f"{_display_path(path)} 的 YAML front matter 无法解析。"
        ) from exc
    if not isinstance(metadata, dict):
        raise ReportDataError(
            f"{_display_path(path)} 的 YAML front matter 必须是键值映射。"
        )
    return metadata, "\n".join(lines[closing_index + 1 :]).strip()


def _required_string(metadata: dict[str, Any], field: str, path: Path) -> str:
    value = metadata.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ReportDataError(
            f"{_display_path(path)} 的 {field} 必须是非空字符串。"
        )
    return value.strip()


def _parse_body(body: str, path: Path) -> dict[str, str]:
    lines = body.splitlines()
    if not lines or lines[0].strip() != "# 错题记录":
        raise ReportDataError(f"{_display_path(path)} 缺少 # 错题记录 标题。")
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if not line.startswith("- ") or "：" not in line:
            continue
        label, value = line[2:].split("：", maxsplit=1)
        if label not in _MISTAKE_FIELDS:
            raise ReportDataError(
                f"{_display_path(path)} 包含不支持的错题字段：{label}。"
            )
        if label in fields:
            raise ReportDataError(
                f"{_display_path(path)} 重复定义了错题字段：{label}。"
            )
        clean_value = value.strip()
        if not clean_value:
            raise ReportDataError(
                f"{_display_path(path)} 的错题字段 {label} 不能为空。"
            )
        fields[label] = clean_value
    missing = [field for field in _MISTAKE_FIELDS if field not in fields]
    if missing:
        raise ReportDataError(
            f"{_display_path(path)} 缺少错题字段：{'、'.join(missing)}。"
        )
    return fields


def _read_mistake_record(path: Path, records_root: Path) -> MistakeRecord:
    display_path = _display_path(path)
    if path.is_symlink():
        raise ReportDataError(f"{display_path} 不能是符号链接。")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReportDataError(f"无法读取 {display_path}。") from exc
    if len(payload) > MAX_MARKDOWN_BYTES:
        raise ReportDataError(f"{display_path} 超过 256 KB，无法用于复盘。")
    try:
        content = payload.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ReportDataError(f"{display_path} 不是 UTF-8 Markdown。") from exc
    metadata, body = _front_matter(content, path)
    unknown = sorted(str(field) for field in set(metadata) - _MISTAKE_METADATA)
    missing = sorted(_MISTAKE_METADATA - set(metadata))
    if unknown:
        raise ReportDataError(
            f"{display_path} 包含不支持的元数据：{'、'.join(unknown)}。"
        )
    if missing:
        raise ReportDataError(
            f"{display_path} 缺少元数据：{'、'.join(missing)}。"
        )
    if type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1:
        raise ReportDataError(f"{display_path} 仅支持 schema_version: 1。")
    mistake_id = _required_string(metadata, "id", path)
    if mistake_id != path.stem:
        raise ReportDataError(f"{display_path} 的 id 必须与文件名一致。")
    subject = _required_string(metadata, "subject", path)
    if subject != path.parent.name:
        raise ReportDataError(f"{display_path} 的 subject 必须与学科目录一致。")
    review_count = metadata["review_count"]
    if type(review_count) is not int or review_count < 0:
        raise ReportDataError(f"{display_path} 的 review_count 必须是非负整数。")
    next_review_at = metadata["next_review_at"]
    if next_review_at is not None and not isinstance(next_review_at, str):
        next_review_at = str(next_review_at)
    source = _required_string(metadata, "source", path)
    source_path = Path(source)
    if source != "chat" and (
        source_path.is_absolute()
        or source_path.parts[:1] != ("inbox",)
        or ".." in source_path.parts
        or source_path.suffix.casefold() != ".md"
    ):
        raise ReportDataError(
            f"{display_path} 的 source 必须是 chat 或 inbox/ 下的 Markdown。"
        )
    fields = _parse_body(body, path)
    try:
        path.resolve().relative_to(records_root.resolve())
    except ValueError as exc:
        raise ReportDataError(f"{display_path} 位于正式错题目录之外。") from exc
    return MistakeRecord(
        mistake_id=mistake_id,
        subject=subject,
        topic=_required_string(metadata, "topic", path),
        status=_required_string(metadata, "status", path),
        created_at=_required_string(metadata, "created_at", path),
        review_count=review_count,
        next_review_at=next_review_at,
        source=source,
        subject_label=fields["学科"],
        problem_type=fields["题型"],
        original_question=fields["原题"],
        student_answer=fields["我的答案"],
        correct_answer=fields["正确答案"],
        correct_reasoning=fields["正确思路"],
        error_reason=fields["错因"],
        knowledge_point=fields["知识点"],
        next_reminder=fields["下次提醒"],
        path=path,
        content_digest=sha256(payload).hexdigest(),
    )


def discover_mistake_records(
    path: str | Path = MISTAKES_RECORDS_PATH,
) -> tuple[MistakeRecord, ...]:
    """读取全部正式错题，任一损坏记录都会终止本次复盘。"""

    records_root = Path(path)
    if records_root.exists() and records_root.is_symlink():
        raise ReportDataError(
            f"{_display_path(records_root)} 不能是符号链接。"
        )
    if not records_root.is_dir():
        raise NoMistakeRecordsError("当前没有正式错题，请先明确要求整理错题。")
    descendants = sorted(records_root.rglob("*"))
    for candidate in descendants:
        if candidate.is_symlink():
            raise ReportDataError(f"{_display_path(candidate)} 不能是符号链接。")
    markdown_paths = [
        candidate
        for candidate in descendants
        if candidate.is_file() and candidate.suffix.casefold() == ".md"
    ]
    if not markdown_paths:
        raise NoMistakeRecordsError("当前没有正式错题，请先明确要求整理错题。")
    for candidate in markdown_paths:
        relative = candidate.relative_to(records_root)
        if len(relative.parts) != 2:
            raise ReportDataError(
                f"{_display_path(candidate)} 必须放在 records/<subject>/ 下。"
            )
    return tuple(
        _read_mistake_record(candidate, records_root)
        for candidate in markdown_paths
    )


def report_source_digest(records: tuple[MistakeRecord, ...]) -> str:
    """计算报告所依据正式记录集合的稳定摘要。"""

    material = "\n".join(
        f"{record.mistake_id}:{record.content_digest}"
        for record in sorted(records, key=lambda item: item.mistake_id)
    )
    return sha256(material.encode("utf-8")).hexdigest()


def render_learning_report(
    records: tuple[MistakeRecord, ...],
    *,
    version: int,
    request_id: str,
    generated_at: str,
    summary: str,
    patterns: list[str],
    action_steps: list[str],
    practice_item: PracticeItem,
) -> str:
    """把已验证数据渲染为不包含练习答案的累计 Markdown 报告。"""

    record_ids = [record.mistake_id for record in records]
    subject_counts = Counter(record.subject for record in records)
    topic_counts = Counter(record.topic for record in records)
    source_ids = practice_item["source_record_ids"]
    if not source_ids or any(source_id not in record_ids for source_id in source_ids):
        raise ReportDataError("个性化题目的来源错题 ID 不属于本次正式记录。")
    if version < 1 or not request_id.strip():
        raise ReportDataError("报告版本和请求 ID 必须有效。")
    lines = [
        "---",
        f"schema_version: {_REPORT_SCHEMA_VERSION}",
        f"version: {version}",
        f"generated_at: {json.dumps(generated_at, ensure_ascii=False)}",
        f"request_id: {json.dumps(request_id, ensure_ascii=False)}",
        f"source_digest: {report_source_digest(records)}",
        "source_record_ids:",
        *(f"  - {json.dumps(record_id, ensure_ascii=False)}" for record_id in record_ids),
        "---",
        "",
        "# 累计学习复盘报告",
        "",
        "## 累计概览",
        "",
        f"- 正式错题：{len(records)} 道",
        "- 学科分布：" + "、".join(
            f"{subject} {count} 道" for subject, count in subject_counts.most_common()
        ),
        "- 知识点分布：" + "、".join(
            f"{topic} {count} 道" for topic, count in topic_counts.most_common()
        ),
        "",
        "## 本次总结",
        "",
        summary.strip(),
        "",
        "## 重点薄弱点",
        "",
        *(f"- {pattern.strip()}" for pattern in patterns),
        "",
        "## 行动建议",
        "",
        *(f"- {step.strip()}" for step in action_steps),
        "",
        "## 本次个性化练习",
        "",
        f"- 学科：{practice_item['subject']}",
        f"- 知识点：{practice_item['topic']}",
        f"- 来源错题：{'、'.join(source_ids)}",
        "",
        practice_item["question"].strip(),
    ]
    return "\n".join(lines).strip() + "\n"


def read_report_snapshot(path: str | Path = LEARNING_REPORT_PATH) -> ReportSnapshot:
    """读取累计报告内容、版本和比较写入所需摘要。"""

    report_path = Path(path)
    if not report_path.exists():
        return ReportSnapshot(content=None, digest=None, version=0, request_id=None)
    if report_path.is_symlink() or not report_path.is_file():
        raise ReportDataError(f"{_display_path(report_path)} 必须是普通文件。")
    try:
        payload = report_path.read_bytes()
        content = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReportDataError(
            f"{_display_path(report_path)} 不是 UTF-8 Markdown。"
        ) from exc
    except OSError as exc:
        raise ReportDataError(f"无法读取 {_display_path(report_path)}。") from exc
    metadata, _ = _front_matter(content, report_path)
    version = metadata.get("version")
    request_id = metadata.get("request_id")
    if type(version) is not int or version < 1:
        raise ReportDataError(f"{_display_path(report_path)} 的 version 无效。")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ReportDataError(f"{_display_path(report_path)} 的 request_id 无效。")
    return ReportSnapshot(
        content=content,
        digest=sha256(payload).hexdigest(),
        version=version,
        request_id=request_id.strip(),
    )


def save_report_atomic(
    path: str | Path,
    content: str,
    *,
    expected_digest: str | None,
    allowed_root: str | Path = STUDENT_ROOT,
) -> ReportSnapshot:
    """在摘要仍匹配时用同目录临时文件原子替换累计报告。"""

    clean_content = content.strip()
    if not clean_content:
        raise ReportDataError("累计报告内容为空，未写入文件。")
    root_path = Path(allowed_root).resolve()
    report_path = Path(path)
    target_path = report_path.resolve()
    try:
        target_path.relative_to(root_path)
    except ValueError as exc:
        raise ReportDataError(f"只能写入 {root_path} 目录及其子目录。") from exc
    if report_path.is_symlink():
        raise ReportDataError(f"{_display_path(report_path)} 不能是符号链接。")
    current_digest = None
    if report_path.exists():
        if not report_path.is_file():
            raise ReportDataError(f"{_display_path(report_path)} 必须是普通文件。")
        current_digest = sha256(report_path.read_bytes()).hexdigest()
    if current_digest != expected_digest:
        raise ReportConflictError("累计报告已被其他操作修改，本次更新已停止。")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report_path.parent.resolve().relative_to(root_path)
    except ValueError as exc:
        raise ReportDataError("报告目录解析到了允许目录之外，未写入文件。") from exc
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=report_path.parent,
            prefix=f".{report_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(f"{clean_content}\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temp_path = Path(temporary.name)
        os.replace(temp_path, report_path)
    except OSError as exc:
        raise ReportDataError(
            f"无法写入 {_display_path(report_path)}，原报告保持不变。"
        ) from exc
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return read_report_snapshot(report_path)
