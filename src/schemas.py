"""跨版本共享的数据契约。"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from src.personalization import OwnerMemoryUpdate


class ChatMessage(TypedDict):
    """V1 多轮对话中由调用方保存的一条消息。"""

    role: Literal["user", "assistant"]
    content: str


CitationField = Literal[
    "标题",
    "关键词",
    "别名",
    "学科",
    "年级",
    "核心规则",
    "例句",
    "易错提醒",
]


class CitationMatch(TypedDict):
    """知识卡中实际触发本轮检索的字段。"""

    field: CitationField
    terms: list[str]
    excerpt: str
    method: Literal["exact", "semantic"]


class Citation(TypedDict):
    """一条可追溯到学生知识卡原文的引用。"""

    id: str
    source: str
    title: str
    matches: list[CitationMatch]


WorkflowIntent = Literal[
    "answer",
    "tutor",
    "organize_mistakes",
    "review",
    "cancel",
    "clarify",
]

WorkflowMode = Literal[
    "idle",
    "tutoring",
    "organizing",
    "practice",
    "review_decision",
]

WaitingFor = Literal["student_message", "review_decision"]


class PracticeItem(TypedDict):
    """复盘后只向学生展示题目，其余字段保留在运行期状态中。"""

    question: str
    expected_answer: str
    reasoning: str
    subject: str
    topic: str
    source_record_ids: list[str]


class PendingReport(TypedDict):
    """在模型生成和原子写入之间保存的可恢复报告草稿。"""

    request_id: str
    version: int
    expected_digest: str | None
    markdown: str
    reply_prefix: str
    reply_summary: str


class WorkflowState(TypedDict):
    """V4 中可被 LangGraph checkpoint 序列化的完整状态。"""

    thread_id: str
    messages: list[ChatMessage]
    current_message: str
    intent: WorkflowIntent
    mode: WorkflowMode
    active_problem: str | None
    active_problem_has_error: bool
    pending_review: bool
    skip_unsaved_for_review: bool
    review_request_id: str | None
    pending_report: PendingReport | None
    practice_item: PracticeItem | None
    citations: list[Citation]
    tool_calls: list[dict[str, Any]]
    saved_record_ids: list[str]
    write_receipts: dict[str, str]
    last_reply: str
    trace: list[dict[str, Any]]
    waiting_for: WaitingFor | None
    error: str | None
    turn_index: int


class AgentResult(TypedDict):
    """V0 到 V4 对上层暴露的统一结果结构。"""

    text: str
    stage: str
    tool_calls: list[dict[str, Any]]
    citations: list[Citation]
    trace: list[dict[str, Any]]
    waiting_for: str | None
    error: str | None
    owner_memory_update: OwnerMemoryUpdate | None
    owner_memory_error: str | None


def new_agent_result(
    stage: str,
    *,
    text: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    citations: list[Citation] | None = None,
    trace: list[dict[str, Any]] | None = None,
    waiting_for: WaitingFor | None = None,
    error: str | None = None,
    owner_memory_update: OwnerMemoryUpdate | None = None,
    owner_memory_error: str | None = None,
) -> AgentResult:
    """创建字段完整的 Agent 结果。"""

    return {
        "text": text,
        "stage": stage,
        "tool_calls": list(tool_calls or []),
        "citations": list(citations or []),
        "trace": list(trace or []),
        "waiting_for": waiting_for,
        "error": error,
        "owner_memory_update": owner_memory_update,
        "owner_memory_error": owner_memory_error,
    }
