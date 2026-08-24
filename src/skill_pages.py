"""Skill 专属页面之间共享的轻量路由数据。"""

from collections.abc import Sequence
from typing import Literal, TypedDict

from src.schemas import AgentResult, ChatMessage


ENGLISH_QUEST_NAME = "english-quest"
ENGLISH_QUEST_PAGE = "app_pages/english_quest.py"


class EnglishQuestSession(TypedDict):
    """课程页和英语闯关页之间传递的会话。"""

    history: list[ChatMessage]
    agent_stage: Literal["V2", "V3"]
    return_page: str
    return_label: str
    source_history_key: str | None
    source_stage: str | None


def result_loaded_skill(result: AgentResult | dict, skill_name: str) -> bool:
    """只依据真实 load_skill 工具调用判断是否进入专属页面。"""

    for call in result.get("tool_calls") or []:
        args = call.get("args")
        if (
            call.get("name") == "load_skill"
            and isinstance(args, dict)
            and args.get("skill_name") == skill_name
        ):
            return True
    return False


def new_english_quest_session(
    *,
    history: Sequence[ChatMessage] = (),
    agent_stage: Literal["V2", "V3"] = "V2",
    return_page: str = "app_pages/lesson_1.py",
    return_label: str = "返回第一课",
    source_history_key: str | None = None,
    source_stage: str | None = None,
) -> EnglishQuestSession:
    """创建可序列化的页面交接数据，并隔离调用方的历史列表。"""

    return {
        "history": [item.copy() for item in history],
        "agent_stage": agent_stage,
        "return_page": return_page,
        "return_label": return_label,
        "source_history_key": source_history_key,
        "source_stage": source_stage,
    }
