from src.schemas import new_agent_result
from src.skill_pages import new_english_quest_session, result_loaded_skill


def test_result_loaded_skill_requires_matching_real_tool_call() -> None:
    result = new_agent_result(
        "V2",
        text="侦探闯关开始。",
        tool_calls=[
            {
                "name": "load_skill",
                "args": {"skill_name": "english-quest"},
            }
        ],
    )

    assert result_loaded_skill(result, "english-quest") is True
    assert result_loaded_skill(result, "sorting-out-mistakes") is False
    assert result_loaded_skill(
        new_agent_result("V2", text="english-quest"),
        "english-quest",
    ) is False
    malformed = new_agent_result(
        "V2",
        tool_calls=[{"name": "load_skill", "args": None}],
    )
    assert result_loaded_skill(malformed, "english-quest") is False


def test_new_english_quest_session_copies_source_history() -> None:
    history = [{"role": "user", "content": "开始闯关"}]

    session = new_english_quest_session(
        history=history,
        agent_stage="V3",
        return_page="app_pages/lesson_2.py",
        return_label="返回第二课",
        source_history_key="lesson2_histories",
        source_stage="V3",
    )
    history[0]["content"] = "外部修改"

    assert session == {
        "history": [{"role": "user", "content": "开始闯关"}],
        "agent_stage": "V3",
        "return_page": "app_pages/lesson_2.py",
        "return_label": "返回第二课",
        "source_history_key": "lesson2_histories",
        "source_stage": "V3",
    }
