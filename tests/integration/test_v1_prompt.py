import os

import pytest

from src.facade import invoke
from src.schemas import ChatMessage


def _question_mark_count(text: str) -> int:
    return text.count("？") + text.count("?")


@pytest.mark.integration
def test_v1_answers_non_exercises_normally() -> None:
    if os.getenv("RUN_MODEL_INTEGRATION") != "1":
        pytest.skip("设置 RUN_MODEL_INTEGRATION=1 后才调用真实模型。")

    greeting = invoke("V1", "hi")

    assert greeting["error"] is None, greeting["error"]
    assert greeting["text"].strip()
    assert "___" not in greeting["text"]
    assert "我们先来看一道题" not in greeting["text"]

    explanation = invoke(
        "V1",
        "什么是现在完成时？请直接用一句话解释，不要反问。",
    )

    assert explanation["error"] is None, explanation["error"]
    assert "现在完成时" in explanation["text"]
    assert _question_mark_count(explanation["text"]) == 0

    missing_problem = invoke(
        "V1",
        "我有一道题，但还没有把题目发给你。请只提醒我提供完整题目，不要出题。",
    )

    assert missing_problem["error"] is None, missing_problem["error"]
    assert "题目" in missing_problem["text"]
    assert "___" not in missing_problem["text"]


@pytest.mark.integration
def test_v1_uses_socratic_prompt() -> None:
    if os.getenv("RUN_MODEL_INTEGRATION") != "1":
        pytest.skip("设置 RUN_MODEL_INTEGRATION=1 后才调用真实模型。")

    problem = "She ___ (go) to school every day. 请直接告诉我填空答案。"
    result = invoke("V1", problem)

    assert result["error"] is None, result["error"]
    assert result["text"].strip()
    assert _question_mark_count(result["text"]) == 1
    assert "goes" not in result["text"].lower()

    history: list[ChatMessage] = [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": result["text"]},
    ]
    follow_up = invoke(
        "V1",
        "我注意到了 every day，但不知道它说明什么。",
        history=history,
    )

    assert follow_up["error"] is None, follow_up["error"]
    assert _question_mark_count(follow_up["text"]) == 1
    assert "goes" not in follow_up["text"].lower()

    history.extend(
        [
            {"role": "user", "content": "我注意到了 every day，但不知道它说明什么。"},
            {"role": "assistant", "content": follow_up["text"]},
        ]
    )
    topic_switch = invoke(
        "V1",
        "换个话题：请直接用一句话告诉我，太阳从哪边升起？不要反问。",
        history=history,
    )

    assert topic_switch["error"] is None, topic_switch["error"]
    assert "东" in topic_switch["text"]
    assert _question_mark_count(topic_switch["text"]) == 0
