from base64 import b64decode
from pathlib import Path
from unittest.mock import Mock, call

import pytest
import streamlit as st
from streamlit.proto.ChatInput_pb2 import ChatInput as ChatInputProto
from streamlit.testing.v1 import AppTest

import src.chat_submission as chat_submission_module
import src.facade as facade_module
import src.progress as progress_module
from src.chat_submission import (
    ACCEPTED_FILE_TYPES,
    MAX_ATTACHMENT_BYTES,
    ChatAttachment,
    ChatSubmission,
    ChatSubmissionError,
)
from src.schemas import new_agent_result


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAGE_PATH = PROJECT_ROOT / "app_pages" / "lesson_2.py"
_TINY_PNG = b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8A"
    "AQUBAScY42YAAAAASUVORK5CYII="
)


def _configure_page(monkeypatch):
    invoke = Mock(
        return_value=new_agent_result(
            "V3",
            text="先观察题目里的次数线索。",
        )
    )
    chat_v4 = Mock(
        return_value=new_agent_result(
            "V4",
            text="你想先从哪个线索开始？",
            waiting_for="student_message",
        )
    )
    load_progress = Mock(return_value=progress_module.default_progress())
    save_progress = Mock(side_effect=lambda progress: progress)

    monkeypatch.setattr(facade_module, "invoke", invoke)
    monkeypatch.setattr(facade_module, "chat_v4", chat_v4)
    monkeypatch.setattr(progress_module, "load_progress", load_progress)
    monkeypatch.setattr(progress_module, "save_progress", save_progress)
    return invoke, chat_v4, load_progress, save_progress


def _widget_by_label(elements, label: str):
    return next(element for element in elements if element.label == label)


def _image_submission(*, text: str = "") -> ChatSubmission:
    return ChatSubmission(
        text=text,
        attachment=ChatAttachment(
            name="question.png",
            media_type="image/png",
            data=_TINY_PNG,
        ),
    )


def test_lesson_2_non_chat_actions_never_call_agents(monkeypatch) -> None:
    invoke, chat_v4, load_progress, save_progress = _configure_page(monkeypatch)

    app = AppTest.from_file(PAGE_PATH).run()

    assert not app.exception
    assert not app.main.title
    assert app.sidebar.header[0].value == "第二课"
    assert not app.main.segmented_control
    assert len(app.sidebar.segmented_control) == 1
    assert not app.main.expander
    assert len(app.main.chat_message) == 1
    assert not app.sidebar.chat_message
    assert app.chat_input[0].placeholder == "问知识点，或发送一道具体题"
    assert app.chat_input[0].proto.accept_file == ChatInputProto.SINGLE
    assert list(app.chat_input[0].proto.file_type) == [
        f".{extension}" for extension in ACCEPTED_FILE_TYPES
    ]
    assert app.chat_input[0].proto.max_upload_size_mb == (
        MAX_ATTACHMENT_BYTES // (1024 * 1024)
    )
    thread_id = app.session_state["thread_id"]
    assert thread_id.startswith("student-")
    invoke.assert_not_called()
    chat_v4.assert_not_called()
    save_progress.assert_not_called()
    load_progress.assert_called_once_with()

    app.run()
    app.segmented_control[0].set_value("V4").run()
    app.segmented_control[0].set_value("V3").run()

    assert not app.exception
    assert app.session_state["thread_id"] == thread_id
    invoke.assert_not_called()
    chat_v4.assert_not_called()
    save_progress.assert_not_called()
    load_progress.assert_called_once_with()


def test_lesson_2_v3_uses_history_and_displays_evidence(monkeypatch) -> None:
    invoke, chat_v4, _, save_progress = _configure_page(monkeypatch)
    invoke.side_effect = [
        new_agent_result(
            "V3",
            text=(
                "three times 表示动作发生过多次。\n\n"
                "知识依据：[english-grammar-present-perfect] 现在完成时"
            ),
            citations=[
                {
                    "id": "english-grammar-present-perfect",
                    "source": (
                        "student/knowledge/english/grammar/present-perfect.md"
                    ),
                    "title": "现在完成时",
                    "matches": [
                        {
                            "field": "例句",
                            "terms": ["three times"],
                            "excerpt": "I have read this book three times.",
                            "method": "exact",
                        }
                    ],
                }
            ],
            trace=[
                {
                    "step": "knowledge_retrieval",
                    "status": "hit",
                    "used_card_ids": ["english-grammar-present-perfect"],
                }
            ],
        ),
        new_agent_result("V3", text="对，次数和现在有关。"),
    ]
    app = AppTest.from_file(PAGE_PATH).run()

    app.chat_input[0].set_value("three times 是什么线索？").run()

    assert not app.exception
    invoke.assert_called_once_with(
        "V3",
        "three times 是什么线索？",
        history=[],
    )
    chat_v4.assert_not_called()
    save_progress.assert_called_once()
    assert "v3" in app.session_state["progress"]["completed_modules"]
    visible_text = [item.value for item in app.markdown]
    assert any("现在完成时" in item for item in visible_text)
    assert any("english-grammar-present-perfect" in item for item in visible_text)
    assert any("I have read this book three times." in item for item in visible_text)

    app.chat_input[0].set_value("所以应该用 have 吗？").run()

    assert invoke.call_count == 2
    assert invoke.call_args_list[1] == call(
        "V3",
        "所以应该用 have 吗？",
        history=[
            {"role": "user", "content": "three times 是什么线索？"},
            {
                "role": "assistant",
                "content": (
                    "three times 表示动作发生过多次。\n\n"
                    "知识依据：[english-grammar-present-perfect] 现在完成时"
                ),
            },
        ],
    )
    chat_v4.assert_not_called()

    app.run()

    assert invoke.call_count == 2
    assert save_progress.call_count == 2


def test_lesson_2_displays_loaded_skill_name(monkeypatch) -> None:
    switch_page = Mock()
    monkeypatch.setattr(st, "switch_page", switch_page)
    invoke, chat_v4, _, _ = _configure_page(monkeypatch)
    invoke.return_value = new_agent_result(
        "V3",
        text="侦探闯关开始。",
        tool_calls=[
            {
                "name": "load_skill",
                "args": {"skill_name": "english-quest"},
            }
        ],
    )
    app = AppTest.from_file(PAGE_PATH).run()

    app.chat_input[0].set_value("玩侦探闯关练英语").run()

    assert not app.exception
    invoke.assert_called_once_with("V3", "玩侦探闯关练英语", history=[])
    chat_v4.assert_not_called()
    visible_text = [item.value for item in app.markdown]
    assert any("english-quest" in item for item in visible_text)
    switch_page.assert_called_once_with("app_pages/english_quest.py")
    quest_session = app.session_state["english_quest_session"]
    assert quest_session["agent_stage"] == "V3"
    assert quest_session["source_history_key"] == "lesson2_histories"
    assert quest_session["history"] == app.session_state[
        "lesson2_histories"
    ]["V3"]


def test_lesson_2_chat_keeps_agent_left_and_student_right(monkeypatch) -> None:
    invoke, _, _, _ = _configure_page(monkeypatch)
    invoke.return_value = new_agent_result("V3", text="先找动词。")
    app = AppTest.from_file(PAGE_PATH).run()

    app.chat_input[0].set_value("这道题怎么做？").run()

    chat_rows = [
        node
        for node in app.main.children.values()
        if node.type == "flex_container"
    ]
    alignments = [
        row.proto.flex_container.Align.Name(row.proto.flex_container.align)
        for row in chat_rows
    ]
    assert alignments == ["ALIGN_START", "ALIGN_END", "ALIGN_START"]
    assert [message.name for message in app.main.chat_message] == [
        "assistant",
        "user",
        "assistant",
    ]


def test_lesson_2_v4_uses_one_thread_and_separate_visible_history(
    monkeypatch,
) -> None:
    invoke, chat_v4, _, save_progress = _configure_page(monkeypatch)
    chat_v4.side_effect = [
        new_agent_result(
            "V4",
            text="你先试着判断它考查什么。",
            waiting_for="student_message",
        ),
        new_agent_result(
            "V4",
            text="这个判断依据是什么？",
            waiting_for="student_message",
        ),
    ]
    app = AppTest.from_file(PAGE_PATH).run()
    app.segmented_control[0].set_value("V4").run()
    thread_id = app.session_state["thread_id"]

    app.chat_input[0].set_value("我有一道数学题").run()
    app.chat_input[0].set_value("我觉得考查勾股定理").run()

    assert not app.exception
    invoke.assert_not_called()
    assert chat_v4.call_args_list == [
        call("我有一道数学题", thread_id),
        call("我觉得考查勾股定理", thread_id),
    ]
    assert app.session_state["lesson2_histories"]["V3"] == []
    assert app.session_state["lesson2_histories"]["V4"] == [
        {"role": "user", "content": "我有一道数学题"},
        {"role": "assistant", "content": "你先试着判断它考查什么。"},
        {"role": "user", "content": "我觉得考查勾股定理"},
        {"role": "assistant", "content": "这个判断依据是什么？"},
    ]
    assert "v4" in app.session_state["progress"]["completed_modules"]
    assert save_progress.call_count == 2

    app.run()

    assert chat_v4.call_count == 2
    assert save_progress.call_count == 2


def test_lesson_2_v3_attachment_is_forwarded_once_and_history_is_text_only(
    monkeypatch,
) -> None:
    invoke, chat_v4, _, save_progress = _configure_page(monkeypatch)
    submission = _image_submission(text="请看图里的题目")
    parser = Mock(return_value=submission)
    monkeypatch.setattr(chat_submission_module, "parse_chat_submission", parser)
    invoke.return_value = new_agent_result("V3", text="先观察图里的已知条件。")
    app = AppTest.from_file(PAGE_PATH).run()

    app.chat_input[0].set_value("测试附件提交").run()

    assert not app.exception
    parser.assert_called_once()
    invoke.assert_called_once_with(
        "V3",
        "请看图里的题目",
        history=[],
        attachment=submission.attachment,
    )
    chat_v4.assert_not_called()
    save_progress.assert_called_once()
    assert app.session_state["lesson2_histories"]["V3"] == [
        {
            "role": "user",
            "content": "请看图里的题目\n\n附件：question.png",
        },
        {"role": "assistant", "content": "先观察图里的已知条件。"},
    ]

    app.run()

    assert invoke.call_count == 1


def test_lesson_2_attachment_parse_error_never_calls_agents_or_saves_progress(
    monkeypatch,
) -> None:
    invoke, chat_v4, _, save_progress = _configure_page(monkeypatch)
    parser = Mock(side_effect=ChatSubmissionError("附件内容已经损坏。"))
    monkeypatch.setattr(chat_submission_module, "parse_chat_submission", parser)
    app = AppTest.from_file(PAGE_PATH).run()

    app.chat_input[0].set_value("尝试上传附件").run()

    assert not app.exception
    parser.assert_called_once()
    invoke.assert_not_called()
    chat_v4.assert_not_called()
    save_progress.assert_not_called()
    assert app.session_state["lesson2_histories"]["V3"] == []
    assert "附件内容已经损坏" in app.error[0].value


def test_lesson_2_v4_attachment_model_error_requires_reselection(
    monkeypatch,
) -> None:
    _, chat_v4, _, save_progress = _configure_page(monkeypatch)
    submission = _image_submission()
    monkeypatch.setattr(
        chat_submission_module,
        "parse_chat_submission",
        Mock(return_value=submission),
    )
    chat_v4.return_value = new_agent_result("V4", error="工作流暂时无法恢复。")
    app = AppTest.from_file(PAGE_PATH).run()
    app.segmented_control[0].set_value("V4").run()
    thread_id = app.session_state["thread_id"]

    app.chat_input[0].set_value("测试附件失败").run()

    assert not app.exception
    chat_v4.assert_called_once_with(
        "",
        thread_id,
        attachment=submission.attachment,
    )
    assert app.session_state["lesson2_histories"]["V4"] == []
    assert "附件未保留，请重新选择附件后发送" in app.error[0].value
    save_progress.assert_not_called()


def test_lesson_2_uses_verified_evidence_and_save_results(monkeypatch) -> None:
    _, chat_v4, _, _ = _configure_page(monkeypatch)
    chat_v4.return_value = new_agent_result(
        "V4",
        text="保存没有完成，请检查错题信息。",
        waiting_for="student_message",
        tool_calls=[
            {"name": "use_knowledge_card", "args": {"card_id": "candidate"}},
            {"name": "save_mistake", "args": {"original_question": "测试题"}},
        ],
        trace=[
            {"step": "route_turn", "status": "routed"},
            {
                "step": "tool_result",
                "name": "save_mistake",
                "status": "failure",
                "content": "测试失败详情",
            },
        ],
    )
    app = AppTest.from_file(PAGE_PATH).run()
    app.segmented_control[0].set_value("V4").run()

    app.chat_input[0].set_value("请整理这道错题").run()

    visible_text = [item.value for item in app.markdown]
    visible_captions = [item.value for item in app.caption]
    assert any("错题保存没有完成" in item for item in visible_text)
    assert not any("错题保存成功" in item for item in visible_text)
    assert any("本轮没有采用知识卡" in item for item in visible_captions)


@pytest.mark.parametrize(
    ("button_label", "decision_message"),
    [
        ("先整理，再复盘", "整理后复盘"),
        ("跳过这题，直接复盘", "跳过当前题直接复盘"),
        ("取消本次复盘", "取消"),
    ],
)
def test_lesson_2_review_choice_sends_exact_same_thread_decision_once(
    monkeypatch,
    button_label: str,
    decision_message: str,
) -> None:
    _, chat_v4, _, _ = _configure_page(monkeypatch)
    chat_v4.side_effect = [
        new_agent_result(
            "V4",
            text=(
                "当前还有一题尚未整理。"
                "请选择整理后复盘，或跳过当前题直接复盘。"
            ),
            waiting_for="review_decision",
        ),
        new_agent_result(
            "V4",
            text="错题已整理，累计复盘报告也已更新。",
            waiting_for="student_message",
        ),
    ]
    app = AppTest.from_file(PAGE_PATH).run()
    app.segmented_control[0].set_value("V4").run()
    thread_id = app.session_state["thread_id"]

    app.chat_input[0].set_value("总结复盘").run()

    assert not app.exception
    assert app.chat_input[0].disabled
    assert _widget_by_label(app.button, "先整理，再复盘")
    assert _widget_by_label(app.button, "跳过这题，直接复盘")
    assert _widget_by_label(app.button, "取消本次复盘")
    chat_v4.assert_called_once_with("总结复盘", thread_id)

    _widget_by_label(app.button, button_label).click().run()

    assert not app.exception
    assert chat_v4.call_args_list == [
        call("总结复盘", thread_id),
        call(decision_message, thread_id),
    ]
    assert not app.chat_input[0].disabled
    assert not app.session_state["lesson2_decision_in_flight"]
    assert app.session_state["lesson2_pending_decision"] is None
    assert app.session_state["lesson2_histories"]["V4"][-2:] == [
        {"role": "user", "content": decision_message},
        {
            "role": "assistant",
            "content": "错题已整理，累计复盘报告也已更新。",
        },
    ]

    app.run()

    assert chat_v4.call_count == 2


def test_lesson_2_v3_error_is_not_added_to_successful_history(
    monkeypatch,
) -> None:
    invoke, _, _, save_progress = _configure_page(monkeypatch)
    invoke.return_value = new_agent_result("V3", error="测试连接失败。")
    app = AppTest.from_file(PAGE_PATH).run()

    app.chat_input[0].set_value("请查这道题").run()

    assert not app.exception
    assert app.session_state["lesson2_histories"]["V3"] == []
    assert app.session_state["lesson2_last_attempts"]["V3"] == "请查这道题"
    assert "测试连接失败" in app.error[0].value
    save_progress.assert_not_called()

    app.run()

    invoke.assert_called_once()
    save_progress.assert_not_called()


def test_lesson_2_v4_completed_error_stays_aligned_with_workflow(
    monkeypatch,
) -> None:
    _, chat_v4, _, save_progress = _configure_page(monkeypatch)
    chat_v4.return_value = new_agent_result(
        "V4",
        text="报告格式损坏，本次没有更新报告。",
        waiting_for="student_message",
        error="错题记录格式损坏。",
    )
    app = AppTest.from_file(PAGE_PATH).run()
    app.segmented_control[0].set_value("V4").run()

    app.chat_input[0].set_value("总结复盘").run()

    assert not app.exception
    assert app.session_state["lesson2_histories"]["V4"] == [
        {"role": "user", "content": "总结复盘"},
        {
            "role": "assistant",
            "content": (
                "报告格式损坏，本次没有更新报告。\n\n"
                "本轮未全部完成：错题记录格式损坏。"
            ),
        },
    ]
    assert app.session_state["lesson2_last_attempts"]["V4"] is None
    assert any("错题记录格式损坏" in item.value for item in app.warning)
    save_progress.assert_not_called()


def test_lesson_2_v4_completed_attachment_error_keeps_reselection_notice(
    monkeypatch,
) -> None:
    _, chat_v4, _, save_progress = _configure_page(monkeypatch)
    submission = _image_submission()
    monkeypatch.setattr(
        chat_submission_module,
        "parse_chat_submission",
        Mock(return_value=submission),
    )
    chat_v4.return_value = new_agent_result(
        "V4",
        text="图片已经分析，但整理没有完成。",
        waiting_for="student_message",
        error="错题信息不足。",
    )
    app = AppTest.from_file(PAGE_PATH).run()
    app.segmented_control[0].set_value("V4").run()

    app.chat_input[0].set_value("测试附件部分失败").run()
    app.run()

    history = app.session_state["lesson2_histories"]["V4"]
    assert "附件未保留，请重新选择附件后发送" in history[-1]["content"]
    assert any(
        "附件未保留，请重新选择附件后发送" in item.value
        for item in app.markdown
    )
    save_progress.assert_not_called()


def test_lesson_2_v4_hard_error_remains_a_transient_attempt(monkeypatch) -> None:
    _, chat_v4, _, save_progress = _configure_page(monkeypatch)
    chat_v4.return_value = new_agent_result(
        "V4",
        error="工作流暂时无法恢复。",
    )
    app = AppTest.from_file(PAGE_PATH).run()
    app.segmented_control[0].set_value("V4").run()

    app.chat_input[0].set_value("继续").run()

    assert not app.exception
    assert app.session_state["lesson2_histories"]["V4"] == []
    assert app.session_state["lesson2_last_attempts"]["V4"] == "继续"
    assert "工作流暂时无法恢复" in app.error[0].value
    save_progress.assert_not_called()

    app.run()

    chat_v4.assert_called_once()
    save_progress.assert_not_called()


def test_lesson_2_v4_transient_decision_error_preserves_known_waiting_state(
    monkeypatch,
) -> None:
    _, chat_v4, _, save_progress = _configure_page(monkeypatch)
    chat_v4.side_effect = [
        new_agent_result(
            "V4",
            text="请选择怎样处理尚未整理的题目。",
            waiting_for="review_decision",
        ),
        new_agent_result("V4", error="工作流暂时无法恢复。"),
    ]
    app = AppTest.from_file(PAGE_PATH).run()
    app.segmented_control[0].set_value("V4").run()
    app.chat_input[0].set_value("总结复盘").run()

    _widget_by_label(app.button, "先整理，再复盘").click().run()

    assert not app.exception
    assert app.session_state["lesson2_waiting_for"]["V4"] == "review_decision"
    assert app.session_state["lesson2_histories"]["V4"] == [
        {"role": "user", "content": "总结复盘"},
        {"role": "assistant", "content": "请选择怎样处理尚未整理的题目。"},
    ]
    assert app.chat_input[0].disabled
    assert _widget_by_label(app.button, "先整理，再复盘")
    assert app.session_state["lesson2_last_attempts"]["V4"] == "整理后复盘"
    assert "工作流暂时无法恢复" in app.error[0].value
    assert chat_v4.call_count == 2
    save_progress.assert_called_once()
