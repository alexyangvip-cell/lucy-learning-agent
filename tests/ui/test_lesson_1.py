from base64 import b64decode
from pathlib import Path
from unittest.mock import Mock

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
PAGE_PATH = PROJECT_ROOT / "app_pages" / "lesson_1.py"
_TINY_PNG = b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8A"
    "AQUBAScY42YAAAAASUVORK5CYII="
)


def _artifact(stage: str, content: str, digest: str) -> facade_module.LessonArtifact:
    if stage == "V1":
        path = "student/prompt.md"
        label = "苏格拉底教练 Prompt"
    else:
        path = "student/skill/sorting-out-mistakes/SKILL.md"
        label = "整理错题 Skill"
    return {
        "stage": stage,
        "label": label,
        "path": path,
        "content": content,
        "digest": digest,
    }


def _configure_page(monkeypatch):
    artifacts = {
        "V1": _artifact("V1", "# 原 Prompt", "prompt-old"),
        "V2": _artifact(
            "V2",
            "---\nname: sorting-out-mistakes\n"
            "description: 整理错题。\n---\n\n# 原 Skill",
            "skill-old",
        ),
    }
    get_artifact = Mock(side_effect=lambda stage: artifacts[stage].copy())
    invoke = Mock(return_value=new_agent_result("V1", text="先观察哪个线索？"))
    save_artifact = Mock()

    def save_side_effect(stage, content, *, expected_digest):
        before = artifacts[stage].copy()
        after = before.copy()
        after["content"] = content.strip()
        after["digest"] = f"{stage.lower()}-new"
        artifacts[stage] = after
        return {
            "before": before,
            "after": after.copy(),
            "changed": before["content"] != after["content"],
            "diff": "-旧内容\n+新内容",
        }

    save_artifact.side_effect = save_side_effect
    load_progress = Mock(return_value=progress_module.default_progress())
    save_progress = Mock(side_effect=lambda progress: progress)

    monkeypatch.setattr(facade_module, "get_lesson_artifact", get_artifact)
    monkeypatch.setattr(facade_module, "save_lesson_artifact", save_artifact)
    monkeypatch.setattr(facade_module, "invoke", invoke)
    monkeypatch.setattr(progress_module, "load_progress", load_progress)
    monkeypatch.setattr(progress_module, "save_progress", save_progress)
    return get_artifact, save_artifact, invoke, load_progress, save_progress


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


def test_lesson_1_non_chat_actions_never_invoke_model(monkeypatch) -> None:
    _, save_artifact, invoke, load_progress, _ = _configure_page(monkeypatch)

    app = AppTest.from_file(PAGE_PATH).run()

    assert not app.exception
    assert not app.main.title
    assert app.sidebar.header[0].value == "第一课"
    assert not app.main.segmented_control
    assert len(app.sidebar.segmented_control) == 1
    assert not app.main.expander
    assert app.sidebar.status[0].label == "比较三个助手的回答"
    assert len(app.main.chat_message) == 1
    assert not app.sidebar.chat_message
    assert app.chat_input[0].placeholder == "和原始普通AI聊天"
    assert app.chat_input[0].proto.accept_file == ChatInputProto.SINGLE
    assert list(app.chat_input[0].proto.file_type) == [
        f".{extension}" for extension in ACCEPTED_FILE_TYPES
    ]
    assert app.chat_input[0].proto.max_upload_size_mb == (
        MAX_ATTACHMENT_BYTES // (1024 * 1024)
    )
    invoke.assert_not_called()
    save_artifact.assert_not_called()
    load_progress.assert_called_once_with()

    app.run()
    app.segmented_control[0].set_value("V1").run()
    _widget_by_label(app.text_area, "写给教学助手的说明书").set_value(
        "# 修改后的 Prompt"
    ).run()

    assert not app.exception
    invoke.assert_not_called()
    save_artifact.assert_not_called()
    load_progress.assert_called_once_with()


def test_lesson_1_saves_training_separately_then_chats_once(monkeypatch) -> None:
    _, save_artifact, invoke, _, save_progress = _configure_page(monkeypatch)
    app = AppTest.from_file(PAGE_PATH).run()
    app.segmented_control[0].set_value("V1").run()
    _widget_by_label(app.text_area, "写给教学助手的说明书").set_value(
        "# 新 Prompt\n\n一次只问一个问题。"
    )

    _widget_by_label(app.button, "保存").click().run()

    assert not app.exception
    save_artifact.assert_called_once_with(
        "V1",
        "# 新 Prompt\n\n一次只问一个问题。",
        expected_digest="prompt-old",
    )
    invoke.assert_not_called()
    save_progress.assert_not_called()

    app.chat_input[0].set_value(
        "She ___ (go) to Shanghai four times."
    ).run()

    assert not app.exception
    save_artifact.assert_called_once()
    invoke.assert_called_once_with(
        "V1",
        "She ___ (go) to Shanghai four times.",
        history=[],
    )
    save_progress.assert_called_once()
    assert app.session_state["lesson1_histories"]["V1"] == [
        {
            "role": "user",
            "content": "She ___ (go) to Shanghai four times.",
        },
        {"role": "assistant", "content": "先观察哪个线索？"},
    ]
    assert "v1" in app.session_state["progress"]["completed_modules"]

    app.run()

    save_artifact.assert_called_once()
    invoke.assert_called_once()
    save_progress.assert_called_once()


def test_lesson_1_save_error_never_sends_a_chat_message(monkeypatch) -> None:
    _, save_artifact, invoke, _, _ = _configure_page(monkeypatch)
    save_artifact.side_effect = facade_module.ArtifactConflictError(
        "文件已被其他页面修改。"
    )
    app = AppTest.from_file(PAGE_PATH).run()
    app.segmented_control[0].set_value("V2").run()

    _widget_by_label(app.button, "保存").click().run()

    assert not app.exception
    save_artifact.assert_called_once()
    invoke.assert_not_called()
    assert "已被其他页面修改" in app.error[0].value


def test_lesson_1_v2_keeps_history_and_displays_loaded_skill_name(
    monkeypatch,
) -> None:
    switch_page = Mock()
    monkeypatch.setattr(st, "switch_page", switch_page)
    _, save_artifact, invoke, _, _ = _configure_page(monkeypatch)
    invoke.return_value = new_agent_result(
        "V2",
        text="侦探闯关开始。",
        tool_calls=[
            {
                "name": "load_skill",
                "args": {"skill_name": "english-quest"},
            }
        ],
    )
    app = AppTest.from_file(PAGE_PATH).run()
    app.segmented_control[0].set_value("V2").run()

    app.chat_input[0].set_value("玩侦探闯关练英语").run()

    assert not app.exception
    save_artifact.assert_not_called()
    invoke.assert_called_once_with("V2", "玩侦探闯关练英语", history=[])
    assert app.session_state["lesson1_histories"]["V1"] == []
    assert app.session_state["lesson1_histories"]["V2"][-1] == {
        "role": "assistant",
        "content": "侦探闯关开始。",
    }
    assert app.session_state["lesson1_last_results"]["V2"]["tool_calls"][0][
        "name"
    ] == "load_skill"
    assert any(
        "english-quest" in markdown.value for markdown in app.markdown
    )
    assert any("load_skill" in markdown.value for markdown in app.markdown)
    switch_page.assert_called_once_with("app_pages/english_quest.py")
    quest_session = app.session_state["english_quest_session"]
    assert quest_session["agent_stage"] == "V2"
    assert quest_session["source_history_key"] == "lesson1_histories"
    assert quest_session["history"] == app.session_state[
        "lesson1_histories"
    ]["V2"]


def test_lesson_1_v2_ordinary_english_question_does_not_switch_page(
    monkeypatch,
) -> None:
    switch_page = Mock()
    monkeypatch.setattr(st, "switch_page", switch_page)
    _, _, invoke, _, _ = _configure_page(monkeypatch)
    invoke.return_value = new_agent_result("V2", text="现在完成时表示过去与现在有关。")
    app = AppTest.from_file(PAGE_PATH).run()
    app.segmented_control[0].set_value("V2").run()

    app.chat_input[0].set_value("什么是现在完成时？").run()

    assert not app.exception
    invoke.assert_called_once_with("V2", "什么是现在完成时？", history=[])
    switch_page.assert_not_called()
    assert "english_quest_session" not in app.session_state


def test_lesson_1_v0_chat_runs_without_saving_markdown(monkeypatch) -> None:
    _, save_artifact, invoke, _, _ = _configure_page(monkeypatch)
    invoke.return_value = new_agent_result("V0", text="普通回答")
    app = AppTest.from_file(PAGE_PATH).run()

    app.chat_input[0].set_value("你好").run()

    assert not app.exception
    save_artifact.assert_not_called()
    invoke.assert_called_once_with("V0", "你好", history=[])
    assert app.session_state["lesson1_histories"]["V0"][-1] == {
        "role": "assistant",
        "content": "普通回答",
    }
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


def test_lesson_1_attachment_is_forwarded_once_and_history_keeps_only_display_text(
    monkeypatch,
) -> None:
    _, save_artifact, invoke, _, save_progress = _configure_page(monkeypatch)
    submission = _image_submission()
    parser = Mock(return_value=submission)
    monkeypatch.setattr(chat_submission_module, "parse_chat_submission", parser)
    invoke.return_value = new_agent_result("V0", text="我看到了这道题。")
    app = AppTest.from_file(PAGE_PATH).run()

    app.chat_input[0].set_value("测试附件提交").run()

    assert not app.exception
    parser.assert_called_once()
    invoke.assert_called_once_with(
        "V0",
        "",
        history=[],
        attachment=submission.attachment,
    )
    save_artifact.assert_not_called()
    save_progress.assert_called_once()
    assert app.session_state["lesson1_histories"]["V0"] == [
        {"role": "user", "content": "附件：question.png"},
        {"role": "assistant", "content": "我看到了这道题。"},
    ]
    assert app.session_state["lesson1_last_attempts"]["V0"] is None

    app.run()

    assert invoke.call_count == 1


def test_lesson_1_attachment_parse_error_never_invokes_model_or_saves_progress(
    monkeypatch,
) -> None:
    _, _, invoke, _, save_progress = _configure_page(monkeypatch)
    parser = Mock(side_effect=ChatSubmissionError("附件类型不受支持。"))
    monkeypatch.setattr(chat_submission_module, "parse_chat_submission", parser)
    app = AppTest.from_file(PAGE_PATH).run()

    app.chat_input[0].set_value("尝试上传附件").run()

    assert not app.exception
    parser.assert_called_once()
    invoke.assert_not_called()
    save_progress.assert_not_called()
    assert app.session_state["lesson1_histories"]["V0"] == []
    assert "附件类型不受支持" in app.error[0].value


def test_lesson_1_attachment_model_error_requires_reselection(monkeypatch) -> None:
    _, _, invoke, _, save_progress = _configure_page(monkeypatch)
    submission = _image_submission(text="请看这道题")
    monkeypatch.setattr(
        chat_submission_module,
        "parse_chat_submission",
        Mock(return_value=submission),
    )
    invoke.return_value = new_agent_result("V0", error="模型暂时不可用。")
    app = AppTest.from_file(PAGE_PATH).run()

    app.chat_input[0].set_value("测试附件失败").run()

    assert not app.exception
    invoke.assert_called_once_with(
        "V0",
        "请看这道题",
        history=[],
        attachment=submission.attachment,
    )
    assert app.session_state["lesson1_histories"]["V0"] == []
    assert "附件未保留，请重新选择附件后发送" in app.error[0].value
    save_progress.assert_not_called()


def test_lesson_1_compares_each_stages_latest_reply(monkeypatch) -> None:
    _, _, invoke, _, _ = _configure_page(monkeypatch)
    invoke.side_effect = [
        new_agent_result("V0", text="V0 的回答"),
        new_agent_result("V1", text="V1 的回答"),
    ]
    app = AppTest.from_file(PAGE_PATH).run()
    app.chat_input[0].set_value("同一道题").run()
    app.segmented_control[0].set_value("V1").run()
    app.chat_input[0].set_value("同一道题").run()

    replies = [markdown.value for markdown in app.markdown]
    assert any("V0 的回答" in reply for reply in replies)
    assert any("V1 的回答" in reply for reply in replies)


def test_lesson_1_shows_memory_changes_and_undoes_once(monkeypatch) -> None:
    _, _, invoke, _, _ = _configure_page(monkeypatch)
    update = {
        "changes": [
            {
                "field": "interests",
                "action": "add",
                "before": [],
                "after": ["天文"],
            }
        ],
        "before_digest": "owner-before",
        "after_digest": "owner-after",
    }
    invoke.return_value = new_agent_result(
        "V1",
        text="我记住了，我们从天文例子开始。",
        owner_memory_update=update,
    )
    undo = Mock()
    monkeypatch.setattr(
        facade_module,
        "undo_owner_memory_update",
        undo,
        raising=False,
    )

    app = AppTest.from_file(PAGE_PATH).run()
    app.segmented_control[0].set_value("V1").run()
    app.chat_input[0].set_value("我喜欢天文").run()

    assert not app.exception
    assert any(
        "兴趣（新增）：从 未记录 改为 天文" in item.value
        for item in app.markdown
    )
    undo_button = _widget_by_label(app.button, "撤销这次记忆更新")
    undo_button.click().run()

    undo.assert_called_once_with(update)
    assert any("这次记忆更新已撤销" in item.value for item in app.success)
    assert not any(
        button.label == "撤销这次记忆更新"
        for button in app.button
    )


def test_lesson_1_warns_when_memory_fails_after_answer(monkeypatch) -> None:
    _, _, invoke, _, _ = _configure_page(monkeypatch)
    invoke.return_value = new_agent_result(
        "V1",
        text="正常回答仍然保留。",
        owner_memory_error="OWNER 文件刚刚被手工修改。",
    )

    app = AppTest.from_file(PAGE_PATH).run()
    app.segmented_control[0].set_value("V1").run()
    app.chat_input[0].set_value("我喜欢天文").run()

    assert not app.exception
    assert app.session_state["lesson1_histories"]["V1"][-1] == {
        "role": "assistant",
        "content": "正常回答仍然保留。",
    }
    assert any(
        "助手已经正常回答，但自动记忆没有更新" in item.value
        for item in app.warning
    )
