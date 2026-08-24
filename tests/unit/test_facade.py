from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, call

import pytest

import src.facade as facade_module
import src.model as model_module
from src.artifacts import ArtifactError
from src.chat_submission import create_chat_attachment
from src.model import ModelConfigurationError
from src.personalization import (
    AgentPersonalization,
    OwnerMemoryChange,
    OwnerMemoryUpdate,
    OwnerProfile,
    PersonalizationDocument,
    PersonalizationError,
    render_owner_markdown,
)
from src.schemas import ChatMessage, new_agent_result


@pytest.fixture
def synthetic_personalization() -> AgentPersonalization:
    """Return a synthetic snapshot so facade tests never read local OWNER data."""

    return AgentPersonalization(
        soul_markdown="SOUL-FACADE-SENTINEL",
        owner=OwnerProfile(
            schema_version=1,
            auto_memory=False,
            preferred_name="OWNER-FACADE-SENTINEL",
            grade_band="初中",
            languages=("中文",),
            interests=("天文学",),
            learning_goals=(),
            strengths=(),
            challenges=(),
            response_preferences=("简洁回答",),
            manual_notes="手写资料。",
        ),
        soul_digest="facade-soul-digest",
        owner_digest="facade-owner-digest",
    )


@pytest.fixture(autouse=True)
def isolate_local_personalization(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_personalization: AgentPersonalization,
) -> None:
    monkeypatch.setattr(
        facade_module,
        "load_personalization",
        lambda: synthetic_personalization,
    )
    monkeypatch.setattr(
        facade_module,
        "_initialize_personalization",
        lambda: synthetic_personalization,
    )


def _owner_document(
    profile: OwnerProfile,
    *,
    digest: str = "owner-editor-digest",
) -> PersonalizationDocument:
    return PersonalizationDocument(
        kind="OWNER",
        content=render_owner_markdown(profile),
        digest=digest,
        path="student/OWNER.md",
    )


def _memory_update() -> OwnerMemoryUpdate:
    return OwnerMemoryUpdate(
        changes=(
            OwnerMemoryChange(
                field="interests",
                action="add",
                before=(),
                after=("天文学",),
            ),
        ),
        before_digest="owner-before",
        after_digest="owner-after",
    )


def test_invoke_rejects_unavailable_stage() -> None:
    result = facade_module.invoke("V4", "测试")

    assert result["stage"] == "V4"
    assert result["error"] == (
        "当前版本仅支持 V0、V1、V2 和 V3，收到的阶段为 V4。"
    )


def test_invoke_routes_v0_without_attachment_using_legacy_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = new_agent_result("V0", text="V0 回复")
    invoke_v0 = Mock(return_value=expected)
    monkeypatch.setattr(facade_module, "invoke_v0", invoke_v0)

    result = facade_module.invoke("V0", "测试")

    assert result is expected
    invoke_v0.assert_called_once_with("测试")


def test_v0_never_loads_personalization_or_runs_memory_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = new_agent_result("V0", text="V0 回复")
    invoke_v0 = Mock(return_value=expected)
    load = Mock(side_effect=AssertionError("V0 must not load personalization"))
    extractor = Mock(side_effect=AssertionError("V0 must not extract memory"))
    monkeypatch.setattr(facade_module, "invoke_v0", invoke_v0)
    monkeypatch.setattr(facade_module, "load_personalization", load)
    monkeypatch.setattr(
        facade_module,
        "extract_and_update_owner_memory",
        extractor,
    )

    result = facade_module.invoke("V0", "我喜欢天文学")

    assert result is expected
    invoke_v0.assert_called_once_with("我喜欢天文学")
    load.assert_not_called()
    extractor.assert_not_called()


@pytest.mark.parametrize(
    ("stage", "function_name"),
    [
        ("V0", "invoke_v0"),
        ("V1", "invoke_v1"),
        ("V2", "invoke_v2"),
        ("V3", "invoke_v3"),
    ],
)
def test_invoke_exactly_forwards_attachment_for_v0_to_v3(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_personalization: AgentPersonalization,
    stage: str,
    function_name: str,
) -> None:
    attachment = create_chat_attachment(
        name="notes.txt",
        media_type="text/plain",
        data=b"attachment body",
    )
    expected = new_agent_result(stage, text=f"{stage} 回复")
    target = Mock(return_value=expected)
    monkeypatch.setattr(facade_module, function_name, target)
    history: list[ChatMessage] = [
        {"role": "user", "content": "上一问"},
        {"role": "assistant", "content": "上一答"},
    ]

    result = facade_module.invoke(
        stage,
        "测试",
        history=history,
        attachment=attachment,
    )

    assert result is expected
    if stage == "V0":
        target.assert_called_once_with("测试", attachment=attachment)
    else:
        target.assert_called_once_with(
            "测试",
            history=history,
            attachment=attachment,
            personalization=synthetic_personalization,
        )


def test_invoke_routes_v1(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_personalization: AgentPersonalization,
) -> None:
    expected = new_agent_result("V1", text="V1 回复")
    received_history = None
    received_personalization = None

    def fake_invoke_v1(message: str, *, history=None, personalization=None):
        nonlocal received_history, received_personalization
        received_history = history
        received_personalization = personalization
        return expected

    monkeypatch.setattr(facade_module, "invoke_v1", fake_invoke_v1)
    history: list[ChatMessage] = [
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一个提示？"},
    ]

    result = facade_module.invoke("v1", "测试", history=history)

    assert result is expected
    assert received_history is history
    assert received_personalization is synthetic_personalization


def test_invoke_routes_v2(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_personalization: AgentPersonalization,
) -> None:
    expected = new_agent_result("V2", text="V2 回复")
    received_history = None
    received_personalization = None

    def fake_invoke_v2(message: str, *, history=None, personalization=None):
        nonlocal received_history, received_personalization
        received_history = history
        received_personalization = personalization
        return expected

    monkeypatch.setattr(facade_module, "invoke_v2", fake_invoke_v2)
    history: list[ChatMessage] = [
        {"role": "user", "content": "请帮我整理错题"},
        {"role": "assistant", "content": "请先发来原题。"},
    ]

    result = facade_module.invoke("v2", "这是原题", history=history)

    assert result is expected
    assert received_history is history
    assert received_personalization is synthetic_personalization


def test_invoke_routes_v3(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_personalization: AgentPersonalization,
) -> None:
    expected = new_agent_result("V3", text="V3 回复")
    received_history = None
    received_personalization = None

    def fake_invoke_v3(message: str, *, history=None, personalization=None):
        nonlocal received_history, received_personalization
        received_history = history
        received_personalization = personalization
        return expected

    monkeypatch.setattr(facade_module, "invoke_v3", fake_invoke_v3)
    history: list[ChatMessage] = [
        {"role": "user", "content": "three times 是什么线索？"},
        {"role": "assistant", "content": "它表示发生了几次？"},
    ]

    result = facade_module.invoke("v3", "为什么？", history=history)

    assert result is expected
    assert received_history is history
    assert received_personalization is synthetic_personalization


def test_invoke_uses_one_shared_personalization_snapshot_for_answer_and_memory(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_personalization: AgentPersonalization,
) -> None:
    load = Mock(return_value=synthetic_personalization)
    answer_profiles: list[AgentPersonalization] = []
    memory_profiles: list[AgentPersonalization] = []

    def fake_invoke_v1(message: str, *, history=None, personalization=None):
        answer_profiles.append(personalization)
        return new_agent_result("V1", text="正常回答")

    def fake_extract(message: str, personalization: AgentPersonalization):
        memory_profiles.append(personalization)
        return None

    monkeypatch.setattr(facade_module, "load_personalization", load)
    monkeypatch.setattr(facade_module, "invoke_v1", fake_invoke_v1)
    monkeypatch.setattr(
        facade_module,
        "extract_and_update_owner_memory",
        fake_extract,
    )

    result = facade_module.invoke("V1", "我喜欢天文学")

    assert result["error"] is None
    load.assert_called_once_with()
    assert answer_profiles == [synthetic_personalization]
    assert memory_profiles == [synthetic_personalization]
    assert answer_profiles[0] is memory_profiles[0]


@pytest.mark.parametrize("stage", ["V1", "V2", "V3"])
def test_profile_load_failure_stops_before_user_facing_agent(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    target = Mock(side_effect=AssertionError("agent must not run"))
    extractor = Mock(side_effect=AssertionError("extractor must not run"))
    monkeypatch.setattr(
        facade_module,
        "load_personalization",
        Mock(side_effect=PersonalizationError("OWNER 读取失败")),
    )
    monkeypatch.setattr(facade_module, f"invoke_{stage.casefold()}", target)
    monkeypatch.setattr(
        facade_module,
        "extract_and_update_owner_memory",
        extractor,
    )

    result = facade_module.invoke(stage, "测试")

    assert result["stage"] == stage
    assert result["error"] == "OWNER 读取失败"
    target.assert_not_called()
    extractor.assert_not_called()


def test_auto_memory_default_off_never_creates_extractor_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_llm = Mock(side_effect=AssertionError("extractor model must not run"))
    monkeypatch.setattr(model_module, "get_llm", get_llm)
    monkeypatch.setattr(
        facade_module,
        "invoke_v1",
        lambda message, **_kwargs: new_agent_result("V1", text="正常回答"),
    )

    result = facade_module.invoke("V1", "我喜欢天文学")

    assert result["error"] is None
    assert result["owner_memory_update"] is None
    assert result["owner_memory_error"] is None
    get_llm.assert_not_called()


def test_successful_owner_memory_update_is_attached_to_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    update = _memory_update()
    answer = new_agent_result("V1", text="正常回答")
    monkeypatch.setattr(facade_module, "invoke_v1", Mock(return_value=answer))
    extractor = Mock(return_value=update)
    monkeypatch.setattr(
        facade_module,
        "extract_and_update_owner_memory",
        extractor,
    )

    result = facade_module.invoke("V1", "我喜欢天文学")

    assert result is answer
    assert result["text"] == "正常回答"
    assert result["error"] is None
    assert result["owner_memory_update"] is update
    assert result["owner_memory_error"] is None


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        (
            PersonalizationError("OWNER 写入冲突，记忆未更新"),
            "OWNER 写入冲突，记忆未更新",
        ),
        (
            RuntimeError("secret extractor failure"),
            "自动记忆提取或保存失败，正常回答不受影响。错误类型：RuntimeError。",
        ),
    ],
)
def test_owner_memory_failure_preserves_answer_with_separate_error(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected_error: str,
) -> None:
    monkeypatch.setattr(
        facade_module,
        "invoke_v1",
        lambda message, **_kwargs: new_agent_result("V1", text="正常回答"),
    )
    monkeypatch.setattr(
        facade_module,
        "extract_and_update_owner_memory",
        Mock(side_effect=failure),
    )

    result = facade_module.invoke("V1", "我喜欢天文学")

    assert result["text"] == "正常回答"
    assert result["error"] is None
    assert result["owner_memory_update"] is None
    assert result["owner_memory_error"] == expected_error
    assert "secret extractor failure" not in result["owner_memory_error"]


def test_memory_extractor_receives_only_current_typed_text_and_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_personalization: AgentPersonalization,
) -> None:
    attachment = create_chat_attachment(
        name="private-notes.md",
        media_type="text/markdown",
        data=b"ATTACHMENT-MEMORY-SENTINEL",
    )
    history: list[ChatMessage] = [
        {"role": "user", "content": "HISTORY-MEMORY-SENTINEL"},
        {"role": "assistant", "content": "上一轮回答"},
    ]
    answer_calls: list[dict] = []
    extractor_calls: list[tuple[str, AgentPersonalization]] = []

    def fake_invoke_v1(message: str, **kwargs):
        answer_calls.append({"message": message, **kwargs})
        return new_agent_result("V1", text="正常回答")

    def fake_extract(message: str, personalization: AgentPersonalization):
        extractor_calls.append((message, personalization))
        return None

    monkeypatch.setattr(facade_module, "invoke_v1", fake_invoke_v1)
    monkeypatch.setattr(
        facade_module,
        "extract_and_update_owner_memory",
        fake_extract,
    )

    result = facade_module.invoke(
        "V1",
        "我喜欢天文学",
        history=history,
        attachment=attachment,
    )

    assert result["error"] is None
    assert answer_calls == [
        {
            "message": "我喜欢天文学",
            "history": history,
            "attachment": attachment,
            "personalization": synthetic_personalization,
        }
    ]
    assert extractor_calls == [
        ("我喜欢天文学", synthetic_personalization)
    ]
    assert "HISTORY-MEMORY-SENTINEL" not in repr(extractor_calls)
    assert "ATTACHMENT-MEMORY-SENTINEL" not in repr(extractor_calls)


def test_chat_v4_delegates_to_workflow(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_personalization: AgentPersonalization,
) -> None:
    expected = new_agent_result(
        "V4",
        text="V4 回复",
        waiting_for="student_message",
    )
    received = None

    def fake_chat_v4(message: str, thread_id: str, *, personalization=None):
        nonlocal received
        received = (message, thread_id, personalization)
        return expected

    monkeypatch.setattr(facade_module, "_chat_v4", fake_chat_v4)

    result = facade_module.chat_v4("继续", "student-1")

    assert result is expected
    assert received == ("继续", "student-1", synthetic_personalization)


def test_chat_v4_exactly_forwards_attachment(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_personalization: AgentPersonalization,
) -> None:
    attachment = create_chat_attachment(
        name="notes.md",
        media_type="text/markdown",
        data=b"# attachment body",
    )
    expected = new_agent_result(
        "V4",
        text="V4 回复",
        waiting_for="student_message",
    )
    chat_v4 = Mock(return_value=expected)
    monkeypatch.setattr(facade_module, "_chat_v4", chat_v4)

    result = facade_module.chat_v4(
        "继续",
        "student-1",
        attachment=attachment,
    )

    assert result is expected
    chat_v4.assert_called_once_with(
        "继续",
        "student-1",
        attachment=attachment,
        personalization=synthetic_personalization,
    )


def test_chat_v4_uses_same_snapshot_and_isolates_memory_extractor_inputs(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_personalization: AgentPersonalization,
) -> None:
    attachment = create_chat_attachment(
        name="v4-private.md",
        media_type="text/markdown",
        data=b"V4-ATTACHMENT-SENTINEL",
    )
    load = Mock(return_value=synthetic_personalization)
    workflow_calls: list[dict] = []
    extractor_calls: list[tuple[str, AgentPersonalization]] = []

    def fake_chat_v4(message: str, thread_id: str, **kwargs):
        workflow_calls.append(
            {"message": message, "thread_id": thread_id, **kwargs}
        )
        return new_agent_result(
            "V4",
            text="V4 正常回答",
            waiting_for="student_message",
        )

    def fake_extract(message: str, personalization: AgentPersonalization):
        extractor_calls.append((message, personalization))
        return _memory_update()

    monkeypatch.setattr(facade_module, "load_personalization", load)
    monkeypatch.setattr(facade_module, "_chat_v4", fake_chat_v4)
    monkeypatch.setattr(
        facade_module,
        "extract_and_update_owner_memory",
        fake_extract,
    )

    result = facade_module.chat_v4(
        "我喜欢天文学",
        "student-v4",
        attachment=attachment,
    )

    assert result["error"] is None
    assert result["owner_memory_update"] == _memory_update()
    load.assert_called_once_with()
    assert workflow_calls == [
        {
            "message": "我喜欢天文学",
            "thread_id": "student-v4",
            "attachment": attachment,
            "personalization": synthetic_personalization,
        }
    ]
    assert extractor_calls == [
        ("我喜欢天文学", synthetic_personalization)
    ]
    assert workflow_calls[0]["personalization"] is extractor_calls[0][1]
    assert "V4-ATTACHMENT-SENTINEL" not in repr(extractor_calls)


def test_chat_v4_profile_failure_stops_before_workflow_and_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = Mock(side_effect=AssertionError("workflow must not run"))
    extractor = Mock(side_effect=AssertionError("extractor must not run"))
    monkeypatch.setattr(
        facade_module,
        "load_personalization",
        Mock(side_effect=PersonalizationError("SOUL 读取失败")),
    )
    monkeypatch.setattr(facade_module, "_chat_v4", workflow)
    monkeypatch.setattr(
        facade_module,
        "extract_and_update_owner_memory",
        extractor,
    )

    result = facade_module.chat_v4("继续", "student-v4")

    assert result["stage"] == "V4"
    assert result["error"] == "SOUL 读取失败"
    workflow.assert_not_called()
    extractor.assert_not_called()


def test_model_configuration_facade_never_adds_secret_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary: facade_module.ModelConfigurationSummary = {
        "provider": "moonshot",
        "configured_providers": ["deepseek", "moonshot"],
    }
    reader = Mock(return_value=summary)
    saver = Mock(return_value=summary)
    monkeypatch.setattr(facade_module, "_get_model_configuration_summary", reader)
    monkeypatch.setattr(facade_module, "_save_model_configuration", saver)

    assert facade_module.get_model_configuration() is summary
    assert facade_module.save_model_configuration("moonshot", "secret") is summary
    assert "secret" not in str(summary)
    reader.assert_called_once_with()
    saver.assert_called_once_with("moonshot", "secret")


def test_model_connection_uses_minimal_v0_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = new_agent_result("V0", text="连接成功")
    invoke_v0 = Mock(return_value=expected)
    monkeypatch.setattr(facade_module, "invoke_v0", invoke_v0)

    result = facade_module.test_model_connection()

    assert result is expected
    invoke_v0.assert_called_once_with("请只回复：连接成功")


def test_initialize_personalization_delegates_to_storage(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_personalization: AgentPersonalization,
) -> None:
    initializer = Mock(return_value=synthetic_personalization)
    monkeypatch.setattr(
        facade_module,
        "_initialize_personalization",
        initializer,
    )

    result = facade_module.initialize_personalization()

    assert result is synthetic_personalization
    initializer.assert_called_once_with()


def test_read_personalization_editor_initializes_then_reads_both_documents(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_personalization: AgentPersonalization,
) -> None:
    soul = PersonalizationDocument(
        kind="SOUL",
        content="# 测试人格\n",
        digest="soul-editor-digest",
        path="student/SOUL.md",
    )
    owner = _owner_document(synthetic_personalization.owner)
    initializer = Mock(return_value=synthetic_personalization)
    reader = Mock(side_effect=[soul, owner])
    monkeypatch.setattr(
        facade_module,
        "_initialize_personalization",
        initializer,
    )
    monkeypatch.setattr(
        facade_module,
        "_read_personalization_document",
        reader,
    )

    result = facade_module.read_personalization_editor()

    assert result == {
        "soul": soul,
        "owner": owner,
        "auto_memory": False,
    }
    initializer.assert_called_once_with()
    assert reader.call_args_list == [
        call("SOUL"),
        call("OWNER"),
    ]


def test_owner_raw_save_cannot_change_auto_memory_authorization(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_personalization: AgentPersonalization,
) -> None:
    current = _owner_document(synthetic_personalization.owner)
    submitted = render_owner_markdown(
        replace(synthetic_personalization.owner, auto_memory=True)
    )
    saver = Mock(side_effect=AssertionError("raw OWNER save must be rejected"))
    monkeypatch.setattr(
        facade_module,
        "_read_personalization_document",
        Mock(return_value=current),
    )
    monkeypatch.setattr(
        facade_module,
        "_save_personalization_document",
        saver,
    )

    with pytest.raises(PersonalizationError, match="独立的自动记忆开关"):
        facade_module.save_personalization_document(
            "OWNER",
            submitted,
            expected_digest=current.digest,
        )

    saver.assert_not_called()


def test_personalization_save_and_restore_delegate_with_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = PersonalizationDocument(
        kind="SOUL",
        content="# 新人格\n",
        digest="new-soul-digest",
        path="student/SOUL.md",
    )
    restored = replace(saved, content="# 模板人格\n", digest="template-digest")
    saver = Mock(return_value=saved)
    restorer = Mock(return_value=restored)
    monkeypatch.setattr(
        facade_module,
        "_save_personalization_document",
        saver,
    )
    monkeypatch.setattr(
        facade_module,
        "_restore_personalization_template",
        restorer,
    )

    save_result = facade_module.save_personalization_document(
        "SOUL",
        "# 新人格",
        expected_digest="old-soul-digest",
    )
    restore_result = facade_module.restore_personalization_template(
        "SOUL",
        expected_digest="new-soul-digest",
    )

    assert save_result is saved
    assert restore_result is restored
    saver.assert_called_once_with(
        "SOUL",
        "# 新人格",
        expected_digest="old-soul-digest",
    )
    restorer.assert_called_once_with(
        "SOUL",
        expected_digest="new-soul-digest",
    )


def test_auto_memory_toggle_clear_and_undo_delegate_safely(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_personalization: AgentPersonalization,
) -> None:
    owner = _owner_document(synthetic_personalization.owner)
    toggled = replace(owner, digest="toggle-digest")
    cleared = replace(owner, digest="clear-digest")
    undone = replace(owner, digest="undo-digest")
    reader = Mock(return_value=owner)
    toggle = Mock(return_value=toggled)
    clear = Mock(return_value=cleared)
    undo = Mock(return_value=undone)
    monkeypatch.setattr(
        facade_module,
        "_read_personalization_document",
        reader,
    )
    monkeypatch.setattr(facade_module, "_set_owner_auto_memory", toggle)
    monkeypatch.setattr(facade_module, "_clear_owner_memory", clear)
    monkeypatch.setattr(facade_module, "_undo_owner_memory_update", undo)
    update = _memory_update()

    toggle_result = facade_module.set_auto_memory(True)
    clear_result = facade_module.clear_auto_memory(
        expected_owner_digest="explicit-clear-digest"
    )
    undo_result = facade_module.undo_owner_memory_update(update)

    assert toggle_result is toggled
    assert clear_result is cleared
    assert undo_result is undone
    toggle.assert_called_once_with(True, expected_digest=owner.digest)
    clear.assert_called_once_with(expected_digest="explicit-clear-digest")
    undo.assert_called_once_with(update)


def test_get_app_status_reports_ready_without_creating_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_personalization: AgentPersonalization,
) -> None:
    for relative_path in facade_module._REQUIRED_APP_FILES:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "3.14.3\n" if path.name == ".python-version" else "ok\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(facade_module, "PROJECT_ROOT", tmp_path)
    initializer = Mock(return_value=synthetic_personalization)
    monkeypatch.setattr(
        facade_module,
        "_initialize_personalization",
        initializer,
    )
    monkeypatch.setattr(facade_module.platform, "python_version", lambda: "3.14.3")
    monkeypatch.setattr(
        facade_module,
        "validate_model_configuration",
        lambda: "deepseek",
    )

    status = facade_module.get_app_status()

    assert status == {
        "ready": True,
        "runtime_ready": True,
        "model_ready": True,
        "python_version": "3.14.3",
        "recommended_python_version": "3.14.3",
        "model_provider": "deepseek",
        "missing_files": [],
        "runtime_errors": [],
        "model_error": None,
        "errors": [],
    }
    initializer.assert_called_once_with()


def test_get_app_status_reports_personalization_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for relative_path in facade_module._REQUIRED_APP_FILES:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "3.14.3\n" if path.name == ".python-version" else "ok\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(facade_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(facade_module.platform, "python_version", lambda: "3.14.3")
    monkeypatch.setattr(
        facade_module,
        "_initialize_personalization",
        Mock(side_effect=PersonalizationError("OWNER 初始化失败")),
    )
    monkeypatch.setattr(
        facade_module,
        "validate_model_configuration",
        lambda: "deepseek",
    )

    status = facade_module.get_app_status()

    assert status["ready"] is False
    assert status["runtime_ready"] is False
    assert status["model_ready"] is True
    assert status["missing_files"] == []
    assert "OWNER 初始化失败" in status["runtime_errors"]


@pytest.mark.parametrize(
    "python_version",
    ["3.11.0", "3.12.9", "3.13.7", "3.14.99"],
)
def test_get_app_status_accepts_supported_python_range(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    python_version: str,
) -> None:
    for relative_path in facade_module._REQUIRED_APP_FILES:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "3.14.3\n" if path.name == ".python-version" else "ok\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(facade_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        facade_module.platform,
        "python_version",
        lambda: python_version,
    )
    monkeypatch.setattr(
        facade_module,
        "validate_model_configuration",
        lambda: "deepseek",
    )

    status = facade_module.get_app_status()

    assert status["ready"] is True
    assert status["runtime_ready"] is True
    assert status["model_ready"] is True
    assert status["errors"] == []


@pytest.mark.parametrize("python_version", ["3.10.20", "3.15.0", "unknown"])
def test_get_app_status_rejects_unsupported_or_invalid_python_versions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    python_version: str,
) -> None:
    for relative_path in facade_module._REQUIRED_APP_FILES:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "3.14.3\n" if path.name == ".python-version" else "ok\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(facade_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        facade_module.platform,
        "python_version",
        lambda: python_version,
    )
    monkeypatch.setattr(
        facade_module,
        "validate_model_configuration",
        lambda: "deepseek",
    )

    status = facade_module.get_app_status()

    assert status["ready"] is False
    assert status["runtime_ready"] is False
    assert status["model_ready"] is True
    assert any("Python 3.11.x 至 3.14.x" in error for error in status["errors"])


def test_get_app_status_rejects_invalid_recommended_python_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for relative_path in facade_module._REQUIRED_APP_FILES:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "invalid\n" if path.name == ".python-version" else "ok\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(facade_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(facade_module.platform, "python_version", lambda: "3.14.3")
    monkeypatch.setattr(
        facade_module,
        "validate_model_configuration",
        lambda: "deepseek",
    )

    status = facade_module.get_app_status()

    assert status["ready"] is False
    assert status["runtime_ready"] is False
    assert status["model_ready"] is True
    assert any(".python-version" in error for error in status["errors"])


def test_get_app_status_explains_missing_configuration_and_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / ".python-version").write_text("3.14.3\n", encoding="utf-8")
    monkeypatch.setattr(facade_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(facade_module.platform, "python_version", lambda: "3.10.0")

    def reject_configuration() -> str:
        raise ModelConfigurationError("缺少测试 API Key。")

    monkeypatch.setattr(
        facade_module,
        "validate_model_configuration",
        reject_configuration,
    )

    status = facade_module.get_app_status()

    assert status["ready"] is False
    assert status["runtime_ready"] is False
    assert status["model_ready"] is False
    assert status["model_provider"] is None
    assert status["model_error"] == "缺少测试 API Key。"
    assert "缺少测试 API Key。" in status["errors"]
    assert "课程运行所需文件不完整。" in status["errors"]
    assert "课程运行所需文件不完整。" in status["runtime_errors"]
    assert "student/prompt.md" in status["missing_files"]
    assert "student/templates/SOUL.md" in status["missing_files"]
    assert "student/templates/OWNER.md" in status["missing_files"]
    assert "student/SOUL.md" in status["missing_files"]
    assert "student/OWNER.md" in status["missing_files"]
    assert any(
        "课程支持 Python 3.11.x 至 3.14.x" in error
        for error in status["errors"]
    )


def _configure_lesson_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path]:
    student_root = tmp_path / "student"
    prompt_path = student_root / "prompt.md"
    skill_path = student_root / "skill" / "sorting-out-mistakes" / "SKILL.md"
    prompt_path.parent.mkdir(parents=True)
    skill_path.parent.mkdir(parents=True)
    prompt_path.write_text("# 原 Prompt\n", encoding="utf-8")
    skill_path.write_text(
        "---\n"
        "name: sorting-out-mistakes\n"
        "description: 整理错题。\n"
        "---\n\n"
        "# 原 Skill\n\n"
        "先观察线索。\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(facade_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        facade_module,
        "_LESSON_ARTIFACTS",
        {
            "V1": facade_module._LessonArtifactSpec(
                stage="V1",
                label="测试 Prompt",
                path=prompt_path,
                kind="prompt",
            ),
            "V2": facade_module._LessonArtifactSpec(
                stage="V2",
                label="测试 Skill",
                path=skill_path,
                kind="skill",
            ),
        },
    )
    return prompt_path, skill_path


def test_lesson_artifact_save_returns_real_diff_and_atomic_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt_path, _ = _configure_lesson_artifacts(monkeypatch, tmp_path)
    before = facade_module.get_lesson_artifact("v1")

    assert before is not None
    change = facade_module.save_lesson_artifact(
        "V1",
        "# 新 Prompt\n\n一次只问一个问题。",
        expected_digest=before["digest"],
    )

    assert change["changed"] is True
    assert change["before"] == before
    assert change["after"]["content"] == "# 新 Prompt\n\n一次只问一个问题。"
    assert "-# 原 Prompt" in change["diff"]
    assert "+# 新 Prompt" in change["diff"]
    assert prompt_path.read_text(encoding="utf-8").endswith("\n")
    assert not list(prompt_path.parent.glob(".prompt.md.*.tmp"))


def test_lesson_artifact_rejects_invalid_skill_without_overwriting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, skill_path = _configure_lesson_artifacts(monkeypatch, tmp_path)
    original = skill_path.read_text(encoding="utf-8")
    before = facade_module.get_lesson_artifact("V2")

    assert before is not None
    with pytest.raises(ArtifactError, match="缺少 YAML frontmatter"):
        facade_module.save_lesson_artifact(
            "V2",
            "# 缺少 frontmatter",
            expected_digest=before["digest"],
        )

    assert skill_path.read_text(encoding="utf-8") == original
    assert not list(skill_path.parent.glob(".SKILL.md.*.tmp"))


def test_lesson_artifact_detects_external_edit_conflict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt_path, _ = _configure_lesson_artifacts(monkeypatch, tmp_path)
    before = facade_module.get_lesson_artifact("V1")

    assert before is not None
    prompt_path.write_text("# 其他页面的新内容\n", encoding="utf-8")

    with pytest.raises(
        facade_module.ArtifactConflictError,
        match="已被其他页面修改",
    ):
        facade_module.save_lesson_artifact(
            "V1",
            "# 当前页面的内容",
            expected_digest=before["digest"],
        )

    assert prompt_path.read_text(encoding="utf-8") == "# 其他页面的新内容\n"


def test_v0_has_no_editable_lesson_artifact() -> None:
    assert facade_module.get_lesson_artifact("V0") is None

    with pytest.raises(ArtifactError, match="V0 没有需要保存"):
        facade_module.save_lesson_artifact(
            "V0",
            "不会保存",
            expected_digest="unused",
        )
