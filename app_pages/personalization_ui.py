"""SOUL、OWNER 编辑器和自动记忆结果的共享 Streamlit 组件。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import streamlit as st

import src.facade as facade


_DOCUMENT_LABELS = {
    "SOUL": "SOUL.md",
    "OWNER": "OWNER.md",
}
_FIELD_LABELS = {
    "preferred_name": "希望使用的称呼",
    "grade_band": "学习阶段",
    "languages": "常用语言",
    "interests": "兴趣",
    "learning_goals": "学习目标",
    "strengths": "擅长的方面",
    "challenges": "正在克服的困难",
    "response_preferences": "回答偏好",
}
_ACTION_LABELS = {
    "add": "新增",
    "append": "新增",
    "clear": "删除",
    "correct": "更新",
    "remove": "删除",
    "delete": "删除",
    "replace": "更新",
    "set": "更新",
    "upsert": "更新",
}


def _value(item: object, name: str, default: Any = None) -> Any:
    """兼容 TypedDict、普通 mapping 和冻结 dataclass。"""

    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _document(snapshot: object, kind: str) -> tuple[str, str]:
    """从 facade 编辑器快照中取得某个文件的正文和摘要。"""

    lower_kind = kind.lower()
    nested = _value(snapshot, lower_kind)
    if nested is not None:
        content = _value(nested, "content", _value(nested, "text", ""))
        digest = _value(nested, "digest", "")
    else:
        content = _value(
            snapshot,
            f"{lower_kind}_text",
            _value(snapshot, f"{lower_kind}_content", ""),
        )
        digest = _value(snapshot, f"{lower_kind}_digest", "")
    return str(content or ""), str(digest or "")


def _auto_memory_enabled(snapshot: object) -> bool:
    """读取首页自动记忆开关的持久值。"""

    return bool(_value(snapshot, "auto_memory", False))


def _editor_key(kind: str) -> str:
    return f"personalization_editor_{kind.lower()}"


def _digest_key(kind: str) -> str:
    return f"personalization_digest_{kind.lower()}"


def _notice_key(kind: str) -> str:
    return f"personalization_notice_{kind.lower()}"


def _error_key(kind: str) -> str:
    return f"personalization_error_{kind.lower()}"


def _set_feedback(
    kind: str,
    *,
    notice: str | None = None,
    error: str | None = None,
) -> None:
    """一次操作只保留成功或失败中的一种反馈。"""

    if notice is None:
        st.session_state.pop(_notice_key(kind), None)
    else:
        st.session_state[_notice_key(kind)] = notice
    if error is None:
        st.session_state.pop(_error_key(kind), None)
    else:
        st.session_state[_error_key(kind)] = error


def _reload_editor(kind: str) -> object:
    """从磁盘重读快照，并同步指定编辑器的基线。"""

    snapshot = facade.read_personalization_editor()
    content, digest = _document(snapshot, kind)
    st.session_state[_editor_key(kind)] = content
    st.session_state[_digest_key(kind)] = digest
    if kind == "OWNER":
        st.session_state.personalization_auto_memory = _auto_memory_enabled(snapshot)
    return snapshot


def _save_document(kind: str) -> None:
    """使用打开编辑器时的摘要执行冲突安全保存。"""

    try:
        facade.save_personalization_document(
            kind,
            st.session_state.get(_editor_key(kind), ""),
            expected_digest=st.session_state.get(_digest_key(kind), ""),
        )
        _reload_editor(kind)
    except Exception as exc:
        _set_feedback(kind, error=f"保存失败。{exc}")
        return
    _set_feedback(kind, notice="已保存。下一条消息会使用新内容。")


def _discard_document(kind: str) -> None:
    """放弃浏览器中的草稿，重新读取磁盘版本。"""

    try:
        _reload_editor(kind)
    except Exception as exc:
        _set_feedback(kind, error=f"重新读取失败。{exc}")
        return
    _set_feedback(kind, notice="已放弃未保存的修改。")


def _request_template_restore(kind: str) -> None:
    st.session_state[f"personalization_restore_pending_{kind.lower()}"] = True


def _cancel_template_restore(kind: str) -> None:
    st.session_state.pop(
        f"personalization_restore_pending_{kind.lower()}",
        None,
    )


def _restore_template(kind: str) -> None:
    """经过二次确认后，用随课程提供的模板替换保存版本。"""

    try:
        facade.restore_personalization_template(
            kind,
            expected_digest=st.session_state.get(_digest_key(kind), ""),
        )
        _reload_editor(kind)
    except Exception as exc:
        _set_feedback(kind, error=f"恢复模板失败。{exc}")
        return
    finally:
        _cancel_template_restore(kind)
    _set_feedback(
        kind,
        notice="已恢复课程模板。下一条消息会使用模板内容。",
    )


def _save_auto_memory_setting() -> None:
    """只通过独立确认操作修改自动记忆开关。"""

    enabled = bool(st.session_state.get("personalization_auto_memory", False))
    try:
        facade.set_auto_memory(
            enabled,
            expected_owner_digest=st.session_state.get(
                _digest_key("OWNER"),
                "",
            ),
        )
        _reload_editor("OWNER")
    except Exception as exc:
        _set_feedback("OWNER", error=f"自动记忆设置未保存。{exc}")
        return
    status = "开启" if enabled else "关闭"
    _set_feedback("OWNER", notice=f"自动记忆已{status}。")


def _request_clear_memory() -> None:
    st.session_state.personalization_clear_memory_pending = True


def _cancel_clear_memory() -> None:
    st.session_state.personalization_clear_memory_pending = False


def _clear_managed_memory() -> None:
    """清除受管字段，保留用户手写正文。"""

    try:
        facade.clear_auto_memory(
            expected_owner_digest=st.session_state.get(
                _digest_key("OWNER"),
                "",
            )
        )
        _reload_editor("OWNER")
    except Exception as exc:
        _set_feedback("OWNER", error=f"自动记忆没有清空。{exc}")
        return
    finally:
        _cancel_clear_memory()
    _set_feedback(
        "OWNER",
        notice="自动记录的学习资料已清空，手写正文保持不变。",
    )


def _render_document_editor(
    snapshot: object,
    kind: str,
    *,
    description: str,
) -> None:
    """渲染一个带摘要冲突保护的 Markdown 编辑器。"""

    content, current_digest = _document(snapshot, kind)
    editor_key = _editor_key(kind)
    digest_key = _digest_key(kind)
    if editor_key not in st.session_state:
        st.session_state[editor_key] = content
    if digest_key not in st.session_state:
        st.session_state[digest_key] = current_digest

    label = _DOCUMENT_LABELS[kind]
    with st.expander(label, icon=":material/edit_note:"):
        st.write(description)
        if kind == "OWNER":
            st.caption(
                "只填写愿意交给当前模型供应商处理的低敏学习资料。"
                "请勿填写密码、联系方式、精确地址、学校、"
                "健康或财务信息。"
            )
        else:
            st.caption(
                "这里可以设置 Agent 的名称、语气和默认表达方式，"
                "不能增加工具权限或改写课程安全规则。"
            )

        error = st.session_state.get(_error_key(kind))
        notice = st.session_state.get(_notice_key(kind))
        if error:
            st.error(error)
        elif notice:
            st.success(notice, icon=":material/check_circle:")

        if st.session_state[digest_key] != current_digest:
            st.warning(
                "这个文件已在别处修改。"
                "请先放弃修改并重新读取，避免覆盖较新的内容。"
            )

        st.text_area(
            f"编辑 {label}",
            key=editor_key,
            height=280,
            max_chars=32 * 1024,
            help="这是保存在 student 文件夹中的 UTF-8 Markdown 文件。",
        )
        dirty = st.session_state[editor_key] != content
        stale = st.session_state[digest_key] != current_digest

        with st.container(horizontal=True, gap="small"):
            st.button(
                "保存",
                key=f"personalization_save_{kind.lower()}",
                type="primary",
                icon=":material/save:",
                disabled=not dirty or stale,
                on_click=_save_document,
                args=(kind,),
            )
            st.button(
                "放弃修改",
                key=f"personalization_discard_{kind.lower()}",
                icon=":material/refresh:",
                disabled=not dirty and not stale,
                on_click=_discard_document,
                args=(kind,),
            )
            st.button(
                "恢复模板",
                key=f"personalization_restore_{kind.lower()}",
                icon=":material/restore_page:",
                disabled=stale,
                on_click=_request_template_restore,
                args=(kind,),
            )

        pending_key = f"personalization_restore_pending_{kind.lower()}"
        if st.session_state.get(pending_key):
            st.warning(f"恢复模板会替换当前保存的 {label} 内容。")
            with st.container(horizontal=True, gap="small"):
                st.button(
                    "确认恢复模板",
                    key=f"personalization_confirm_restore_{kind.lower()}",
                    type="primary",
                    on_click=_restore_template,
                    args=(kind,),
                )
                st.button(
                    "取消",
                    key=f"personalization_cancel_restore_{kind.lower()}",
                    on_click=_cancel_template_restore,
                    args=(kind,),
                )

        if dirty:
            st.caption("当前有未保存的修改。")
        else:
            st.caption(
                "编辑内容不会调用模型，保存后从下一条消息开始生效。"
            )


def _render_auto_memory_controls(snapshot: object) -> None:
    """渲染明确授权、关闭和清空自动记忆的控件。"""

    owner_content, owner_current_digest = _document(snapshot, "OWNER")
    owner_editor = st.session_state.get(_editor_key("OWNER"), owner_content)
    owner_base_digest = st.session_state.get(
        _digest_key("OWNER"),
        owner_current_digest,
    )
    owner_dirty = owner_editor != owner_content
    owner_stale = owner_base_digest != owner_current_digest
    persisted_enabled = _auto_memory_enabled(snapshot)
    if "personalization_auto_memory" not in st.session_state:
        st.session_state.personalization_auto_memory = persisted_enabled

    st.markdown("**自动记忆**")
    st.write(
        "开启后，只有本轮输入框中的第一人称低敏学习资料"
        "可能被提取。"
        "附件、旧对话、SOUL 和 OWNER 手写正文不会用于自动提取。"
    )
    st.toggle(
        "允许 Agent 自动更新 OWNER 的受管学习资料",
        key="personalization_auto_memory",
        help="默认关闭。每次候选提取会额外调用一次当前模型。",
    )
    requested_enabled = bool(st.session_state.personalization_auto_memory)
    setting_changed = requested_enabled != persisted_enabled
    if requested_enabled and setting_changed:
        st.info(
            "确认开启即表示你同意把符合候选条件的本轮纯文本"
            "发送给当前模型供应商进行结构化提取。"
        )
    if owner_dirty or owner_stale:
        st.caption(
            "请先保存或放弃 OWNER 的其他修改，再更改自动记忆。"
        )
    st.button(
        "确认开启" if requested_enabled else "确认关闭",
        key="personalization_save_auto_memory",
        type="primary" if requested_enabled else "secondary",
        disabled=not setting_changed or owner_dirty or owner_stale,
        on_click=_save_auto_memory_setting,
    )

    st.divider()
    st.markdown("**清空自动记忆**")
    st.write(
        "只清空 OWNER 顶部由系统管理的学习资料，"
        "不会删除手写正文，也不会修改 SOUL。"
    )
    st.button(
        "准备清空",
        key="personalization_request_clear_memory",
        icon=":material/delete_sweep:",
        disabled=owner_dirty or owner_stale,
        on_click=_request_clear_memory,
    )
    if st.session_state.get("personalization_clear_memory_pending"):
        st.warning("这会删除所有自动管理的学习资料。")
        with st.container(horizontal=True, gap="small"):
            st.button(
                "确认清空",
                key="personalization_confirm_clear_memory",
                type="primary",
                on_click=_clear_managed_memory,
            )
            st.button(
                "取消",
                key="personalization_cancel_clear_memory",
                on_click=_cancel_clear_memory,
            )


def render_personalization_settings(provider_label: str | None = None) -> None:
    """渲染首页的 Agent 个性化设置区域。"""

    st.subheader("Agent 个性化")
    provider = provider_label or "当前配置的模型供应商"
    st.warning(
        f"V1 至 V4 回答时会把保存的 SOUL 和 OWNER 内容发送给 {provider}。"
        "V0 不读取这些文件。Git 忽略不等于加密，"
        "请不要在其中保存秘密或高敏信息。",
        icon=":material/privacy_tip:",
    )
    try:
        facade.initialize_personalization()
        snapshot = facade.read_personalization_editor()
    except Exception as exc:
        st.error(f"个性化文件暂时无法读取。{exc}")
        st.caption(
            "修复 student/SOUL.md 和 student/OWNER.md 后重新加载页面。"
        )
        return

    _render_document_editor(
        snapshot,
        "SOUL",
        description=(
            "定义 Agent 的名称、语气、表达习惯和默认回答方式。"
        ),
    )
    _render_document_editor(
        snapshot,
        "OWNER",
        description="记录希望 Agent 在后续学习中记住的个人资料。",
    )
    with st.expander("自动记忆与清理", icon=":material/memory:"):
        _render_auto_memory_controls(snapshot)


def _format_change_value(value: object) -> str:
    """紧凑显示单个变更值，避免让长列表破坏窄屏布局。"""

    if value is None or value == "":
        return "未记录"
    if isinstance(value, (list, tuple, set)):
        rendered = "、".join(str(item) for item in value) or "未记录"
    else:
        rendered = str(value)
    if len(rendered) > 160:
        return rendered[:157] + "..."
    return rendered


def _undo_memory_update(update: object, state_key: str, digest: str) -> None:
    """撤销当前结果的记忆更新，并记录只属于 UI 的操作状态。"""

    try:
        facade.undo_owner_memory_update(update)
    except Exception as exc:
        st.session_state[f"{state_key}_undo_error"] = str(exc)
        st.session_state[f"{state_key}_undo_error_digest"] = digest
        return
    st.session_state[f"{state_key}_undone_digest"] = digest
    st.session_state.pop(f"{state_key}_undo_error", None)
    st.session_state.pop(f"{state_key}_undo_error_digest", None)


def render_owner_memory_result(result: object | None, *, state_key: str) -> None:
    """展示本轮字段级自动记忆结果和摘要安全的一次撤销。"""

    if not result:
        return
    memory_error = _value(result, "owner_memory_error")
    update = _value(result, "owner_memory_update")
    if memory_error:
        st.warning(
            "助手已经正常回答，但自动记忆没有更新。"
            f"原因：{memory_error}"
        )
    if not update:
        return

    changes = list(_value(update, "changes", ()) or ())
    if not changes:
        return
    digest = str(_value(update, "after_digest", ""))
    undone_digest = st.session_state.get(f"{state_key}_undone_digest")
    undo_error = st.session_state.get(f"{state_key}_undo_error")
    undo_error_digest = st.session_state.get(
        f"{state_key}_undo_error_digest"
    )

    with st.container(border=True):
        st.markdown("**Agent 刚刚更新了这些记忆**")
        for change in changes:
            field = str(_value(change, "field", ""))
            action = str(_value(change, "action", "update"))
            before = _format_change_value(_value(change, "before"))
            after = _format_change_value(_value(change, "after"))
            field_label = _FIELD_LABELS.get(field, field or "学习资料")
            action_label = _ACTION_LABELS.get(action, "更新")
            st.markdown(
                f"- {field_label}（{action_label}）：从 {before} 改为 {after}"
            )
        st.write(
            "这里只显示本轮变更，不展示完整 OWNER 或提取证据。"
        )

        if undone_digest == digest:
            st.success("这次记忆更新已撤销。")
            return
        if undo_error and undo_error_digest == digest:
            st.error(
                "无法安全撤销这次更新。"
                f"{undo_error} 请到首页的 Agent 个性化区域处理。"
            )
        st.button(
            "撤销这次记忆更新",
            key=f"{state_key}_undo_memory",
            icon=":material/undo:",
            on_click=_undo_memory_update,
            args=(update, state_key, digest),
        )
