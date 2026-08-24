"""第二课：用对话体验知识查证和用户主导的学习 Workflow。"""

from collections.abc import Iterator
from contextlib import contextmanager
from uuid import uuid4

import streamlit as st

from src.chat_submission import (
    ACCEPTED_FILE_TYPES,
    MAX_ATTACHMENT_BYTES,
    ChatSubmissionError,
    parse_chat_submission,
)
from src.facade import chat_v4, invoke
from src.progress import (
    ProgressDataError,
    complete_module,
    default_progress,
    load_progress,
    save_progress,
)
from src.skill_pages import (
    ENGLISH_QUEST_NAME,
    ENGLISH_QUEST_PAGE,
    new_english_quest_session,
    result_loaded_skill,
)

from app_pages.personalization_ui import render_owner_memory_result


STAGES = ("V3", "V4")
STAGE_LABELS = {
    "V3": "第 1 关 · 会查资料的助手",
    "V4": "第 2 关 · 会整理和复盘的学习教练",
}
STAGE_SELECTOR_LABELS = {
    "V3": "1 名师智能体",
    "V4": "2 个性化智能体",
}
STAGE_SUMMARIES = {
    "V3": (
        "回答前会先查看你的知识卡。"
        "真正用到资料时，它会告诉你依据来自哪里。"
    ),
    "V4": (
        "你决定什么时候继续答疑、什么时候保存错题，"
        "以及什么时候生成复盘。"
    ),
}
STAGE_CHALLENGES = {
    "V3": "发一道题，看看助手是否找到了可以支持回答的知识卡。",
    "V4": "完成一次答疑、整理错题、总结复盘和个性化练习。",
}
STAGE_INTROS = {
    "V3": (
        "Hi，我是你的「名师智能体」教练🦉。我会根据自己多年的教学经验，陪你一起进步✌️。"
    ),
    "V4": (
        "Hey，我是你的「个性化智能体」教练🦉。"
        "我可以陪你一步步想题，只有你明确提出“整理错题”或"
        "“总结复盘”时，我才会保存或整理错题本🐸。"
    ),
}
CHAT_PLACEHOLDERS = {
    "V3": "问知识点，或发送一道具体题",
    "V4": "问一道题，或明确说“整理错题”“总结复盘”",
}
WORKFLOW_GUIDE = (
    "1. 先发一道题。\n"
    "2. 告诉助手你的想法或答案。\n"
    "3. 需要保存时，明确说“请整理这道错题”。\n"
    "4. 需要复盘时，明确说“总结复盘”。\n"
    "5. 最后回答助手生成的新练习。"
)
TOOL_DESCRIPTIONS = {
    "load_skill": "尝试读取任务需要的 Skill",
    "load_mistake_file": "尝试读取等待整理的错题材料",
    "use_knowledge_card": "尝试查看一张可能有帮助的知识卡",
}
WORKFLOW_STEP_DESCRIPTIONS = {
    "route_turn": "判断你想做什么",
    "knowledge_retrieval": "查找可以使用的知识卡",
    "coach": "答疑或陪你思考",
    "organize_mistakes": "处理错题整理请求",
    "prepare_review": "准备学习复盘和新练习",
    "persist_report": "更新学习复盘",
    "clarify": "请你进一步说明需求",
    "cancel": "取消当前任务",
}
REVIEW_DECISIONS = (
    ("先整理，再复盘", "整理后复盘", ":material/save:"),
    (
        "跳过这题，直接复盘",
        "跳过当前题直接复盘",
        ":material/skip_next:",
    ),
    ("取消本次复盘", "取消", ":material/close:"),
)


def initialize_session_state() -> None:
    """集中初始化第二课所需的 Session State。"""

    st.session_state.setdefault("lesson2_stage", "V3")
    st.session_state.setdefault(
        "lesson2_histories",
        {stage: [] for stage in STAGES},
    )
    st.session_state.setdefault(
        "lesson2_last_results",
        {stage: None for stage in STAGES},
    )
    st.session_state.setdefault(
        "lesson2_last_attempts",
        {stage: None for stage in STAGES},
    )
    st.session_state.setdefault(
        "lesson2_last_errors",
        {stage: None for stage in STAGES},
    )
    st.session_state.setdefault(
        "lesson2_waiting_for",
        {stage: None for stage in STAGES},
    )
    st.session_state.setdefault("lesson2_pending_decision", None)
    st.session_state.setdefault("lesson2_decision_in_flight", False)
    st.session_state.setdefault("lesson2_progress_error", None)
    st.session_state.setdefault("thread_id", f"student-{uuid4().hex}")

    if "progress" not in st.session_state:
        try:
            st.session_state.progress = load_progress()
            st.session_state.progress_error = None
        except ProgressDataError as exc:
            st.session_state.progress = default_progress()
            st.session_state.progress_error = str(exc)


def _turn_completed(stage: str, result: dict) -> bool:
    """判断结果是否已经成为后端会话中的完整一轮。"""

    if not result.get("error"):
        return True
    return bool(
        stage == "V4"
        and str(result.get("text", "")).strip()
        and result.get("waiting_for")
        in {"student_message", "review_decision"}
    )


def _save_completed_progress(stage: str) -> None:
    """只在成功完成模型回合后保存对应课程进度。"""

    if st.session_state.get("progress_error"):
        st.session_state.lesson2_progress_error = (
            "原课程进度文件无法读取，因此本次没有覆盖它。"
            f"原因：{st.session_state.progress_error}"
        )
        return
    try:
        updated = complete_module(st.session_state.progress, stage.lower())
        st.session_state.progress = save_progress(updated)
        st.session_state.lesson2_progress_error = None
    except ProgressDataError as exc:
        st.session_state.lesson2_progress_error = str(exc)


def _record_result(stage: str, message: str, result: dict) -> bool:
    """记录已完成回合，并保持 V3 历史和 V4 后端线程一致。"""

    st.session_state.lesson2_last_results[stage] = result
    completed = _turn_completed(stage, result)
    if not completed:
        st.session_state.lesson2_last_errors[stage] = (
            result.get("error") or "这条消息没有发送成功。"
        )
        return False

    reply = str(result.get("text") or result.get("error") or "本轮没有回复。")
    if result.get("text") and result.get("error"):
        reply = f"{reply}\n\n本轮未全部完成：{result['error']}"
    st.session_state.lesson2_histories[stage].extend(
        [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ]
    )
    st.session_state.lesson2_last_attempts[stage] = None
    st.session_state.lesson2_last_errors[stage] = result.get("error")
    st.session_state.lesson2_waiting_for[stage] = result.get("waiting_for")
    if not result.get("error"):
        _save_completed_progress(stage)
    return True


def _latest_turn_trace(stage: str, result: dict | None) -> list[dict]:
    """V4 trace 会累计，只返回从最近一次路由开始的事件。"""

    if not result:
        return []
    trace = list(result.get("trace") or [])
    if stage != "V4":
        return trace
    for index in range(len(trace) - 1, -1, -1):
        if trace[index].get("step") == "route_turn":
            return trace[index:]
    return trace


def _safe_trace(trace: list[dict]) -> list[dict]:
    """保留教学所需字段，不展示学生答案或内部写入数据。"""

    safe_keys = {
        "step",
        "status",
        "name",
        "intent",
        "confidence",
        "explicit_write",
        "version",
        "used_card_ids",
    }
    return [
        {key: item[key] for key in safe_keys if key in item}
        for item in trace
    ]


def _verified_save_statuses(trace: list[dict]) -> list[str]:
    """只依据确定性工具结果描述错题是否保存成功。"""

    labels = {
        "success": "错题保存成功",
        "failure": "错题保存没有完成",
        "unknown": "尝试保存错题，但结果无法核验",
    }
    return [
        labels.get(str(item.get("status")), "错题保存结果未知")
        for item in trace
        if item.get("step") == "tool_result"
        and item.get("name") == "save_mistake"
    ]


def _queue_review_decision(decision: str) -> None:
    """先锁定复盘选择，让下一次脚本运行只提交这一个决定。"""

    if st.session_state.lesson2_decision_in_flight:
        return
    st.session_state.lesson2_pending_decision = decision
    st.session_state.lesson2_decision_in_flight = True


@contextmanager
def _chat_bubble(role: str) -> Iterator[None]:
    """用原生布局让助手消息靠左、学生消息靠右。"""

    is_user = role == "user"
    alignment = "right" if is_user else "left"
    avatar = ":material/person:" if is_user else ":material/smart_toy:"
    with st.container(horizontal_alignment=alignment, gap=None):
        with st.chat_message(role, avatar=avatar, width="content"):
            yield


st.set_page_config(
    page_title="第二课 - 查证与复盘",
    page_icon=":material/psychology:",
    layout="centered",
)

initialize_session_state()

with st.sidebar:
    st.header("第二课")
    st.caption("让助手学会查证和复盘")
    # st.write(
    #    "同一个学习助手继续升级。"
    #    "先让它学会查资料，再让它按你的决定整理和复盘。"
    #)

    stage = st.segmented_control(
        "选择一关",
        STAGES,
        format_func=lambda value: STAGE_SELECTOR_LABELS[value],
        required=True,
        key="lesson2_stage",
        width="stretch",
        persist_state="session",
        disabled=st.session_state.lesson2_decision_in_flight,
    )
    stage = stage or "V3"

    st.markdown(f"**{STAGE_LABELS[stage]}**")
    st.caption(STAGE_SUMMARIES[stage])
    st.caption(f"本关挑战：{STAGE_CHALLENGES[stage]}")

    if stage == "V3":
        st.info(
            "知识卡是带标签的学习笔记。"
            "只有真正被采用的卡片，才会出现在本轮依据中。",
            icon=":material/library_books:",
        )
    else:
        with st.expander("怎样完成这一关", icon=":material/route:"):
            st.markdown(WORKFLOW_GUIDE)
            st.caption(
                "普通答疑不会自动保存。"
                "写入错题和更新复盘都需要你明确提出。"
            )

    last_result = st.session_state.lesson2_last_results[stage]
    current_trace = _latest_turn_trace(stage, last_result)
    waiting_for = st.session_state.lesson2_waiting_for[stage]
    completed_error = bool(
        last_result
        and last_result.get("error")
        and _turn_completed(stage, last_result)
    )

    if stage == "V4":
        if waiting_for == "review_decision":
            st.info(
                "现在需要你选择怎样处理尚未整理的题目。",
                icon=":material/touch_app:",
            )
        else:
            st.caption("当前状态：等你继续发消息")

    if completed_error:
        st.warning(
            "本轮已经结束，但任务没有全部完成。"
            f"原因：{last_result['error']}"
        )

    with st.expander("这次参考了什么", icon=":material/menu_book:"):
        citations = list(last_result.get("citations") or []) if last_result else []
        if citations:
            for citation in citations:
                st.markdown(f"**《{citation.get('title', '未命名知识卡')}》**")
                matches = citation.get("matches") or []
                for match in matches:
                    field = match.get("field", "内容")
                    excerpt = str(match.get("excerpt", "")).strip()
                    if excerpt:
                        st.markdown(f"- {field}：{excerpt}")
        elif last_result:
            st.caption("本轮没有采用知识卡。")
        else:
            st.caption("发送消息后，这里会展示真正采用的知识卡。")

    activity_items: list[str] = []
    if last_result:
        for tool_call in last_result.get("tool_calls") or []:
            name = tool_call.get("name")
            if name in TOOL_DESCRIPTIONS:
                description = TOOL_DESCRIPTIONS[name]
                if name == "load_skill":
                    skill_name = tool_call.get("args", {}).get("skill_name")
                    if skill_name:
                        description = f"尝试读取 Skill：`{skill_name}`"
                if description not in activity_items:
                    activity_items.append(description)
        activity_items.extend(_verified_save_statuses(current_trace))

    with st.expander("这次助手做了什么", icon=":material/bolt:"):
        if activity_items:
            for item in activity_items:
                st.markdown(f"- {item}")
        else:
            st.caption("本轮没有需要单独说明的工具操作。")

    if stage == "V4":
        report_updated = any(
            item.get("step") == "persist_report"
            and item.get("status") == "complete"
            for item in current_trace
        )
        with st.expander("我的学习复盘", icon=":material/assignment:"):
            if report_updated:
                st.success("本轮学习复盘已更新，新练习已经发到聊天中。")
            else:
                st.caption("明确说“总结复盘”后，这里会显示本轮更新状态。")
            st.caption(
                "回答新练习只会继续辅导，"
                "不会自动保存成错题，也不会自动修改报告。"
            )

        with st.expander("看看助手幕后的流程", icon=":material/account_tree:"):
            steps = []
            for item in current_trace:
                description = WORKFLOW_STEP_DESCRIPTIONS.get(item.get("step"))
                if description and item.get("status") == "error":
                    description = f"{description}（没有完成）"
                if description and description not in steps:
                    steps.append(description)
            if steps:
                for index, description in enumerate(steps, start=1):
                    st.markdown(f"{index}. {description}")
                st.markdown(f"{len(steps) + 1}. 等待你继续")
            else:
                st.caption("发送消息后，这里会显示助手刚刚走过的步骤。")

    with st.expander("查看开发者信息", icon=":material/code:"):
        st.caption("这里使用的是简化后的运行记录，不包含隐藏答案。")
        if last_result:
            st.code(
                f"stage: {last_result.get('stage')}\n"
                f"waiting_for: {last_result.get('waiting_for')}",
                language="yaml",
            )
            tool_names = [
                call.get("name", "unknown")
                for call in last_result.get("tool_calls") or []
            ]
            if tool_names:
                st.write("工具名称：" + "、".join(tool_names))
            citation_sources = [
                citation.get("source", "")
                for citation in last_result.get("citations") or []
                if citation.get("source")
            ]
            if citation_sources:
                st.write("知识卡来源：")
                for source in citation_sources:
                    st.code(source, language=None)
            if current_trace:
                st.json(_safe_trace(current_trace), expanded=False)
        else:
            st.caption("还没有本轮运行记录。")

    completed_modules = set(st.session_state.progress["completed_modules"])
    completed_count = sum(
        stage_name.lower() in completed_modules for stage_name in STAGES
    )
    st.progress(
        completed_count / len(STAGES),
        text=f"第二课已体验 {completed_count}/2 关",
    )

    if st.session_state.lesson2_progress_error:
        st.warning(
            "助手已经回答，但课程进度没有保存。"
            f"原因：{st.session_state.lesson2_progress_error}"
        )

history = st.session_state.lesson2_histories[stage]

with _chat_bubble("assistant"):
    st.write(STAGE_INTROS[stage])

for chat_message in history:
    with _chat_bubble(chat_message["role"]):
        st.write(chat_message["content"])

last_attempt = st.session_state.lesson2_last_attempts[stage]
last_error = st.session_state.lesson2_last_errors[stage]
if last_error and last_attempt:
    with _chat_bubble("user"):
        st.write(last_attempt)
    with _chat_bubble("assistant"):
        st.error(f"这条消息没有发送成功。{last_error}")

render_owner_memory_result(
    st.session_state.lesson2_last_results[stage],
    state_key=f"lesson2_{stage.lower()}",
)

needs_review_decision = stage == "V4" and waiting_for == "review_decision"
if needs_review_decision:
    with _chat_bubble("assistant"):
        st.caption("请选择一个回复，助手会按你的决定继续。")
        for label, decision, icon in REVIEW_DECISIONS:
            st.button(
                label,
                key=f"lesson2_review_{decision}",
                icon=icon,
                width="stretch",
                type="primary" if decision == "整理后复盘" else "secondary",
                disabled=st.session_state.lesson2_decision_in_flight,
                on_click=_queue_review_decision,
                args=(decision,),
            )

decision_message = (
    st.session_state.lesson2_pending_decision
    if needs_review_decision and st.session_state.lesson2_decision_in_flight
    else None
)

chat_placeholder = CHAT_PLACEHOLDERS[stage]
if needs_review_decision:
    chat_placeholder = "请先选择一种复盘方式"

chat_value = st.chat_input(
    chat_placeholder,
    key=f"lesson2_chat_{stage.lower()}",
    max_chars=8_000,
    max_upload_size=MAX_ATTACHMENT_BYTES // (1024 * 1024),
    accept_file=True,
    file_type=list(ACCEPTED_FILE_TYPES),
    disabled=needs_review_decision,
    submit_mode="disable",
)
raw_submission = decision_message if decision_message is not None else chat_value

if raw_submission is not None:
    try:
        submission = parse_chat_submission(raw_submission)
    except ChatSubmissionError as exc:
        raw_text = (
            raw_submission
            if isinstance(raw_submission, str)
            else getattr(raw_submission, "text", "")
        )
        st.session_state.lesson2_last_attempts[stage] = (
            raw_text.strip() or "附件提交"
        )
        st.session_state.lesson2_last_errors[stage] = str(exc)
        st.rerun()

    st.session_state.lesson2_last_attempts[stage] = submission.display_text
    st.session_state.lesson2_last_errors[stage] = None
    with _chat_bubble("user"):
        st.write(submission.display_text)
        if submission.attachment and submission.attachment.is_image:
            st.image(submission.attachment.data)
    with _chat_bubble("assistant"):
        with st.spinner("助手正在思考…", show_time=True):
            try:
                if stage == "V3":
                    conversation = [item.copy() for item in history]
                    if submission.attachment is None:
                        result = invoke(
                            "V3",
                            submission.text,
                            history=conversation,
                        )
                    else:
                        result = invoke(
                            "V3",
                            submission.text,
                            history=conversation,
                            attachment=submission.attachment,
                        )
                elif submission.attachment is None:
                    result = chat_v4(
                        submission.text,
                        st.session_state.thread_id,
                    )
                else:
                    result = chat_v4(
                        submission.text,
                        st.session_state.thread_id,
                        attachment=submission.attachment,
                    )
            finally:
                if decision_message is not None:
                    st.session_state.lesson2_pending_decision = None
                    st.session_state.lesson2_decision_in_flight = False
        if result.get("error") and submission.attachment is not None:
            result = {
                **result,
                "error": (
                    f"{result['error']} "
                    "附件未保留，请重新选择附件后发送。"
                ),
            }
        completed = _record_result(stage, submission.display_text, result)
        if completed:
            st.write(result.get("text") or result.get("error"))
            if result.get("error"):
                st.warning(result["error"])
            elif stage == "V3" and result_loaded_skill(
                result,
                ENGLISH_QUEST_NAME,
            ):
                st.session_state.english_quest_session = (
                    new_english_quest_session(
                        history=st.session_state.lesson2_histories[stage],
                        agent_stage="V3",
                        return_page="app_pages/lesson_2.py",
                        return_label="返回第二课",
                        source_history_key="lesson2_histories",
                        source_stage=stage,
                    )
                )
                st.switch_page(ENGLISH_QUEST_PAGE)
        else:
            st.error(
                "这条消息没有发送成功。"
                f"{result.get('error') or '请稍后重试。'}"
            )
    st.rerun()
