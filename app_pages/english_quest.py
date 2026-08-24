"""english-quest Skill 的专属互动页面。"""

from collections.abc import Iterator
from contextlib import contextmanager
import runpy

import streamlit as st

from src.artifacts import SKILLS_PATH
from src.facade import invoke
from src.skill_pages import (
    ENGLISH_QUEST_NAME,
    new_english_quest_session,
    result_loaded_skill,
)

from app_pages.personalization_ui import render_owner_memory_result


QUEST_ROOT = SKILLS_PATH / ENGLISH_QUEST_NAME
QUEST_STATE_SCRIPT = QUEST_ROOT / "scripts" / "quest_state.py"
QUEST_BANNER = QUEST_ROOT / "assets" / "detective-board.svg"
TOPIC_OPTIONS = ("语法", "词汇", "阅读")
THEME_OPTIONS = ("侦探破案", "奇幻冒险", "荒岛生存", "校园寻宝")


def initialize_session_state() -> None:
    """让页面既能接受课程页交接，也能被直接打开。"""

    st.session_state.setdefault(
        "english_quest_session",
        new_english_quest_session(),
    )
    st.session_state.setdefault("english_quest_last_result", None)
    st.session_state.setdefault("english_quest_last_attempt", None)
    st.session_state.setdefault("english_quest_last_error", None)


def _derive_quest_state(history: list[dict]) -> tuple[dict, str | None]:
    """通过 Skill 自带脚本提取状态，失败时向学生说明资源问题。"""

    try:
        namespace = runpy.run_path(str(QUEST_STATE_SCRIPT))
        derive = namespace["derive_quest_state"]
        state = derive(history)
        required_keys = {
            "level",
            "hearts",
            "xp",
            "complete",
            "has_status_line",
            "template_complete",
            "deviations",
        }
        if not isinstance(state, dict) or not required_keys <= state.keys():
            raise ValueError("脚本没有返回完整的游戏状态。")
    except Exception as exc:
        return {
            "level": 0,
            "hearts": 3,
            "xp": 0,
            "complete": False,
            "has_status_line": False,
            "template_complete": False,
            "deviations": ["状态脚本执行失败"],
        }, str(exc)
    return state, None


def _sync_source_history() -> None:
    """把专属页中的新回合同步回进入页面前的课程对话。"""

    session = st.session_state.english_quest_session
    history_key = session.get("source_history_key")
    source_stage = session.get("source_stage")
    histories = st.session_state.get(history_key) if history_key else None
    if isinstance(histories, dict) and source_stage in histories:
        histories[source_stage] = [
            item.copy() for item in session["history"]
        ]


def _render_status_panel(state: dict) -> None:
    """在桌面和窄屏上都紧凑展示三项游戏状态。"""

    hearts = "❤️" * state["hearts"] or "0"
    with st.container(
        horizontal=True,
        horizontal_alignment="distribute",
        gap="small",
    ):
        st.metric("关卡", f"{state['level']}/5", border=True)
        st.metric("生命", hearts, border=True)
        st.metric("经验", f"{state['xp']} XP", border=True)


def _reset_quest() -> None:
    """保留来源页面信息，只清空本局对话。"""

    session = st.session_state.english_quest_session
    session["history"] = []
    st.session_state.english_quest_last_result = None
    st.session_state.english_quest_last_attempt = None
    st.session_state.english_quest_last_error = None
    _sync_source_history()


def _run_turn(message: str) -> None:
    """调用原 V2/V3 Agent，并让完整历史在专属页继续。"""

    session = st.session_state.english_quest_session
    history = session["history"]
    st.session_state.english_quest_last_attempt = message
    st.session_state.english_quest_last_error = None
    result = invoke(
        session["agent_stage"],
        message,
        history=[item.copy() for item in history],
    )
    st.session_state.english_quest_last_result = result
    if result["error"]:
        st.session_state.english_quest_last_error = result["error"]
        return
    history.extend(
        [
            {"role": "user", "content": message},
            {"role": "assistant", "content": result["text"]},
        ]
    )
    st.session_state.english_quest_last_attempt = None
    _sync_source_history()


@contextmanager
def _chat_bubble(role: str) -> Iterator[None]:
    """保持学生消息靠右，任务主持人消息靠左。"""

    is_user = role == "user"
    alignment = "right" if is_user else "left"
    avatar = ":material/person:" if is_user else ":material/search:"
    with st.container(horizontal_alignment=alignment, gap=None):
        with st.chat_message(role, avatar=avatar, width="content"):
            yield


st.set_page_config(
    page_title="英语剧情闯关",
    page_icon=":material/stadia_controller:",
    layout="centered",
)

initialize_session_state()
quest_session = st.session_state.english_quest_session
history = quest_session["history"]
quest_state, state_error = _derive_quest_state(history)

with st.sidebar:
    if st.button(
        quest_session["return_label"],
        icon=":material/arrow_back:",
        width="stretch",
    ):
        _sync_source_history()
        st.switch_page(quest_session["return_page"])

    st.header("English Quest")
    st.caption("这个页面由 `english-quest` Skill 激活。")
    with st.expander("这个 Skill 由什么组成", icon=":material/deployed_code:"):
        st.markdown("**SKILL.md**：决定何时触发、怎样出题和怎样反馈。")
        st.markdown("**scripts/quest_state.py**：把对话转换成关卡、生命和 XP。")
        st.markdown("**assets/detective-board.svg**：提供任务页的视觉主题。")
        st.caption("脚本只负责界面状态，英语判题仍由 Agent 完成。")

    if state_error:
        st.error(f"Skill 状态脚本读取失败：{state_error}")

    if history and not quest_state["template_complete"]:
        with st.expander("Skill 输出模板漂移", expanded=False, icon=":material/warning:"):
            st.warning(
                "最新一局回复未通过 SKILL.md 的 4 段模板校验（这只是软提示，不影响答题和判分）："
            )
            for deviation in quest_state["deviations"]:
                st.markdown(f"- {deviation}")

    last_result = st.session_state.english_quest_last_result
    if last_result and result_loaded_skill(
        last_result,
        "sorting-out-mistakes",
    ):
        st.success("Agent 已切换到 `sorting-out-mistakes`。")

st.image(QUEST_BANNER, width="stretch")
st.title("英语剧情闯关")
st.caption("每次只破解一个英语挑战。答错先拿提示，完成五关后获得任务报告。")

_render_status_panel(quest_state)
st.progress(
    quest_state["level"] / 5,
    text=("任务已完成" if quest_state["complete"] else "案件调查中"),
)

if not history:
    st.subheader("选择你的第一项任务")
    st.write("可以先选一个方向，也可以直接写下正在学的知识点。")

with st.expander("任务设置", expanded=not history, icon=":material/tune:"):
    with st.form("english_quest_setup"):
        topic_type = st.selectbox(
            "练习方向",
            TOPIC_OPTIONS,
            key="english_quest_topic_type",
        )
        knowledge_point = st.text_input(
            "知识点",
            value="现在完成时",
            max_chars=120,
            key="english_quest_knowledge_point",
        )
        theme = st.selectbox(
            "剧情主题",
            THEME_OPTIONS,
            key="english_quest_theme",
        )
        start_clicked = st.form_submit_button(
            "开始任务",
            icon=":material/play_arrow:",
            type="primary",
            width="stretch",
        )

if start_clicked:
    if history:
        _reset_quest()
    target = knowledge_point.strip() or topic_type
    opening_message = f"我们玩一个{theme}闯关游戏练英语{target}。"
    with st.spinner("正在生成第一条线索…", show_time=True):
        _run_turn(opening_message)
    st.rerun()

if history:
    for chat_message in history:
        with _chat_bubble(chat_message["role"]):
            st.write(chat_message["content"])

last_attempt = st.session_state.english_quest_last_attempt
last_error = st.session_state.english_quest_last_error
if last_error and last_attempt:
    with _chat_bubble("user"):
        st.write(last_attempt)
    with _chat_bubble("assistant"):
        st.error(f"这条消息没有发送成功。{last_error}")

render_owner_memory_result(
    st.session_state.english_quest_last_result,
    state_key="english_quest",
)

if history:
    submitted_message = None
    if quest_state["complete"]:
        st.success("五关完成。你可以再玩一局，或把本局错题交给另一个 Skill。")
        with st.container(horizontal=True):
            if st.button(
                "再玩一局",
                icon=":material/replay:",
                width="stretch",
            ):
                _reset_quest()
                st.rerun()
            if st.button(
                "整理本局错题",
                icon=":material/library_add:",
                type="primary",
                width="stretch",
            ):
                submitted_message = "把刚才答错的题整理进错题本。"
    else:
        submitted_message = st.chat_input(
            "输入答案，或说“给我一个提示”",
            key="english_quest_chat",
            max_chars=2_000,
            submit_mode="disable",
        )

    if submitted_message is not None:
        clean_message = submitted_message.strip()
        if not clean_message:
            st.session_state.english_quest_last_attempt = submitted_message
            st.session_state.english_quest_last_error = "请输入内容后再发送。"
            st.rerun()
        with st.spinner("侦探正在核对线索…", show_time=True):
            _run_turn(clean_message)
        st.rerun()
