"""青少年个性化学习 Agent 的课程首页。"""

from uuid import uuid4

import streamlit as st

from src.facade import (
    AppStatus,
    ModelConfigurationError,
    get_app_status,
    get_model_configuration,
    save_model_configuration,
    test_model_connection,
)
from src.progress import ProgressDataError, default_progress, load_progress

from app_pages.personalization_ui import render_personalization_settings


st.set_page_config(
    page_title="学习 Agent",
    page_icon=":material/explore:",
    layout="wide",
)

PROVIDER_LABELS = {
    "deepseek": "DeepSeek",
    "moonshot": "Kimi",
    "gemini": "Gemini",
}


def initialize_session_state() -> None:
    """集中初始化首页当前需要的 Session State。"""

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"student-{uuid4().hex}"
    if "progress" not in st.session_state:
        try:
            st.session_state.progress = load_progress()
            st.session_state.progress_error = None
        except ProgressDataError as exc:
            st.session_state.progress = default_progress()
            st.session_state.progress_error = str(exc)


def render_model_configuration(status: AppStatus) -> AppStatus:
    """展示本机模型配置表单，且永不回显已保存的 Key。"""

    try:
        configuration = get_model_configuration()
        configuration_error = None
    except ModelConfigurationError as exc:
        configuration = {
            "provider": "deepseek",
            "configured_providers": [],
        }
        configuration_error = str(exc)

    configured_labels = [
        PROVIDER_LABELS[provider]
        for provider in configuration["configured_providers"]
    ]
    with st.expander("模型配置", expanded=not status["model_ready"]):
        st.write("选择模型供应商，再填写该供应商的 API Key。")
        st.caption(
            "API Key 只保存在这台电脑的 .env 文件中，"
            "页面不会显示已保存的 Key。"
        )
        if configured_labels:
            st.caption("已保存：" + "、".join(configured_labels))
        if configuration_error is not None:
            st.warning(configuration_error)
        elif status["model_error"] is not None:
            st.warning(status["model_error"])

        provider_options = list(PROVIDER_LABELS)
        provider_index = provider_options.index(configuration["provider"])
        with st.form("model_configuration", clear_on_submit=True):
            provider = st.selectbox(
                "模型供应商",
                provider_options,
                index=provider_index,
                format_func=PROVIDER_LABELS.__getitem__,
            )
            api_key = st.text_input(
                "API Key",
                type="password",
                autocomplete="new-password",
                placeholder="已保存时可留空",
                help="留空会继续使用该供应商已保存的 Key。",
            )
            st.caption(
                "中国大陆学员建议使用 DeepSeek 或 Kimi。"
                "Gemini 仅供符合官方地区和年龄要求的"
                "教师或开发者测试。"
            )
            submitted = st.form_submit_button("保存并测试连接")

        if submitted:
            try:
                save_model_configuration(provider, api_key)
            except ModelConfigurationError as exc:
                st.error(f"配置未保存。{exc}")
            else:
                connection = test_model_connection()
                if connection["error"]:
                    st.error(
                        "配置已保存，但连接测试失败。"
                        f"{connection['error']}"
                    )
                else:
                    st.success("配置已保存，连接测试成功。")
                status = get_app_status()

    return status


def render_runtime_status(status: AppStatus) -> None:
    """展示配置状态和可以直接执行的修复步骤。"""

    if status["runtime_ready"] and status["model_ready"]:
        provider = status["model_provider"] or "未识别"
        provider_label = PROVIDER_LABELS.get(provider, provider)
        st.success(
            "课程文件和模型配置已就绪，"
            f"当前模型供应商：{provider_label}。"
        )
        return

    if status["runtime_ready"]:
        st.info("课程程序已就绪。请在上方完成模型配置。")
        return

    st.error("运行环境还差一步，请先完成下面的修复。")
    for error in status["runtime_errors"]:
        st.write(f"- {error}")
    if status["missing_files"]:
        st.write("缺少的课程文件：")
        for path in status["missing_files"]:
            st.code(path, language=None)

    with st.expander("家长修复步骤", expanded=True):
        st.markdown(
            "1. 确认当前版本为 Python 3.11.x 至 3.14.x。\n"
            "2. 如果课程文件缺失，请从原始课程包恢复对应文件。\n"
            "3. 恢复文件后重新启动 Streamlit。"
        )


def render_course_card(
    title: str,
    subtitle: str,
    modules: tuple[str, ...],
    completed_modules: set[str],
) -> None:
    """用原生 Streamlit 组件展示一节课的完成进度。"""

    completed_count = sum(module in completed_modules for module in modules)
    progress_value = completed_count / len(modules)
    with st.container(border=True):
        st.subheader(title)
        st.caption(subtitle)
        st.progress(
            progress_value,
            text=f"已完成 {completed_count}/{len(modules)} 个模块",
        )
        st.write(" · ".join(module.upper() for module in modules))


initialize_session_state()
progress = st.session_state.progress
completed_modules = set(progress["completed_modules"])

st.title("你的学习 Agent")
st.write("从会聊天的 AI 开始，一步步训练出真正懂你的学习助手。")

if st.session_state.progress_error:
    st.warning(
        "课程进度文件暂时无法读取，首页已使用默认进度。"
        f"原因：{st.session_state.progress_error}"
    )

app_status = get_app_status()
app_status = render_model_configuration(app_status)
render_runtime_status(app_status)

current_provider = app_status["model_provider"]
current_provider_label = (
    PROVIDER_LABELS.get(current_provider, current_provider)
    if current_provider
    else None
)
render_personalization_settings(current_provider_label)

lesson_names = {"lesson_1": "第一课", "lesson_2": "第二课"}
metric_columns = st.columns(3)
metric_columns[0].metric("当前课程", lesson_names[progress["current_lesson"]])
metric_columns[1].metric("已完成模块", f"{len(completed_modules)}/5")
metric_columns[2].metric("最近位置", progress["last_location"].upper())

st.subheader("课程路线")
lesson_columns = st.columns(2)
with lesson_columns[0]:
    render_course_card(
        "第一课：训练你的学习助手",
        "比较三个关卡，理解教学说明书和技能怎样改变助手。",
        ("v0", "v1", "v2"),
        completed_modules,
    )
with lesson_columns[1]:
    render_course_card(
        "第二课：让助手学会查证和复盘",
        "让助手查找知识卡，并根据你的决定完成长期学习复盘。",
        ("v3", "v4"),
        completed_modules,
    )

st.success("两节课都已经可以体验。请从左侧导航选择课程。")
