"""青少年个性化学习 Agent 的 Streamlit 入口。"""

import streamlit as st


page = st.navigation(
    [
        st.Page(
            "app_pages/home.py",
            title="课程首页",
            icon=":material/home:",
            default=True,
        ),
        st.Page(
            "app_pages/lesson_1.py",
            title="第一课：苏格拉底智能体与错题整理智能体",
            icon=":material/school:",
        ),
        st.Page(
            "app_pages/lesson_2.py",
            title="第二课：名师智能体与个性化智能体",
            icon=":material/psychology:",
        ),
        st.Page(
            "app_pages/english_quest.py",
            title="英语剧情闯关",
            icon=":material/stadia_controller:",
        ),
    ],
    position="sidebar",
)
page.run()
