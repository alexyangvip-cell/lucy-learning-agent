import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "student"
    / "skill"
    / "english-quest"
    / "scripts"
    / "quest_state.py"
)


def _derive(history: list[dict]) -> dict:
    namespace = runpy.run_path(str(SCRIPT_PATH))
    return namespace["derive_quest_state"](history)


def _expected_state(**overrides: object) -> dict:
    state = {
        "level": 0,
        "hearts": 3,
        "xp": 0,
        "complete": False,
        "has_status_line": False,
        "template_complete": False,
        "deviations": [],
    }
    state.update(overrides)
    return state


def test_quest_state_script_extracts_latest_visible_status() -> None:
    history = [
        {"role": "assistant", "content": "关卡：1/5 生命：❤️❤️❤️ 经验：0 XP"},
        {"role": "user", "content": "B"},
        {"role": "assistant", "content": "关卡：2/5 生命：❤️❤️❤️ 经验：10 XP"},
    ]

    assert _derive(history) == _expected_state(
        level=2,
        hearts=3,
        xp=10,
        # 旧版消息没有 ### 段落标题，应记为漂移。
        has_status_line=True,
        template_complete=False,
        deviations=[
            "缺少反馈段标题（### ✅ 反馈 / ### ❌ 首答反馈 / ### 🔁 重试反馈 / ### 🆘 救援反馈 之一）",
            "缺少挑战段标题（### 🎯 第 N 关）",
        ],
    )


def test_quest_state_script_handles_retry_and_final_report() -> None:
    retry_history = [
        {"role": "assistant", "content": "第 3 关\n生命：2\nXP：20"},
    ]
    completed_history = retry_history + [
        {
            "role": "assistant",
            "content": "任务报告\n最终 XP：40\n剩余生命：2",
        }
    ]

    assert _derive(retry_history) == _expected_state(
        level=3,
        hearts=2,
        xp=20,
        # 首行是 "第 3 关"，不是完整状态行；也没段落标题。
        deviations=[
            "首行不是状态行（关卡/生命/经验 缺一不可）",
            "缺少反馈段标题（### ✅ 反馈 / ### ❌ 首答反馈 / ### 🔁 重试反馈 / ### 🆘 救援反馈 之一）",
            "缺少挑战段标题（### 🎯 第 N 关）",
        ],
    )
    assert _derive(completed_history) == _expected_state(
        level=5,
        hearts=2,
        xp=40,
        complete=True,
        deviations=[
            "首行不是状态行（关卡/生命/经验 缺一不可）",
            "缺少任务报告段标题（### 📋 任务报告 / ### ✅ 已掌握 / ### ⚠️ 易错点 / ### 🎯 下一步 之一）",
        ],
    )


def test_quest_state_script_accepts_natural_status_without_colons() -> None:
    history = [
        {
            "role": "assistant",
            "content": "关卡：1/5 生命：❤️❤️❤️ 经验：0 XP",
        },
        {
            "role": "assistant",
            "content": "关卡 1/5 生命 ❤️❤️❤️ 获得 10 XP",
        },
        {
            "role": "assistant",
            "content": "关卡 2/5 生命 ❤️❤️ 获得 10 XP",
        },
    ]

    assert _derive(history) == _expected_state(
        level=2,
        hearts=2,
        xp=20,
        # 老式省略冒号且把 XP 表达成"获得 …"——脚本仍能解析出 level/hearts/xp，
        # 但首行不再满足 SKILL.md 的强制状态行格式，记为漂移。
        deviations=[
            "首行不是状态行（关卡/生命/经验 缺一不可）",
            "缺少反馈段标题（### ✅ 反馈 / ### ❌ 首答反馈 / ### 🔁 重试反馈 / ### 🆘 救援反馈 之一）",
            "缺少挑战段标题（### 🎯 第 N 关）",
        ],
    )


def test_quest_state_script_recognises_compliant_template() -> None:
    """最新一条 assistant 消息遵循 4 段模板（状态行 + 反馈 + 剧情 + 挑战）时应全绿。"""

    compliant = (
        "关卡：2/5 生命：❤️❤️❤️ 经验：10 XP\n\n"
        "### ✅ 反馈\n"
        "- 反馈：went 是过去式。\n"
        "- 考点：一般过去时 yesterday。\n\n"
        "### 🎬 剧情\n"
        "NPC Sam 推了推眼镜。\n\n"
        "### 🎯 第 3 关\n"
        "你跑去食堂……\n\n"
        "你选哪一个？"
    )
    history = [
        {"role": "user", "content": "B"},
        {"role": "assistant", "content": compliant},
    ]

    state = _derive(history)
    assert state["level"] == 2
    assert state["hearts"] == 3
    assert state["xp"] == 10
    assert state["has_status_line"] is True
    assert state["template_complete"] is True
    assert state["deviations"] == []


def test_quest_state_script_flags_drift_status_line() -> None:
    """首行状态行缺生命或经验时仍要单独记漂移。"""

    history = [
        {
            "role": "assistant",
            "content": (
                "关卡：3/5\n\n"
                "### ✅ 反馈\n- 反馈：…\n\n"
                "### 🎯 第 4 关\n你选哪一个？"
            ),
        }
    ]
    state = _derive(history)
    assert state["has_status_line"] is False
    assert state["template_complete"] is True
    assert any("首行" in d for d in state["deviations"])


def test_quest_state_script_flags_missing_section_headings() -> None:
    """状态行在但缺反馈段或挑战段，应各自加一条 deviations。"""

    history = [
        {
            "role": "assistant",
            "content": (
                "关卡：3/5 生命：❤️❤️❤️ 经验：30 XP\n\n"
                "你直奔图书馆，线索指向另一处。\n\n"
                "I ___ this book three times."
            ),
        }
    ]
    state = _derive(history)
    assert state["has_status_line"] is True
    assert state["template_complete"] is False
    assert len(state["deviations"]) == 2
    assert any("反馈段" in d for d in state["deviations"])
    assert any("挑战段" in d for d in state["deviations"])
