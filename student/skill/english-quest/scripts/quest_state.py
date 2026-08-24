"""从 english-quest 对话中提取供互动页面展示的游戏状态。"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from typing import TypedDict


class QuestState(TypedDict):
    """页面展示所需的确定性状态。"""

    level: int
    hearts: int
    xp: int
    complete: bool
    has_status_line: bool
    template_complete: bool
    deviations: list[str]


_LEVEL_PATTERNS = (
    re.compile(r"关卡\s*[：:]?\s*(\d+)\s*/\s*5"),
    re.compile(r"第\s*(\d+)\s*关"),
)
_HEARTS_NUMBER_PATTERN = re.compile(r"(?:生命|剩余生命)\s*[：:]?\s*(\d+)")
_HEARTS_EMOJI_PATTERN = re.compile(r"生命\s*[：:]?\s*((?:❤️|❤|♥️?)+)")
_EXPLICIT_XP_PATTERNS = (
    re.compile(r"经验\s*[：:]?\s*(\d+)\s*XP", re.IGNORECASE),
    re.compile(r"(?:最终\s*)?XP\s*[：:]?\s*(\d+)", re.IGNORECASE),
)
_XP_GAIN_PATTERN = re.compile(r"获得\s*(\d+)\s*XP", re.IGNORECASE)
_COMPLETE_MARKERS = ("任务报告", "任务完成", "闯关完成")

# 模板段落标题，用于轻量模板校验，不阻断流程。
_FEEDBACK_HEADING = re.compile(
    r"###\s+(?:✅\s*(?:首答)?反馈|❌\s*首答反馈|🔁\s*重试反馈|🆘\s*救援反馈)"
)
_CHALLENGE_HEADING = re.compile(r"###\s+🎯\s*第\s*\d+\s*关")
_REPORT_HEADINGS = (
    re.compile(r"###\s+📋\s*任务报告"),
    re.compile(r"###\s+✅\s*已掌握"),
    re.compile(r"###\s+⚠️\s*易错点"),
    re.compile(r"###\s+🎯\s*下一步"),
)


def _assistant_texts(history: Sequence[dict[str, object]]) -> list[str]:
    return [
        str(item.get("content", ""))
        for item in history
        if item.get("role") == "assistant" and item.get("content")
    ]


def _latest_match(
    texts: Sequence[str],
    patterns: Sequence[re.Pattern[str]],
) -> int | None:
    for text in reversed(texts):
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return int(match.group(1))
    return None


def _latest_hearts(texts: Sequence[str]) -> int | None:
    for text in reversed(texts):
        emoji_match = _HEARTS_EMOJI_PATTERN.search(text)
        if emoji_match:
            return len(re.findall(r"❤️|❤|♥️?", emoji_match.group(1)))
        number_match = _HEARTS_NUMBER_PATTERN.search(text)
        if number_match:
            return int(number_match.group(1))
    return None


def _latest_xp(texts: Sequence[str]) -> int:
    for index in range(len(texts) - 1, -1, -1):
        for pattern in _EXPLICIT_XP_PATTERNS:
            match = pattern.search(texts[index])
            if match:
                later_gains = sum(
                    int(gain.group(1))
                    for text in texts[index + 1 :]
                    for gain in _XP_GAIN_PATTERN.finditer(text)
                )
                return int(match.group(1)) + later_gains
    return sum(
        int(match.group(1))
        for text in texts
        for match in _XP_GAIN_PATTERN.finditer(text)
    )


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line
    return ""


def _is_status_line(line: str) -> bool:
    """一行同时包含关卡、生命、经验三要素才算状态行。"""

    has_level = any(pattern.search(line) for pattern in _LEVEL_PATTERNS)
    has_hearts = bool(
        _HEARTS_NUMBER_PATTERN.search(line)
        or _HEARTS_EMOJI_PATTERN.search(line)
    )
    has_xp = any(pattern.search(line) for pattern in _EXPLICIT_XP_PATTERNS)
    return bool(has_level and has_hearts and has_xp)


def _detect_template_drift(
    latest_text: str,
    complete: bool,
) -> tuple[bool, bool, list[str]]:
    """检查最新 assistant 消息是否遵循 SKILL.md 的 4 段模板，返回 (has_status_line, template_complete, deviations)。"""

    deviations: list[str] = []
    first_line = _first_nonempty_line(latest_text)
    has_status_line = bool(first_line) and _is_status_line(first_line)
    if not has_status_line:
        deviations.append("首行不是状态行（关卡/生命/经验 缺一不可）")

    if complete:
        title_hits = any(
            pattern.search(latest_text) for pattern in _REPORT_HEADINGS
        )
        if not title_hits:
            deviations.append(
                "缺少任务报告段标题（### 📋 任务报告 / ### ✅ 已掌握 / ### ⚠️ 易错点 / ### 🎯 下一步 之一）"
            )
        template_complete = title_hits
    else:
        has_feedback = bool(_FEEDBACK_HEADING.search(latest_text))
        has_challenge = bool(_CHALLENGE_HEADING.search(latest_text))
        if not has_feedback:
            deviations.append(
                "缺少反馈段标题（### ✅ 反馈 / ### ❌ 首答反馈 / ### 🔁 重试反馈 / ### 🆘 救援反馈 之一）"
            )
        if not has_challenge:
            deviations.append("缺少挑战段标题（### 🎯 第 N 关）")
        template_complete = has_feedback and has_challenge

    return has_status_line, template_complete, deviations


def derive_quest_state(history: Sequence[dict[str, object]]) -> QuestState:
    """读取对话，不猜测答题对错，只返回可验证的展示状态。"""

    texts = _assistant_texts(history)
    latest_text = texts[-1] if texts else ""
    complete = any(marker in latest_text for marker in _COMPLETE_MARKERS)
    level = _latest_match(texts, _LEVEL_PATTERNS)
    hearts = _latest_hearts(texts)
    xp = _latest_xp(texts)
    has_status_line, template_complete, deviations = _detect_template_drift(
        latest_text, complete
    )
    return {
        "level": 5 if complete else max(0, min(level or 0, 5)),
        "hearts": max(0, min(hearts if hearts is not None else 3, 3)),
        "xp": max(0, xp),
        "complete": complete,
        "has_status_line": has_status_line,
        "template_complete": template_complete,
        "deviations": deviations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="提取英语闯关页面状态")
    parser.add_argument("history_json", help="包含对话历史的 JSON 文件")
    args = parser.parse_args()
    with open(args.history_json, encoding="utf-8") as history_file:
        history = json.load(history_file)
    print(json.dumps(derive_quest_state(history), ensure_ascii=False))


if __name__ == "__main__":
    main()
