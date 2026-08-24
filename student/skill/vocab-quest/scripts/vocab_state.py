"""从 vocab-quest 对话中提取供互动页面展示的游戏状态。"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from typing import TypedDict


class VocabState(TypedDict):
    """页面展示所需的确定性状态。"""

    round: int
    hp: int
    xp: int
    streak: int
    complete: bool
    has_status_line: bool
    template_complete: bool
    deviations: list[str]


_ROUND_PATTERNS = (
    re.compile(r"轮次\s*[：:]?\s*(\d+)\s*/\s*5"),
    re.compile(r"第\s*(\d+)\s*词"),
)
_HP_NUMBER_PATTERN = re.compile(r"(?:记忆值|剩余记忆值)\s*[：:]?\s*(\d+)")
_HP_EMOJI_PATTERN = re.compile(r"记忆值\s*[：:]?\s*((?:❤️|❤|♥️?)+)")
_EXPLICIT_XP_PATTERNS = (
    re.compile(r"经验\s*[：:]?\s*(\d+)\s*XP", re.IGNORECASE),
    re.compile(r"(?:最终\s*)?XP\s*[：:]?\s*(\d+)", re.IGNORECASE),
)
_XP_GAIN_PATTERN = re.compile(r"获得\s*(\d+)\s*XP", re.IGNORECASE)
_STREAK_PATTERNS = (
    re.compile(r"连击\s*[：:]?\s*(\d+)"),
    re.compile(r"(\d+)\s*🔥"),
)
_COMPLETE_MARKERS = ("闯关报告", "任务完成", "背词完成")


# 模板段落标题，用于轻量模板校验，不阻断流程。
_FEEDBACK_HEADING = re.compile(
    r"###\s+(?:✅\s*(?:反馈)?|❌\s*首答反馈|🔁\s*重试反馈|🆘\s*救援反馈)"
)
_WORD_CARD_HEADING = re.compile(r"###\s+📇\s*单词卡")
_CHALLENGE_HEADING = re.compile(r"###\s+🎯\s*第\s*\d+\s*词")
_REPORT_HEADINGS = (
    re.compile(r"###\s+📋\s*闯关报告"),
    re.compile(r"###\s+✅\s*已掌握"),
    re.compile(r"###\s+⚠️\s*易错词"),
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


def _latest_hp(texts: Sequence[str]) -> int | None:
    for text in reversed(texts):
        emoji_match = _HP_EMOJI_PATTERN.search(text)
        if emoji_match:
            return len(re.findall(r"❤️|❤|♥️?", emoji_match.group(1)))
        number_match = _HP_NUMBER_PATTERN.search(text)
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


def _latest_streak(texts: Sequence[str]) -> int:
    for text in reversed(texts):
        for pattern in _STREAK_PATTERNS:
            match = pattern.search(text)
            if match:
                return int(match.group(1))
    return 0


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line
    return ""


def _is_status_line(line: str) -> bool:
    """一行同时包含轮次、记忆值、经验、连击四要素才算状态行。"""

    has_round = any(pattern.search(line) for pattern in _ROUND_PATTERNS)
    has_hp = bool(
        _HP_NUMBER_PATTERN.search(line) or _HP_EMOJI_PATTERN.search(line)
    )
    has_xp = any(pattern.search(line) for pattern in _EXPLICIT_XP_PATTERNS)
    has_streak = any(pattern.search(line) for pattern in _STREAK_PATTERNS)
    return bool(has_round and has_hp and has_xp and has_streak)


def _detect_template_drift(
    latest_text: str,
    complete: bool,
) -> tuple[bool, bool, list[str]]:
    """检查最新 assistant 消息是否遵循 SKILL.md 的 4 段模板，返回 (has_status_line, template_complete, deviations)。"""

    deviations: list[str] = []
    first_line = _first_nonempty_line(latest_text)
    has_status_line = bool(first_line) and _is_status_line(first_line)
    if not has_status_line:
        deviations.append("首行不是状态行（轮次/记忆值/经验/连击 缺一不可）")

    if complete:
        title_hits = any(
            pattern.search(latest_text) for pattern in _REPORT_HEADINGS
        )
        if not title_hits:
            deviations.append(
                "缺少闯关报告段标题（### 📋 闯关报告 / ### ✅ 已掌握 / ### ⚠️ 易错词 / ### 🎯 下一步 之一）"
            )
        template_complete = title_hits
    else:
        has_feedback = bool(_FEEDBACK_HEADING.search(latest_text))
        has_card = bool(_WORD_CARD_HEADING.search(latest_text))
        has_challenge = bool(_CHALLENGE_HEADING.search(latest_text))
        if not has_feedback:
            deviations.append(
                "缺少反馈段标题（### ✅ 反馈 / ### ❌ 首答反馈 / ### 🔁 重试反馈 / ### 🆘 救援反馈 之一）"
            )
        if not has_card:
            deviations.append("缺少单词卡段标题（### 📇 单词卡）")
        if not has_challenge:
            deviations.append("缺少挑战段标题（### 🎯 第 N 词）")
        template_complete = has_feedback and has_card and has_challenge

    return has_status_line, template_complete, deviations


def derive_vocab_state(history: Sequence[dict[str, object]]) -> VocabState:
    """读取对话，不猜测答题对错，只返回可验证的展示状态。"""

    texts = _assistant_texts(history)
    latest_text = texts[-1] if texts else ""
    complete = any(marker in latest_text for marker in _COMPLETE_MARKERS)
    round_ = _latest_match(texts, _ROUND_PATTERNS)
    hp = _latest_hp(texts)
    xp = _latest_xp(texts)
    streak = _latest_streak(texts)
    has_status_line, template_complete, deviations = _detect_template_drift(
        latest_text, complete
    )
    return {
        "round": 5 if complete else max(0, min(round_ or 0, 5)),
        "hp": max(0, min(hp if hp is not None else 3, 3)),
        "xp": max(0, xp),
        "streak": max(0, streak),
        "complete": complete,
        "has_status_line": has_status_line,
        "template_complete": template_complete,
        "deviations": deviations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="提取背单词闯关页面状态")
    parser.add_argument("history_json", help="包含对话历史的 JSON 文件")
    args = parser.parse_args()
    with open(args.history_json, encoding="utf-8") as history_file:
        history = json.load(history_file)
    print(json.dumps(derive_vocab_state(history), ensure_ascii=False))


if __name__ == "__main__":
    main()
