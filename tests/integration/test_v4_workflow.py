import os
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
import pytest

import src.workflow as workflow_module


def _write_mistake(records_root: Path) -> None:
    path = records_root / "math" / "mistake-integration.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "schema_version: 1\n"
        "id: mistake-integration\n"
        "subject: math\n"
        "topic: linear-equations\n"
        "status: needs-review\n"
        'created_at: "2026-08-15"\n'
        "review_count: 0\n"
        "next_review_at: null\n"
        'source: "chat"\n'
        "---\n\n"
        "# 错题记录\n\n"
        "- 学科：数学\n"
        "- 题型：一元一次方程\n"
        "- 原题：x + 2 = 5\n"
        "- 我的答案：x = 2\n"
        "- 正确答案：x = 3\n"
        "- 正确思路：等式两边同时减去 2。\n"
        "- 错因：移项时计算错误。\n"
        "- 知识点：一元一次方程\n"
        "- 下次提醒：把答案代回原式检查。\n",
        encoding="utf-8",
    )


@pytest.mark.integration
def test_v4_generates_grounded_report_and_practice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if os.getenv("RUN_MODEL_INTEGRATION") != "1":
        pytest.skip("设置 RUN_MODEL_INTEGRATION=1 后才调用真实模型。")

    records_root = tmp_path / "student" / "mistakes" / "records"
    report_path = tmp_path / "student" / "reports" / "learning-review.md"
    _write_mistake(records_root)
    monkeypatch.setattr(workflow_module, "MISTAKES_RECORDS_PATH", records_root)
    monkeypatch.setattr(workflow_module, "LEARNING_REPORT_PATH", report_path)
    monkeypatch.setattr(
        workflow_module,
        "_V4_GRAPH",
        workflow_module.build_v4_graph(checkpointer=InMemorySaver()),
    )

    result = workflow_module.chat_v4("总结复盘", "integration-review")

    assert result["error"] is None, result["error"]
    assert result["waiting_for"] == "student_message"
    assert "本次个性化练习" in result["text"]
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "mistake-integration" in report
    assert "## 本次个性化练习" in report
