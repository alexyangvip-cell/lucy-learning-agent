import os

import pytest

from src.facade import invoke


@pytest.mark.integration
def test_v3_retrieves_and_cites_default_knowledge_card() -> None:
    if os.getenv("RUN_MODEL_INTEGRATION") != "1":
        pytest.skip("设置 RUN_MODEL_INTEGRATION=1 后才调用真实模型。")

    result = invoke(
        "V3",
        "I ____ (read) this book three times. 请提示我先观察哪个线索。",
    )

    assert result["error"] is None, result["error"]
    assert result["text"].strip()
    assert "[english-grammar-present-perfect]" in result["text"]
    assert result["citations"][0]["source"] == (
        "student/knowledge/english/grammar/present-perfect.md"
    )
    assert result["citations"][0]["title"] == "现在完成时"
    evidence_fields = {
        match["field"] for match in result["citations"][0]["matches"]
    }
    assert evidence_fields
    assert evidence_fields <= {"核心规则", "例句", "易错提醒"}
    assert result["trace"][0]["status"] == "hit"
