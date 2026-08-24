import os

from dotenv import dotenv_values
import pytest

from src.facade import invoke
import src.model as model_module


@pytest.mark.integration
def test_v0_connects_to_selected_model() -> None:
    if os.getenv("RUN_MODEL_INTEGRATION") != "1":
        pytest.skip("设置 RUN_MODEL_INTEGRATION=1 后才调用真实模型。")

    result = invoke("V0", "请只回复：V0 连接成功")

    assert result["error"] is None, result["error"]
    assert result["text"].strip()


@pytest.mark.integration
def test_deepseek_v4_flash_connects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.getenv("RUN_DEEPSEEK_INTEGRATION") != "1":
        pytest.skip("设置 RUN_DEEPSEEK_INTEGRATION=1 后才调用真实 DeepSeek。")

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        api_key = str(
            dotenv_values(model_module.ENV_FILE).get("DEEPSEEK_API_KEY") or ""
        ).strip()
    if not api_key:
        pytest.fail("真实 DeepSeek 测试需要 DEEPSEEK_API_KEY。", pytrace=False)

    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", api_key)
    del api_key

    content = None
    try:
        llm = model_module.get_llm()
        assert model_module.get_model_name(llm) == "deepseek-v4-flash"
        response = llm.invoke("请只回复：DeepSeek 连接成功")
        content = response.content
    except Exception as exc:
        error_type = type(exc).__name__
    else:
        error_type = None

    if error_type is not None:
        pytest.fail(
            "DeepSeek V4 Flash 连通性检查失败，"
            f"异常类型：{error_type}。",
            pytrace=False,
        )

    if not content or (isinstance(content, str) and not content.strip()):
        pytest.fail("DeepSeek V4 Flash 返回了空内容。", pytrace=False)
