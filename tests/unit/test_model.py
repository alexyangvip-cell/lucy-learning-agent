import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from dotenv import dotenv_values

import src.model as model_module


_MODEL_ENVIRONMENT_VARIABLES = (
    "MODEL_PROVIDER",
    "DEEPSEEK_API_KEY",
    "MOONSHOT_API_KEY",
    "GEMINI_API_KEY",
    "MODEL_NAME",
    "MODEL_TEMPERATURE",
    "MODEL_TIMEOUT_SECONDS",
    "MODEL_MAX_RETRIES",
)


@pytest.fixture(autouse=True)
def isolate_model_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_example_file = tmp_path / ".env.example"
    env_example_file.write_text(
        "# 可选模型配置\n"
        "MODEL_PROVIDER=deepseek\n"
        "DEEPSEEK_API_KEY=\n"
        "MOONSHOT_API_KEY=\n"
        "GEMINI_API_KEY=\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(model_module, "ENV_FILE", env_file)
    monkeypatch.setattr(
        model_module,
        "ENV_EXAMPLE_FILE",
        env_example_file,
        raising=False,
    )
    for name in _MODEL_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("provider", "key_name"),
    [
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("moonshot", "MOONSHOT_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
    ],
)
def test_get_llm_requires_selected_provider_key(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    key_name: str,
) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", provider)

    with pytest.raises(model_module.ModelConfigurationError, match=key_name):
        model_module.get_llm()


def test_get_llm_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "unknown")

    with pytest.raises(
        model_module.ModelConfigurationError,
        match="deepseek、moonshot 或 gemini",
    ):
        model_module.get_llm()


def test_get_llm_defaults_to_deepseek(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    constructor = Mock(return_value=object())
    monkeypatch.setattr(model_module, "ChatDeepSeek", constructor)

    model_module.get_llm()

    constructor.assert_called_once_with(
        model="deepseek-v4-flash",
        temperature=0.0,
        timeout=45.0,
        max_retries=2,
        api_key="deepseek-key",
        extra_body={"thinking": {"type": "disabled"}},
    )


def test_get_llm_prefers_running_process_environment_over_env_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_module.ENV_FILE.write_text(
        "MODEL_PROVIDER=deepseek\nDEEPSEEK_API_KEY=file-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "process-key")
    constructor = Mock(return_value=object())
    monkeypatch.setattr(model_module, "ChatDeepSeek", constructor)

    model_module.get_llm()

    assert constructor.call_args.kwargs["api_key"] == "process-key"


def test_validate_model_configuration_does_not_construct_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    constructor = Mock()
    monkeypatch.setattr(model_module, "ChatDeepSeek", constructor)

    provider = model_module.validate_model_configuration()

    assert provider == "deepseek"
    constructor.assert_not_called()


def test_get_llm_selects_moonshot_and_only_requires_its_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "moonshot")
    monkeypatch.setenv("MOONSHOT_API_KEY", "moonshot-key")
    monkeypatch.setenv("MODEL_NAME", "ignored-model")
    monkeypatch.setenv("MODEL_TEMPERATURE", "0")
    constructor = Mock(return_value=object())
    monkeypatch.setattr(model_module, "ChatOpenAI", constructor)

    model_module.get_llm()

    constructor.assert_called_once_with(
        model="kimi-k2.6",
        api_key="moonshot-key",
        base_url="https://api.moonshot.cn/v1",
        timeout=45.0,
        max_retries=2,
        use_responses_api=False,
        extra_body={"thinking": {"type": "disabled"}},
    )


def test_get_llm_selects_gemini_and_only_requires_its_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "GEMINI")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("MODEL_NAME", "ignored-model")
    monkeypatch.setenv("MODEL_TEMPERATURE", "0")
    constructor = Mock(return_value=object())
    monkeypatch.setattr(model_module, "ChatGoogleGenerativeAI", constructor)

    model_module.get_llm()

    constructor.assert_called_once_with(
        model="gemini-3.6-flash",
        api_key="gemini-key",
        timeout=45.0,
        max_retries=2,
    )


@pytest.mark.parametrize(
    ("provider", "key_name", "expected_name"),
    [
        ("deepseek", "DEEPSEEK_API_KEY", "deepseek-v4-flash"),
        ("moonshot", "MOONSHOT_API_KEY", "kimi-k2.6"),
        ("gemini", "GEMINI_API_KEY", "gemini-3.6-flash"),
    ],
)
def test_get_model_name_supports_all_providers(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    key_name: str,
    expected_name: str,
) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", provider)
    monkeypatch.setenv(key_name, "test-key")

    llm = model_module.get_llm()

    assert model_module.get_model_name(llm) == expected_name


def test_get_llm_uses_shared_timeout_and_retry_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_TEMPERATURE", "0.2")
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("MODEL_MAX_RETRIES", "3")
    constructor = Mock(return_value=object())
    monkeypatch.setattr(model_module, "ChatDeepSeek", constructor)

    model_module.get_llm()

    constructor.assert_called_once_with(
        model="deepseek-v4-flash",
        temperature=0.2,
        timeout=30.0,
        max_retries=3,
        api_key="test-key",
        extra_body={"thinking": {"type": "disabled"}},
    )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("MODEL_TEMPERATURE", "warm", "MODEL_TEMPERATURE 必须是数字"),
        ("MODEL_TIMEOUT_SECONDS", "0", "MODEL_TIMEOUT_SECONDS 必须大于或等于"),
        ("MODEL_MAX_RETRIES", "1.5", "MODEL_MAX_RETRIES 必须是整数"),
    ],
)
def test_get_llm_rejects_invalid_numeric_settings(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv(name, value)

    with pytest.raises(model_module.ModelConfigurationError, match=message):
        model_module.get_llm()


def test_model_configuration_summary_never_contains_api_keys() -> None:
    secret = "summary-must-not-contain-this-key"
    model_module.ENV_FILE.write_text(
        "MODEL_PROVIDER=moonshot\n"
        "DEEPSEEK_API_KEY=deepseek-key\n"
        f"MOONSHOT_API_KEY={secret}\n",
        encoding="utf-8",
    )

    summary = model_module.get_model_configuration_summary()

    assert summary == {
        "provider": "moonshot",
        "configured_providers": ["deepseek", "moonshot"],
    }
    assert secret not in str(summary)


def test_save_model_configuration_creates_env_from_example_and_is_immediate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "new-moonshot-key"
    constructor = Mock(return_value=object())
    monkeypatch.setattr(model_module, "ChatOpenAI", constructor)

    summary = model_module.save_model_configuration("moonshot", secret)
    model_module.get_llm()

    assert summary == {
        "provider": "moonshot",
        "configured_providers": ["moonshot"],
    }
    assert model_module.ENV_FILE.is_file()
    contents = model_module.ENV_FILE.read_text(encoding="utf-8")
    assert "# 可选模型配置" in contents
    assert dotenv_values(model_module.ENV_FILE)["MOONSHOT_API_KEY"] == secret
    assert constructor.call_args.kwargs["api_key"] == secret
    assert not list(model_module.ENV_FILE.parent.glob("..env.*.tmp"))
    if os.name != "nt":
        assert model_module.ENV_FILE.stat().st_mode & 0o077 == 0


def test_save_model_configuration_preserves_other_keys_and_unknown_settings() -> None:
    model_module.ENV_FILE.write_text(
        "# 保留这条注释\n"
        "MODEL_PROVIDER=deepseek\n"
        "DEEPSEEK_API_KEY=existing-deepseek-key\n"
        "CUSTOM_SETTING=keep-me\n",
        encoding="utf-8",
    )

    model_module.save_model_configuration("gemini", "new-gemini-key")

    values = dotenv_values(model_module.ENV_FILE)
    contents = model_module.ENV_FILE.read_text(encoding="utf-8")
    assert values["MODEL_PROVIDER"] == "gemini"
    assert values["DEEPSEEK_API_KEY"] == "existing-deepseek-key"
    assert values["GEMINI_API_KEY"] == "new-gemini-key"
    assert values["CUSTOM_SETTING"] == "keep-me"
    assert "# 保留这条注释" in contents


def test_save_model_configuration_blank_key_reuses_saved_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_module.ENV_FILE.write_text(
        "MODEL_PROVIDER=deepseek\nMOONSHOT_API_KEY=saved-moonshot-key\n",
        encoding="utf-8",
    )
    constructor = Mock(return_value=object())
    monkeypatch.setattr(model_module, "ChatOpenAI", constructor)

    model_module.save_model_configuration("moonshot", "")
    model_module.get_llm()

    assert dotenv_values(model_module.ENV_FILE)["MOONSHOT_API_KEY"] == (
        "saved-moonshot-key"
    )
    assert constructor.call_args.kwargs["api_key"] == "saved-moonshot-key"


def test_save_model_configuration_rejects_blank_missing_key() -> None:
    with pytest.raises(
        model_module.ModelConfigurationError,
        match="MOONSHOT_API_KEY",
    ):
        model_module.save_model_configuration("moonshot", "")

    assert not model_module.ENV_FILE.exists()


@pytest.mark.parametrize(
    ("provider", "api_key", "message"),
    [
        ("unknown", "safe-key", "deepseek、moonshot 或 gemini"),
        ("deepseek", "line-one\nline-two", "不能包含换行符"),
        ("deepseek", "nul\0key", "不能包含空字符"),
    ],
)
def test_save_model_configuration_rejects_invalid_input_without_leaking_key(
    provider: str,
    api_key: str,
    message: str,
) -> None:
    with pytest.raises(model_module.ModelConfigurationError, match=message) as error:
        model_module.save_model_configuration(provider, api_key)

    assert api_key not in str(error.value)
    assert not model_module.ENV_FILE.exists()


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows 普通用户可能无法创建符号链接",
)
def test_save_model_configuration_rejects_symlink() -> None:
    target = model_module.ENV_FILE.parent / "outside.env"
    target.write_text("MODEL_PROVIDER=deepseek\n", encoding="utf-8")
    model_module.ENV_FILE.symlink_to(target)

    with pytest.raises(model_module.ModelConfigurationError, match="符号链接"):
        model_module.save_model_configuration("deepseek", "new-key")

    assert target.read_text(encoding="utf-8") == "MODEL_PROVIDER=deepseek\n"


def test_get_llm_reloads_env_file_on_each_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor = Mock(return_value=object())
    monkeypatch.setattr(model_module, "ChatDeepSeek", constructor)
    model_module.ENV_FILE.write_text(
        "MODEL_PROVIDER=deepseek\nDEEPSEEK_API_KEY=first-key\n",
        encoding="utf-8",
    )

    model_module.get_llm()
    model_module.ENV_FILE.write_text(
        "MODEL_PROVIDER=deepseek\nDEEPSEEK_API_KEY=second-key\n",
        encoding="utf-8",
    )
    model_module.get_llm()

    received_keys = [call.kwargs["api_key"] for call in constructor.call_args_list]
    assert received_keys == ["first-key", "second-key"]
