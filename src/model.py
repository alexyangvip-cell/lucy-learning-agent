"""所有学习 Agent 共用的模型配置入口。"""

import os
import tempfile
from pathlib import Path
from typing import Literal, TypedDict, cast

from dotenv import dotenv_values, set_key
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE_FILE = PROJECT_ROOT / ".env.example"
ModelProvider = Literal["deepseek", "moonshot", "gemini"]
MODEL_PROVIDER_KEYS: dict[ModelProvider, str] = {
    "deepseek": "DEEPSEEK_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
_MODEL_SETTING_NAMES = (
    "MODEL_PROVIDER",
    *MODEL_PROVIDER_KEYS.values(),
    "MODEL_TEMPERATURE",
    "MODEL_TIMEOUT_SECONDS",
    "MODEL_MAX_RETRIES",
)
_MAX_ENV_FILE_BYTES = 64 * 1024


class ModelConfigurationSummary(TypedDict):
    """可安全展示的模型配置摘要，不包含 API Key。"""

    provider: ModelProvider
    configured_providers: list[ModelProvider]


class ModelConfigurationError(ValueError):
    """模型环境变量缺失或无效。"""


def _read_dotenv_file(path: Path, *, label: str) -> dict[str, str]:
    if path.is_symlink():
        raise ModelConfigurationError(f"{label} 不能是符号链接。")
    if not path.exists():
        return {}
    if not path.is_file():
        raise ModelConfigurationError(f"{label} 必须是普通文件。")
    try:
        if path.stat().st_size > _MAX_ENV_FILE_BYTES:
            raise ModelConfigurationError(
                f"{label} 超过 64 KB，请缩短后重试。"
            )
        values = dotenv_values(path, encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ModelConfigurationError(
            f"无法读取 {label}，请检查文件权限和编码。"
        ) from exc
    return {key: value for key, value in values.items() if value is not None}


def _read_model_environment() -> dict[str, str]:
    """每次都重新读取 .env，同时保留运行进程环境的优先级。"""

    values = _read_dotenv_file(ENV_FILE, label=".env")
    for name in _MODEL_SETTING_NAMES:
        if name in os.environ:
            values[name] = os.environ[name]
    return values


def _normalize_provider(provider: str) -> ModelProvider:
    normalized = provider.strip().lower() if isinstance(provider, str) else ""
    if normalized not in MODEL_PROVIDER_KEYS:
        raise ModelConfigurationError(
            "MODEL_PROVIDER 必须是 deepseek、moonshot 或 gemini。"
        )
    return cast(ModelProvider, normalized)


def get_model_configuration_summary() -> ModelConfigurationSummary:
    """返回前端可展示的配置状态，永不返回 API Key。"""

    values = _read_model_environment()
    raw_provider = values.get("MODEL_PROVIDER", "deepseek").strip().lower()
    provider = cast(
        ModelProvider,
        raw_provider if raw_provider in MODEL_PROVIDER_KEYS else "deepseek",
    )
    configured_providers = [
        candidate
        for candidate, key_name in MODEL_PROVIDER_KEYS.items()
        if values.get(key_name, "").strip()
    ]
    return {
        "provider": provider,
        "configured_providers": configured_providers,
    }


def _read_env_text(path: Path, *, label: str) -> str:
    _read_dotenv_file(path, label=label)
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ModelConfigurationError(
            f"无法读取 {label}，请检查文件权限和编码。"
        ) from exc


def save_model_configuration(
    provider: str,
    api_key: str,
) -> ModelConfigurationSummary:
    """安全保存所选供应商和 Key，并立即同步到当前进程。"""

    normalized_provider = _normalize_provider(provider)
    if not isinstance(api_key, str):
        raise ModelConfigurationError("API Key 必须是文本。")
    if "\n" in api_key or "\r" in api_key:
        raise ModelConfigurationError("API Key 不能包含换行符。")
    if "\0" in api_key:
        raise ModelConfigurationError("API Key 不能包含空字符。")

    clean_api_key = api_key.strip()
    key_name = MODEL_PROVIDER_KEYS[normalized_provider]
    current_values = _read_model_environment()
    if not clean_api_key and not current_values.get(key_name, "").strip():
        raise ModelConfigurationError(
            f"尚未保存 {key_name}，请填写 API Key 后再试。"
        )

    if ENV_FILE.exists():
        base_content = _read_env_text(ENV_FILE, label=".env")
    else:
        if not ENV_EXAMPLE_FILE.is_file():
            raise ModelConfigurationError(
                "缺少 .env.example，请从原始课程包恢复后重试。"
            )
        base_content = _read_env_text(ENV_EXAMPLE_FILE, label=".env.example")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=ENV_FILE.parent,
            prefix=f".{ENV_FILE.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(base_content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temp_path = Path(temporary.name)

        if clean_api_key:
            set_key(
                temp_path,
                key_name,
                clean_api_key,
                quote_mode="always",
                encoding="utf-8",
            )
        set_key(
            temp_path,
            "MODEL_PROVIDER",
            normalized_provider,
            quote_mode="never",
            encoding="utf-8",
        )
        saved_values = _read_dotenv_file(temp_path, label="新的 .env")
        effective_api_key = (
            saved_values.get(key_name, "").strip()
            or current_values.get(key_name, "").strip()
        )
        if not effective_api_key:
            raise ModelConfigurationError(
                f"尚未保存 {key_name}，请填写 API Key 后再试。"
            )
        os.replace(temp_path, ENV_FILE)
    except ModelConfigurationError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ModelConfigurationError(
            "无法保存 .env，原配置保持不变。请检查文件权限。"
        ) from exc
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    os.environ["MODEL_PROVIDER"] = normalized_provider
    os.environ[key_name] = effective_api_key
    return get_model_configuration_summary()


def get_model_name(llm: BaseChatModel) -> str:
    """返回不同 LangChain 模型适配器使用的模型名称。"""

    model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None)
    if not isinstance(model_name, str) or not model_name:
        raise ModelConfigurationError("当前模型适配器未提供可显示的模型名称。")
    return model_name


def _read_float(
    values: dict[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
) -> float:
    raw_value = values.get(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ModelConfigurationError(
            f"{name} 必须是数字，当前值为 {raw_value!r}。请修改 .env。"
        ) from exc
    if value < minimum:
        raise ModelConfigurationError(
            f"{name} 必须大于或等于 {minimum}，当前值为 {value}。请修改 .env。"
        )
    return value


def _read_int(
    values: dict[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    raw_value = values.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ModelConfigurationError(
            f"{name} 必须是整数，当前值为 {raw_value!r}。请修改 .env。"
        ) from exc
    if value < minimum:
        raise ModelConfigurationError(
            f"{name} 必须大于或等于 {minimum}，当前值为 {value}。请修改 .env。"
        )
    return value


def _validate_model_environment(values: dict[str, str]) -> ModelProvider:
    provider = _normalize_provider(values.get("MODEL_PROVIDER", "deepseek"))
    key_name = MODEL_PROVIDER_KEYS[provider]
    if not values.get(key_name, "").strip():
        raise ModelConfigurationError(
            f"MODEL_PROVIDER={provider}，但未找到 {key_name}。"
            "请在首页填写模型配置。"
        )
    return provider


def validate_model_configuration() -> ModelProvider:
    """校验模型供应商和 API Key，但不创建模型实例。"""

    return _validate_model_environment(_read_model_environment())


def get_llm() -> BaseChatModel:
    """根据 MODEL_PROVIDER 创建所有 V0-V4 共用的聊天模型实例。"""

    values = _read_model_environment()
    provider = _validate_model_environment(values)
    api_key = values[MODEL_PROVIDER_KEYS[provider]].strip()

    timeout = _read_float(values, "MODEL_TIMEOUT_SECONDS", 45.0, minimum=0.1)
    max_retries = _read_int(values, "MODEL_MAX_RETRIES", 2, minimum=0)

    if provider == "deepseek":
        temperature = _read_float(values, "MODEL_TEMPERATURE", 0.0, minimum=0.0)
        return ChatDeepSeek(
            model="deepseek-v4-flash",
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            api_key=api_key,
            extra_body={"thinking": {"type": "disabled"}},
        )

    if provider == "moonshot":
        return ChatOpenAI(
            model="kimi-k2.6",
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
            timeout=timeout,
            max_retries=max_retries,
            use_responses_api=False,
            extra_body={"thinking": {"type": "disabled"}},
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )
