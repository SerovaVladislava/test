"""Настройки из переменных окружения и файла .env (ключ API в коде не хранится)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
PROVIDERS = ("anthropic", "openrouter", "ollama")
JSON_MODES = ("json_schema", "json_object")

# Модель и таймаут по умолчанию зависят от провайдера: локальная модель на CPU
# пишет карточку минутами, а не секундами.
DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "openrouter": "nvidia/nemotron-3.5-super-vl:free",
                  "ollama": "qwen2.5:7b"}
DEFAULT_TIMEOUTS = {"anthropic": 180.0, "openrouter": 180.0, "ollama": 1800.0}
CREDENTIAL_HINTS = {
    "anthropic": "не задан ANTHROPIC_API_KEY. Скопируйте .env.example в .env и укажите ключ "
                 "или выберите бесплатный вариант: CARD_PROVIDER=openrouter или CARD_PROVIDER=ollama",
    "openrouter": "не задан OPENROUTER_API_KEY. Создайте бесплатный ключ на https://openrouter.ai/keys "
                  "и впишите его в .env",
}


class ConfigError(Exception):
    """Некорректная настройка в окружении."""


@dataclass(frozen=True)
class Settings:
    provider: str = "anthropic"
    model: str = DEFAULT_MODELS["anthropic"]
    effort: str = "medium"
    max_attempts: int = 3
    fallbacks: bool = True
    timeout: float = DEFAULT_TIMEOUTS["anthropic"]
    ollama_url: str = "http://127.0.0.1:11434"
    num_ctx: int = 8192
    openrouter_url: str = "https://openrouter.ai/api/v1"
    json_mode: str = "json_schema"

    @property
    def has_credentials(self) -> bool:
        if self.provider == "ollama":
            return True  # локальной модели ключ не нужен
        if self.provider == "openrouter":
            return bool(os.getenv("OPENROUTER_API_KEY"))
        return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))

    @property
    def credentials_hint(self) -> str:
        return CREDENTIAL_HINTS.get(self.provider, "")


def _int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as e:
        raise ConfigError(f"{name} должно быть целым числом, получено {raw!r}") from e
    if value < minimum:
        raise ConfigError(f"{name} должно быть не меньше {minimum}")
    return value


def load_settings() -> Settings:
    # Переменные, уже заданные в окружении, имеют приоритет над .env.
    load_dotenv(override=False)

    provider = (os.getenv("CARD_PROVIDER", "").strip() or Settings.provider).lower()
    if provider not in PROVIDERS:
        raise ConfigError(f"CARD_PROVIDER должно быть одним из {', '.join(PROVIDERS)}")

    effort = os.getenv("CARD_EFFORT", "").strip() or Settings.effort
    if effort not in EFFORT_LEVELS:
        raise ConfigError(f"CARD_EFFORT должно быть одним из {', '.join(EFFORT_LEVELS)}")

    fallbacks = (os.getenv("CARD_FALLBACKS", "").strip() or "default").lower()
    if fallbacks not in ("default", "off"):
        raise ConfigError("CARD_FALLBACKS должно быть 'default' или 'off'")

    timeout_raw = os.getenv("CARD_TIMEOUT", "").strip()
    try:
        timeout = float(timeout_raw) if timeout_raw else DEFAULT_TIMEOUTS[provider]
    except ValueError as e:
        raise ConfigError(f"CARD_TIMEOUT должно быть числом, получено {timeout_raw!r}") from e

    ollama_url = os.getenv("CARD_OLLAMA_URL", "").strip() or Settings.ollama_url
    openrouter_url = os.getenv("CARD_OPENROUTER_URL", "").strip() or Settings.openrouter_url
    for name, url in (("CARD_OLLAMA_URL", ollama_url), ("CARD_OPENROUTER_URL", openrouter_url)):
        if not url.startswith(("http://", "https://")):
            raise ConfigError(f"{name} должен начинаться с http:// или https://")

    json_mode = (os.getenv("CARD_JSON_MODE", "").strip() or Settings.json_mode).lower()
    if json_mode not in JSON_MODES:
        raise ConfigError(f"CARD_JSON_MODE должно быть одним из {', '.join(JSON_MODES)}")

    return Settings(
        provider=provider,
        model=os.getenv("CARD_MODEL", "").strip() or DEFAULT_MODELS[provider],
        effort=effort,
        max_attempts=_int("CARD_MAX_ATTEMPTS", Settings.max_attempts, minimum=1),
        fallbacks=fallbacks == "default",
        timeout=timeout,
        ollama_url=ollama_url.rstrip("/"),
        num_ctx=_int("CARD_NUM_CTX", Settings.num_ctx, minimum=2048),
        openrouter_url=openrouter_url.rstrip("/"),
        json_mode=json_mode,
    )
