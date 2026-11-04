import pytest

from card_generator import config
from card_generator.config import ConfigError, load_settings

ENV_VARS = ("CARD_PROVIDER", "CARD_MODEL", "CARD_EFFORT", "CARD_MAX_ATTEMPTS", "CARD_FALLBACKS",
            "CARD_TIMEOUT", "CARD_OLLAMA_URL", "CARD_NUM_CTX", "CARD_OPENROUTER_URL", "CARD_JSON_MODE",
            "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENROUTER_API_KEY")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    # Настоящий .env разработчика не должен влиять на тесты.
    monkeypatch.setattr(config, "load_dotenv", lambda **kwargs: None)
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_are_anthropic():
    s = load_settings()
    assert (s.provider, s.model, s.timeout) == ("anthropic", "claude-opus-5-5", 180.0)
    assert not s.has_credentials


def test_ollama_defaults(monkeypatch):
    monkeypatch.setenv("CARD_PROVIDER", "Ollama")
    s = load_settings()
    assert (s.provider, s.model, s.timeout, s.num_ctx) == ("ollama", "qwen2.5:7b", 1800.0, 8192)
    assert s.has_credentials  # ключ не нужен


def test_openrouter_defaults_and_key(monkeypatch):
    monkeypatch.setenv("CARD_PROVIDER", "openrouter")
    s = load_settings()
    assert (s.model, s.json_mode, s.openrouter_url) == ("nvidia/nemotron-3.5-super-vl:free", "json_schema",
                                                        "https://openrouter.ai/api/v1")
    assert not s.has_credentials and "OPENROUTER_API_KEY" in s.credentials_hint
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert s.has_credentials


def test_explicit_values_win(monkeypatch):
    monkeypatch.setenv("CARD_PROVIDER", "ollama")
    monkeypatch.setenv("CARD_MODEL", "qwen2.5:3b")
    monkeypatch.setenv("CARD_TIMEOUT", "600")
    monkeypatch.setenv("CARD_OLLAMA_URL", "http://192.168.1.10:11434/")
    s = load_settings()
    assert (s.model, s.timeout, s.ollama_url) == ("qwen2.5:3b", 600.0, "http://192.168.1.10:11434")


@pytest.mark.parametrize("name, value", [
    ("CARD_PROVIDER", "openai"),
    ("CARD_JSON_MODE", "xml"),
    ("CARD_OPENROUTER_URL", "openrouter.ai"),
    ("CARD_EFFORT", "extreme"),
    ("CARD_MAX_ATTEMPTS", "0"),
    ("CARD_TIMEOUT", "долго"),
    ("CARD_OLLAMA_URL", "localhost:11434"),
    ("CARD_NUM_CTX", "1024"),
])
def test_invalid_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError, match=name):
        load_settings()
