"""Разбор ответов API и маппинг ошибок — на заглушке клиента, без сети."""

import io
import json
import urllib.error
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from card_generator.config import Settings
from card_generator.llm import (FALLBACK_BETA, INLINE_CARD_SCHEMA, AnthropicCardLLM, InvalidLLMOutput, LLMError,
                                OllamaCardLLM, OpenRouterCardLLM, inline_refs, make_llm)

from .conftest import make_card


class StubClient:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def response(content, stop_reason="end_turn", iterations=None):
    usage = SimpleNamespace(input_tokens=10, output_tokens=20, cache_read_input_tokens=5,
                            cache_creation_input_tokens=0, iterations=iterations)
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage, model="claude-opus-5-5",
                           stop_details=None)


def text(t):
    return SimpleNamespace(type="text", text=t)


def llm_with(result, **settings):
    client = StubClient(result)
    return AnthropicCardLLM(Settings(**settings), client=client), client


def test_request_shape_and_parsing():
    llm, client = llm_with(response([SimpleNamespace(type="thinking"), text(make_card().model_dump_json())]))
    out = llm.generate("system", "user")
    assert out.card.seo_title == make_card().seo_title
    assert out.input_tokens == 15 and out.output_tokens == 20
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["effort"] == "medium"
    assert call["fallbacks"] == "default" and call["betas"] == [FALLBACK_BETA]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_fallbacks_can_be_disabled():
    llm, client = llm_with(response([text(make_card().model_dump_json())]), fallbacks=False)
    llm.generate("s", "u")
    assert "fallbacks" not in client.calls[0] and "betas" not in client.calls[0]


def test_text_after_fallback_block_is_used():
    content = [text('{"partial": '), SimpleNamespace(type="fallback"), text(make_card().model_dump_json())]
    llm, _ = llm_with(response(content, iterations=[SimpleNamespace(type="fallback_message")]))
    out = llm.generate("s", "u")
    assert out.fallback_used


@pytest.mark.parametrize("resp, error", [
    (response([text("{}")], stop_reason="max_tokens"), InvalidLLMOutput),
    (response([text("не JSON")]), InvalidLLMOutput),
    (response([text('{"seo_title": "x"}')]), InvalidLLMOutput),
    (response([]), InvalidLLMOutput),
    (response([text("")], stop_reason="refusal"), LLMError),
])
def test_bad_responses(resp, error):
    llm, _ = llm_with(resp)
    with pytest.raises(error):
        llm.generate("s", "u")


def _status_error(cls, status):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("error", response=httpx2.Response(status, request=request), body=None)


@pytest.mark.parametrize("exc, message", [
    (_status_error(anthropic.AuthenticationError, 401), "API-ключ"),
    (_status_error(anthropic.NotFoundError, 404), "Модель не найдена"),
    (_status_error(anthropic.RateLimitError, 429), "лимит"),
    (_status_error(anthropic.InternalServerError, 500), "500"),
    (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com")), "соединения"),
])
def test_api_errors_are_mapped(exc, message):
    llm, _ = llm_with(exc)
    with pytest.raises(LLMError, match=message):
        llm.generate("s", "u")


# --- Ollama ---

OLLAMA_SETTINGS = Settings(provider="ollama", model="qwen2.5:7b", num_ctx=8192)


def ollama_with(result):
    calls = []

    def post(path, payload):
        calls.append((path, payload))
        if isinstance(result, Exception):
            raise result
        return result

    return OllamaCardLLM(OLLAMA_SETTINGS, post=post), calls


def ollama_body(content, done_reason="stop"):
    return {"model": "qwen2.5:7b", "message": {"role": "assistant", "content": content},
            "done_reason": done_reason, "prompt_eval_count": 2500, "eval_count": 1400}


def test_ollama_request_shape_and_parsing():
    llm, calls = ollama_with(ollama_body(make_card().model_dump_json()))
    out = llm.generate("system", "user")
    assert out.card.seo_title == make_card().seo_title
    assert (out.model, out.input_tokens, out.output_tokens) == ("qwen2.5:7b", 2500, 1400)
    path, payload = calls[0]
    assert path == "/api/chat" and payload["stream"] is False
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert payload["format"] is INLINE_CARD_SCHEMA
    assert payload["options"]["num_ctx"] == 8192


def test_inline_refs_removes_references():
    assert "$ref" not in json.dumps(INLINE_CARD_SCHEMA) and "$defs" not in INLINE_CARD_SCHEMA
    # Порядок полей сохраняется — он задаёт порядок генерации (факты раньше текстов).
    assert list(INLINE_CARD_SCHEMA["properties"])[:2] == ["key_specs", "data_gaps"]
    schema = {"$defs": {"A": {"type": "string"}}, "type": "object",
              "properties": {"a": {"$ref": "#/$defs/A"}, "b": {"type": "array", "items": {"$ref": "#/$defs/A"}}}}
    assert inline_refs(schema)["properties"] == {"a": {"type": "string"},
                                                 "b": {"type": "array", "items": {"type": "string"}}}


@pytest.mark.parametrize("body", [
    ollama_body(make_card().model_dump_json()[:200], done_reason="length"),
    ollama_body("не JSON"),
    ollama_body(""),
    {"model": "qwen2.5:7b"},
])
def test_ollama_bad_responses(body):
    llm, _ = ollama_with(body)
    with pytest.raises(InvalidLLMOutput):
        llm.generate("s", "u")


def _http_error(code, message):
    return urllib.error.HTTPError("http://127.0.0.1:11434/api/chat", code, "error", {},
                                  io.BytesIO(json.dumps({"error": message}).encode()))


@pytest.mark.parametrize("exc, message", [
    (_http_error(404, 'model "qwen2.5:7b" not found, try pulling it first'), "ollama pull qwen2.5:7b"),
    (_http_error(500, "model requires more system memory"), "more system memory"),
    (urllib.error.URLError(ConnectionRefusedError(10061, "refused")), "Нет соединения с Ollama"),
    (TimeoutError("timed out"), "не ответила"),
    (ConnectionResetError("reset"), "некорректный ответ"),
])
def test_ollama_errors_are_mapped(exc, message):
    llm, _ = ollama_with(exc)
    with pytest.raises(LLMError, match=message):
        llm.generate("s", "u")


# --- OpenRouter (OpenAI-совместимый API) ---

def openrouter_with(*results, **settings):
    """results — ответы или исключения по очереди; последний повторяется."""
    calls, sleeps = [], []
    queue = list(results)

    def post(url, payload, headers):
        calls.append((url, payload, headers))
        result = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(result, Exception):
            raise result
        return result

    s = Settings(provider="openrouter", model="google/gemma-4-31b-it:free", **settings)
    return OpenRouterCardLLM(s, post=post, sleep=sleeps.append), calls, sleeps


def chat_body(content, finish_reason="stop"):
    return {"model": "google/gemma-4-31b-it", "choices": [{"finish_reason": finish_reason,
            "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 3000, "completion_tokens": 1200}}


def _or_error(code, message="error"):
    return urllib.error.HTTPError("https://openrouter.ai/api/v1/chat/completions", code, "error", {},
                                  io.BytesIO(json.dumps({"error": {"message": message, "code": code}}).encode()))


def test_openrouter_request_shape_and_parsing(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    llm, calls, _ = openrouter_with(chat_body(make_card().model_dump_json()))
    out = llm.generate("system", "user")
    assert out.card.seo_title == make_card().seo_title
    assert (out.model, out.input_tokens, out.output_tokens) == ("google/gemma-4-31b-it", 3000, 1200)
    url, payload, headers = calls[0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-or-test"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"] is INLINE_CARD_SCHEMA
    assert payload["messages"][0] == {"role": "system", "content": "system"}


def test_openrouter_json_object_mode_puts_schema_in_prompt():
    llm, calls, _ = openrouter_with(chat_body(make_card().model_dump_json()), json_mode="json_object")
    llm.generate("system", "user")
    payload = calls[0][1]
    assert payload["response_format"] == {"type": "json_object"}
    assert '"key_specs"' in payload["messages"][0]["content"]


@pytest.mark.parametrize("wrap", ["```json\n{}\n```", "Вот карточка:\n{}\nГотово."])
def test_json_inside_fences_or_text_is_parsed(wrap):
    llm, _, _ = openrouter_with(chat_body(wrap.replace("{}", make_card().model_dump_json())))
    assert llm.generate("s", "u").card == make_card()


def test_openrouter_retries_rate_limit_then_succeeds():
    llm, calls, sleeps = openrouter_with(_or_error(429), _or_error(503), chat_body(make_card().model_dump_json()))
    llm.generate("s", "u")
    assert len(calls) == 3 and sleeps == [10, 30]


@pytest.mark.parametrize("exc, message", [
    (_or_error(429, "Rate limit exceeded: free-models-per-day"), "лимит"),
    (_or_error(401, "No auth credentials found"), "Ключ OpenRouter отклонён"),
    (_or_error(404, "No endpoints found that support the requested parameters"), "CARD_JSON_MODE"),
    (_or_error(400, "Invalid model"), "Invalid model"),
    (urllib.error.URLError(OSError("getaddrinfo failed")), "Нет соединения"),
])
def test_openrouter_errors_are_mapped(exc, message):
    llm, _, _ = openrouter_with(exc)
    with pytest.raises(LLMError, match=message):
        llm.generate("s", "u")


@pytest.mark.parametrize("body, error", [
    ({"error": {"message": "Provider returned error", "code": 502}}, LLMError),
    (chat_body(make_card().model_dump_json()[:300], finish_reason="length"), InvalidLLMOutput),
    (chat_body(None), InvalidLLMOutput),
    ({"choices": []}, InvalidLLMOutput),
])
def test_openrouter_bad_responses(body, error):
    llm, _, _ = openrouter_with(body)
    with pytest.raises(error):
        llm.generate("s", "u")


def test_make_llm_picks_provider():
    assert isinstance(make_llm(OLLAMA_SETTINGS), OllamaCardLLM)
    assert isinstance(make_llm(Settings(provider="openrouter")), OpenRouterCardLLM)
    assert isinstance(make_llm(Settings()), AnthropicCardLLM)
