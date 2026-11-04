"""Обращение к LLM со строгой JSON-схемой ответа.

Три провайдера с общим интерфейсом CardLLM:
- AnthropicCardLLM — Claude через официальный SDK Anthropic (нужен ключ API);
- OpenRouterCardLLM — OpenAI-совместимый API, по умолчанию OpenRouter (есть бесплатные модели);
- OllamaCardLLM — локальная модель через Ollama (бесплатно, без ключа, без интернета).
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import anthropic
from pydantic import ValidationError

from .config import Settings
from .models import ProductCard

# Схема ответа строится из Pydantic-модели; transform_schema приводит её к формату,
# который принимает API (additionalProperties: false, без неподдерживаемых ограничений).
CARD_SCHEMA = anthropic.transform_schema(ProductCard)
MAX_TOKENS = 16_000
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Лимит длины ответа локальной модели: карточка занимает ~1,5–2 тыс. токенов,
# лимит страхует от зацикливания небольшой модели.
OLLAMA_MAX_OUTPUT = 4096
OLLAMA_TEMPERATURE = 0.3


class LLMError(Exception):
    """Ошибка API или отказ модели — повтор с тем же запросом не поможет."""


class InvalidLLMOutput(Exception):
    """Ответ получен, но не соответствует схеме — имеет смысл повторить генерацию."""


@dataclass
class LLMResponse:
    card: ProductCard
    model: str
    input_tokens: int
    output_tokens: int
    fallback_used: bool = False


class CardLLM(Protocol):
    model: str

    def generate(self, system: str, user: str) -> LLMResponse: ...


class AnthropicCardLLM:
    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None):
        self.settings = settings
        self.model = settings.model
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            # Ключ берётся из окружения (ANTHROPIC_API_KEY) — в коде его нет.
            self._client = anthropic.Anthropic(timeout=self.settings.timeout, max_retries=2)
        return self._client

    def generate(self, system: str, user: str) -> LLMResponse:
        request: dict = {
            "model": self.settings.model,
            "max_tokens": MAX_TOKENS,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "effort": self.settings.effort,
                "format": {"type": "json_schema", "schema": CARD_SCHEMA},
            },
        }
        if self.settings.fallbacks:
            # При отказе классификатора безопасности API сам повторит запрос на рекомендованной модели.
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"

        try:
            response = self.client.beta.messages.create(**request)
        except anthropic.CredentialsError as e:
            raise LLMError("Не найден API-ключ: укажите ANTHROPIC_API_KEY в файле .env или в окружении") from e
        except anthropic.AuthenticationError as e:
            raise LLMError("API-ключ Anthropic отклонён: проверьте ANTHROPIC_API_KEY") from e
        except anthropic.PermissionDeniedError as e:
            raise LLMError(f"Нет доступа к модели {self.settings.model}: {e.message}") from e
        except anthropic.NotFoundError as e:
            raise LLMError(f"Модель не найдена: {self.settings.model}") from e
        except anthropic.BadRequestError as e:
            raise LLMError(f"API отклонил запрос: {e.message}") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Превышен лимит запросов к API Anthropic, повторите позже") from e
        except anthropic.APITimeoutError as e:
            raise LLMError(f"API не ответил за {self.settings.timeout:.0f} с, повторите позже") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Нет соединения с API Anthropic: проверьте интернет или прокси") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Ошибка API Anthropic ({e.status_code}): {e.message}") from e

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            raise LLMError(f"Модель отклонила запрос (категория: {category or 'не указана'})")
        if response.stop_reason == "max_tokens":
            raise InvalidLLMOutput("Ответ модели обрезан по лимиту длины — JSON неполный")

        # Если сработал fallback, в content сначала может идти частичный ответ первой модели,
        # затем блок fallback и полный ответ второй. Берём текст после последнего fallback.
        texts: list[str] = []
        for block in response.content:
            if block.type == "fallback":
                texts = []
            elif block.type == "text":
                texts.append(block.text)
        card = _parse_card("".join(texts))

        usage = response.usage
        iterations = getattr(usage, "iterations", None) or []
        return LLMResponse(
            card=card,
            model=response.model,
            input_tokens=(usage.input_tokens or 0)
            + (getattr(usage, "cache_read_input_tokens", 0) or 0)
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0),
            output_tokens=usage.output_tokens or 0,
            fallback_used=any(getattr(it, "type", None) == "fallback_message" for it in iterations),
        )


def inline_refs(schema: dict) -> dict:
    """Подставляет определения из $defs вместо ссылок $ref: не все движки
    структурированного вывода (грамматики llama.cpp в старых Ollama) понимают ссылки."""
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                return resolve(copy.deepcopy(defs[ref.removeprefix("#/$defs/")]))
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


INLINE_CARD_SCHEMA = inline_refs(CARD_SCHEMA)

PostFn = Callable[[str, dict], dict]


class OllamaCardLLM:
    """Локальная модель через HTTP API Ollama (/api/chat).

    Формат ответа задаётся JSON-схемой в параметре format: Ollama ограничивает генерацию
    грамматикой, поэтому структура JSON и допустимые значения enum соблюдаются всегда,
    а содержание проверяет postcheck, как и для Claude."""

    def __init__(self, settings: Settings, post: PostFn | None = None):
        self.settings = settings
        self.model = settings.model
        self._post = post or self._http_post

    def _http_post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.settings.ollama_url + path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.settings.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def generate(self, system: str, user: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": INLINE_CARD_SCHEMA,
            "stream": False,
            # num_ctx обязателен: по умолчанию у Ollama 2048 токенов, и длинный промпт
            # молча обрезается с начала — модель теряет системные правила.
            "options": {"num_ctx": self.settings.num_ctx, "num_predict": OLLAMA_MAX_OUTPUT,
                        "temperature": OLLAMA_TEMPERATURE},
        }
        url = self.settings.ollama_url
        try:
            body = self._post("/api/chat", payload)
        except urllib.error.HTTPError as e:
            detail = _ollama_error_text(e)
            if e.code == 404:
                raise LLMError(f"Модель {self.model} не найдена в Ollama: "
                               f"выполните «ollama pull {self.model}»") from e
            raise LLMError(f"Ошибка Ollama ({e.code}): {detail}") from e
        except TimeoutError as e:
            raise LLMError(f"Ollama не ответила за {self.settings.timeout:.0f} с: увеличьте CARD_TIMEOUT "
                           "или возьмите модель поменьше") from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise LLMError(f"Ollama не ответила за {self.settings.timeout:.0f} с") from e
            raise LLMError(f"Нет соединения с Ollama ({url}): запустите Ollama "
                           "или проверьте CARD_OLLAMA_URL") from e
        except (OSError, ValueError) as e:
            # Разрыв соединения (например, процесс модели упал из-за нехватки памяти) или не-JSON в ответе.
            raise LLMError(f"Ollama вернула некорректный ответ: {e}") from e

        if body.get("done_reason") == "length":
            raise InvalidLLMOutput("Ответ модели обрезан по лимиту длины — JSON неполный")
        card = _parse_card((body.get("message") or {}).get("content") or "")
        return LLMResponse(
            card=card,
            model=body.get("model") or self.model,
            input_tokens=body.get("prompt_eval_count") or 0,
            output_tokens=body.get("eval_count") or 0,
        )


def _ollama_error_text(error: urllib.error.HTTPError) -> str:
    try:
        return json.loads(error.read().decode("utf-8")).get("error") or error.reason
    except (OSError, ValueError, AttributeError):
        return str(error.reason)


# Временные ошибки, которые имеет смысл переждать: лимит бесплатного тарифа и сбои провайдера модели.
RETRYABLE_STATUSES = {429, 502, 503}
RETRY_DELAYS = (10, 30)
OPENROUTER_MAX_TOKENS = 16_000  # у моделей с рассуждениями в лимит входят и они

HeaderPostFn = Callable[[str, dict, dict], dict]


class OpenRouterCardLLM:
    """OpenAI-совместимый Chat Completions API (по умолчанию OpenRouter).

    Схема передаётся в response_format. Если модель поддерживает только JSON-режим без схемы
    (CARD_JSON_MODE=json_object), схема дополнительно приводится в системном сообщении."""

    def __init__(self, settings: Settings, post: HeaderPostFn | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.settings = settings
        self.model = settings.model
        self._post = post or self._http_post
        self._sleep = sleep

    def _http_post(self, url: str, payload: dict, headers: dict) -> dict:
        request = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                         headers=headers)
        with urllib.request.urlopen(request, timeout=self.settings.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def generate(self, system: str, user: str) -> LLMResponse:
        if self.settings.json_mode == "json_schema":
            response_format = {"type": "json_schema",
                               "json_schema": {"name": "product_card", "strict": True, "schema": INLINE_CARD_SCHEMA}}
        else:
            response_format = {"type": "json_object"}
            system += ("\n\nОтвет — один JSON-объект строго по этой JSON-схеме, без пояснений и без Markdown:\n"
                       + json.dumps(INLINE_CARD_SCHEMA, ensure_ascii=False))
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": response_format,
            "temperature": OLLAMA_TEMPERATURE,
            "max_tokens": OPENROUTER_MAX_TOKENS,
        }
        headers = {
            "Content-Type": "application/json",
            # Ключ берётся из окружения (OPENROUTER_API_KEY) — в коде его нет.
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}",
            "X-Title": "card-generator",
        }
        body = self._request(payload, headers)

        if body.get("error"):
            raise LLMError(f"Ошибка провайдера модели: {_error_message(body)}")
        choice = (body.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            raise InvalidLLMOutput("Ответ модели обрезан по лимиту длины — JSON неполный")
        message = choice.get("message") or {}
        if message.get("refusal"):
            raise LLMError(f"Модель отклонила запрос: {message['refusal']}")
        card = _parse_card(message.get("content") or "")
        usage = body.get("usage") or {}
        return LLMResponse(
            card=card,
            model=body.get("model") or self.model,
            input_tokens=usage.get("prompt_tokens") or 0,
            output_tokens=usage.get("completion_tokens") or 0,
        )

    def _request(self, payload: dict, headers: dict) -> dict:
        url = self.settings.openrouter_url + "/chat/completions"
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                return self._post(url, payload, headers)
            except urllib.error.HTTPError as e:
                detail = _http_error_text(e)
                if e.code in RETRYABLE_STATUSES and attempt < len(RETRY_DELAYS):
                    self._sleep(RETRY_DELAYS[attempt])
                    continue
                raise LLMError(_openrouter_error(e.code, detail, self.model)) from e
            except TimeoutError as e:
                raise LLMError(f"API не ответил за {self.settings.timeout:.0f} с, повторите позже") from e
            except urllib.error.URLError as e:
                raise LLMError(f"Нет соединения с {self.settings.openrouter_url}: проверьте интернет "
                               "или прокси") from e
            except (OSError, ValueError) as e:
                raise LLMError(f"API вернул некорректный ответ: {e}") from e
        raise AssertionError("unreachable")


def _openrouter_error(status: int, detail: str, model: str) -> str:
    if status == 401:
        return "Ключ OpenRouter отклонён: проверьте OPENROUTER_API_KEY в .env"
    if status == 402:
        return "Недостаточно кредитов OpenRouter: выберите бесплатную модель (с суффиксом :free)"
    if status == 404:
        return (f"Модель {model} не найдена или не поддерживает ответ по JSON-схеме: "
                f"попробуйте CARD_JSON_MODE=json_object ({detail})")
    if status == 429:
        return "Превышен лимит запросов к бесплатной модели, повторите через несколько минут"
    return f"Ошибка API ({status}): {detail}"


def _http_error_text(error: urllib.error.HTTPError) -> str:
    try:
        return _error_message(json.loads(error.read().decode("utf-8"))) or str(error.reason)
    except (OSError, ValueError, AttributeError):
        return str(error.reason)


def _error_message(body: dict) -> str:
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error or "")


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _parse_card(text: str) -> ProductCard:
    text = text.strip()
    # Модели без строгого режима иногда оборачивают JSON в ```json … ``` или добавляют фразу до/после.
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1)
    elif text and not text.startswith("{") and "{" in text and "}" in text:
        text = text[text.index("{"):text.rindex("}") + 1]
    if not text:
        raise InvalidLLMOutput("Модель вернула пустой ответ")
    try:
        return ProductCard.model_validate_json(text)
    except ValidationError as e:
        raise InvalidLLMOutput(f"Ответ модели не соответствует схеме: {e.error_count()} ошибок") from e


def make_llm(settings: Settings) -> CardLLM:
    if settings.provider == "ollama":
        return OllamaCardLLM(settings)
    if settings.provider == "openrouter":
        return OpenRouterCardLLM(settings)
    return AnthropicCardLLM(settings)
