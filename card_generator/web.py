"""Простая веб-форма и JSON API (Flask)."""

from __future__ import annotations

import json
import logging

from flask import Flask, jsonify, render_template, request

from .config import load_settings
from .input_validation import InputError, parse_json_text, validate_product
from .llm import AnthropicCardLLM, InvalidLLMOutput, LLMError, OpenRouterCardLLM, make_llm
from .render import to_markdown
from .service import CardGenerator

log = logging.getLogger(__name__)

SAMPLE_JSON = {
    "name": "Манометр показывающий DM 02",
    "manufacturer": "Манотомь",
    "specs": {
        "диапазон измерений": "0–1,6 МПа",
        "класс точности": "1,5",
        "присоединение": "G1/2",
        "диаметр корпуса": "100 мм",
    },
    "description": "Прибор предназначен для измерения избыточного давления...",
}


def parse_specs_text(text: str) -> dict[str, str]:
    """Строки вида «название: значение» → словарь. Ошибки — с номером строки."""
    specs: dict[str, str] = {}
    errors: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        sep = next((s for s in (":", "\t", " — ", " - ", "=") if s in line), None)
        if sep is None:
            errors.append(f"Характеристики, строка {number}: нужен формат «название: значение»")
            continue
        key, value = (part.strip() for part in line.split(sep, 1))
        specs[key] = value
    if errors:
        raise InputError(errors)
    return specs


def create_app(generator: CardGenerator | None = None) -> Flask:
    app = Flask(__name__)
    app.json.ensure_ascii = False
    app.json.sort_keys = False
    settings = load_settings()
    if generator is None:
        generator = CardGenerator(make_llm(settings), max_attempts=settings.max_attempts)

    @app.context_processor
    def wait_hint():
        if settings.provider == "ollama":
            return {"wait_hint": "Локальная модель на CPU пишет карточку несколько минут…"}
        return {"wait_hint": "Генерация обычно занимает 30–90 секунд…"}

    def run(raw: dict):
        """(result, None) или (None, (http_status, payload))."""
        if isinstance(generator.llm, (AnthropicCardLLM, OpenRouterCardLLM)) and not settings.has_credentials:
            hint = settings.credentials_hint
            return None, (503, {"error": "no_credentials", "messages": [hint[:1].upper() + hint[1:]]})
        try:
            return generator.generate(raw), None
        except InputError as e:
            return None, (400, {"error": "invalid_input", "messages": e.errors})
        except (LLMError, InvalidLLMOutput) as e:
            log.warning("Ошибка генерации: %s", e)
            return None, (502, {"error": "llm_error", "messages": [str(e)]})

    @app.get("/")
    def index():
        sample = json.dumps(SAMPLE_JSON, ensure_ascii=False, indent=2)
        return render_template("index.html", sample_json=sample, form={}, errors=[], mode="fields")

    @app.post("/")
    def submit():
        mode = request.form.get("mode", "fields")
        form = request.form.to_dict()
        try:
            if mode == "json":
                raw = parse_json_text(form.get("json", ""))
            else:
                raw = {
                    "name": form.get("name", ""),
                    "manufacturer": form.get("manufacturer", ""),
                    "specs": parse_specs_text(form.get("specs", "")),
                    "description": form.get("description", ""),
                }
        except InputError as e:
            return _form_error(form, mode, e.errors), 400

        result, error = run(raw)
        if error:
            status, payload = error
            return _form_error(form, mode, payload["messages"]), status
        return render_template(
            "result.html",
            result=result,
            markdown=to_markdown(result),
            result_json=result.model_dump_json(indent=2),
        )

    def _form_error(form: dict, mode: str, errors: list[str]):
        sample = json.dumps(SAMPLE_JSON, ensure_ascii=False, indent=2)
        return render_template("index.html", sample_json=sample, form=form, errors=errors, mode=mode)

    @app.post("/api/generate")
    def api_generate():
        try:
            raw = parse_json_text(request.get_data(as_text=True))
        except InputError as e:
            return jsonify(error="invalid_input", messages=e.errors), 400
        result, error = run(raw)
        if error:
            status, payload = error
            return jsonify(payload), status
        return app.response_class(result.model_dump_json(indent=2), mimetype="application/json")

    @app.post("/api/check")
    def api_check():
        try:
            report = validate_product(parse_json_text(request.get_data(as_text=True)))
        except InputError as e:
            return jsonify(error="invalid_input", messages=e.errors), 400
        return jsonify(product=report.product.model_dump(), gaps=[g.model_dump() for g in report.gaps],
                       notes=report.notes)

    return app
