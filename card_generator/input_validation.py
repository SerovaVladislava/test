"""Разбор и проверка входного JSON до обращения к LLM."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .models import DataGap, ProductInput

MAX_DESCRIPTION_CHARS = 30_000
MAX_SPECS = 200

# Значения, которые означают «не заполнено».
EMPTY_MARKERS = {"", "-", "—", "–", "н/д", "нет данных", "данных нет", "n/a", "na", "null", "none", "?"}

KNOWN_FIELDS = {"name", "manufacturer", "specs", "description"}

# Сообщения json.JSONDecodeError — по-английски; переводим частые случаи для контент-менеджера.
_JSON_ERRORS = {
    "Expecting property name enclosed in double quotes": "ожидается имя поля в двойных кавычках "
                                                         "(возможно, лишняя запятая перед закрывающей скобкой)",
    "Expecting ',' delimiter": "пропущена запятая между элементами",
    "Expecting ':' delimiter": "пропущено двоеточие после имени поля",
    "Expecting value": "ожидается значение",
    "Unterminated string starting at": "незакрытая строка (нет закрывающей кавычки)",
    "Invalid control character at": "недопустимый символ внутри строки (например, перенос строки)",
    "Extra data": "лишние данные после окончания JSON",
}

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"[ \t ]+")


class InputError(Exception):
    """Входные данные нельзя обработать. Содержит список понятных пользователю сообщений."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass
class InputReport:
    product: ProductInput
    gaps: list[DataGap] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_sparse(self) -> bool:
        """Нет ни характеристик, ни описания — карточка будет опираться только на название."""
        return not self.product.specs and not self.product.description


def parse_json_text(text: str) -> Any:
    if not text or not text.strip():
        raise InputError(["Пустой ввод: ожидается JSON с данными о товаре"])
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        reason = _JSON_ERRORS.get(e.msg, e.msg)
        raise InputError([f"Некорректный JSON: {reason} (строка {e.lineno}, столбец {e.colno})"]) from e


def _clean_text(value: str) -> str:
    text = html.unescape(_TAG_RE.sub(" ", value))
    lines = [_SPACE_RE.sub(" ", line).strip() for line in text.replace("\r", "").split("\n")]
    # Схлопываем пустые строки, сохраняя абзацы.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in EMPTY_MARKERS)


def _scalar_to_str(value: Any) -> str:
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return _clean_text(str(value))


def _flatten_specs(raw: Any, errors: list[str]) -> list[tuple[str, Any]]:
    """Приводит specs к списку пар (название, значение).

    Поддерживаются: {"название": "значение"}, вложенные словари
    ({"питание": {"напряжение": "24 В"}}) и список [{"name": ..., "value": ...}].
    """
    pairs: list[tuple[str, Any]] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix} / {k}" if prefix else str(k), v)
        elif isinstance(value, list) and all(not isinstance(x, (dict, list)) for x in value):
            pairs.append((prefix, "; ".join(_scalar_to_str(x) for x in value if not _is_empty(x)) or None))
        elif isinstance(value, list):
            errors.append(f"specs.{prefix}: вложенные списки объектов не поддерживаются")
        else:
            pairs.append((prefix, value))

    if isinstance(raw, dict):
        walk("", raw)
    elif isinstance(raw, list):
        for i, item in enumerate(raw):
            if isinstance(item, dict) and "name" in item:
                pairs.append((str(item["name"]), item.get("value")))
            else:
                errors.append(f"specs[{i}]: ожидается объект вида {{\"name\": ..., \"value\": ...}}")
    else:
        errors.append("specs: ожидается объект {\"характеристика\": \"значение\"} или список пар name/value")
    return pairs


def validate_product(raw: Any) -> InputReport:
    """Проверяет один товар. Бросает InputError, если генерация невозможна;
    недостающие, но необязательные данные возвращает как пробелы (gaps)."""
    if not isinstance(raw, dict):
        raise InputError([f"Ожидается JSON-объект с данными о товаре, получено: {type(raw).__name__}"])

    errors: list[str] = []
    gaps: list[DataGap] = []
    notes: list[str] = []

    # name — единственное обязательное поле: без него неясно, о каком товаре писать.
    name_raw = raw.get("name")
    if _is_empty(name_raw) or not isinstance(name_raw, (str, int, float)):
        errors.append("Поле name (название товара) обязательно и должно быть непустой строкой")
        name = ""
    else:
        name = _clean_text(str(name_raw))

    manufacturer_raw = raw.get("manufacturer")
    manufacturer: str | None = None
    if _is_empty(manufacturer_raw):
        gaps.append(DataGap(field="Производитель", status="отсутствует",
                            comment="Поле manufacturer не заполнено — укажите производителя"))
    elif not isinstance(manufacturer_raw, (str, int, float)):
        errors.append("Поле manufacturer должно быть строкой")
    else:
        manufacturer = _clean_text(str(manufacturer_raw))

    specs: dict[str, str] = {}
    specs_raw = raw.get("specs")
    if specs_raw is None or specs_raw == {} or specs_raw == []:
        gaps.append(DataGap(field="Технические характеристики", status="отсутствует",
                            comment="Блок specs пуст — добавьте характеристики из паспорта прибора"))
    else:
        pairs = _flatten_specs(specs_raw, errors)
        if len(pairs) > MAX_SPECS:
            errors.append(f"specs: слишком много характеристик ({len(pairs)}), максимум {MAX_SPECS}")
        for key, value in pairs:
            key = _clean_text(key)
            if not key:
                errors.append("specs: найдена характеристика без названия")
                continue
            if _is_empty(value):
                gaps.append(DataGap(field=key, status="отсутствует",
                                    comment="Характеристика указана во входных данных без значения"))
                continue
            if isinstance(value, (dict, list)):
                errors.append(f"specs.{key}: неподдерживаемый тип значения")
                continue
            if key in specs:
                notes.append(f"Характеристика «{key}» указана несколько раз, использовано последнее значение")
            specs[key] = _scalar_to_str(value)

    description: str | None = None
    description_raw = raw.get("description")
    if _is_empty(description_raw):
        gaps.append(DataGap(field="Описание", status="отсутствует",
                            comment="Нет текстового описания или фрагмента паспорта — назначение и область "
                                    "применения будут описаны только в общих словах"))
    elif not isinstance(description_raw, str):
        errors.append("Поле description должно быть строкой")
    else:
        description = _clean_text(description_raw)
        if len(description) > MAX_DESCRIPTION_CHARS:
            errors.append(f"description: слишком длинный текст ({len(description)} символов), "
                          f"максимум {MAX_DESCRIPTION_CHARS}. Оставьте фрагмент паспорта с характеристиками")

    extra: dict[str, str] = {}
    for key, value in raw.items():
        if key in KNOWN_FIELDS or _is_empty(value):
            continue
        if isinstance(value, (str, int, float, bool)):
            extra[str(key)] = _scalar_to_str(value)
        else:
            notes.append(f"Поле «{key}» пропущено: поддерживаются только строковые и числовые значения")

    if errors:
        raise InputError(errors)

    product = ProductInput(name=name, manufacturer=manufacturer, specs=specs,
                           description=description, extra=extra)
    return InputReport(product=product, gaps=gaps, notes=notes)


def load_products(text: str) -> list[Any]:
    """Разбирает JSON с одним товаром (объект) или несколькими (массив объектов)."""
    data = parse_json_text(text)
    if isinstance(data, list):
        if not data:
            raise InputError(["Пустой массив: нет ни одного товара"])
        return data
    return [data]
