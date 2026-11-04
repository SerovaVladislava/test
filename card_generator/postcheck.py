"""Детерминированная проверка ответа LLM: длины, полнота и отсутствие выдуманных значений."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from .models import NO_DATA, CheckIssue, KeySpec, ProductCard, ProductInput

TITLE_MAX = 70
DESCRIPTION_MIN = 800
DESCRIPTION_MAX = 1200
QUERIES_MIN = 3
QUERIES_MAX = 5

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
# «1 000» → «1000»: разделитель разрядов (обычный, неразрывный или узкий пробел).
_THOUSANDS_RE = re.compile(r"(?<=\d)[   ](?=\d{3}(?!\d))")
_SUPERLATIVE_RE = re.compile(
    r"\b(лучш\w*|самы[йе] (точн|надёжн|надежн|дешёв|дешев)\w*|номер один|идеальн\w*|уникальн\w*"
    r"|непревзойд\w*|высок\w* точност\w*)|№\s?1\b",
    re.IGNORECASE)
_MARKDOWN_RE = re.compile(r"(\*\*|__|^#{1,6}\s|^\s*[-*]\s)", re.MULTILINE)
# Иероглифы, хангыль, арабское письмо и т. п.: небольшие модели иногда вставляют их посреди русского текста.
_FOREIGN_SCRIPT_RE = re.compile(r"[֐-ࣿ฀-๿ᄀ-ᇿ぀-ヿ㄀-㆏"
                                r"㐀-䶿一-鿿가-힯豈-﫿]+")
# Запятая внутри слова — десятичная («ДИ1,0»), поэтому токены режем только по пробелам и скобкам,
# а знаки препинания на краях снимаем отдельно.
_TOKEN_RE = re.compile(r"[^\s()\[\]{};«»\"']+")
_EDGE_PUNCT = ".,:!?"
_LETTER_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")
_DIGIT_RE = re.compile(r"\d")


def _norm_number(raw: str) -> str:
    try:
        return format(Decimal(raw.replace(",", ".")).normalize(), "f")
    except InvalidOperation:
        return raw


def _numbers(text: str) -> set[str]:
    return {_norm_number(m) for m in _NUM_RE.findall(text)}


def _input_text(product: ProductInput) -> str:
    chunks = [product.name, product.manufacturer or "", product.description or ""]
    chunks += [f"{k} {v}" for k, v in product.specs.items()]
    chunks += [f"{k} {v}" for k, v in product.content_extra.items()]
    return "\n".join(chunks)


def _numbers_both_readings(text: str) -> set[str]:
    # Учитываем оба прочтения «1 000»: и как 1000, и как отдельные числа.
    return _numbers(text) | _numbers(_THOUSANDS_RE.sub("", text))


def source_numbers(product: ProductInput) -> set[str]:
    """Все числа из входных данных — то, что модели разрешено упоминать."""
    return _numbers_both_readings(_input_text(product))


def _is_code(token: str) -> bool:
    """Код модификации: буквы и цифры, соединённые дефисами («ПД100И-ДИ1,0-111», «МП4-У»)."""
    return "-" in token and bool(_LETTER_RE.search(token)) and bool(_DIGIT_RE.search(token))


def modification_codes(product: ProductInput) -> tuple[list[str], set[str]]:
    """Коды модификаций во входных данных и числа, которые встречаются только внутри них.

    Такие числа модель может упоминать лишь в составе кода: «ДИ1,0» → «класс точности 1,0» —
    это уже расшифровка кода, которой во входных данных нет."""
    text = _input_text(product)
    tokens = {t.strip(_EDGE_PUNCT) for t in _TOKEN_RE.findall(text)}
    codes = sorted((t for t in tokens if _is_code(t)), key=len, reverse=True)
    if not codes:
        return [], set()
    outside = text
    for code in codes:
        outside = outside.replace(code, " ")
    inside = set().union(*(_numbers(code) for code in codes))
    return codes, inside - _numbers_both_readings(outside)


def _without_code_mentions(text: str, codes: list[str]) -> str:
    """Убирает из текста упоминания кода целиком или его части («ПД100И», «ПД100И-ДИ1,0»)."""
    lowered = [c.lower() for c in codes]

    def replace(m: re.Match) -> str:
        token = m.group(0).strip(_EDGE_PUNCT).lower()
        is_code_part = _LETTER_RE.search(token) and _DIGIT_RE.search(token) and any(token in c for c in lowered)
        return " " if is_code_part else m.group(0)

    return _TOKEN_RE.sub(replace, text)


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", "", value.lower().replace("ё", "е"))


def _snippet(text: str, number: str) -> str:
    """Фрагмент текста вокруг числа — чтобы в отчёте было видно, о чём речь."""
    for m in _NUM_RE.finditer(text):
        if _norm_number(m.group()) == number:
            start = max(0, m.start() - 20)
            end = min(len(text), m.end() + 15)
            return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")
    return number


def normalize_card(card: ProductCard) -> ProductCard:
    """Безопасные автоисправления, не меняющие смысл: пробелы, дубли запросов, пометка «данных нет»."""
    card = card.model_copy(deep=True)
    card.seo_title = " ".join(card.seo_title.split()).rstrip(".")
    card.seo_description = re.sub(r"[ \t]+", " ", card.seo_description).strip()

    seen: set[str] = set()
    queries = []
    for q in card.search_queries:
        q = " ".join(q.split())
        if q and q.lower() not in seen:
            seen.add(q.lower())
            queries.append(q)
    card.search_queries = queries

    for spec in card.key_specs:
        spec.name = spec.name.strip()
        spec.value = spec.value.strip()
        is_missing = spec.source == NO_DATA or _norm_text(spec.value) in {_norm_text(NO_DATA), "", "-", "—"}
        if is_missing:
            spec.value = NO_DATA
            spec.source = NO_DATA
    return card


def complete_input_specs(card: ProductCard, product: ProductInput) -> list[str]:
    """Дописывает в key_specs входные характеристики, которые модель потеряла или изменила.

    Это известные факты, поэтому их восстанавливает код, а не повторный запрос к модели
    (небольшие локальные модели теряют их стабильно). Меняет card на месте и возвращает
    список исправлений для отчёта."""
    fixes: list[str] = []
    for key, value in product.specs.items():
        needle = _norm_text(value)
        if any(needle in _norm_text(s.value) for s in card.key_specs if s.source != NO_DATA):
            continue
        same_name = next((s for s in card.key_specs if _norm_text(s.name) == _norm_text(key)), None)
        if same_name is not None:
            old = same_name.value
            same_name.value, same_name.source = value, "характеристики"
            fixes.append(f"Значение «{same_name.name}» заменено на входное: «{old}» → «{value}»")
            continue
        # Вставляем после характеристик со значениями, перед строками «данных нет».
        position = max((i + 1 for i, s in enumerate(card.key_specs) if s.source != NO_DATA), default=0)
        card.key_specs.insert(position, KeySpec(name=key[:1].upper() + key[1:], value=value,
                                                source="характеристики"))
        fixes.append(f"Добавлена характеристика из входных данных: «{key}: {value}»")
    return fixes


def check_card(card: ProductCard, product: ProductInput) -> list[CheckIssue]:
    issues: list[CheckIssue] = []

    def add(code: str, field: str, message: str) -> None:
        issues.append(CheckIssue(code=code, field=field, message=message))

    # --- Длины и количество ---
    title_len = len(card.seo_title)
    if not card.seo_title:
        add("empty_field", "seo_title", "SEO-заголовок пустой")
    elif title_len > TITLE_MAX:
        add("title_length", "seo_title",
            f"SEO-заголовок {title_len} символов при лимите {TITLE_MAX} — сократите минимум на "
            f"{title_len - TITLE_MAX} символов")

    desc_len = len(card.seo_description)
    if desc_len < DESCRIPTION_MIN:
        add("description_length", "seo_description",
            f"SEO-описание {desc_len} символов, нужно {DESCRIPTION_MIN}–{DESCRIPTION_MAX} — не хватает примерно "
            f"{1000 - desc_len} символов. Напишите описание заново: 3 абзаца по 3–4 предложения (назначение; "
            f"характеристики из данных; применение и кому подходит), без новых характеристик и чисел")
    elif desc_len > DESCRIPTION_MAX:
        add("description_length", "seo_description",
            f"SEO-описание {desc_len} символов, нужно {DESCRIPTION_MIN}–{DESCRIPTION_MAX} — сократите минимум на "
            f"{desc_len - DESCRIPTION_MAX} символов (ориентир — около 1000)")

    if not QUERIES_MIN <= len(card.search_queries) <= QUERIES_MAX:
        add("queries_count", "search_queries",
            f"Поисковых запросов {len(card.search_queries)} (после удаления дублей), нужно "
            f"{QUERIES_MIN}–{QUERIES_MAX}")

    if not card.commercial.audience or not card.commercial.tasks or not card.commercial.pitch.strip():
        add("empty_field", "commercial", "Commercial-блок заполнен не полностью (audience, tasks, pitch)")

    if not card.key_specs:
        add("empty_field", "key_specs", "Список ключевых характеристик пуст")

    # --- Оформление и тон ---
    if _MARKDOWN_RE.search(card.seo_description):
        add("markdown", "seo_description", "SEO-описание должно быть обычным текстом, без Markdown-разметки")

    for field, text in _texts(card):
        m = _SUPERLATIVE_RE.search(text)
        if m:
            add("superlative", field, f"Рекламная превосходная степень без подтверждения: «{m.group(0)}»")

    flagged: set[str] = set()
    for field, text in _all_strings(card):
        m = _FOREIGN_SCRIPT_RE.search(text)
        if m and field not in flagged:
            flagged.add(field)
            add("foreign_script", field,
                f"В тексте есть символы не русского и не латинского алфавита («{m.group(0)}») — напишите по-русски")

    # --- Выдуманные значения: каждое число в ответе должно встречаться во входных данных ---
    allowed = source_numbers(product)
    codes, code_only = modification_codes(product)
    for field, text in _texts(card, include_specs=True):
        for number in sorted(_numbers(text) - allowed):
            add("ungrounded_value", field,
                f"Значение «{_snippet(text, number)}» не найдено во входных данных — удалите его "
                f"или замените формулировкой без чисел")
        if code_only:
            stripped = _without_code_mentions(text, codes)
            for number in sorted(_numbers(stripped) & code_only):
                add("code_value", field,
                    f"Значение «{_snippet(stripped, number)}» есть во входных данных только внутри кода "
                    f"модификации ({', '.join(codes)}) — не расшифровывайте код: упоминайте его целиком "
                    f"как модель, а параметры по коду вынесите в data_gaps")

    # --- Полнота: все входные характеристики должны попасть в key_specs ---
    spec_values = [_norm_text(s.value) for s in card.key_specs if s.source != NO_DATA]
    for key, value in product.specs.items():
        needle = _norm_text(value)
        if not any(needle in v for v in spec_values):
            add("missing_input_spec", "key_specs",
                f"Характеристика из входных данных не попала в список (или её значение изменено): "
                f"«{key}: {value}»")
    return issues


def _texts(card: ProductCard, include_specs: bool = False) -> list[tuple[str, str]]:
    texts = [("seo_title", card.seo_title), ("seo_description", card.seo_description)]
    texts += [("search_queries", q) for q in card.search_queries]
    texts += [("commercial.audience", a) for a in card.commercial.audience]
    texts += [("commercial.tasks", t) for t in card.commercial.tasks]
    texts.append(("commercial.pitch", card.commercial.pitch))
    if include_specs:
        texts += [("key_specs", f"{s.name}: {s.value}") for s in card.key_specs if s.source != NO_DATA]
    return texts


def _all_strings(card: ProductCard) -> list[tuple[str, str]]:
    """Все строки карточки с именем верхнеуровневого поля — для проверок, не зависящих от смысла поля."""
    found: list[tuple[str, str]] = []

    def walk(field: str, node) -> None:
        if isinstance(node, str):
            found.append((field, node))
        elif isinstance(node, dict):
            for value in node.values():
                walk(field, value)
        elif isinstance(node, list):
            for value in node:
                walk(field, value)

    for field, value in card.model_dump().items():
        walk(field, value)
    return found


def card_stats(card: ProductCard) -> dict[str, int]:
    return {
        "seo_title_chars": len(card.seo_title),
        "seo_description_chars": len(card.seo_description),
        "search_queries": len(card.search_queries),
        "key_specs": len(card.key_specs),
        "key_specs_missing": sum(1 for s in card.key_specs if s.source == NO_DATA),
    }
