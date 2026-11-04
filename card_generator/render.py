"""Представление результата в Markdown — для копирования в CMS или отправки на вычитку."""

from __future__ import annotations

from .models import NO_DATA, CardResult
from .postcheck import DESCRIPTION_MAX, DESCRIPTION_MIN, TITLE_MAX

STATUS_LABELS = {
    "ok": "✅ Автопроверки пройдены — черновик готов к вычитке",
    "needs_review": "⚠️ Требует внимания — см. раздел «Замечания автопроверки»",
}


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def to_markdown(result: CardResult) -> str:
    card, stats = result.card, result.stats
    out: list[str] = [f"# {result.product.name}", ""]
    out.append(f"**Статус:** {STATUS_LABELS[result.status]}")
    if result.product.manufacturer:
        out.append(f"**Производитель:** {result.product.manufacturer}")
    out += ["", "## SEO-заголовок", "", card.seo_title, "",
            f"*{stats['seo_title_chars']} / {TITLE_MAX} символов*", ""]

    out += ["## SEO-описание", "", card.seo_description, "",
            f"*{stats['seo_description_chars']} символов (норма {DESCRIPTION_MIN}–{DESCRIPTION_MAX})*", ""]

    out += ["## Ключевые характеристики", "", "| Характеристика | Значение | Источник |", "|---|---|---|"]
    for spec in card.key_specs:
        value = f"**{NO_DATA}**" if spec.source == NO_DATA else _cell(spec.value)
        source = "—" if spec.source == NO_DATA else spec.source
        out.append(f"| {_cell(spec.name)} | {value} | {source} |")
    out.append("")

    out += ["## Поисковые запросы", ""]
    out += [f"- {q}" for q in card.search_queries]
    out.append("")

    out += ["## Для менеджера", "", "**Кому подходит:**", ""]
    out += [f"- {a}" for a in card.commercial.audience]
    out += ["", "**Какие задачи решает:**", ""]
    out += [f"- {t}" for t in card.commercial.tasks]
    out += ["", f"**Как представить клиенту:** {card.commercial.pitch}", ""]

    out += ["## Чего не хватает в данных", ""]
    if card.data_gaps:
        out += ["| Поле | Статус | Что сделать |", "|---|---|---|"]
        out += [f"| {_cell(g.field)} | {g.status} | {_cell(g.comment)} |" for g in card.data_gaps]
    else:
        out.append("Пробелов в данных не найдено.")
    out.append("")

    if result.issues:
        out += ["## Замечания автопроверки", ""]
        out += [f"- `{i.field}` — {i.message}" for i in result.issues]
        out.append("")

    if result.autofixes:
        out += ["## Исправлено автоматически", ""]
        out += [f"- {fix}" for fix in result.autofixes]
        out.append("")

    meta = result.meta
    fallback = ", сработал fallback" if meta.fallback_used else ""
    out.append(f"<sub>Модель: {meta.model}; попыток: {meta.attempts}; токены: {meta.input_tokens} вход / "
               f"{meta.output_tokens} выход{fallback}</sub>")
    return "\n".join(out) + "\n"
