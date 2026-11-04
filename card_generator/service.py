"""Оркестрация: проверка входа → LLM → автопроверка → повтор с замечаниями."""

from __future__ import annotations

import logging
from typing import Any

from .input_validation import InputReport, validate_product
from .llm import CardLLM, InvalidLLMOutput
from .models import CardResult, DataGap, GenerationMeta, ProductCard
from .postcheck import card_stats, check_card, complete_input_specs, normalize_card
from .prompts import SYSTEM_PROMPT, build_user_prompt

log = logging.getLogger(__name__)


class CardGenerator:
    def __init__(self, llm: CardLLM, max_attempts: int = 3):
        self.llm = llm
        self.max_attempts = max_attempts

    def generate(self, raw_product: Any) -> CardResult:
        """raw_product — словарь из входного JSON. InputError пробрасывается наружу,
        LLMError — тоже (повтор того же запроса не поможет)."""
        report = validate_product(raw_product)

        best: ProductCard | None = None
        best_issues: list = []
        best_fixes: list[str] = []
        review: list[str] = []
        draft: ProductCard | None = None
        meta = GenerationMeta(model=self.llm.model, attempts=0)
        last_invalid: InvalidLLMOutput | None = None

        for attempt in range(1, self.max_attempts + 1):
            meta.attempts = attempt
            prompt = build_user_prompt(report, previous_draft=draft, review_issues=review)
            try:
                response = self.llm.generate(SYSTEM_PROMPT, prompt)
            except InvalidLLMOutput as e:
                log.warning("Попытка %d: %s", attempt, e)
                last_invalid = e
                review = [f"Предыдущий ответ не удалось разобрать ({e}). Верни ответ строго по схеме."]
                continue

            meta.model = response.model
            meta.input_tokens += response.input_tokens
            meta.output_tokens += response.output_tokens
            meta.fallback_used = meta.fallback_used or response.fallback_used

            card = normalize_card(response.card)
            fixes = complete_input_specs(card, report.product)
            issues = check_card(card, report.product)
            log.info("Попытка %d: замечаний автопроверки — %d, автоисправлений — %d",
                     attempt, len(issues), len(fixes))
            if best is None or len(issues) <= len(best_issues):
                best, best_issues, best_fixes = card, issues, fixes
            if not issues:
                break
            draft, review = card, [i.message for i in issues]

        if best is None:
            # Ни одного разборчивого ответа за все попытки.
            raise last_invalid or InvalidLLMOutput("Модель не вернула ответ")

        best.data_gaps = merge_gaps(report, best.data_gaps)
        return CardResult(
            status="ok" if not best_issues else "needs_review",
            product=report.product,
            card=best,
            issues=best_issues,
            autofixes=best_fixes,
            stats=card_stats(best),
            meta=meta,
        )


def merge_gaps(report: InputReport, llm_gaps: list[DataGap]) -> list[DataGap]:
    """Пробелы, найденные при проверке входа, + найденные моделью, без дублей."""
    merged: list[DataGap] = []
    seen: set[str] = set()
    for gap in [*report.gaps, *llm_gaps]:
        key = gap.field.strip().lower()
        if key not in seen:
            seen.add(key)
            merged.append(gap)
    return merged
