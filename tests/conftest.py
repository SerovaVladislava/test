from __future__ import annotations

import copy

import pytest

from card_generator.llm import LLMResponse
from card_generator.models import ProductCard

DM02 = {
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

_SENTENCE = ("Манометр показывающий DM 02 производства Манотомь предназначен для измерения избыточного "
             "давления неагрессивных сред. ")

VALID_CARD = {
    "key_specs": [
        {"name": "Диапазон измерений", "value": "0–1,6 МПа", "source": "характеристики"},
        {"name": "Класс точности", "value": "1,5", "source": "характеристики"},
        {"name": "Присоединение", "value": "G1/2", "source": "характеристики"},
        {"name": "Диаметр корпуса", "value": "100 мм", "source": "характеристики"},
        {"name": "Степень защиты корпуса", "value": "данных нет", "source": "данных нет"},
    ],
    "data_gaps": [
        {"field": "Степень защиты корпуса", "status": "отсутствует", "comment": "Запросить IP по паспорту"},
    ],
    "search_queries": ["манометр dm 02", "манометр dm 02 манотомь", "манометр 0–1,6 мпа",
                       "купить манометр dm 02"],
    "seo_title": "Манометр показывающий DM 02 0–1,6 МПа, кл. 1,5 — Манотомь",
    "seo_description": (_SENTENCE * 8).strip(),
    "commercial": {
        "audience": ["Службы КИПиА промышленных предприятий", "Монтажные организации"],
        "tasks": ["Визуальный контроль давления в трубопроводах"],
        "pitch": "Показывающий манометр DM 02 с диапазоном 0–1,6 МПа и классом точности 1,5.",
    },
}


def make_card(**overrides) -> ProductCard:
    data = copy.deepcopy(VALID_CARD)
    data.update(overrides)
    return ProductCard.model_validate(data)


class FakeLLM:
    """Возвращает заранее заданные ответы по очереди и запоминает промпты."""

    model = "fake-model"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, system: str, user: str) -> LLMResponse:
        self.prompts.append(user)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return LLMResponse(card=item, model=self.model, input_tokens=100, output_tokens=50)


@pytest.fixture
def dm02() -> dict:
    return copy.deepcopy(DM02)
