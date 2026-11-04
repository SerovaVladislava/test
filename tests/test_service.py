import pytest

from card_generator.input_validation import InputError
from card_generator.llm import InvalidLLMOutput, LLMError
from card_generator.service import CardGenerator

from .conftest import FakeLLM, make_card


def test_first_attempt_ok(dm02):
    llm = FakeLLM(make_card())
    result = CardGenerator(llm).generate(dm02)
    assert result.status == "ok"
    assert result.meta.attempts == 1
    assert result.stats["seo_title_chars"] == len(result.card.seo_title)
    assert "<product_data>" in llm.prompts[0]
    assert "<previous_draft>" not in llm.prompts[0]


def test_lost_input_spec_is_restored_without_retry(dm02):
    card = make_card()
    card.key_specs = [s for s in card.key_specs if s.value != "100 мм"]
    result = CardGenerator(FakeLLM(card)).generate(dm02)
    assert result.status == "ok" and result.meta.attempts == 1
    assert result.autofixes == ["Добавлена характеристика из входных данных: «диаметр корпуса: 100 мм»"]


def test_retry_with_review_feedback(dm02):
    llm = FakeLLM(make_card(seo_title="М" * 80), make_card())
    result = CardGenerator(llm).generate(dm02)
    assert result.status == "ok"
    assert result.meta.attempts == 2
    assert result.meta.input_tokens == 200
    assert "<previous_draft>" in llm.prompts[1]
    assert "80 символов" in llm.prompts[1]


def test_needs_review_after_all_attempts(dm02):
    llm = FakeLLM(make_card(seo_title="Манометр DM 02 на 16 бар"))
    result = CardGenerator(llm, max_attempts=3).generate(dm02)
    assert result.status == "needs_review"
    assert result.meta.attempts == 3
    assert [i.code for i in result.issues] == ["ungrounded_value"]


def test_best_attempt_is_returned(dm02):
    worse = make_card(seo_title="М" * 80, search_queries=["один"])
    better = make_card(seo_title="М" * 80)
    result = CardGenerator(FakeLLM(better, worse, worse), max_attempts=3).generate(dm02)
    assert len(result.issues) == 1


def test_invalid_output_is_retried(dm02):
    llm = FakeLLM(InvalidLLMOutput("обрезан"), make_card())
    result = CardGenerator(llm).generate(dm02)
    assert result.status == "ok"
    assert result.meta.attempts == 2
    assert "не удалось разобрать" in llm.prompts[1]


def test_invalid_output_every_time(dm02):
    with pytest.raises(InvalidLLMOutput):
        CardGenerator(FakeLLM(InvalidLLMOutput("мусор")), max_attempts=2).generate(dm02)


def test_api_error_is_not_retried(dm02):
    llm = FakeLLM(LLMError("ключ отклонён"), make_card())
    with pytest.raises(LLMError):
        CardGenerator(llm).generate(dm02)
    assert len(llm.prompts) == 1


def test_input_error_before_llm():
    llm = FakeLLM(make_card())
    with pytest.raises(InputError):
        CardGenerator(llm).generate({"manufacturer": "ОВЕН"})
    assert llm.prompts == []


def test_gaps_are_merged_without_duplicates(dm02):
    dm02.pop("description")
    llm = FakeLLM(make_card(data_gaps=[
        {"field": "описание", "status": "отсутствует", "comment": "дубль"},
        {"field": "Степень защиты корпуса", "status": "отсутствует", "comment": "Запросить IP"},
    ]))
    result = CardGenerator(llm).generate(dm02)
    assert [g.field for g in result.card.data_gaps] == ["Описание", "Степень защиты корпуса"]
    assert "<input_check>" in llm.prompts[0]


def test_sparse_product_prompt_has_no_service_fields():
    llm = FakeLLM(make_card())
    product = {"name": "Манометр показывающий МП4-У 0...60 кгс/см² кл.т 1,5", "manufacturer": "Манотомь",
               "specs": {}, "url": "https://example.ru/mp4-u/", "feed_offer_id": "205847"}
    CardGenerator(llm, max_attempts=1).generate(product)
    prompt = llm.prompts[0]
    assert "Нет ни характеристик, ни описания" in prompt
    assert "example.ru" not in prompt and "205847" not in prompt
