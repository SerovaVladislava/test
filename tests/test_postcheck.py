import pytest

from card_generator.input_validation import validate_product
from card_generator.models import Commercial, KeySpec, ProductInput
from card_generator.postcheck import check_card, complete_input_specs, normalize_card, source_numbers

from .conftest import _SENTENCE, make_card


def codes(issues):
    return [i.code for i in issues]


def product(dm02) -> ProductInput:
    return validate_product(dm02).product


def test_valid_card_passes(dm02):
    assert check_card(normalize_card(make_card()), product(dm02)) == []


def test_invented_value_is_flagged(dm02):
    card = make_card(seo_description=(_SENTENCE * 8) + "Рабочая температура среды до 60 °C.")
    issues = check_card(card, product(dm02))
    assert codes(issues) == ["ungrounded_value"]
    assert "60 °C" in issues[0].message


def test_unit_conversion_is_flagged(dm02):
    card = make_card(seo_title="Манометр DM 02 на 16 бар")
    issues = check_card(card, product(dm02))
    assert "ungrounded_value" in codes(issues)


def test_invented_spec_value_is_flagged(dm02):
    card = make_card()
    card.key_specs.append(KeySpec(name="Степень защиты", value="IP65", source="описание"))
    assert codes(check_card(card, product(dm02))) == ["ungrounded_value"]


def test_number_formats_are_equivalent():
    p = ProductInput(name="DT-8865", specs={"диапазон": "-50...+1 000 °C", "погрешность": "1,5 %"})
    nums = source_numbers(p)
    assert {"50", "1000", "1", "0", "1.5", "8865"} <= nums


def test_service_fields_do_not_allow_numbers():
    p = ProductInput(name="Манометр", extra={"url": "https://x.ru/pd100/", "category": "Манометры 2024"})
    nums = source_numbers(p)
    assert "100" not in nums and "2024" in nums


def test_lengths_and_counts(dm02):
    card = make_card(seo_title="М" * 71, seo_description="Короткое описание.",
                     search_queries=["манометр", "Манометр", "манометр dm 02"])
    issues = check_card(normalize_card(card), product(dm02))
    assert set(codes(issues)) == {"title_length", "description_length", "queries_count"}


def test_missing_input_spec(dm02):
    card = make_card()
    card.key_specs = [s for s in card.key_specs if s.value != "100 мм"]
    issues = check_card(card, product(dm02))
    assert codes(issues) == ["missing_input_spec"]
    assert "100 мм" in issues[0].message


def test_complete_input_specs_restores_lost_and_changed_values(dm02):
    card = make_card()
    card.key_specs = [s for s in card.key_specs if s.value != "100 мм"]
    card.key_specs[0].value = "0-1.6 МПа"  # модель «пересказала» значение
    fixes = complete_input_specs(card, product(dm02))
    assert len(fixes) == 2 and "100 мм" in fixes[1]
    # Возвращённая характеристика встаёт перед строками «данных нет».
    assert [s.value for s in card.key_specs] == ["0–1,6 МПа", "1,5", "G1/2", "100 мм", "данных нет"]
    assert check_card(card, product(dm02)) == []


def test_complete_input_specs_keeps_complete_card(dm02):
    card = make_card()
    assert complete_input_specs(card, product(dm02)) == []
    assert card == make_card()


def test_superlative_and_markdown(dm02):
    card = make_card(seo_description="**Лучший** манометр. " + _SENTENCE * 8)
    assert set(codes(check_card(card, product(dm02)))) == {"superlative", "markdown"}


PD100 = {"name": "Датчик давления взрывозащищенный, искробезопасный ПД100И-ДИ1,0-111-0,5-ЕХI (М23)",
         "manufacturer": "ОВЕН"}


def pd100_card(**overrides):
    data = dict(
        key_specs=[{"name": "Модель", "value": "ПД100И-ДИ1,0-111-0,5-ЕХI (М23)", "source": "название товара"}],
        search_queries=["датчик давления пд100и", "овен пд100и-ди1,0-111-0,5-ехi", "пд100и цена"],
        seo_title="Датчик давления ПД100И-ДИ1,0-111-0,5-ЕХI (М23) ОВЕН",
        seo_description="Датчик давления ОВЕН ПД100И (М23) во взрывозащищённом исполнении. " * 14,
        commercial={"audience": ["Службы КИПиА"], "tasks": ["Контроль давления"], "pitch": "Датчик ОВЕН ПД100И."},
    )
    data.update(overrides)
    return make_card(**data)


def test_modification_code_mentions_are_allowed():
    assert check_card(pd100_card(), validate_product(PD100).product) == []


def test_value_decoded_from_modification_code_is_flagged():
    card = pd100_card(commercial={"audience": ["Службы КИПиА"], "tasks": ["Контроль давления"],
                                  "pitch": "Датчик с классом точности 1,0 и выходом 0,5."})
    issues = check_card(card, validate_product(PD100).product)
    assert codes(issues) == ["code_value", "code_value"]
    assert "ПД100И-ДИ1,0-111-0,5-ЕХI" in issues[0].message


def test_foreign_script_is_flagged_once_per_field(dm02):
    card = make_card()
    card.key_specs.append(KeySpec(name="Степень защиты от溅射", value="данных нет", source="данных нет"))
    card.key_specs.append(KeySpec(name="Защита 溅射", value="данных нет", source="данных нет"))
    issues = check_card(card, product(dm02))
    assert codes(issues) == ["foreign_script"] and "溅射" in issues[0].message


@pytest.mark.parametrize("phrase", ["Идеально подходит для котельных.", "Обеспечивает высокую точность измерений."])
def test_marketing_evaluations_are_flagged(dm02, phrase):
    card = make_card(seo_description=_SENTENCE * 8 + phrase)
    assert codes(check_card(card, product(dm02))) == ["superlative"]


def test_normalize_marks_missing_values():
    card = make_card()
    card.key_specs.append(KeySpec(name="Материал корпуса", value="Данных нет", source="характеристики"))
    card.key_specs.append(KeySpec(name="Масса", value="будет позже", source="данных нет"))
    card.commercial = Commercial(audience=["a"], tasks=["b"], pitch="c")
    fixed = normalize_card(card)
    assert [(s.value, s.source) for s in fixed.key_specs[-2:]] == [("данных нет", "данных нет")] * 2
