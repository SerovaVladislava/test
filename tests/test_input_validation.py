from pathlib import Path

import pytest

from card_generator.input_validation import InputError, load_products, parse_json_text, validate_product

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "input"


def test_broken_json_reports_position():
    with pytest.raises(InputError) as e:
        parse_json_text((EXAMPLES / "invalid" / "broken_json.json").read_text(encoding="utf-8"))
    assert "Некорректный JSON" in e.value.errors[0]
    assert "строка" in e.value.errors[0]


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_empty_input(text):
    with pytest.raises(InputError, match="Пустой ввод"):
        parse_json_text(text)


@pytest.mark.parametrize("raw", [["a"], "строка", 42, None])
def test_not_an_object(raw):
    with pytest.raises(InputError, match="JSON-объект"):
        validate_product(raw)


@pytest.mark.parametrize("name", [None, "", "  ", "-", {"x": 1}])
def test_name_is_required(name):
    with pytest.raises(InputError, match="name"):
        validate_product({"name": name, "manufacturer": "ОВЕН"})


def test_full_product_has_no_gaps(dm02):
    report = validate_product(dm02)
    assert report.gaps == []
    assert report.product.specs["класс точности"] == "1,5"
    assert not report.is_sparse


def test_missing_fields_become_gaps():
    report = validate_product({"name": "Манометр МП4-У"})
    fields = {g.field for g in report.gaps}
    assert fields == {"Производитель", "Технические характеристики", "Описание"}
    assert report.is_sparse


def test_empty_spec_values_become_gaps_not_specs():
    report = validate_product({"name": "ТБ-1", "specs": {"диапазон": "0...120 °C", "длина": "", "класс": "-",
                                                         "вес": None}})
    assert report.product.specs == {"диапазон": "0...120 °C"}
    assert {"длина", "класс", "вес"} <= {g.field for g in report.gaps}


def test_specs_list_and_nested_forms():
    report = validate_product({
        "name": "ТРМ1",
        "specs": [{"name": "Питание", "value": "220 В"}, {"name": "Входов", "value": 1}],
    })
    assert report.product.specs == {"Питание": "220 В", "Входов": "1"}

    report = validate_product({"name": "ТРМ1", "specs": {"питание": {"напряжение": "24 В", "ток": 0.5},
                                                         "интерфейсы": ["RS-485", "USB"]}})
    assert report.product.specs == {"питание / напряжение": "24 В", "питание / ток": "0.5",
                                    "интерфейсы": "RS-485; USB"}


def test_specs_wrong_type():
    with pytest.raises(InputError, match="specs"):
        validate_product({"name": "ДТС", "specs": "диапазон 0-100"})


def test_description_html_is_stripped():
    report = validate_product({"name": "ТБ-1", "description": "<p>Термометр&nbsp;для <b>воды</b></p>"})
    assert report.product.description == "Термометр для воды"


def test_extra_fields_are_kept():
    report = validate_product({"name": "ПД100", "category": "Датчики давления", "sku": 161871,
                               "nested": {"a": 1}})
    assert report.product.extra == {"category": "Датчики давления", "sku": "161871"}
    assert any("nested" in n for n in report.notes)


def test_example_file_with_errors():
    products = load_products((EXAMPLES / "invalid" / "incomplete_and_wrong_types.json").read_text(encoding="utf-8"))
    assert len(products) == 2
    with pytest.raises(InputError) as e:
        validate_product(products[0])
    assert len(e.value.errors) == 2  # пустое name и specs строкой

    report = validate_product(products[1])
    assert report.product.description.startswith("Термометр для измерения")
    assert {"Производитель", "длина погружной части", "класс точности"} <= {g.field for g in report.gaps}
