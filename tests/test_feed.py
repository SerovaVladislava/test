from pathlib import Path

import pytest

from card_generator.feed import products_by_ids, search_offers
from card_generator.input_validation import validate_product

FEED = Path(__file__).parent / "fixtures" / "feed_sample.xml"


def test_offer_to_product_without_specs():
    [mp4u] = products_by_ids(FEED, ["205847"])
    assert mp4u == {
        "name": "Манометр показывающий МП4-У 0...60 кгс/см² кл.т 1,5",
        "manufacturer": "Манотомь",
        "specs": {},
        "category": "Теплотехнические приборы, газоанализаторы, счетчики / Манометры показывающие",
        "url": "https://www.souz-pribor.ru/product/mp4-u/",
        "feed_offer_id": "205847",
    }


def test_sku_and_service_params():
    [pd100] = products_by_ids(FEED, ["472022"])
    assert pd100["sku"] == "161871"
    assert pd100["specs"] == {}  # «Код товара» и «Артикул» — не характеристики прибора
    assert pd100["name"].startswith("Датчик давления взрывозащищенный, искробезопасный ПД100И")


def test_params_with_units_and_description():
    [offer] = products_by_ids(FEED, ["900001"])
    assert offer["specs"] == {"Выходной сигнал": "4...20 мА", "Диапазон": "0...1 МПа"}
    report = validate_product(offer)
    assert report.product.description == "Описание из фида."


def test_order_is_preserved_and_missing_reported():
    products = products_by_ids(FEED, ["472022", "205847"])
    assert [p["feed_offer_id"] for p in products] == ["472022", "205847"]
    with pytest.raises(KeyError, match="1"):
        products_by_ids(FEED, ["205847", "1"])


def test_search():
    assert [r[0] for r in search_offers(FEED, "овен")] == ["472022"]
    assert [r[0] for r in search_offers(FEED, "датчик давления")] == ["472022", "900001"]
