"""Импорт товаров из YML-фида (формат Яндекс.Маркета) во входной JSON сервиса."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

# Служебные параметры фида — не технические характеристики прибора.
SERVICE_PARAMS = {"код товара"}
SKU_PARAMS = {"артикул"}


def _text(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def load_categories(path: str | Path) -> dict[str, tuple[str, str | None]]:
    """id → (название, id родителя). Категории идут в фиде до товаров, дальше не читаем."""
    categories: dict[str, tuple[str, str | None]] = {}
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "category":
            categories[el.get("id", "")] = (" ".join((el.text or "").split()), el.get("parentId"))
        elif el.tag == "categories":
            break
    return categories


def category_path(categories: dict[str, tuple[str, str | None]], category_id: str) -> str:
    names: list[str] = []
    seen: set[str] = set()
    current: str | None = category_id
    while current and current in categories and current not in seen:
        seen.add(current)
        name, current = categories[current]
        names.append(name)
    return " / ".join(reversed(names))


def iter_offers(path: str | Path) -> Iterator[ET.Element]:
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "offer":
            yield el
            el.clear()


def offer_to_product(offer: ET.Element, categories: dict[str, tuple[str, str | None]]) -> dict:
    """Собирает входной JSON сервиса из оффера. Ничего не додумывает: чего нет в фиде, того нет в JSON."""
    model = _text(offer.find("model")) or _text(offer.find("name"))
    type_prefix = _text(offer.find("typePrefix"))
    if type_prefix and type_prefix.lower() not in model.lower():
        name = f"{type_prefix[:1].upper()}{type_prefix[1:]} {model}"
    else:
        name = model

    product: dict = {"name": name, "manufacturer": _text(offer.find("vendor"))}

    specs: dict[str, str] = {}
    sku = ""
    for param in offer.findall("param"):
        key = (param.get("name") or "").strip()
        value = _text(param)
        unit = param.get("unit")
        if not key or not value or key.lower() in SERVICE_PARAMS:
            continue
        if key.lower() in SKU_PARAMS:
            sku = value
            continue
        specs[key] = f"{value} {unit}" if unit else value
    product["specs"] = specs

    description = _text(offer.find("description"))
    if description:
        product["description"] = description

    if category_id := _text(offer.find("categoryId")):
        product["category"] = category_path(categories, category_id)
    if sku or (sku := _text(offer.find("vendorCode"))):
        product["sku"] = sku
    if url := _text(offer.find("url")):
        product["url"] = url
    product["feed_offer_id"] = offer.get("id", "")
    return product


def products_by_ids(path: str | Path, offer_ids: list[str]) -> list[dict]:
    categories = load_categories(path)
    wanted = set(offer_ids)
    found: dict[str, dict] = {}
    for offer in iter_offers(path):
        offer_id = offer.get("id", "")
        if offer_id in wanted:
            found[offer_id] = offer_to_product(offer, categories)
            if len(found) == len(wanted):
                break
    missing = [i for i in offer_ids if i not in found]
    if missing:
        raise KeyError(f"В фиде не найдены офферы: {', '.join(missing)}")
    return [found[i] for i in offer_ids]


def search_offers(path: str | Path, query: str, limit: int = 20) -> list[tuple[str, str, str]]:
    """Поиск по подстроке в названии/типе/производителе. Возвращает (id, название, производитель)."""
    needle = query.lower()
    results: list[tuple[str, str, str]] = []
    for offer in iter_offers(path):
        name = _text(offer.find("name"))
        haystack = " ".join([name, _text(offer.find("typePrefix")), _text(offer.find("vendor"))]).lower()
        if needle in haystack:
            results.append((offer.get("id", ""), name, _text(offer.find("vendor"))))
            if len(results) >= limit:
                break
    return results
