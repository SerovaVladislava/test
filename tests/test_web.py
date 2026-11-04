import json

import pytest

from card_generator.service import CardGenerator
from card_generator.web import create_app

from .conftest import DM02, FakeLLM, make_card


@pytest.fixture
def client():
    app = create_app(CardGenerator(FakeLLM(make_card())))
    return app.test_client()


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Сгенерировать" in r.get_data(as_text=True)


def test_api_generate(client):
    r = client.post("/api/generate", data=json.dumps(DM02, ensure_ascii=False),
                    content_type="application/json")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["card"]["seo_title"]


@pytest.mark.parametrize("payload, message", [
    ("{broken", "Некорректный JSON"),
    ('{"manufacturer": "ОВЕН"}', "name"),
    ("", "Пустой ввод"),
])
def test_api_generate_invalid(client, payload, message):
    r = client.post("/api/generate", data=payload, content_type="application/json")
    assert r.status_code == 400
    assert message in r.get_json()["messages"][0]


def test_api_check(client):
    r = client.post("/api/check", data='{"name": "МП4-У"}', content_type="application/json")
    assert r.status_code == 200
    assert {g["field"] for g in r.get_json()["gaps"]} == {"Производитель", "Технические характеристики",
                                                           "Описание"}


def test_form_fields(client):
    r = client.post("/", data={
        "mode": "fields", "name": DM02["name"], "manufacturer": DM02["manufacturer"],
        "specs": "диапазон измерений: 0–1,6 МПа\nкласс точности: 1,5\nприсоединение: G1/2\n"
                 "диаметр корпуса: 100 мм",
        "description": DM02["description"],
    })
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "Автопроверки пройдены" in html and "данных нет" in html


def test_form_bad_specs_line(client):
    r = client.post("/", data={"mode": "fields", "name": "Манометр", "specs": "просто текст"})
    assert r.status_code == 400
    assert "строка 1" in r.get_data(as_text=True)
