"""Модели данных: входной товар, ответ LLM и итоговый результат."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

NO_DATA = "данных нет"

# Служебные поля: сохраняются в результате, но не передаются модели и не считаются
# источником технических значений (иначе, например, цифры из URL «разрешили» бы модели их упомянуть).
SERVICE_FIELDS = {"url", "feed_offer_id", "picture", "image", "price"}

SpecSource = Literal["характеристики", "описание", "название товара", "данных нет"]
GapStatus = Literal["отсутствует", "требует уточнения"]


# ---------- Вход ----------


class ProductInput(BaseModel):
    """Нормализованные входные данные о товаре."""

    name: str
    manufacturer: str | None = None
    specs: dict[str, str] = Field(default_factory=dict)
    description: str | None = None
    # Прочие поля из входного JSON (категория, артикул, ссылка и т. п.) — тоже исходные данные.
    extra: dict[str, str] = Field(default_factory=dict)

    @property
    def content_extra(self) -> dict[str, str]:
        return {k: v for k, v in self.extra.items() if k.lower() not in SERVICE_FIELDS}

    def to_prompt_dict(self) -> dict:
        data: dict = {"name": self.name, "manufacturer": self.manufacturer or NO_DATA}
        data["specs"] = self.specs or {}
        data["description"] = self.description or NO_DATA
        if self.content_extra:
            data["extra"] = self.content_extra
        return data


# ---------- Ответ LLM (структурированный вывод по JSON-схеме) ----------
# Порядок полей важен: модель генерирует JSON по порядку схемы. Сначала она
# фиксирует факты (характеристики и пробелы), затем запросы, и только потом
# пишет тексты, опираясь на уже собранные факты.


class KeySpec(BaseModel):
    name: str = Field(description="Название характеристики")
    value: str = Field(description="Значение дословно из входных данных или «данных нет»")
    source: SpecSource = Field(description="Откуда взято значение")


class DataGap(BaseModel):
    field: str = Field(description="Чего не хватает")
    status: GapStatus
    comment: str = Field(description="Что именно запросить у поставщика или найти в паспорте")


class Commercial(BaseModel):
    audience: list[str] = Field(description="Кому подходит прибор, 2–4 пункта")
    tasks: list[str] = Field(description="Какие задачи решает, 2–4 пункта")
    pitch: str = Field(description="1–2 предложения для менеджера, только на основе данных")


class ProductCard(BaseModel):
    key_specs: list[KeySpec]
    data_gaps: list[DataGap]
    search_queries: list[str] = Field(description="3–5 поисковых запросов")
    seo_title: str = Field(description="SEO-заголовок, не длиннее 70 символов с пробелами")
    seo_description: str = Field(description="SEO-описание, 800–1200 символов с пробелами, без Markdown")
    commercial: Commercial


# ---------- Итог ----------


class CheckIssue(BaseModel):
    code: str
    field: str
    message: str


class GenerationMeta(BaseModel):
    model: str
    attempts: int
    input_tokens: int = 0
    output_tokens: int = 0
    fallback_used: bool = False


class CardResult(BaseModel):
    # ok — все автоматические проверки пройдены; needs_review — после всех попыток остались замечания.
    status: Literal["ok", "needs_review"]
    product: ProductInput
    card: ProductCard
    issues: list[CheckIssue] = Field(default_factory=list)
    # Что код поправил в ответе модели сам (например, вернул потерянную входную характеристику).
    autofixes: list[str] = Field(default_factory=list)
    stats: dict[str, int] = Field(default_factory=dict)
    meta: GenerationMeta
