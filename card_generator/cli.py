"""Командная строка: python -m card_generator <команда> ..."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from .config import ConfigError, load_settings
from .feed import products_by_ids, search_offers
from .input_validation import InputError, load_products, validate_product
from .llm import InvalidLLMOutput, LLMError, make_llm
from .render import to_markdown
from .service import CardGenerator


def _slug(text: str) -> str:
    translit = str.maketrans("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                             "abvgdeejziiklmnoprstufhccss_y_eua")
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower().translate(translit)).strip("-")
    return slug[:60] or "product"


def cmd_generate(args: argparse.Namespace) -> int:
    settings = load_settings()
    if not settings.has_credentials:
        print(f"Ошибка: {settings.credentials_hint}", file=sys.stderr)
        return 2

    try:
        products = load_products(Path(args.input).read_text(encoding="utf-8-sig"))
    except InputError as e:
        print(f"Ошибка во входных данных: {e}", file=sys.stderr)
        return 2

    generator = CardGenerator(make_llm(settings), max_attempts=settings.max_attempts)
    print(f"Модель: {settings.model} ({settings.provider})", file=sys.stderr)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    exit_code = 0
    for index, raw in enumerate(products, start=1):
        label = raw.get("name", f"товар #{index}") if isinstance(raw, dict) else f"товар #{index}"
        print(f"[{index}/{len(products)}] {label} …", file=sys.stderr)
        try:
            result = generator.generate(raw)
        except InputError as e:
            print(f"  пропущен — ошибка во входных данных: {e}", file=sys.stderr)
            exit_code = 1
            continue
        except (LLMError, InvalidLLMOutput) as e:
            print(f"  не удалось сгенерировать: {e}", file=sys.stderr)
            exit_code = 1
            continue

        stem = args.name if args.name and len(products) == 1 else _slug(result.product.name)
        (out_dir / f"{stem}.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8")
        (out_dir / f"{stem}.md").write_text(to_markdown(result), encoding="utf-8")
        print(f"  {result.status}: {out_dir / stem}.json, .md (попыток: {result.meta.attempts}, "
              f"замечаний: {len(result.issues)}, пробелов в данных: {len(result.card.data_gaps)})",
              file=sys.stderr)
        if result.status != "ok":
            exit_code = max(exit_code, 1)
    return exit_code


def cmd_check(args: argparse.Namespace) -> int:
    """Проверка входного JSON без обращения к LLM."""
    try:
        products = load_products(Path(args.input).read_text(encoding="utf-8-sig"))
    except InputError as e:
        print(f"Ошибка во входных данных: {e}")
        return 2
    exit_code = 0
    for index, raw in enumerate(products, start=1):
        try:
            report = validate_product(raw)
        except InputError as e:
            print(f"[{index}] ОШИБКА:")
            for err in e.errors:
                print(f"  - {err}")
            exit_code = 2
            continue
        print(f"[{index}] {report.product.name}: OK, характеристик — {len(report.product.specs)}")
        for gap in report.gaps:
            print(f"  - {gap.field}: {gap.status} — {gap.comment}")
        for note in report.notes:
            print(f"  · {note}")
    return exit_code


def cmd_from_feed(args: argparse.Namespace) -> int:
    try:
        products = products_by_ids(args.feed, args.offer_id)
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 2
    data = products[0] if len(products) == 1 else products
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Сохранено: {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


def cmd_search_feed(args: argparse.Namespace) -> int:
    for offer_id, name, vendor in search_offers(args.feed, args.query, args.limit):
        print(f"{offer_id}\t{vendor}\t{name}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .web import create_app

    create_app().run(host=args.host, port=args.port, debug=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m card_generator",
                                     description="Черновики карточек товаров КИП с помощью LLM "
                                                 "(Claude или локальная модель в Ollama)")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="сгенерировать карточки из JSON (объект или массив товаров)")
    p.add_argument("input", help="путь к входному JSON")
    p.add_argument("-o", "--out", default="output", help="папка для результатов (по умолчанию output/)")
    p.add_argument("--name", help="имя файлов результата (для одного товара)")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("check", help="проверить входной JSON без обращения к LLM")
    p.add_argument("input")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("from-feed", help="выгрузить товары из YML-фида во входной JSON")
    p.add_argument("feed", help="путь к YML-фиду (feed.xml)")
    p.add_argument("--offer-id", action="append", required=True, help="id оффера, можно несколько раз")
    p.add_argument("-o", "--out", help="куда сохранить JSON (по умолчанию — в stdout)")
    p.set_defaults(func=cmd_from_feed)

    p = sub.add_parser("search-feed", help="найти офферы в фиде по названию, типу или производителю")
    p.add_argument("feed")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_search_feed)

    p = sub.add_parser("serve", help="запустить веб-форму")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows-консоль по умолчанию не в UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"Ошибка конфигурации: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"Файл не найден: {e.filename}", file=sys.stderr)
        return 2
