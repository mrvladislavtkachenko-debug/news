"""Командная строка анализатора.

    python3 analyze_channel.py data/channel.md
    python3 analyze_channel.py data/channel.md --json report.json --md report.md
    python3 analyze_channel.py data/channel.md --explain 205
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .analyze import AnalyzerConfig, analyze
from .report import render_markdown, to_dict, to_json
from .tgparser import load_channel


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze_channel",
        description="Локальный аудит Telegram-канала по фиксированному промту (без LLM и без сети).",
    )
    p.add_argument("input", nargs="?", help="путь к Markdown-экспорту канала")
    p.add_argument("--md", help="куда сохранить Markdown-отчёт (по умолчанию stdout)")
    p.add_argument("--json", dest="json_path", help="куда сохранить JSON со всеми метриками")
    p.add_argument("--explain", metavar="POST_ID", help="показать, из чего сложились оценки конкретного поста")
    p.add_argument(
        "--own-domains",
        default="",
        help="домены автора через запятую, например sdelaem.agency,molyanov.notion.site",
    )
    p.add_argument(
        "--handle",
        default="",
        help="username канала (подставится в отчёт, если в экспорте его нет, "
             "и позволит отличать собственные ссылки от внешних)",
    )
    p.add_argument("--quiet", action="store_true", help="не печатать служебную статистику в stderr")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input:
        build_parser().print_help()
        return 2

    started = time.time()
    try:
        channel = load_channel(args.input)
    except OSError as exc:
        print(f"Не удалось прочитать файл: {exc}", file=sys.stderr)
        return 1
    if not channel.posts:
        print(
            f"В файле {args.input} не найдено ни одного поста. "
            "Ожидается Markdown-экспорт с заголовками вида «## Post 123».",
            file=sys.stderr,
        )
        return 1
    if args.handle.strip() and not channel.username:
        channel.username = args.handle.strip().lstrip("@")
    config = AnalyzerConfig(
        own_domains=[d.strip() for d in args.own_domains.split(",") if d.strip()],
        channel_handle=args.handle.strip().lstrip("@"),
    )
    analysis = analyze(channel, config)
    elapsed = time.time() - started

    if args.explain:
        import json as _json

        try:
            payload = analysis.explain(args.explain)
        except KeyError as exc:
            ids = ", ".join(str(p.post.number) for p in analysis.posts[:10])
            print(f"Пост {exc.args[0].split()[1]} не найден. Доступные id: {ids} …", file=sys.stderr)
            return 1
        print(_json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    markdown = render_markdown(analysis)
    if args.md:
        Path(args.md).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write(markdown)
    else:
        print(markdown)

    if args.json_path:
        Path(args.json_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_path, "w", encoding="utf-8") as fh:
            fh.write(to_json(analysis))

    if not args.quiet:
        print(
            f"[textforge] постов разобрано: {len(analysis.posts)} из заявленных {channel.posts_declared}; "
            f"предупреждений: {len(channel.warnings)}; время: {elapsed:.2f} c",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
