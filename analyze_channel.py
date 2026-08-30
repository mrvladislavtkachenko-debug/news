#!/usr/bin/env python3
"""Точка входа: локальный аудит Markdown-экспорта Telegram-канала.

Примеры:
    python3 analyze_channel.py data/channel.md
    python3 analyze_channel.py data/channel.md --md out/report.md --json out/report.json
    python3 analyze_channel.py data/channel.md --explain 205
    python3 analyze_channel.py data/channel.md --own-domains sdelaem.agency,molyanov.notion.site
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from textforge.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
