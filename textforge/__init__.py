"""textforge — локальный, детерминированный анализ текстовых баз.

Пакет реализует «логический аппарат» для обработки больших текстовых файлов
без обращения к LLM: разбор структуры, словари признаков, скоринг, отчёт.

Точка входа для аудита Telegram-канала:

    from textforge import analyze_file, render_markdown

    result = analyze_file("data/channel.md")
    print(result.markdown)
"""

from dataclasses import dataclass
from typing import Any

from .analyze import AnalyzerConfig, ChannelAnalysis, ChannelAnalyzer, analyze
from .report import render_markdown, to_dict, to_json
from .tgparser import Channel, Post, load_channel, parse_channel

__all__ = [
    "AnalyzerConfig",
    "AnalysisResult",
    "ChannelAnalyzer",
    "Channel",
    "Post",
    "analyze",
    "analyze_file",
    "load_channel",
    "parse_channel",
    "render_markdown",
    "to_dict",
    "to_json",
]

__version__ = "0.1.0"


@dataclass
class AnalysisResult:
    """Результат `analyze_file`: отчёт в трёх видах сразу."""

    markdown: str
    data: dict
    analysis: ChannelAnalysis


def analyze_file(
    path: str, *, own_domains: list[str] | None = None, handle: str = ""
) -> AnalysisResult:
    """Однострочный API: прочитать экспорт канала и получить отчёт, dict и сам анализ."""
    channel = load_channel(path)
    config = AnalyzerConfig(own_domains=list(own_domains or []), channel_handle=handle)
    analysis = analyze(channel, config)
    return AnalysisResult(
        markdown=render_markdown(analysis), data=to_dict(analysis), analysis=analysis
    )
