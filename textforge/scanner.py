"""Быстрый сканер словарей.

:mod:`textforge.match` удобен для точечного поиска с подробной статистикой, но для
тысяч постов и сотен терминов он медленный (проход по токенам на каждый термин).

``FastLexicon`` компилирует словарь один раз и считает все вхождения за один проход
по документу:

* однословные термины -> сравнение с частотным словарём основ (O(уникальных токенов));
* фразы -> один регулярный проход по нормализованному тексту;
* ``re:`` -> отдельная регулярка.

Сложность на документ: O(токены + число фраз), что позволяет обрабатывать экспорт
на несколько тысяч постов за секунды.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import Doc, stem

_WORD = r"[^\s]"


def _term_to_regex(term: str) -> re.Pattern:
    """«ключевая ставка» -> регулярка с допуском в 2 слова и префиксным матчингом основ."""
    words = [w for w in term.replace("*", " ").split() if w]
    parts = []
    for w in words:
        base = stem(w.lower())
        # усечённый префикс основы: «сила» -> «сил» ловит «силы», «силой»
        prefix = base[: max(3, len(base) - 2)]
        parts.append(re.escape(prefix) + r"[\w\-]*")
    glue = r"\s+(?:" + _WORD + r"+\s+){0,2}"
    return re.compile(glue.join(parts), re.I)


@dataclass
class LexiconHits:
    """group -> term -> число вхождений."""

    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def get(self, group: str) -> dict[str, int]:
        return self.counts.get(group, {})

    def total(self, group: str) -> int:
        return sum(self.counts.get(group, {}).values())

    def terms(self, group: str) -> list[str]:
        return sorted(self.counts.get(group, {}))

    def has(self, group: str) -> bool:
        return bool(self.counts.get(group))


class FastLexicon:
    def __init__(self, groups: dict[str, list[str]]):
        self.groups = groups
        # основа -> [(группа, термин)]
        self.singles: dict[str, list[tuple[str, str]]] = {}
        # (группа, термин, регулярка)
        self.phrases: list[tuple[str, str, re.Pattern]] = []
        self.regexes: list[tuple[str, str, re.Pattern]] = []

        registered: set[tuple[str, str]] = set()
        for group, terms in groups.items():
            for raw in terms:
                term = str(raw).strip()
                if not term:
                    continue
                if term.lower().startswith("re:"):
                    self.regexes.append((group, term, re.compile(term[3:].strip(), re.I)))
                    continue
                words = [w for w in term.replace("*", " ").split() if w]
                if not words:
                    continue
                if len(words) == 1:
                    key = (group, stem(words[0].lower()))
                    # «кейс» и «кейсы» дают одну основу — иначе вхождение посчитается дважды
                    if key in registered:
                        continue
                    registered.add(key)
                    self.singles.setdefault(key[1], []).append((group, term))
                else:
                    sig = (group, tuple(stem(w.lower()) for w in words))
                    if sig in registered:
                        continue
                    registered.add(sig)
                    self.phrases.append((group, term, _term_to_regex(term)))

    # ------------------------------------------------------------------
    def scan(self, doc: Doc) -> LexiconHits:
        counts: dict[str, dict[str, int]] = {}

        def bump(group: str, term: str, n: int = 1) -> None:
            counts.setdefault(group, {})
            counts[group][term] = counts[group].get(term, 0) + n

        freq: dict[str, int] = {}
        for s in doc.stems:
            freq[s] = freq.get(s, 0) + 1

        for base, n in freq.items():
            for group, term in self.singles.get(base, ()):
                bump(group, term, n)
            if len(base) < 4:
                continue
            seen: set[tuple[str, str]] = set()
            # суффиксное совпадение: «сделай»(сдела) находит термин «делай»(дела)
            for start in range(1, len(base) - 3):
                for group, term in self.singles.get(base[start:], ()):
                    if (group, term) in seen:
                        continue
                    seen.add((group, term))
                    bump(group, term, n)
            # префиксное совпадение для терминов с «*»: «выбор*» -> «выборная»
            for prefix_len in range(4, len(base)):
                for group, term in self.singles.get(base[:prefix_len], ()):
                    if term.endswith("*") and (group, term) not in seen:
                        seen.add((group, term))
                        bump(group, term, n)

        text = doc.text.lower()
        for group, term, rx in self.phrases:
            hits = len(rx.findall(text))
            if hits:
                bump(group, term, hits)
        for group, term, rx in self.regexes:
            hits = len(rx.findall(text))
            if hits:
                bump(group, term, hits)

        return LexiconHits(counts)


def merged(*group_maps: dict[str, list[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for gm in group_maps:
        out.update(gm)
    return out
