"""Матчинг терминов: ядро «логического аппарата».

Поддерживаемые формы термина:

* ``выборы``             — точное совпадение по основе (окончание не важно);
* ``выбор*``             — префиксное совпадение по основе (выборы, выборов, выборная);
* ``центральный банк``   — фраза: основы идут подряд, допускается разрыв до 2 слов;
* ``"инфляц"~8``         — термин + окно: считается только внутри 8 токенов (для near);
* ``re:инфляц|рост цен`` — регулярное выражение по нормализованному тексту;
* ``!спецоперация``      — отрицание (используется в блоках `none`).

Скоринг: ``weight * (1 + ln(1 + count))``, совпадение в первом предложении (заголовок/лид)
умножается на ``head_bonus``. Логарифм нужен, чтобы 40 упоминаний одного слова
не задавили все остальные признаки.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .text import Doc, stem, tokenize

MAX_GAP = 2  # сколько посторонних слов допускается между словами фразы


@dataclass
class TermMatch:
    term: str
    count: int
    weight: float
    in_head: bool
    positions: list[int] = field(default_factory=list)
    snippet: str = ""

    @property
    def score(self) -> float:
        base = self.weight * (1.0 + math.log(1.0 + self.count))
        return round(base * (1.5 if self.in_head else 1.0), 4)


Part = tuple[str, str]  # ("stem", "основ") | ("glob", "префикс")


@dataclass
class Term:
    source: str
    kind: str  # exact | glob | phrase | regex
    parts: list[Part] = field(default_factory=list)
    regex: re.Pattern | None = None
    weight: float = 1.0
    gap: int = MAX_GAP

    def __str__(self) -> str:  # pragma: no cover - удобно в отладке
        return self.source


_REGEX_PREFIX = "re:"


def compile_term(spec: str | dict) -> Term:
    """Собирает термин из конфига. Строка или словарь {t, w, weight, gap}."""
    weight = 1.0
    gap = MAX_GAP
    if isinstance(spec, dict):
        raw = str(spec.get("t") or spec.get("term") or "")
        weight = float(spec.get("w", spec.get("weight", 1.0)))
        gap = int(spec.get("gap", MAX_GAP))
    else:
        raw = str(spec)

    raw = raw.strip()
    if not raw:
        raise ValueError("пустой термин в конфигурации")

    if raw.lower().startswith(_REGEX_PREFIX):
        pattern = raw[len(_REGEX_PREFIX) :].strip()
        return Term(source=raw, kind="regex", regex=re.compile(pattern, re.I), weight=weight)

    words = [w for w in raw.strip("* ").split() if w]
    if not words:
        raise ValueError(f"не разобран термин: {raw!r}")

    # Префикс/окончание: «выбор*» -> ("glob", "выбор"); «выборы» -> ("stem", "выбор")
    parts: list[Part] = []
    for i, w in enumerate(words):
        glob = raw.endswith("*") and i == len(words) - 1 and len(words) == 1
        if glob:
            parts.append(("glob", stem(w.lower())))
        else:
            parts.append(("stem", stem(w.lower())))
    kind = "glob" if any(k == "glob" for k, _ in parts) else "exact"
    if len(parts) > 1:
        kind = "phrase"
    return Term(source=raw, kind=kind, parts=parts, weight=weight, gap=gap)


def _snippet(doc: Doc, i: int, j: int, pad: int = 5) -> str:
    a = max(0, i - pad)
    b = min(len(doc.tokens), j + 1 + pad)
    s = " ".join(doc.tokens[a:b])
    return (s[:160] + "…") if len(s) > 160 else s


def _part_hit(stem_token: str, part: Part) -> bool:
    kind, value = part
    if kind == "glob":
        return stem_token.startswith(value)
    return stem_token == value


def match_term(doc: Doc, term: Term) -> TermMatch | None:
    """Ищет термин в документе и возвращает статистику совпадений."""
    head_len = len(tokenize(doc.head))

    if term.kind == "regex":
        assert term.regex is not None
        positions = [m.start() for m in term.regex.finditer(doc.text)]
        if not positions:
            return None
        snippet = doc.text[max(0, positions[0] - 40) : positions[0] + 90].strip()
        in_head = bool(term.regex.search(doc.head))
        return TermMatch(
            term=term.source,
            count=len(positions),
            weight=term.weight,
            in_head=in_head,
            positions=positions[:10],
            snippet=("…" if positions[0] > 40 else "") + snippet.replace("\n", " "),
        )

    stems = doc.stems
    n = len(stems)
    gap = term.gap
    positions: list[int] = []

    if len(term.parts) == 1:
        part = term.parts[0]
        for i, s in enumerate(stems):
            if _part_hit(s, part):
                positions.append(i)
        if not positions:
            return None
        last = positions[0]
    else:
        # Фраза: первая часть, затем остальные в пределах окна `gap`.
        k = len(term.parts)
        i = 0
        while i < n:
            if _part_hit(stems[i], term.parts[0]):
                matched = [i]
                cursor = i
                ok = True
                for part in term.parts[1:]:
                    found = -1
                    for j in range(cursor + 1, min(n, cursor + 1 + gap + 1)):
                        if _part_hit(stems[j], part):
                            found = j
                            break
                    if found < 0:
                        ok = False
                        break
                    matched.append(found)
                    cursor = found
                if ok:
                    positions.append(i)
                    i = matched[-1]
                else:
                    i += 1
            else:
                i += 1
        if not positions:
            return None
        last = positions[0]

    return TermMatch(
        term=term.source,
        count=len(positions),
        weight=term.weight,
        in_head=last < head_len,
        positions=positions[:10],
        snippet=_snippet(doc, positions[0], positions[0] + len(term.parts) - 1),
    )


def match_terms(doc: Doc, specs: list[str | dict]) -> list[TermMatch]:
    out: list[TermMatch] = []
    for spec in specs:
        term = compile_term(spec)
        m = match_term(doc, term)
        if m and m.count:
            out.append(m)
    return out


def score_terms(doc: Doc, specs: list[str | dict]) -> tuple[float, list[TermMatch]]:
    matches = match_terms(doc, specs)
    return round(sum(m.score for m in matches), 4), matches


def near(doc: Doc, specs: list[str | dict], window: int = 12) -> bool:
    """Все термины должны встретиться внутри окна из `window` токенов.

    Это самый сильный структурный признак: «инфляция» и «ключевая ставка» рядом —
    почти наверняка экономика, а не просто упоминание.
    """
    hits: list[list[int]] = []
    for spec in specs:
        term = compile_term(spec)
        m = match_term(doc, term)
        if not m or not m.positions:
            return False
        hits.append(m.positions)
    if not hits:
        return False
    first = hits[0]
    for start in first:
        if all(any(start <= p < start + window for p in hs) for hs in hits):
            return True
    return False
