"""Базовые операции над текстом: нормализация, токенизация, стемминг, предложения.

Всё на стандартной библиотеке, без внешних зависимостей. Спроектировано под русский
текст (новости), но работает и с латиницей.

Важно: нормализация НИКОГДА не трогает оригинал записи. Оригинальный текст хранится
в поле `raw`, а для матчинга строится отдельная нормализованная проекция.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# Нормализация
# --------------------------------------------------------------------------------------

_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_ANY_TAG_RE = re.compile(r"<[^>]{0,500}>")
_ENTITY_RE = re.compile(r"&(?:#[0-9]{1,7}|#x[0-9a-fA-F]{1,7}|[a-zA-Z][a-zA-Z0-9]{1,31});")
_NBSP_RE = re.compile(r"[\u00a0\u2007\u202f]")
_WS_RE = re.compile(r"[ \t\r\f\v]{2,}")
_EMPTY_LINES_RE = re.compile(r"\n{3,}")

# Типографские кавычки и дефисы -> канонические, чтобы регуляризации писались один раз.
_QUOTES = {
    "\u00ab": '"',  # «
    "\u00bb": '"',  # »
    "\u201e": '"',  # „
    "\u201c": '"',  # “
    "\u201d": '"',  # ”
    "\u2018": "'",
    "\u2019": "'",  # ’
    "\u2013": "-",  # –
    "\u2014": "-",  # —
    "\u2015": "-",  # ―
    "\u2212": "-",  # −
}

# Аббревиатуры, после которых точка НЕ заканчивает предложение.
_ABBREV = {
    "т.д", "т.п", "т.е", "т.к", "г", "гг", "ул", "пр", "пер", "обл", "р-н", "им",
    "см", "ср", "ст", "стр", "млн", "млрд", "тыс", "трлн", "руб", "долл", "евро",
    "г-н", "г-жа", "проф", "д-р", "акад", "мин", "макс", "прибл", "ок", "др", "проч",
    "no", "vs", "etc", "e.g", "i.e", "mr", "mrs", "ms", "dr", "jr", "sr", "inc", "ltd",
}

_ABBREV_RE = re.compile(
    r"(?:^|[\s(\[])([\wа-яё.\-]{1,10})\.\s", re.I
)


def strip_html(text: str) -> str:
    """Убирает script/style целиком, остальные теги — поштучно, разворачивает сущности."""
    if "<" not in text:
        return _ENTITY_RE.sub(lambda m: html.unescape(m.group(0)), text)
    text = _TAG_RE.sub(" ", text)
    text = _ANY_TAG_RE.sub(" ", text)
    return html.unescape(text)


def normalize(
    text: str,
    *,
    strip_tags: bool = True,
    collapse_ws: bool = True,
    fold_quotes: bool = True,
    keep_case: bool = True,
) -> str:
    """Каноническая проекция текста для сравнения и поиска.

    Порядок операций фиксирован — от него зависит воспроизводимость скоринга.
    """
    if text is None:
        return ""
    if strip_tags:
        text = strip_html(text)
    # NFKC приводит "ﬁ" -> "fi", полноширинные символы -> ASCII и т.п.
    text = unicodedata.normalize("NFKC", text)
    text = _NBSP_RE.sub(" ", text)
    if fold_quotes:
        for src, dst in _QUOTES.items():
            text = text.replace(src, dst)
    if collapse_ws:
        text = _WS_RE.sub(" ", text)
        text = text.replace(" \n", "\n").replace("\n ", "\n")
        text = _EMPTY_LINES_RE.sub("\n\n", text)
    if not keep_case:
        text = text.casefold()
    return text.strip()


def sentences(text: str) -> list[str]:
    """Разбивает на предложения, не ломаясь об аббревиатуры и десятичные дроби."""
    if not text:
        return []
    out: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in ".!?…":
            # «…» считаем одним знаком
            j = i
            while j + 1 < n and text[j + 1] == ".":
                j += 1
            # десятичная дробь: 3.14
            if text[i] == "." and i > 0 and text[i - 1].isdigit() and j + 1 < n and text[j + 1].isdigit():
                i = j + 1
                continue
            tail = text[j + 1 : j + 2]
            if tail and not (tail.isspace() or tail in ")]\"'»"):
                i = j + 1
                continue
            before = text[start:j + 1]
            m = None
            for m in _ABBREV_RE.finditer(before):
                pass
            if m and m.group(1).lower() in _ABBREV and len(before.split()) <= 4:
                i = j + 1
                continue
            chunk = text[start : j + 1].strip()
            if chunk:
                out.append(chunk)
            i = j + 1
            while i < n and text[i] in " \n":
                i += 1
            start = i
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


# --------------------------------------------------------------------------------------
# Токенизация
# --------------------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[0-9][0-9 .,/\-]*[0-9]|[0-9]+|[\w][\w\-']*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Слова и числа. Числа вида `1 000 000` и `12.03.2024` остаются одним токеном."""
    return _TOKEN_RE.findall(text or "")


# --------------------------------------------------------------------------------------
# Стоп-слова (для скоринга и дедупликации)
# --------------------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset(
    """
    и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
    только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если
    уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей
    может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз
    тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом
    один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец
    два об другой хоть после над больше тот через эти нас про всего них какая много
    разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой
    им более всегда конечно всю между это эта эти этот этой этого эту эти да же ли бы
    the a an and or of to in on for with by as at is are was were be been from that this
    it its not but have has had will would can could they their he she his her we our you
    """.split()
)


# --------------------------------------------------------------------------------------
# Лёгкий русский стеммер (вариация алгоритма Портера)
# --------------------------------------------------------------------------------------

def _m(stem: str, suffixes: list[str]) -> str:
    """Отрезает самый длинный подходящий суффикс. Возврат None-эквивалента не нужен."""
    for suf in suffixes:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


_PERF_GERUND = [
    "ившись", "ывшись", "ивши", "ывши", "ив", "ыв", "ивш", "ывш",
]
_ADJ = [
    "ейшими", "ыми", "ими", "ейших", "ейшим", "ыми", "ейшее", "ейшей", "ейшего",
    "ейшему", "ейшую", "ейшие", "ейше", "ейш",
    "его", "ому", "ими", "ыми", "его", "ого", "ему", "ому",
    "ее", "ие", "ые", "ое", "ей", "ий", "ый", "ой", "их", "ых", "ую", "юю",
    "ая", "яя", "ою", "ею", "ем", "им", "ым", "ом",
]
_NOUN = [
    "иями", "ями", "ами", "иям", "ям", "ием", "ем", "ам", "ом", "ах", "ях",
    "ию", "ью", "ия", "ья", "у", "ю", "я", "а", "о", "ев", "ов", "ие", "ье",
    "е", "и", "ы", "й", "ь",
]
_VERB = [
    "ейте", "уйте", "или", "ыли", "ила", "ыла", "ена", "ите", "ило", "ыло", "ено",
    "ует", "уют", "ены", "ить", "ыть", "ишь", "ую", "ю", "у", "а", "я", "и", "ы",
    "й", "л", "ем", "ей", "уй", "ил", "ыл", "им", "ым", "ен", "ят", "ыт", "ит", "ыт",
]
_SUPER = ["ейш", "ейше"]
_DERIV = ["ость", "остей", "ост"]

_VOWELS = "аеёиоуыэюя"


def stem(token: str) -> str:
    """Усекает слово до основы. Токены короче 5 символов не трогаем — иначе
    «он» и «она» схлопнутся и начнутся ложные срабатывания."""
    t = token.lower()
    if len(t) < 5 or t in STOPWORDS:
        return t
    # зона RV: всё после первой гласной
    idx = next((i for i, ch in enumerate(t) if ch in _VOWELS), -1)
    if idx < 0 or idx >= len(t) - 1:
        return t
    rv = t[idx + 1 :]
    head = t[: idx + 1]
    base = _m(rv, _PERF_GERUND)
    if base == rv:
        base = _m(rv, _ADJ)
        base = _m(base, _NOUN)
        if base == rv:
            base = _m(rv, _VERB)
    base = _m(base, _SUPER)
    base = _m(base, _DERIV)
    if len(head + base) < 4:
        # слишком агрессивное усечение — лучше вернуть более длинную форму
        return head + _m(rv, _NOUN) if len(head + _m(rv, _NOUN)) >= 4 else t
    return head + base


# --------------------------------------------------------------------------------------
# Документ: то, что реально летает по конвейеру
# --------------------------------------------------------------------------------------


@dataclass
class Doc:
    """Запись + её производные представления.

    `raw`     — оригинал (то, что печатаем в отчёт);
    `text`    — нормализованная проекция (то, по чему ищем);
    `tokens`  — токены нормализованного текста;
    `stems`   — основы токенов (для матчинга с окончанием *);
    `meta`    — произвольные поля (номер строки, заголовок и т.п.).
    """

    id: str
    raw: str
    text: str = ""
    tokens: list[str] = field(default_factory=list)
    stems: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text:
            self.text = normalize(self.raw)
        if not self.tokens:
            self.tokens = tokenize(self.text)
        if not self.stems:
            self.stems = [stem(t) for t in self.tokens]

    _head: str | None = field(default=None, repr=False, compare=False)

    @property
    def head(self) -> str:
        """Первое предложение — обычно заголовок/лид, совпадения там весят больше."""
        if self._head is None:
            sents = sentences(self.text)
            self._head = sents[0] if sents else self.text[:120]
        return self._head

    def window(self, size: int = 120) -> str:
        s = " ".join(self.text.split())
        return s if len(s) <= size else s[: size - 1].rstrip() + "…"


def lower(text: str) -> str:
    return (text or "").casefold()


def plural(n: int, forms: tuple[str, str, str]) -> str:
    """Русское склонение: plural(2, ("пост", "поста", "постов")) -> "поста"."""
    n = abs(int(n))
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        return forms[0]
    if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        return forms[1]
    return forms[2]
