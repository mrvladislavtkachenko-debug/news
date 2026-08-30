"""Парсер Markdown-экспорта Telegram-канала.

Формат, который ожидает парсер (это стандартный вывод экспортера):

    # Telegram Channel: Название

    ## Channel Information

    - **Title:** Название
    - **Username:** @channel
    - **Collected:** 2026-08-30 17:12:53 RTZ 1 (зима)
    - **Posts exported:** 3879

    ## Post 2

    **Date:** 2021-08-28 18:18:39 RTZ 1 (зима)
    **Link:** [https://t.me/channel/2](https://t.me/channel/2)
    **Views:** 4 104
    **Forwards:** 79
    **Reactions:** 🔥 30, 👍 9, ❤ 3
    **Edited:** 2022-06-06 13:21:01 RTZ 1 (зима)
    **Media:** Photo

    ### Text

    текст поста...

Парсер намеренно «всеядный»: пропущенные поля не роняют разбор, а помечаются как None.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

POST_SPLIT_RE = re.compile(r"^##\s+Post\s+(\d+)\s*$", re.M)
HEADER_RE = re.compile(r"^\s*-\s+\*\*(.+?):\*\*\s*(.*)$", re.M)
FIELD_RE = re.compile(r"^\*\*(.+?):\*\*\s*(.*)$", re.M)
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")
NUM_RE = re.compile(r"[\d][\d\s\u00a0]*")
REACTION_ITEM_RE = re.compile(r"(\S+?)\s+(\d+)")
MEDIA_NOTE_RE = re.compile(r"^\*This post contains .*\*$", re.M)


def _to_int(raw: str | None) -> int:
    if not raw:
        return 0
    digits = NUM_RE.search(raw.replace("\u00a0", " "))
    if not digits:
        return 0
    try:
        return int(digits.group(0).replace(" ", ""))
    except ValueError:
        return 0


def _parse_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    m = DATE_RE.search(raw)
    if not m:
        return None
    try:
        return datetime.fromisoformat(m.group(1).replace(" ", "T"))
    except ValueError:
        return None


def _parse_reactions(raw: str | None) -> dict[str, int]:
    """'🔥 30, 👍 9, custom_emoji:123 1' -> {'🔥': 30, '👍': 9, 'custom_emoji:123': 1}"""
    out: dict[str, int] = {}
    if not raw:
        return out
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = REACTION_ITEM_RE.match(chunk)
        if not m:
            continue
        emoji, count = m.group(1), _to_int(m.group(2))
        out[emoji] = out.get(emoji, 0) + count
    return out


@dataclass
class Post:
    number: int
    date: datetime | None = None
    edited: datetime | None = None
    link: str = ""
    views: int = 0
    forwards: int = 0
    reactions: dict[str, int] = field(default_factory=dict)
    media: str | None = None
    text: str = ""
    raw: str = ""

    # -- производные, считаются один раз ---------------------------------------
    @property
    def reactions_total(self) -> int:
        return sum(self.reactions.values())

    @property
    def engagement(self) -> float:
        """(реакции + репосты) / просмотры."""
        if not self.views:
            return 0.0
        return round((self.reactions_total + self.forwards) / self.views, 4)

    @property
    def forward_rate(self) -> float:
        return round(self.forwards / self.views, 4) if self.views else 0.0

    @property
    def is_media_only(self) -> bool:
        return not self.text.strip() or bool(MEDIA_NOTE_RE.search(self.text))

    @property
    def title(self) -> str:
        """Первая строка поста без эмодзи — используется как «название»."""
        for line in self.text.splitlines():
            line = line.strip()
            if not line or line.startswith("*This post"):
                continue
            line = _EMOJI_RE.sub("", line).strip(" -—•·")
            if len(line) >= 4:
                return line[:110]
        return f"Пост {self.number}"

    @property
    def slug(self) -> str:
        return str(self.number)


_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\u200d\u20e3\ufe0f\u2b00-\u2bff"
    "]+",
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text)


@dataclass
class Channel:
    title: str = ""
    username: str = ""
    source: str = ""
    collected: str = ""
    posts_declared: int = 0
    posts: list[Post] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # -- базовая статистика ----------------------------------------------------
    @property
    def first_date(self) -> datetime | None:
        dates = [p.date for p in self.posts if p.date]
        return min(dates) if dates else None

    @property
    def last_date(self) -> datetime | None:
        dates = [p.date for p in self.posts if p.date]
        return max(dates) if dates else None

    @property
    def span_days(self) -> float:
        if not (self.first_date and self.last_date):
            return 0.0
        return max((self.last_date - self.first_date).total_seconds() / 86400, 0.0)

    @property
    def weeks(self) -> float:
        return self.span_days / 7.0

    @property
    def posts_per_week(self) -> float:
        if self.weeks <= 0:
            return 0.0
        return round(len(self.posts) / self.weeks, 2)

    def posts_in_period(self, start: datetime, end: datetime) -> list[Post]:
        return [p for p in self.posts if p.date and start <= p.date <= end]


def _extract_text_block(body: str) -> str:
    """Возвращает текст после заголовка '### Text'.

    Если заголовка нет, значит у поста только медиа — служебные поля текстом не считаются.
    """
    m = re.search(r"^###\s+Text\s*$", body, re.M)
    if not m:
        return ""
    text = body[m.end() :]
    # обрезаем по разделителю следующего поста
    text = re.split(r"^\s*---\s*$", text, maxsplit=1, flags=re.M)[0]
    # служебную пометку экспортера убираем из полезного текста, но храним факт медиа
    text = MEDIA_NOTE_RE.sub("", text)
    return text.strip()


def _extract_media(body: str) -> str | None:
    m = re.search(r"^\*\*Media:\*\*\s*(.+)$", body, re.M)
    if m:
        return m.group(1).strip()
    note = MEDIA_NOTE_RE.search(body)
    if note:
        inner = note.group(0).strip("*").strip()
        return inner
    return None


def parse_channel(markdown: str) -> Channel:
    """Разбирает весь файл. Не бросает исключений на «кривых» постах — копит warnings."""
    channel = Channel()

    head = POST_SPLIT_RE.split(markdown, maxsplit=1)[0]
    for m in HEADER_RE.finditer(head):
        key, value = m.group(1).strip().lower(), m.group(2).strip()
        if key == "title":
            channel.title = value
        elif key == "username":
            channel.username = value.lstrip("@")
        elif key == "source":
            channel.source = value.lstrip("@")
        elif key == "collected":
            channel.collected = value
        elif key in ("posts exported", "posts"):
            channel.posts_declared = _to_int(value)
    if not channel.title:
        t = re.search(r"^#\s+Telegram Channel:\s*(.+)$", head, re.M)
        if t:
            channel.title = t.group(1).strip()

    chunks = POST_SPLIT_RE.split(markdown)[1:]
    if not chunks:
        channel.warnings.append("не найдено ни одного блока '## Post N'")
        return channel

    for i in range(0, len(chunks), 2):
        number_raw, body = chunks[i], chunks[i + 1] if i + 1 < len(chunks) else ""
        try:
            number = int(number_raw)
        except ValueError:
            channel.warnings.append(f"пропущен блок с нечисловым номером: {number_raw!r}")
            continue

        fields: dict[str, str] = {}
        for m in FIELD_RE.finditer(body):
            fields[m.group(1).strip().lower()] = m.group(2).strip()

        link = ""
        raw_link = fields.get("link", "")
        lm = LINK_RE.search(raw_link)
        link = lm.group(2).strip() if lm else raw_link.strip()

        post = Post(
            number=number,
            date=_parse_date(fields.get("date")),
            edited=_parse_date(fields.get("edited")),
            link=link,
            views=_to_int(fields.get("views")),
            forwards=_to_int(fields.get("forwards")),
            reactions=_parse_reactions(fields.get("reactions")),
            media=_extract_media(body),
            text=_extract_text_block(body),
            raw=body.strip(),
        )
        if post.date is None:
            channel.warnings.append(f"пост {number}: не распознана дата")
        channel.posts.append(post)

    channel.posts.sort(key=lambda p: (p.date or datetime.min, p.number))
    return channel


def load_channel(path: str) -> Channel:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        return parse_channel(fh.read())
