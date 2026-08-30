"""Ядро анализа: признаки поста, индексы, вердикт.

Всё считается детерминированно из текста — без внешних моделей и без сети.
Каждый индекс собирается из именованных сигналов, поэтому любое число в отчёте
можно развернуть в список постов-доказательств (см. ``ChannelAnalysis.explain``).

Шкала 0..100. Популярность (просмотры/репосты) в индексы КАЧЕСТВА не входит —
это прямое требование промта; вовлечённость участвует только в рейтинге «лучшие посты»
с малым весом и только после порога по полезности.
"""

from __future__ import annotations

import bisect
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urlparse

from . import lexicons as LX
from .dedup import DedupResult, Deduper
from .extract import extract_money, extract_percent
from .scanner import FastLexicon
from .text import Doc, plural, sentences
from .tgparser import Channel, Post, strip_emoji

# --------------------------------------------------------------------------------------
# Константы скоринга (крутить здесь, а не в логике)
# --------------------------------------------------------------------------------------

MIN_WORDS_SHORT = 40          # пост короче — «заметка», не материал
PRACTICAL_DENSITY_CAP = 35.0  # максимум баллов за плотность практических сигналов
MOTIVATION_PENALTY_CAP = 30.0
VAGUE_PENALTY_CAP = 15.0
AD_USEFULNESS_FACTOR = 0.65   # штраф за то, что полезность спрятана в рекламе
CLAIM_PENALTY_CAP = 25.0
MANIP_PENALTY_CAP = 25.0

BEST_POST_MIN_SCORE = 55.0
PRACTICE_MIN_USEFULNESS = 60.0

VERDICT_GREEN = (70.0, 62.0, 58.0, 0.20)   # usefulness, expertise, trust, ad_share
VERDICT_YELLOW = 48.0
VERDICT_ORANGE = 32.0


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return round(max(lo, min(hi, value)), 1)


def _per100(hits: int, words: int) -> float:
    if words <= 0:
        return 0.0
    return hits / (words / 100.0)


# --------------------------------------------------------------------------------------
# Структурные признаки поста
# --------------------------------------------------------------------------------------

_NUM_LINE = re.compile(r"^\s*\d{1,3}\s*[.)]\s")
_QUESTION_WORDS = re.compile(
    r"\b(как|что|почему|зачем|кто|где|когда|сколько|какой|какая|какие|чем|куда|откуда)\b", re.I
)
_IMPERATIVE_END = ("й", "и", "те", "ись", "итесь")
_URL_RE = re.compile(r"https?://[^\s\)\]]+", re.I)
_FIRST_PERSON = re.compile(r"\b(я|мне|меня|мой|моя|моё|мое|мы|нас|нам|наш|наша|наше)\b", re.I)


@dataclass
class Structure:
    words: int = 0
    chars: int = 0
    sentences: int = 0
    paragraphs: int = 0
    numbered_lines: int = 0
    list_lines: int = 0
    subheadings: int = 0
    questions: int = 0
    imperatives: int = 0
    emoji: int = 0
    links: list[str] = field(default_factory=list)
    money: list[dict] = field(default_factory=list)
    percent: list[dict] = field(default_factory=list)
    first_person: int = 0

    @property
    def first_person_density(self) -> float:
        return _per100(self.first_person, self.words)


def structure_of(doc: Doc) -> Structure:
    st = Structure()
    st.words = len(doc.tokens)
    st.chars = len(doc.text)
    st.sentences = len(sentences(doc.text))
    st.links = _URL_RE.findall(doc.text)
    st.money = extract_money(doc)
    st.percent = extract_percent(doc)

    lines = [ln.strip() for ln in doc.text.splitlines() if ln.strip()]
    st.paragraphs = max(1, len([ln for ln in doc.text.split("\n\n") if ln.strip()]))

    short_run = 0
    for i, ln in enumerate(lines):
        if _NUM_LINE.match(ln):
            st.numbered_lines += 1
        if len(ln) <= 90:
            short_run += 1
        else:
            short_run = 0
        if short_run >= 3:
            st.list_lines += 1
        # подзаголовок: короткая строка без точки, за которой идёт развёрнутый абзац.
        # В экспорте Telegram маркеры списков теряются, поэтому ориентируемся на форму.
        plain = strip_emoji(ln).strip()
        if (
            8 <= len(plain) <= 70
            and not plain.endswith((".", "!", "?", ":", ","))
            and not plain.startswith(("→", "http"))
            and i + 1 < len(lines)
            and (len(lines[i + 1]) >= 20 or bool(_NUM_LINE.match(lines[i + 1])))
        ):
            st.subheadings += 1

    st.questions = len(re.findall(r"\?", doc.text)) + len(_QUESTION_WORDS.findall(doc.text))
    st.emoji = sum(1 for ch in doc.text if ord(ch) > 0x2600)
    st.first_person = len(_FIRST_PERSON.findall(doc.text))

    for tok in doc.tokens:
        low = tok.lower()
        if len(low) >= 4 and low.endswith(_IMPERATIVE_END) and low not in ("если", "или", "они", "все", "всю"):
            st.imperatives += 1
    return st


# --------------------------------------------------------------------------------------
# Анализ одного поста
# --------------------------------------------------------------------------------------


@dataclass
class PostAnalysis:
    post: Post
    doc: Doc
    struct: Structure
    hits: dict[str, dict[str, int]] = field(default_factory=dict)

    category: str = "прочее"
    category_scores: dict[str, float] = field(default_factory=dict)
    topics: list[str] = field(default_factory=list)
    audiences: list[str] = field(default_factory=list)

    usefulness: float = 0.0
    originality: float = 0.0
    expertise: float = 0.0
    trust: float = 0.0
    depth: float = 0.0

    claims: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    limits: list[str] = field(default_factory=list)
    flags: dict[str, bool] = field(default_factory=dict)
    composite: float = 0.0

    # -- дубликаты (MinHash + LSH, считаются до скоринга) -----------------------
    is_duplicate: bool = False
    duplicate_of: str | None = None
    dup_similarity: float = 0.0

    # -- ссылки -----------------------------------------------------------------
    @property
    def link(self) -> str:
        return self.post.link or f"https://t.me/{self.post.number}"

    @property
    def title(self) -> str:
        return self.post.title

    def group_total(self, group: str) -> int:
        return sum(self.hits.get(group, {}).values())

    def group_terms(self, group: str, limit: int = 4) -> list[str]:
        items = sorted(self.hits.get(group, {}).items(), key=lambda kv: -kv[1])
        return [k for k, _ in items[:limit]]


@dataclass
class AnalyzerConfig:
    own_domains: list[str] = field(default_factory=list)
    channel_handle: str = ""
    min_words_short: int = MIN_WORDS_SHORT


class ChannelAnalyzer:
    """Считает признаки по всем постам и агрегирует их до уровня канала."""

    def __init__(self, config: AnalyzerConfig | None = None):
        self.config = config or AnalyzerConfig()
        self.lexicon = FastLexicon(
            {
                **{f"cat:{k}": v for k, v in LX.CATEGORY_TERMS.items()},
                **{f"topic:{k}": v for k, v in LX.TOPICS.items()},
                **{f"aud:{k}": v for k, v in LX.AUDIENCE.items()},
                **{f"flag:{k}": v for k, v in LX.RED_FLAGS.items()},
                "practical": LX.PRACTICAL_TERMS,
                "motivation": LX.MOTIVATION_TERMS,
                "vague": LX.VAGUE_TERMS,
                "expert": LX.EXPERT_TERMS,
                "authority": LX.AUTHORITY_APPEAL,
                "original": LX.ORIGINAL_TERMS,
                "derivative": LX.DERIVATIVE_TERMS,
                "claims": LX.CLAIM_TERMS,
                "evidence": LX.EVIDENCE_TERMS,
                "transparency": LX.TRANSPARENCY_TERMS,
                "manipulation": LX.MANIPULATION_TERMS,
            }
        )

    # ---------------------------------------------------------------- ссылки
    def _own_domain_ratio(self, channel: Channel) -> dict[str, float]:
        counter: Counter[str] = Counter()
        total = 0
        for post in channel.posts:
            for url in _URL_RE.findall(post.text):
                host = (urlparse(url).netloc or "").lower().removeprefix("www.")
                if host:
                    counter[host] += 1
                    total += 1
        if not total:
            return {}
        return {host: cnt / total for host, cnt in counter.items()}

    def _is_own_link(self, url: str, host_share: dict[str, float]) -> bool:
        host = (urlparse(url).netloc or "").lower().removeprefix("www.")
        if not host:
            return False
        handle = self.config.channel_handle
        if handle and handle.lower() in host:
            return True
        if f"t.me/{handle}".lower() in url.lower() and handle:
            return True
        for dom in self.config.own_domains:
            if host == dom or host.endswith("." + dom) or dom in host:
                return True
        # регулярно повторяющаяся собственная площадка
        return host_share.get(host, 0.0) >= 0.08

    # ---------------------------------------------------------------- категории
    def _category(self, pa: PostAnalysis) -> tuple[str, dict[str, float]]:
        scores: dict[str, float] = {}
        words = max(pa.struct.words, 1)
        for cat, terms in LX.CATEGORY_TERMS.items():
            hits = sum(pa.hits.get(f"cat:{cat}", {}).values())
            scores[cat] = hits * (100.0 / words) * 8.0
        # реклама по одному слову «курс» не определяется — понижаем вклад лексики
        scores["реклама/продажа"] *= 0.6
        # первое лицо есть почти в каждом посте — сам по себе это не «личный опыт»
        scores["личный опыт"] *= 0.5

        s = pa.struct
        st = LX.CATEGORY_STRUCT
        if s.numbered_lines >= 3 or s.list_lines >= 4 or s.subheadings >= 3:
            scores["гайд/инструкция"] += 8.0
        elif s.numbered_lines >= 2 or s.list_lines >= 3:
            scores["гайд/инструкция"] += st["гайд/инструкция"]["has_numbered_list"]
        if s.imperatives >= 3:
            scores["гайд/инструкция"] += st["гайд/инструкция"]["has_imperatives"]
        if s.words > 300:
            scores["гайд/инструкция"] += st["гайд/инструкция"]["long"]
        if (s.money or s.percent) and pa.group_total("cat:кейс/разбор") >= 2:
            scores["кейс/разбор"] += 6.0
        elif s.money or s.percent:
            scores["кейс/разбор"] += st["кейс/разбор"]["has_money"]
        if s.first_person_density >= 4:
            scores["личный опыт"] += st["личный опыт"]["first_person_dense"]
        if s.questions >= 2 and s.words < 250:
            scores["инсайт/мнение"] += st["инсайт/мнение"]["has_question"]
        has_cta = bool(
            re.search(
                r"\b(подпишись|подписывайся|подпишитесь|переходи|жми|забирай|регистрируй|успей|го)\b|по ссылке|читать тут",
                pa.doc.text,
                re.I,
            )
        )
        own_links = sum(1 for u in s.links if pa.flags.get("own_link_" + u))
        discount = bool(re.search(r"\b(скидк|промокод|распродаж|со скидкой)\w*", pa.doc.text, re.I))
        if own_links:
            scores["реклама/продажа"] += st["реклама/продажа"]["own_links"]
        if has_cta:
            scores["реклама/продажа"] += st["реклама/продажа"]["has_cta"]
        if discount and (s.money or re.search(r"\d+\s*%", pa.doc.text)):
            scores["реклама/продажа"] += st["реклама/продажа"]["has_price"]

        ranked_cats = sorted(scores, key=lambda k: -scores[k])
        best = ranked_cats[0]
        # короткий пост со скидкой и призывом — это продажа, даже если лексики мало
        if discount and (has_cta or own_links) and s.words < 200:
            best = "реклама/продажа"
        # гейт: рекламу ставим только при структурном подтверждении, иначе берём следующую категорию
        if best == "реклама/продажа" and not (own_links or has_cta or discount):
            for alt in ranked_cats[1:]:
                if scores[alt] >= 1.0:
                    best = alt
                    break
        if scores[best] < 1.0:
            return "прочее", scores
        return best, {k: round(v, 2) for k, v in scores.items()}

    # ---------------------------------------------------------------- индексы
    def _usefulness(self, pa: PostAnalysis) -> float:
        s, h = pa.struct, pa.hits
        score = 0.0
        score += min(_per100(sum(h.get("practical", {}).values()), s.words) * 7.0, PRACTICAL_DENSITY_CAP)
        score += min(s.subheadings * 4.0, 12.0)
        score += min(s.numbered_lines * 3.0, 10.0)
        score += min(s.list_lines * 0.8, 6.0)
        score += min(len(re.findall(r"\d[\d\s]*", pa.doc.text)) * 0.5, 8.0)
        score += min(len(s.money) * 2.0 + len(s.percent) * 2.0, 8.0)
        score += min(len(s.links) * 1.5, 5.0)
        score += min(math.log1p(s.words) * 5.0, 22.0)
        score -= min(_per100(sum(h.get("motivation", {}).values()), s.words) * 12.0, MOTIVATION_PENALTY_CAP)
        score -= min(_per100(sum(h.get("vague", {}).values()), s.words) * 3.0, VAGUE_PENALTY_CAP)
        if s.words < self.config.min_words_short:
            score *= 0.55
        if pa.category == "реклама/продажа":
            score *= AD_USEFULNESS_FACTOR
        return _clip(score)

    def _originality(self, pa: PostAnalysis) -> float:
        s, h = pa.struct, pa.hits
        score = 45.0
        score += min(sum(h.get("original", {}).values()) * 4.0, 25.0)
        score += min(s.first_person_density * 2.2, 18.0)
        score -= min(sum(h.get("derivative", {}).values()) * 5.0, 35.0)
        if pa.category == "новость/обзор":
            score = min(score, 42.0)
        if pa.category == "реклама/продажа":
            score = min(score, 50.0)
        if s.words > 500 and s.first_person_density >= 3:
            score += 8.0
        if pa.is_duplicate:
            # пост повторяет уже опубликованный: оригинальности у него быть не может
            score = min(score, 22.0 if pa.dup_similarity >= 1.0 else 34.0)
        return _clip(score)

    def _expertise(self, pa: PostAnalysis) -> float:
        s, h = pa.struct, pa.hits
        score = 30.0
        causal = sum(h.get("expert", {}).values())
        score += min(_per100(causal, s.words) * 5.0, 28.0)
        score += min(len(s.money) * 2.5 + len(s.percent) * 2.5, 14.0)
        score += min(sum(h.get("transparency", {}).values()) * 4.0, 18.0)
        score += min(math.log1p(s.words) * 2.0, 10.0)
        score -= min(sum(h.get("authority", {}).values()) * 4.0, 15.0)
        claims = sum(h.get("claims", {}).values())
        evidence = sum(h.get("evidence", {}).values())
        if claims and claims > evidence:
            score -= min((claims - evidence) * 4.0, CLAIM_PENALTY_CAP)
        if s.words < self.config.min_words_short:
            score *= 0.7
        return _clip(score)

    def _trust(self, pa: PostAnalysis) -> float:
        s, h = pa.struct, pa.hits
        score = 55.0
        score += min(sum(h.get("evidence", {}).values()) * 4.0, 20.0)
        score += min(sum(h.get("transparency", {}).values()) * 4.5, 22.0)
        ext_links = sum(1 for u in s.links if not pa.flags.get("own_link_" + u))
        score += min(ext_links * 2.5, 10.0)
        score -= min(sum(h.get("claims", {}).values()) * 4.0, CLAIM_PENALTY_CAP + 5.0)
        score -= min(sum(h.get("manipulation", {}).values()) * 6.0, MANIP_PENALTY_CAP)
        if s.words < self.config.min_words_short:
            score -= 5.0
        return _clip(score)

    def _depth(self, pa: PostAnalysis) -> float:
        s = pa.struct
        score = 0.0
        score += min(math.log1p(s.words) * 12.0, 45.0)
        score += min(s.sentences * 0.6, 20.0)
        score += min(pa.group_total("expert") * 2.0, 20.0)
        score += min(len(re.findall(r"\bнапример\b", pa.doc.text, re.I)) * 5.0, 15.0)
        return _clip(score)

    # ---------------------------------------------------------------- публичный API
    def analyze_post(
        self,
        post: Post,
        host_share: dict[str, float],
        dup: DedupResult | None = None,
    ) -> PostAnalysis:
        doc = Doc(str(post.number), post.text)
        struct = structure_of(doc)
        hits_obj = self.lexicon.scan(doc)
        pa = PostAnalysis(post=post, doc=doc, struct=struct, hits=hits_obj.counts)
        if dup is not None and dup.is_duplicate:
            pa.is_duplicate = True
            pa.duplicate_of = dup.duplicate_of
            pa.dup_similarity = dup.similarity

        for url in struct.links:
            pa.flags["own_link_" + url] = self._is_own_link(url, host_share)

        pa.topics = [
            g[6:]
            for g in sorted(pa.hits, key=lambda g: -sum(pa.hits[g].values()))
            if g.startswith("topic:") and sum(pa.hits[g].values()) >= 2
        ][:5]
        pa.audiences = [g[4:] for g in pa.hits if g.startswith("aud:") and sum(pa.hits[g].values()) >= 2]

        pa.claims = pa.group_terms("claims", 5)
        pa.evidence = pa.group_terms("evidence", 5)
        pa.limits = pa.group_terms("transparency", 5)

        pa.category, pa.category_scores = self._category(pa)
        if post.is_media_only or struct.words < 15:
            pa.category = "прочее"
        pa.flags["is_ad"] = pa.category == "реклама/продажа"
        pa.flags["is_news"] = pa.category == "новость/обзор"
        pa.flags["is_personal"] = pa.category == "личный опыт"
        pa.flags["is_short"] = struct.words < self.config.min_words_short
        pa.flags["has_case"] = bool(struct.money or struct.percent) and pa.group_total("cat:кейс/разбор") >= 2
        pa.flags["media_only"] = post.is_media_only

        pa.usefulness = self._usefulness(pa)
        pa.originality = self._originality(pa)
        pa.expertise = self._expertise(pa)
        pa.trust = self._trust(pa)
        pa.depth = self._depth(pa)
        return pa

    def run(self, channel: Channel) -> "ChannelAnalysis":
        cfg = self.config
        if not cfg.channel_handle:
            cfg.channel_handle = channel.username or channel.source
        host_share = self._own_domain_ratio(channel)

        # дедупликация считается до скоринга: оригинальность зависит от её результата
        dup_map = (
            Deduper().run([Doc(str(p.number), p.text) for p in channel.posts])
            if len(channel.posts) > 1
            else {}
        )
        posts = [
            self.analyze_post(p, host_share, dup_map.get(str(p.number)))
            for p in channel.posts
        ]
        return ChannelAnalysis(channel=channel, posts=posts, host_share=host_share).compute()


# --------------------------------------------------------------------------------------
# Агрегация по каналу
# --------------------------------------------------------------------------------------


def _weighted_mean(pairs: list[tuple[float, float]]) -> float:
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return 0.0
    return sum(v * w for v, w in pairs) / total_w


def _spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(sum((v - mx) ** 2 for v in rx) * sum((v - my) ** 2 for v in ry))
    return round(num / den, 3) if den else 0.0


SUBJECTS = [
    "прогрев", "контент-план", "телефон", "рассрочк", "книг", "формул", "шаблон",
    "таймер", "хэштег", "бесплатн продвижен", "перфекционизм", "дизайн",
]
POSITIVE = ["работает", "офигенн", "круто", "полезно", "приносит", "эффективно", "надо", "стоит", "кайф"]
NEGATIVE = ["не работает", "плохо", "хрень", "бесполезно", "манипуляц", "зло", "фигня", "не надо", "не ведись", "развод"]


@dataclass
class ChannelAnalysis:
    channel: Channel
    posts: list[PostAnalysis]
    host_share: dict[str, float] = field(default_factory=dict)

    indices: dict[str, float] = field(default_factory=dict)
    category_share: dict[str, float] = field(default_factory=dict)
    ad_share: float = 0.0
    forward_rate: float = 0.0
    engagement: float = 0.0
    topics_by_freq: list[tuple[str, int]] = field(default_factory=list)
    topics_by_popularity: list[tuple[str, float]] = field(default_factory=list)
    best_posts: list[PostAnalysis] = field(default_factory=list)
    practices: list[PostAnalysis] = field(default_factory=list)
    red_flags: dict[str, dict] = field(default_factory=dict)
    trend: dict = field(default_factory=dict)
    popularity_vs_quality: dict = field(default_factory=dict)
    author: dict = field(default_factory=dict)
    audiences: list[tuple[str, int]] = field(default_factory=list)
    verdict: dict = field(default_factory=dict)
    claim_density: float = 0.0
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def compute(self) -> "ChannelAnalysis":
        ch, posts = self.channel, self.posts
        if not posts:
            self.verdict = {"label": "🔴 МАЛО ПОЛЕЗНОГО КОНТЕНТА", "reason": "в файле не найдено постов"}
            return self

        weights = [max(pa.struct.words, 20) for pa in posts]

        self.indices = {
            "usefulness": round(_weighted_mean([(pa.usefulness, w) for pa, w in zip(posts, weights)]), 1),
            "expertise": round(_weighted_mean([(pa.expertise, w) for pa, w in zip(posts, weights)]), 1),
            "originality": round(_weighted_mean([(pa.originality, w) for pa, w in zip(posts, weights)]), 1),
            "trust": round(_weighted_mean([(pa.trust, w) for pa, w in zip(posts, weights)]), 1),
            "depth": round(_weighted_mean([(pa.depth, w) for pa, w in zip(posts, weights)]), 1),
        }

        # ---- категории и реклама
        cat_counter: Counter[str] = Counter(pa.category for pa in posts)
        self.category_share = {
            cat: round(100.0 * cnt / len(posts), 1)
            for cat, cnt in cat_counter.most_common()
        }
        ads = [pa for pa in posts if pa.flags.get("is_ad")]
        self.ad_share = round(100.0 * len(ads) / len(posts), 1)

        total_views = sum(pa.post.views for pa in posts) or 1
        self.forward_rate = round(100.0 * sum(pa.post.forwards for pa in posts) / total_views, 2)
        self.engagement = round(
            100.0 * sum(pa.post.reactions_total + pa.post.forwards for pa in posts) / total_views, 2
        )

        # ---- темы: частотность и популярность
        freq: Counter[str] = Counter()
        pop: dict[str, list[float]] = {}
        for pa in posts:
            for topic in pa.topics:
                freq[topic] += 1
                pop.setdefault(topic, []).append(float(pa.post.views))
        self.topics_by_freq = freq.most_common(6)
        self.topics_by_popularity = sorted(
            ((t, round(sum(v) / len(v))) for t, v in pop.items() if len(v) >= 3),
            key=lambda kv: -kv[1],
        )[:6]

        # ---- аудитории
        aud: Counter[str] = Counter()
        for pa in posts:
            for a in pa.audiences:
                aud[a] += 1
        self.audiences = [(a, c) for a, c in aud.most_common(5) if c >= max(3, len(posts) * 0.03)]

        # ---- штраф за маркетинговые заявления (уровень канала)
        claim_posts = [pa for pa in posts if pa.claims]
        claim_density = len(claim_posts) / len(posts)
        manip_posts = [pa for pa in posts if pa.group_total("manipulation")]
        penalty = min(claim_density * 30.0, 18.0) + min(len(manip_posts) / len(posts) * 20.0, 8.0)
        self.indices["trust"] = round(max(0.0, self.indices["trust"] - penalty), 1)
        self.indices["expertise"] = round(max(0.0, self.indices["expertise"] - penalty * 0.6), 1)
        if self.ad_share > 35:
            self.indices["usefulness"] = round(
                max(0.0, self.indices["usefulness"] - (self.ad_share - 35) * 0.4), 1
            )
        self.claim_density = round(100.0 * claim_density, 1)

        # ---- лучшие посты и практики
        eng_values = sorted(pa.post.engagement for pa in posts)

        def eng_norm(pa: PostAnalysis) -> float:
            if not eng_values:
                return 0.0
            return 100.0 * bisect.bisect_left(eng_values, pa.post.engagement) / max(len(eng_values) - 1, 1)

        ranked = sorted(
            posts,
            key=lambda pa: -(
                0.5 * pa.usefulness + 0.2 * pa.expertise + 0.2 * pa.originality + 0.1 * eng_norm(pa)
            ),
        )
        for pa in ranked:
            pa.composite = round(
                0.5 * pa.usefulness + 0.2 * pa.expertise + 0.2 * pa.originality + 0.1 * eng_norm(pa), 1
            )
        self.best_posts = [pa for pa in ranked if pa.composite >= BEST_POST_MIN_SCORE][:5]

        practices = [
            pa
            for pa in ranked
            if pa.usefulness >= PRACTICE_MIN_USEFULNESS
            and pa.category in ("гайд/инструкция", "кейс/разбор")
            and not pa.flags.get("is_ad")
            and not pa.flags.get("is_short")
        ]
        self.practices = practices[:5]

        # ---- популярность против пользы
        views = [float(pa.post.views) for pa in posts]
        usef = [pa.usefulness for pa in posts]
        corr = _spearman(views, usef)
        top_views = {pa.post.number for pa in sorted(posts, key=lambda p: -p.post.views)[:10]}
        top_useful = {pa.post.number for pa in sorted(posts, key=lambda p: -p.usefulness)[:10]}
        overlap = len(top_views & top_useful)
        self.popularity_vs_quality = {
            "spearman": corr,
            "overlap_top10": overlap,
            "interpretation": (
                "популярность почти не связана с полезностью"
                if abs(corr) < 0.2
                else (
                    "умеренная связь популярности и полезности"
                    if abs(corr) < 0.5
                    else "популярность и полезность заметно связаны"
                )
            ),
        }

        # ---- тренд
        dated = [pa for pa in posts if pa.post.date]
        if len(dated) >= 6:
            mid = dated[0].post.date + (dated[-1].post.date - dated[0].post.date) / 2
            first = [pa for pa in dated if pa.post.date <= mid]
            second = [pa for pa in dated if pa.post.date > mid]
            f_weeks = max((mid - dated[0].post.date).days / 7, 1)
            s_weeks = max((dated[-1].post.date - mid).days / 7, 1)
            f_rate, s_rate = len(first) / f_weeks, len(second) / s_weeks
            f_views = sum(p.post.views for p in first) / max(len(first), 1)
            s_views = sum(p.post.views for p in second) / max(len(second), 1)
            ratio = s_rate / f_rate if f_rate else 1.0
            label = "📈 растущий" if ratio > 1.25 else ("📉 снижающийся" if ratio < 0.75 else "➡️ стабильный")
            self.trend = {
                "label": label,
                "first_half_per_week": round(f_rate, 2),
                "second_half_per_week": round(s_rate, 2),
                "first_half_avg_views": round(f_views),
                "second_half_avg_views": round(s_views),
                "note": "рассчитано по частоте публикаций и средним просмотрам двух половин выборки",
            }
        else:
            self.trend = {"label": "❓ недостаточно данных", "note": "меньше 6 датированных постов"}

        # ---- красные флаги
        self.red_flags = {}
        for flag_name in LX.RED_FLAGS:
            hits = [pa for pa in posts if pa.hits.get(f"flag:{flag_name}")]
            if not hits:
                continue
            share = len(hits) / len(posts)
            # на малых выборках одиночные попадания не показываем — иначе отчёт шумит
            if len(hits) < 3 or share < 0.05:
                continue
            self.red_flags[flag_name] = {
                "count": len(hits),
                "share": round(100.0 * share, 1),
                "examples": [
                    {"id": pa.post.number, "link": pa.link, "terms": pa.group_terms(f"flag:{flag_name}", 3)}
                    for pa in hits[:3]
                ],
            }

        # «постоянные продажи» считаем по назначению поста, а не по слову «курс» в тексте
        self.red_flags["постоянные продажи"] = {
            "count": len(ads),
            "share": self.ad_share,
            "examples": [
                {"id": pa.post.number, "link": pa.link, "terms": ["продажа собственного продукта"]}
                for pa in ads[:3]
            ],
        } if ads else {"count": 0, "share": 0.0, "examples": []}
        if not ads:
            self.red_flags.pop("постоянные продажи")

        # ---- противоречия (12-й красный флаг из промта)
        pairs = self._find_contradictions()
        self.contradictions = [text for text, _ in pairs]
        if self.contradictions:
            ids = sorted({i for _, post_ids in pairs for i in post_ids})
            self.red_flags["противоречия между постами"] = {
                "count": len(ids),
                "share": round(100.0 * len(ids) / max(len(posts), 1), 1),
                "examples": [
                    {"id": i, "link": "", "terms": ["позиция по одной теме меняется от поста к посту"]}
                    for i in ids[:3]
                ],
            }

        # ---- автор
        self.author = self._author_block(claim_posts)

        # ---- факты и заявления
        self.facts = self._facts_block()
        self.claims = self._claims_block(claim_posts)

        # ---- сильные/слабые стороны
        self.strengths, self.weaknesses = self._sides()

        # ---- вердикт
        self.verdict = self._verdict()
        return self

    # ------------------------------------------------------------------
    def _find_contradictions(self) -> list[tuple[str, list[int]]]:
        """Возвращает (описание, номера постов) для каждой найденной пары противоречий."""
        out: list[tuple[str, list[int]]] = []
        for subj in SUBJECTS:
            pos, neg = [], []
            for pa in self.posts:
                low = pa.doc.text.lower()
                if subj not in low:
                    continue
                window_hits = [low[max(0, m - 90) : m + 90] for m in [low.find(subj)]]
                text = " ".join(window_hits)
                if any(w in text for w in POSITIVE) and not any(w in text for w in NEGATIVE):
                    pos.append(pa)
                elif any(w in text for w in NEGATIVE):
                    neg.append(pa)
            if pos and neg:
                involved = neg[:2] + pos[:2]
                text = (
                    f"«{subj}»: посты {', '.join(str(p.post.number) for p in neg[:2])} оценивают негативно, "
                    f"а {', '.join(str(p.post.number) for p in pos[:2])} — позитивно"
                )
                out.append((text, [p.post.number for p in involved]))
        return out[:5]

    def _author_block(self, claim_posts: list[PostAnalysis]) -> dict:
        ch = self.channel
        first_person_posts = [pa for pa in self.posts if pa.struct.first_person >= 3]
        selfdesc: Counter[str] = Counter()
        patterns = {
            "руководитель агентства": r"\b(руковожу|основал|запустил агентство|мо[её] агентств|наше агентство)\b",
            "редактор": r"\b(редактор|главред|редактур)\w*",
            "копирайтер": r"\b(копирайтер|копирайтинг)\w*",
            "маркетолог": r"\b(маркетолог|контент-маркетинг)\w*",
            "предприниматель": r"\b(предпринимател|бизнес)\w*",
        }
        for pa in self.posts:
            for name, rx in patterns.items():
                if re.search(rx, pa.doc.text, re.I):
                    selfdesc[name] += 1
        positioning = ", ".join(name for name, _ in selfdesc.most_common(3)) or "нет данных"

        clients = Counter()
        canon = {
            "яндекс": "Яндекс", "билайн": "Билайн", "авито": "Авито", "альфа": "Альфа-Банк",
            "сбер": "Сбер", "тинькофф": "Тинькофф", "вконтакте": "ВКонтакте", "дзен": "Дзен",
        }
        for pa in self.posts:
            found: set[str] = set()
            for m in re.finditer(
                r"\b(Яндекс\w*|Билайн\w*|Авито\w*|Альфа-Банк\w*|Сбер\w*|Тинькофф\w*|ВКонтакте|Дзен\w*)\b",
                pa.doc.text,
                re.I,
            ):
                word = m.group(1).lower()
                for key, name in canon.items():
                    if word.startswith(key):
                        found.add(name)
                        break
            # sorted(): порядок обхода set зависит от PYTHONHASHSEED,
            # а Counter.most_common() сохраняет порядок вставки для равных счётчиков
            for name in sorted(found):
                clients[name] += 1
        return {
            "name": ch.title or ch.username or "нет данных",
            "handle": f"@{ch.username}" if ch.username else "нет данных",
            "subscribers": "нет данных",
            "positioning": positioning,
            "personal_brand": round(100.0 * len(first_person_posts) / max(len(self.posts), 1), 1),
            "clients_mentioned": [
                f"{name} ({cnt})"
                for name, cnt in sorted(clients.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
            ],
        }

    def _facts_block(self) -> list[str]:
        total = len(self.posts)
        with_numbers = [pa for pa in self.posts if pa.struct.money or pa.struct.percent]
        with_steps = [pa for pa in self.posts if pa.struct.numbered_lines >= 3 or pa.struct.list_lines >= 4]
        with_links = [pa for pa in self.posts if pa.struct.links]
        with_limits = [pa for pa in self.posts if pa.limits]
        def loc(n: int) -> str:
            return f"{n} {plural(n, ('посте', 'постах', 'постах'))}"

        return [
            f"в выборке {total} {plural(total, ('пост', 'поста', 'постов'))}; конкретные числа (деньги/проценты) "
            f"встречаются в {loc(len(with_numbers))}",
            f"структурированные инструкции (нумерованные списки/перечни) — в {loc(len(with_steps))}",
            f"ссылки на внешние материалы приведены в {loc(len(with_links))}",
            f"оговорки и признание ограничений («не факт», «не сработало», «риск») — в {loc(len(with_limits))}",
        ]

    def _claims_block(self, claim_posts: list[PostAnalysis]) -> list[str]:
        out: list[str] = []
        money_claims = [
            pa
            for pa in self.posts
            if pa.struct.money and re.search(r"\b(заработал|выручк|оборот|прибыл|сделали|получили)\w*", pa.doc.text, re.I)
        ]
        if money_claims:
            refs = ", ".join(str(pa.post.number) for pa in money_claims[:5])
            out.append(
                f"автор заявляет финансовые результаты (выручка/прибыль/оборот) в постах {refs}; "
                "подтверждающих документов (отчёты, скриншоты выгрузок) в выборке не обнаружено"
            )
        if claim_posts:
            out.append(
                f"маркетинговые формулировки («гарантированно», «успей», «100%», «скидка») встречаются в "
                f"{len(claim_posts)} {plural(len(claim_posts), ('посте', 'постах', 'постах'))} из {len(self.posts)}"
            )
        out.append(
            "количество подписчиков, верификация и упоминания выступлений в файле не приведены — "
            "проверить их по данным нельзя"
        )
        return out

    def _sides(self) -> tuple[list[str], list[str]]:
        total = len(self.posts)
        idx = self.indices
        with_numbers = sum(1 for pa in self.posts if pa.struct.money or pa.struct.percent)
        with_steps = sum(1 for pa in self.posts if pa.struct.numbered_lines >= 3 or pa.struct.list_lines >= 4)
        with_limits = sum(1 for pa in self.posts if pa.limits)
        shorts = sum(1 for pa in self.posts if pa.flags.get("is_short"))
        strengths = [
            f"{round(100 * with_steps / total)}% публикаций ({with_steps} из {total}) содержат пошаговые инструкции "
            f"или структурированные перечни, а не только общие рассуждения.",
            f"Конкретика: числа и денежные суммы приведены в {with_numbers} "
            f"{plural(with_numbers, ('посте', 'постах', 'постах'))}, что позволяет проверять часть утверждений.",
            f"Индекс полезности {idx['usefulness']}/100 при доле рекламного контента {self.ad_share}% — практическая часть ленты преобладает над продающей.",
        ]
        if with_limits >= 5:
            strengths.append(
                f"Автор фиксирует ограничения и неудачи в {with_limits} "
            f"{plural(with_limits, ('посте', 'постах', 'постах'))} — это повышает доверие."
            )
        weaknesses = [
            f"{round(100 * shorts / total)}% публикаций ({shorts} из {total}) короче {MIN_WORDS_SHORT} слов — "
            f"это заметки без практической ценности.",
            f"Аргументация опирается на личный опыт: индекс экспертности {idx['expertise']}/100, независимых источников и данных в выборке мало.",
            f"Реклама и продажа собственных продуктов занимают {self.ad_share}% ленты.",
        ]
        if self.claim_density > 15:
            weaknesses.append(
                f"Маркетинговые заявления встречаются в {self.claim_density}% постов, при этом подтверждения к ним не приведены."
            )
        if self.contradictions:
            weaknesses.append(
                f"Найдены расхождения в позициях по {len(self.contradictions)} темам (см. раздел противоречий)."
            )
        dups = [pa for pa in self.posts if pa.is_duplicate]
        if dups:
            n = len(dups)
            weaknesses.append(
                f"{n} {plural(n, ('пост', 'поста', 'постов'))} практически "
                f"{'повторяет ранее опубликованный' if n == 1 else 'повторяют ранее опубликованные'}"
                f" — оригинальность этой части ленты нулевая."
            )
        return strengths[:5], weaknesses[:5]

    def _verdict(self) -> dict:
        idx = self.indices
        u, e, t = idx["usefulness"], idx["expertise"], idx["trust"]
        g_u, g_e, g_t, g_ad = VERDICT_GREEN
        if u >= g_u and e >= g_e and t >= g_t and self.ad_share <= g_ad * 100:
            label = "🟢 СТОИТ ИЗУЧАТЬ"
        elif u >= VERDICT_YELLOW:
            label = "🟡 ЕСТЬ ПОЛЕЗНЫЕ ИДЕИ"
        elif u >= VERDICT_ORANGE:
            label = "🟠 ПОЛЕЗНО, НО ПЕРЕОЦЕНЕНО"
        else:
            label = "🔴 МАЛО ПОЛЕЗНОГО КОНТЕНТА"
        return {
            "label": label,
            "usefulness": u,
            "expertise": e,
            "trust": t,
            "ad_share": self.ad_share,
        }

    # ------------------------------------------------------------------
    def explain(self, post_id: str | int) -> dict:
        """Развёртка одного поста: почему такие оценки и какие термины сработали."""
        target = str(post_id)
        for pa in self.posts:
            if pa.post.slug == target:
                return {
                    "id": pa.post.number,
                    "link": pa.link,
                    "category": pa.category,
                    "category_scores": pa.category_scores,
                    "scores": {
                        "usefulness": pa.usefulness,
                        "originality": pa.originality,
                        "expertise": pa.expertise,
                        "trust": pa.trust,
                        "depth": pa.depth,
                    },
                    "signals": {
                        g: dict(sorted(pa.hits[g].items(), key=lambda kv: -kv[1])[:8])
                        for g in sorted(pa.hits)
                        if not g.startswith(("cat:", "topic:", "aud:", "flag:")) and pa.hits[g]
                    },
                    "duplicate": {
                        "is_duplicate": pa.is_duplicate,
                        "duplicate_of": pa.duplicate_of,
                        "similarity": pa.dup_similarity,
                    },
                    "structure": {
                        "words": pa.struct.words,
                        "sentences": pa.struct.sentences,
                        "numbered_lines": pa.struct.numbered_lines,
                        "list_lines": pa.struct.list_lines,
                        "links": len(pa.struct.links),
                        "money": len(pa.struct.money),
                    },
                }
        raise KeyError(f"пост {post_id} не найден")


def analyze(channel: Channel, config: AnalyzerConfig | None = None) -> ChannelAnalysis:
    return ChannelAnalyzer(config).run(channel)
