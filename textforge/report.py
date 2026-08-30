"""Сборка отчёта в формате, который требует промт, плюс машиночитаемый JSON.

Отчёт собирается только из посчитанных чисел и реальных ссылок на посты:
шаблоны подставляют факты, а не «красивые» формулировки.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass

from .analyze import ChannelAnalysis, PostAnalysis
from .tgparser import strip_emoji
from .lexicons import NOT_FOR_AUDIENCE
from .text import plural as _plural

LINE = "━━━━━━━━━━━━━━━━━━"

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _level(value: float) -> str:
    if value >= 70:
        return "🟢 высокая"
    if value >= 45:
        return "🟡 средняя"
    return "🔴 низкая"


def _clean(text: str) -> str:
    return " ".join(strip_emoji(text).split()).strip()


def _body_sentences(pa: PostAnalysis, min_len: int = 30) -> list[str]:
    """Предложения поста без первой строки (она обычно дублирует заголовок).

    Разбиваем построчно: подзаголовки в постах стоят отдельной строкой без точки,
    и если сначала склеить всё пробелом, подзаголовок слипается со следующим
    предложением («Сделай кейсы без клиентов Хочешь писать статьи…»).
    """
    lines = [ln.strip() for ln in pa.doc.text.splitlines() if ln.strip()]
    body_lines = lines[1:] if len(lines) > 1 else [pa.doc.text]
    out: list[str] = []
    for line in body_lines:
        for chunk in _SENT_SPLIT.split(line):
            cleaned = _clean(chunk)
            if len(cleaned) >= min_len and cleaned not in out:
                out.append(cleaned)
    return out


_IMPERATIVE_START = re.compile(
    r"^(подготовь\w*|напиши\w*|сделай\w*|используй\w*|проверь\w*|настрой\w*|собери\w*|добавь\w*|"
    r"посмотри\w*|запиши\w*|разберись|задавай\w*|предложи\w*|договорись|возьми|открой|поставь\w*|"
    r"публикуй\w*|показывай\w*|рассказывай\w*|делай|делите|выделяй\w*|минимизируй\w*|отдыхай\w*|"
    r"не бойся|бер\w*|пиши\w*|учись|найди|зови\w*|упоминай\w*)\b",
    re.I,
)
_BAD_START = ("—", "-", "нет", "но ", "частая ошибка", "кстати", "upd")


def _score_action(sent: str) -> float:
    score = 0.0
    if _IMPERATIVE_START.match(sent):
        score += 2.5
    for marker in ("надо ", "нужно ", "шаг", "чтобы ", "порядок", "схема"):
        if marker in sent.lower():
            score += 1.0
    if sent.lower().startswith(_BAD_START):
        score -= 3.0
    if len(sent) < 40:
        score -= 0.5
    return score


def _cut(sent: str, limit: int) -> str:
    sent = sent.rstrip(",;: ")
    return sent if len(sent) <= limit else sent[: limit - 1].rstrip() + "…"


def _pick(sentences_pool: list[str], markers: tuple[str, ...], used: set[int], limit: int = 200) -> str | None:
    for i, sent in enumerate(sentences_pool):
        if i in used:
            continue
        low = sent.lower()
        if any(m in low for m in markers):
            used.add(i)
            return _cut(sent, limit)
    return None


def _first_sentences(text: str, n: int = 2, max_len: int = 220) -> str:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    out: list[str] = []
    total = 0
    for p in parts:
        p = _clean(p)
        if not p:
            continue
        out.append(p)
        total += len(p)
        if len(out) >= n or total >= max_len:
            break
    joined = " ".join(out)
    return joined[:max_len].rstrip(",;: ") + ("…" if len(joined) > max_len else "")


def _sentence_with(text: str, markers: tuple[str, ...], fallback: str, max_len: int = 200) -> str:
    for part in _SENT_SPLIT.split(text):
        low = part.lower()
        if any(m in low for m in markers):
            cleaned = _clean(part)
            if 25 <= len(cleaned) <= max_len + 60:
                return cleaned[:max_len].rstrip(",;: ")
    return fallback


def _practice_block(index: int, pa: PostAnalysis) -> str:
    title = _clean(strip_emoji(pa.title))[:90]
    pool = _body_sentences(pa)
    used: set[int] = set()

    what = None
    if pool:
        best_i, best_score = 0, -99.0
        for i, sent in enumerate(pool[:12]):
            sc = _score_action(sent)
            if sc > best_score:
                best_i, best_score = i, sc
        if best_score > 0:
            used.add(best_i)
            what = _cut(pool[best_i], 200)
    what = what or (pool[0] if pool else "В посте описан конкретный порядок действий.")
    why = _pick(
        pool,
        ("чтобы ", "потому что", "это полезно", "благодаря", "экономит", "помогает", "дает ", "даёт ", "зачем"),
        used,
    ) or "Практика прикладная: в посте описаны конкретные действия, а не общая идея."
    limits = None
    for i, sent in enumerate(pool):
        if i in used or sent.startswith(("—", "-")):
            continue
        low = sent.lower()
        if any(m in low for m in ("не подходит", "риск", "не сработает", "минус", "ограничен", "зависит от", "не обеща", "но это", "не факт")):
            used.add(i)
            limits = _cut(sent, 200)
            break
    limits = limits or (
        "Ограничение: вывод получен на опыте одного автора в одной нише — "
        "перед применением нужно проверить на своих данных."
    )
    return (
        f"**{index}. {title}**\n\n"
        f"{what}\n\n"
        f"💡 {why}\n\n"
        f"⚠️ {limits}\n\n"
        f"🔗 {pa.link}"
    )


def _best_post_line(pa: PostAnalysis) -> str:
    title = _clean(strip_emoji(pa.title))[:80]
    pool = _body_sentences(pa, min_len=25)
    reason = _pick(pool, ("как ", "потому что", "чтобы ", "правило", "схема", "шаг", "вывод"), set(), limit=150) or (
        "Содержит конкретные действия и числа, а не общие рассуждения."
    )
    views = f"{pa.post.views:,}".replace(",", " ")
    fw = f"{pa.post.forwards:,}".replace(",", " ")
    return (
        f"• [{title}]({pa.link}) — 👁 {views} · 🔁 {fw}\n"
        f"{reason}"
    )


def render_markdown(a: ChannelAnalysis) -> str:
    ch = a.channel
    idx = a.indices
    out: list[str] = []
    add = out.append

    # ---------------------------------------------------------------- сводка
    add(f'📊 **СВОДКА ПО КАНАЛУ «{ch.title or "без названия"}»**')
    add("")
    add(f"🔗 @{ch.username or ch.source or 'нет данных'} · 👥 нет данных")
    add("")
    add(f"📈 ~{ch.posts_per_week} постов в неделю · проанализировано постов: {len(a.posts)}")
    add("")
    add(f"🎯 **Индекс полезности: {idx['usefulness']}/100** ({_level(idx['usefulness'])})")
    add("")
    add(
        f"🧠 Экспертность: {idx['expertise']}/100 · 💎 Оригинальность: {idx['originality']}/100 · "
        f"🛡 Доверие: {idx['trust']}/100"
    )
    add("")
    add(
        f"📢 Реклама/продажи: ~{a.ad_share}% · 🔁 Репосты: ~{a.forward_rate}% · "
        f"💬 Вовлечённость: {a.engagement}%"
    )
    add("")
    feed = " · ".join(f"{cat} {share}%" for cat, share in list(a.category_share.items())[:4])
    add(f"🧩 Лента: {feed}")
    add("")
    add(LINE)

    # ---------------------------------------------------------------- автор
    au = a.author
    add("")
    add("👤 **АВТОР**")
    add("")
    add(f"**{au.get('name', 'нет данных')}**")
    add("")
    add(f"*{au.get('positioning', 'нет данных')}*")
    add("")
    add(
        f"Личный бренд: {au.get('personal_brand', 0)}% постов написаны от первого лица "
        f"с опорой на собственный опыт. Подписчики и верификация в файле не указаны."
    )
    add("")
    add("**Что подтверждается контентом:**")
    for fact in a.facts[:4]:
        add(f"• {fact}")
    add("")
    add("**Что остаётся заявлением автора:**")
    for claim in a.claims[:4]:
        add(f"• {claim}")
    if au.get("clients_mentioned"):
        add("")
        add("Упомянутые клиенты/площадки (упоминание, не подтверждение): " + ", ".join(au["clients_mentioned"]))
    add("")
    add(LINE)

    # ---------------------------------------------------------------- о чём канал
    add("")
    add("📚 **О ЧЁМ КАНАЛ**")
    add("")
    add("🏷 " + ", ".join(name for name, _ in a.topics_by_freq[:5]))
    add("")
    freq_txt = ", ".join(
        f"{name} — {cnt} {_plural(cnt, ('пост', 'поста', 'постов'))}" for name, cnt in a.topics_by_freq[:5]
    )
    add(f"Читатель получает прикладные материалы о работе с текстами и продвижении: {freq_txt}.")
    add("")
    add(LINE)

    # ---------------------------------------------------------------- развитие
    add("")
    add("📈 **РАЗВИТИЕ**")
    add("")
    add(f"*Тренд: {a.trend.get('label', '❓ недостаточно данных')}*")
    add("")
    if "first_half_per_week" in a.trend:
        add(
            f"Первая половина выборки: {a.trend['first_half_per_week']} постов/неделю "
            f"(средние просмотры {a.trend['first_half_avg_views']}), "
            f"вторая: {a.trend['second_half_per_week']} постов/неделю "
            f"(средние просмотры {a.trend['second_half_avg_views']}). {a.trend['note'].capitalize()}."
        )
    else:
        add(a.trend.get("note", ""))
    add("")
    add("Рост частоты публикаций не означает роста качества: индексы считаются по содержанию, а не по объёму.")
    add("")
    add(LINE)

    # ---------------------------------------------------------------- практики
    add("")
    add("🧰 **ЧТО ДЕЙСТВИТЕЛЬНО МОЖНО ЗАБРАТЬ СЕБЕ**")
    add("")
    if a.practices:
        for i, pa in enumerate(a.practices, start=1):
            add(_practice_block(i, pa))
            add("")
    else:
        add("В анализируемой выборке не обнаружено практик, которые можно уверенно рекомендовать к адаптации.")
        add("")
    add(LINE)

    # ---------------------------------------------------------------- лучшие посты
    add("")
    add("🔥 **ЛУЧШИЕ ПОСТЫ**")
    add("")
    if a.best_posts:
        for pa in a.best_posts:
            add(_best_post_line(pa))
            add("")
    else:
        add("Постов, которые уверенно проходят порог по совокупности пользы и глубины, не найдено.")
        add("")
    add(LINE)

    # ---------------------------------------------------------------- популярность vs польза
    pq = a.popularity_vs_quality
    add("")
    add("⚖️ **ПОПУЛЯРНОСТЬ VS ПОЛЬЗА**")
    add("")
    add(
        f"Ранговая корреляция просмотров и полезности: {pq.get('spearman', 0)} — {pq.get('interpretation', '')}. "
        f"Пересечение топ-10 по просмотрам и топ-10 по полезности: {pq.get('overlap_top10', 0)} из 10."
    )
    add("")
    add(LINE)

    # ---------------------------------------------------------------- сильные/слабые
    add("")
    add("💪 **СИЛЬНЫЕ СТОРОНЫ**")
    add("")
    for s in a.strengths:
        add(f"• {s}")
    add("")
    add("⚠️ **СЛАБЫЕ СТОРОНЫ**")
    add("")
    for w in a.weaknesses:
        add(f"• {w}")
    add("")
    add(LINE)

    # ---------------------------------------------------------------- красные флаги
    add("")
    add("🚩 **КРАСНЫЕ ФЛАГИ**")
    add("")
    if a.red_flags:
        for name, info in sorted(a.red_flags.items(), key=lambda kv: -kv[1]["count"])[:5]:
            refs = ", ".join(str(e["id"]) for e in info["examples"])
            add(
                f"• {name}: {info['count']} {_plural(info['count'], ('пост', 'поста', 'постов'))} "
                f"({info['share']}% выборки), например {refs}"
            )
        add("")
    else:
        add("**Существенных красных флагов в анализируемой выборке не обнаружено.**")
        add("")
    if a.contradictions:
        add("Противоречия между постами:")
        for c in a.contradictions:
            add(f"• {c}")
        add("")
    add(LINE)

    # ---------------------------------------------------------------- аудитория
    add("")
    add("👀 **КОМУ СТОИТ ЧИТАТЬ**")
    add("")
    if a.audiences:
        for name, cnt in a.audiences[:4]:
            add(f"✅ {name} — тема затрагивается в {cnt} постах")
    else:
        add("✅ тем, кто работает с текстами, контентом и продвижением")
    add("")
    for line in NOT_FOR_AUDIENCE[:2]:
        add(f"❌ {line}")
    add("")
    add(LINE)

    # ---------------------------------------------------------------- доверие
    add("")
    add("🛡 **ДОВЕРИЕ**")
    add("")
    add(f"**{idx['trust']}/100 — {_level(idx['trust']).split(' ', 1)[1]}**")
    add("")
    evidence_posts = sum(1 for pa in a.posts if pa.evidence)
    limits_posts = sum(1 for pa in a.posts if pa.limits)
    claim_posts = sum(1 for pa in a.posts if pa.claims)
    add(
        f"Повышают доверие: конкретные цифры и оговорки — ограничения признаются в {limits_posts} постах, "
        f"ссылки на источники встречаются в {evidence_posts} постах. "
        f"Снижают: маркетинговые заявления в {claim_posts} постах без подтверждающих материалов."
    )
    if a.unverifiable:
        add("")
        add("Не подтверждается по имеющимся данным: " + "; ".join(a.unverifiable) + ".")
    add("")
    add(
        "Индекс отражает качество сигналов доверия в доступном контенте и не означает, "
        "что утверждения автора являются истинными."
    )
    add("")
    add(LINE)

    # ---------------------------------------------------------------- вердикт
    add("")
    add("⭐ **ВЕРДИКТ**")
    add("")
    v = a.verdict
    add(
        f"Полезность {v['usefulness']}/100, экспертность {v['expertise']}/100, доверие {v['trust']}/100 "
        f"при доле продаж {v['ad_share']}%. "
        f"Порог по полезности проходят {len(a.practices)} {_plural(len(a.practices), ('пост', 'поста', 'постов'))}; "
        f"доказательная база строится на личном опыте автора, а не на проверяемых данных."
    )
    add("")
    add(
        "Тратить время стоит выборочно: брать структурные гайды и разборы, пропускать мотивационные заметки "
        "и анонсы собственных продуктов. Наиболее полезен канал тем, кто сам пишет тексты или ведёт контент "
        "для бизнеса; переоценена здесь связка «личный успех → универсальный совет»."
    )
    add("")
    add(f"**{v['label']}**")
    add("")

    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------------------


def _jsonable(obj):
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def to_dict(a: ChannelAnalysis) -> dict:
    ch = a.channel
    dups = [pa for pa in a.posts if pa.is_duplicate]
    data = {
        "channel": {
            "title": ch.title,
            "username": ch.username,
            "collected": ch.collected,
            "posts_declared": ch.posts_declared,
            "posts_analyzed": len(a.posts),
            "first_post": ch.first_date,
            "last_post": ch.last_date,
            "span_days": round(ch.span_days, 1),
            "posts_per_week": ch.posts_per_week,
            "warnings": ch.warnings,
        },
        "indices": a.indices,
        "categories": a.category_share,
        "ad_share": a.ad_share,
        "claim_density": a.claim_density,
        "duplicates": {
            "count": len(dups),
            "share": round(100.0 * len(dups) / max(len(a.posts), 1), 1),
            "posts": [
                {
                    "id": pa.post.number,
                    "duplicate_of": pa.duplicate_of,
                    "similarity": pa.dup_similarity,
                }
                for pa in dups[:10]
            ],
        },
        "forward_rate": a.forward_rate,
        "engagement": a.engagement,
        "topics_by_frequency": [{"topic": t, "posts": c} for t, c in a.topics_by_freq],
        "topics_by_avg_views": [{"topic": t, "avg_views": v} for t, v in a.topics_by_popularity],
        "audiences": [{"group": g, "posts": c} for g, c in a.audiences],
        "trend": a.trend,
        "popularity_vs_quality": a.popularity_vs_quality,
        "best_posts": [
            {
                "id": pa.post.number,
                "link": pa.link,
                "title": _clean(strip_emoji(pa.title)),
                "views": pa.post.views,
                "forwards": pa.post.forwards,
                "composite": pa.composite,
                "usefulness": pa.usefulness,
            }
            for pa in a.best_posts
        ],
        "practices": [
            {"id": pa.post.number, "link": pa.link, "title": _clean(strip_emoji(pa.title)), "usefulness": pa.usefulness}
            for pa in a.practices
        ],
        "red_flags": a.red_flags,
        "contradictions": a.contradictions,
        "unverifiable": a.unverifiable,
        "author": a.author,
        "facts": a.facts,
        "claims": a.claims,
        "strengths": a.strengths,
        "weaknesses": a.weaknesses,
        "verdict": a.verdict,
        "posts": [
            {
                "id": pa.post.number,
                "date": pa.post.date,
                "link": pa.link,
                "views": pa.post.views,
                "forwards": pa.post.forwards,
                "reactions": pa.post.reactions_total,
                "words": pa.struct.words,
                "category": pa.category,
                "usefulness": pa.usefulness,
                "originality": pa.originality,
                "expertise": pa.expertise,
                "trust": pa.trust,
                "depth": pa.depth,
                "topics": pa.topics,
                "claims": pa.claims,
                "duplicate_of": pa.duplicate_of,
            }
            for pa in a.posts
        ],
    }
    # даты и другие не-JSON типы приводим сразу, чтобы to_dict и to_json не расходились
    return _jsonable(data)


def to_json(a: ChannelAnalysis, indent: int = 2) -> str:
    return json.dumps(to_dict(a), indent=indent, ensure_ascii=False, default=_jsonable)
