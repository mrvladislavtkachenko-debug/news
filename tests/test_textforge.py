"""Тесты ядра: парсер, словари, скоринг, отчёт.

Запуск:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textforge import analyze_file  # noqa: E402
from textforge.analyze import AnalyzerConfig, analyze, structure_of  # noqa: E402
from textforge.dedup import Deduper  # noqa: E402
from textforge.extract import extract_money, extract_percent  # noqa: E402
from textforge.report import render_markdown, to_dict, to_json  # noqa: E402
from textforge.scanner import FastLexicon  # noqa: E402
from textforge.text import Doc, plural, sentences, stem  # noqa: E402
from textforge.tgparser import load_channel, parse_channel  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_channel.md"

FIXTURE = """# Telegram Channel: Тестовый

## Channel Information

- **Title:** Тестовый
- **Username:** @test_channel
- **Source:** @test_channel
- **Collected:** 2026-08-30 17:12:53 RTZ 1 (зима)
- **Posts exported:** 3

---

## Post 1

**Date:** 2024-01-01 10:00:00 RTZ 1 (зима)

**Link:** [https://t.me/test_channel/1](https://t.me/test_channel/1)

**Views:** 1 000

**Forwards:** 50

**Reactions:** 🔥 10, 👍 5

### Text

Как настроить рекламу: пошаговая инструкция

1. Собери семантику запросов.
2. Настрой таргетинг по аудиториям.
3. Проверь метрики через неделю.

Например, стоимость клика снизилась на 30%, а конверсия выросла. Это работает, потому что
аудитория уже прогрета. Но метод не подходит для холодного трафика — есть риск слить бюджет.

---

## Post 2

**Date:** 2024-01-08 10:00:00 RTZ 1 (зима)

**Link:** [https://t.me/test_channel/2](https://t.me/test_channel/2)

**Views:** 2 000

**Forwards:** 10

**Reactions:** 👍 3

### Text

Скидка 50% только сегодня! Успей купить курс, осталось 30 мест.

Го читать подробности и забирай доступ по ссылке: https://test-channel.example/checkout

---

## Post 3

**Date:** 2024-01-15 10:00:00 RTZ 1 (зима)

**Link:** [https://t.me/test_channel/3](https://t.me/test_channel/3)

**Views:** 1 500

**Forwards:** 0

**Media:** Photo

*This post contains photo, but the media file was not downloaded.*

---
"""


class TextToolsTest(unittest.TestCase):
    def test_stem_unifies_word_forms(self):
        self.assertEqual(stem("копирайтинга"), stem("копирайтинг"))
        self.assertEqual(stem("агентства"), stem("агентство"))

    def test_stem_leaves_short_words_alone(self):
        self.assertEqual(stem("он"), "он")
        self.assertEqual(stem("она"), "она")

    def test_sentences_respects_abbreviations(self):
        parts = sentences("Мы работали с ОАО «Ромашка» и получили результат. Потом отдохнули.")
        self.assertEqual(len(parts), 2)

    def test_plural(self):
        self.assertEqual(plural(1, ("пост", "поста", "постов")), "пост")
        self.assertEqual(plural(3, ("пост", "поста", "постов")), "поста")
        self.assertEqual(plural(11, ("пост", "поста", "постов")), "постов")
        self.assertEqual(plural(22, ("пост", "поста", "постов")), "поста")


class ExtractTest(unittest.TestCase):
    def test_money_multiplier(self):
        doc = Doc("1", "Мы сделали 1,5 млн рублей выручки и потратили 250 тыс. руб.")
        values = [m["value"] for m in extract_money(doc)]
        self.assertIn(1_500_000.0, values)
        self.assertIn(250_000.0, values)

    def test_percent(self):
        doc = Doc("1", "Конверсия выросла на 12,5 процента")
        self.assertEqual(extract_percent(doc)[0]["value"], 12.5)


class ScannerTest(unittest.TestCase):
    def test_same_stem_terms_are_not_double_counted(self):
        lex = FastLexicon({"g": ["кейс", "кейсы"]})
        hits = lex.scan(Doc("1", "Это кейс. Ещё один кейс."))
        self.assertEqual(hits.total("g"), 2)

    def test_prefix_form_matches(self):
        lex = FastLexicon({"g": ["выбор*"]})
        hits = lex.scan(Doc("1", "Выборы и выборная кампания"))
        self.assertEqual(hits.total("g"), 2)

    def test_prefixed_verb_form_matches(self):
        lex = FastLexicon({"g": ["делай"]})
        hits = lex.scan(Doc("1", "Сделай это сам"))
        self.assertGreaterEqual(hits.total("g"), 1)

    def test_phrase_with_gap(self):
        lex = FastLexicon({"g": ["целевая аудитория"]})
        hits = lex.scan(Doc("1", "Твоя целевая и лояльная аудитория"))
        self.assertEqual(hits.total("g"), 1)

    def test_phrase_outside_gap_window_does_not_match(self):
        lex = FastLexicon({"g": ["целевая аудитория"]})
        hits = lex.scan(Doc("1", "целевая и очень сильно лояльная аудитория"))
        self.assertEqual(hits.total("g"), 0)


class ParserTest(unittest.TestCase):
    def setUp(self):
        self.ch = parse_channel(FIXTURE)

    def test_channel_header(self):
        self.assertEqual(self.ch.title, "Тестовый")
        self.assertEqual(self.ch.username, "test_channel")
        self.assertEqual(self.ch.posts_declared, 3)

    def test_posts_and_numbers(self):
        self.assertEqual(len(self.ch.posts), 3)
        first = self.ch.posts[0]
        self.assertEqual(first.views, 1000)
        self.assertEqual(first.forwards, 50)
        self.assertEqual(first.reactions, {"🔥": 10, "👍": 5})
        self.assertEqual(first.link, "https://t.me/test_channel/1")

    def test_media_only_post_has_no_text(self):
        third = [p for p in self.ch.posts if p.number == 3][0]
        self.assertEqual(third.text, "")
        self.assertTrue(third.is_media_only)
        self.assertEqual(third.media, "Photo")

    def test_engagement(self):
        first = self.ch.posts[0]
        self.assertAlmostEqual(first.engagement, 0.065, places=3)

    def test_weeks_and_frequency(self):
        self.assertEqual(round(self.ch.span_days), 14)
        self.assertEqual(self.ch.posts_per_week, 1.5)

    def test_broken_date_produces_warning_not_crash(self):
        broken = FIXTURE.replace("2024-01-01 10:00:00", "не дата")
        ch = parse_channel(broken)
        self.assertEqual(len(ch.posts), 3)
        self.assertTrue(any("не распознана дата" in w for w in ch.warnings))


class StructureTest(unittest.TestCase):
    def test_numbered_lines_and_subheadings(self):
        doc = Doc("1", "Заголовок раздела\n\n1. Первый пункт.\n2. Второй пункт.\n3. Третий пункт.\n\nОбычный абзац текста.")
        st = structure_of(doc)
        self.assertEqual(st.numbered_lines, 3)
        self.assertGreaterEqual(st.subheadings, 1)


class DedupTest(unittest.TestCase):
    def test_exact_duplicate_detected(self):
        d = Deduper(k=3, num_perm=32, threshold=0.6, min_tokens=3)
        text = "раз два три четыре пять шесть семь"
        d.add("a", text)
        res = d.add("b", text)
        self.assertTrue(res.exact)
        self.assertEqual(res.duplicate_of, "a")

    def test_near_duplicate_detected(self):
        d = Deduper(k=3, num_perm=64, threshold=0.5, min_tokens=3)
        base = " ".join(f"слово{i}" for i in range(40))
        d.add("a", base)
        res = d.add("b", base.replace("слово5", "словоX"))
        self.assertTrue(res.near)
        self.assertGreater(res.similarity, 0.5)


class AnalyzeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ch = parse_channel(FIXTURE)
        cls.a = analyze(cls.ch, AnalyzerConfig(own_domains=["test-channel.example"]))

    def test_categories_assigned(self):
        by_id = {pa.post.number: pa for pa in self.a.posts}
        self.assertEqual(by_id[1].category, "гайд/инструкция")
        self.assertEqual(by_id[2].category, "реклама/продажа")
        self.assertEqual(by_id[3].category, "прочее")

    def test_indices_within_scale(self):
        for name, value in self.a.indices.items():
            self.assertGreaterEqual(value, 0.0, name)
            self.assertLessEqual(value, 100.0, name)

    def test_ad_share_counts_only_sales(self):
        self.assertAlmostEqual(self.a.ad_share, round(100 / 3, 1), places=1)

    def test_ad_post_penalised_in_usefulness(self):
        by_id = {pa.post.number: pa for pa in self.a.posts}
        self.assertLess(by_id[2].usefulness, by_id[1].usefulness)

    def test_verdict_label_is_one_of_four(self):
        self.assertIn(
            self.a.verdict["label"],
            {"🟢 СТОИТ ИЗУЧАТЬ", "🟡 ЕСТЬ ПОЛЕЗНЫЕ ИДЕИ", "🟠 ПОЛЕЗНО, НО ПЕРЕОЦЕНЕНО", "🔴 МАЛО ПОЛЕЗНОГО КОНТЕНТА"},
        )

    def test_practices_exclude_ads(self):
        for pa in self.a.practices:
            self.assertNotEqual(pa.category, "реклама/продажа")

    def test_explain_returns_evidence(self):
        data = self.a.explain(1)
        self.assertEqual(data["id"], 1)
        self.assertIn("usefulness", data["scores"])
        self.assertTrue(data["signals"])

    def test_popularity_not_part_of_quality_indices(self):
        """Просмотры не должны влиять на полезность: два одинаковых текста с разным охватом."""
        base = parse_channel(FIXTURE)
        boosted = parse_channel(FIXTURE.replace("**Views:** 1 000", "**Views:** 999 999"))
        a1 = analyze(base, AnalyzerConfig())
        a2 = analyze(boosted, AnalyzerConfig())
        self.assertEqual(
            [p.usefulness for p in a1.posts],
            [p.usefulness for p in a2.posts],
        )

    def test_sample_file_end_to_end(self):
        result = analyze_file(str(SAMPLE))
        self.assertIn("СВОДКА ПО КАНАЛУ", result.markdown)
        self.assertEqual(result.data["channel"]["posts_analyzed"], 12)


class ReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = analyze(parse_channel(FIXTURE), AnalyzerConfig())
        cls.md = render_markdown(cls.a)

    def test_all_required_sections_present(self):
        for section in (
            "📊 **СВОДКА ПО КАНАЛУ",
            "👤 **АВТОР**",
            "📚 **О ЧЁМ КАНАЛ**",
            "📈 **РАЗВИТИЕ**",
            "🧰 **ЧТО ДЕЙСТВИТЕЛЬНО МОЖНО ЗАБРАТЬ СЕБЕ**",
            "🔥 **ЛУЧШИЕ ПОСТЫ**",
            "⚖️ **ПОПУЛЯРНОСТЬ VS ПОЛЬЗА**",
            "💪 **СИЛЬНЫЕ СТОРОНЫ**",
            "🚩 **КРАСНЫЕ ФЛАГИ**",
            "👀 **КОМУ СТОИТ ЧИТАТЬ**",
            "🛡 **ДОВЕРИЕ**",
            "⭐ **ВЕРДИКТ**",
        ):
            self.assertIn(section, self.md)

    def test_verdict_rendered(self):
        self.assertIn(self.a.verdict["label"], self.md)

    def test_json_is_serialisable_and_complete(self):
        payload = to_json(self.a)
        data = json.loads(payload)
        self.assertEqual(data["channel"]["posts_analyzed"], 3)
        self.assertIn("indices", data)
        self.assertIn("red_flags", data)
        self.assertEqual(len(data["posts"]), 3)

    def test_to_dict_dates_are_strings(self):
        data = to_dict(self.a)
        self.assertIsInstance(data["channel"]["first_post"], str)


class SampleDataTest(unittest.TestCase):
    def test_sample_channel_parses_cleanly(self):
        ch = load_channel(str(SAMPLE))
        self.assertEqual(ch.username, "molyanov_blog")
        self.assertEqual(ch.posts_declared, 3879)
        self.assertEqual(len(ch.posts), 12)
        self.assertEqual(ch.warnings, [])


class OwnLinkTest(unittest.TestCase):
    """--own-domains и определение собственных ссылок (влияют на «доверие» через ext_links)."""

    def analyzer(self, domains):
        from textforge.analyze import ChannelAnalyzer

        return ChannelAnalyzer(
            AnalyzerConfig(own_domains=list(domains), channel_handle="molyanov_blog")
        )

    def test_flag_marks_domain_as_own(self):
        share = {"vc.ru": 0.02, "example.org": 0.02}
        self.assertFalse(self.analyzer([])._is_own_link("https://vc.ru/x", share))
        self.assertTrue(self.analyzer(["vc.ru"])._is_own_link("https://vc.ru/x", share))

    def test_subdomain_of_own_domain(self):
        share = {"sdelaem.vc.ru": 0.02}
        self.assertTrue(self.analyzer(["vc.ru"])._is_own_link("https://sdelaem.vc.ru/x", share))

    def test_channel_handle_is_own(self):
        share = {}
        a = self.analyzer([])
        self.assertTrue(a._is_own_link("https://t.me/molyanov_blog/5", share))
        self.assertFalse(a._is_own_link("https://t.me/someone_else/5", share))

    def test_repeated_host_counts_as_own_platform(self):
        # эвристика: домен, на который приходится >=8% всех ссылок, считается своей площадкой
        self.assertTrue(self.analyzer([])._is_own_link("https://vc.ru/x", {"vc.ru": 0.5}))
        self.assertFalse(self.analyzer([])._is_own_link("https://vc.ru/x", {"vc.ru": 0.02}))

    def test_external_links_raise_trust(self):
        ch = load_channel(str(SAMPLE))
        base = analyze(ch, AnalyzerConfig())
        # помечаем все домены как свои -> внешних ссылок не остаётся, доверие не должно вырасти
        own_all = analyze(
            ch, AnalyzerConfig(own_domains=["t.me", "molyanov.blog", "sdelaem.agency"])
        )
        self.assertLessEqual(own_all.indices["trust"], base.indices["trust"] + 0.01)


class PublicApiTest(unittest.TestCase):
    """analyze_file из README обязан работать ровно так, как там написано."""

    def test_analyze_file_returns_three_views(self):
        import textforge

        result = textforge.analyze_file(str(SAMPLE))
        self.assertIsInstance(result, textforge.AnalysisResult)
        self.assertIn("📊 **СВОДКА ПО КАНАЛУ", result.markdown)
        self.assertEqual(result.markdown, textforge.render_markdown(result.analysis))
        self.assertEqual(
            result.data["indices"]["usefulness"], result.analysis.indices["usefulness"]
        )

    def test_to_dict_and_to_json_agree(self):
        result = analyze_file(str(SAMPLE))
        self.assertEqual(json.loads(to_json(result.analysis)), result.data)


DUP_FIXTURE = """# Exported channel history

Posts exported: 3

## Post 1
**Date:** 2024-05-01 10:00:00
**Views:** 1000

### Text

Кейс: было 10 лидов, стало 40. Мы изменили оффер, переписали лендинг и получили рост 40%.
Потому что старый оффер обещал слишком много. Вот что сделали: 1) убрали гарантии, 2) добавили цены.

## Post 2
**Date:** 2024-05-02 10:00:00
**Views:** 900

### Text

Кейс: было 10 лидов, стало 40. Мы изменили оффер, переписали лендинг и получили рост 40%.
Потому что старый оффер обещал слишком много. Вот что сделали: 1) убрали гарантии, 2) добавили цены.

## Post 3
**Date:** 2024-05-03 10:00:00
**Views:** 800

### Text

Совершенно другая тема: как считать юнит-экономику. Сделай таблицу, посчитай маржу, проверь когорты.
"""


CONTRA_FIXTURE = """# Exported channel history

Posts exported: 3

## Post 1
**Date:** 2024-05-01 10:00:00
**Views:** 1000

### Text

Прогрев работает отлично и приносит продажи. Надо обязательно прогревать аудиторию перед запуском.

## Post 2
**Date:** 2024-06-01 10:00:00
**Views:** 900

### Text

Прогрев не работает, это манипуляция и бесполезно. Не ведись на такие схемы, это развод.

## Post 3
**Date:** 2024-07-01 10:00:00
**Views:** 800

### Text

Как считать юнит-экономику: сделай таблицу, посчитай маржу, проверь когорты по месяцам.
"""


class PromptQuotasTest(unittest.TestCase):
    """Числовые требования промта: 3-7 тем, 3-5 сильных и слабых сторон, честные флаги."""

    @classmethod
    def setUpClass(cls):
        cls.analysis = analyze(load_channel(str(SAMPLE)))
        cls.md = render_markdown(cls.analysis)

    def _bullets(self, header, *stops):
        section = self.md.split(header, 1)[1]
        for stop in stops:
            if stop in section:
                section = section.split(stop, 1)[0]
        return [l for l in section.splitlines() if l.startswith("•")]

    def test_topics_between_three_and_seven(self):
        import re

        line = re.search(r"🏷 (.+)", self.md)
        self.assertIsNotNone(line, "в отчёте нет строки с темами")
        topics = [t.strip() for t in line.group(1).split(",") if t.strip()]
        self.assertGreaterEqual(len(topics), 3, topics)
        self.assertLessEqual(len(topics), 7, topics)

    def test_strengths_between_three_and_five(self):
        items = self._bullets("💪 **СИЛЬНЫЕ СТОРОНЫ**", "⚠️ **СЛАБЫЕ СТОРОНЫ**")
        self.assertGreaterEqual(len(items), 3, items)
        self.assertLessEqual(len(items), 5, items)

    def test_weaknesses_between_three_and_five(self):
        items = self._bullets("⚠️ **СЛАБЫЕ СТОРОНЫ**", "━" * 18)
        self.assertGreaterEqual(len(items), 3, items)
        self.assertLessEqual(len(items), 5, items)

    def test_red_flags_either_substantiated_or_explicitly_absent(self):
        section = self.md.split("🚩 **КРАСНЫЕ ФЛАГИ**", 1)[1].split("━" * 18)[0]
        bullets = [l for l in section.splitlines() if l.startswith("•")]
        if not bullets:
            self.assertIn("Существенных красных флагов", section)
            return
        # каждый показанный флаг обязан ссылаться на реально существующие посты
        known = {str(pa.post.number) for pa in self.analysis.posts}
        for flag, info in self.analysis.red_flags.items():
            self.assertGreaterEqual(info["count"], 1, flag)
            for ex in info["examples"]:
                self.assertIn(str(ex["id"]), known, f"флаг {flag} ссылается на несуществующий пост")


class DegenerateInputTest(unittest.TestCase):
    """Один пост, битые даты, пост только с медиа — не должны ронять конвейер."""

    ONE_POST = """# Exported channel history

Posts exported: 1

## Post 1
**Date:** not-a-date
**Views:** abc

### Text

Один единственный пост про копирайтинг.
"""

    MEDIA_ONLY = """# Exported channel history

Posts exported: 2

## Post 1
**Date:** 2024-05-01 10:00:00
**Views:** 100
**Media:** photo

## Post 2
**Date:** 2024-05-02 10:00:00
**Views:** 200

### Text

Короткий.
"""

    def _analyze(self, text):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.md"
            path.write_text(text, encoding="utf-8")
            analysis = analyze(load_channel(str(path)))
            return analysis, render_markdown(analysis), to_dict(analysis)

    def test_single_post_with_broken_date_and_views(self):
        analysis, md, data = self._analyze(self.ONE_POST)
        self.assertEqual(len(data["posts"]), 1)
        self.assertEqual(analysis.trend["label"], "❓ недостаточно данных")
        self.assertEqual(md.count("━" * 18), 11, "отчёт должен быть полным даже на одном посте")
        for value in analysis.indices.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 100.0)

    def test_media_only_post_does_not_crash(self):
        analysis, md, data = self._analyze(self.MEDIA_ONLY)
        self.assertEqual(len(data["posts"]), 2)
        self.assertEqual(data["posts"][0]["words"], 0)
        self.assertEqual(data["posts"][0]["category"], "прочее")
        self.assertEqual(md.count("━" * 18), 11)

    def test_indices_always_in_range(self):
        for text in (self.ONE_POST, self.MEDIA_ONLY):
            analysis, _, _ = self._analyze(text)
            for name, value in analysis.indices.items():
                self.assertGreaterEqual(value, 0.0, name)
                self.assertLessEqual(value, 100.0, name)


class AudienceSectionTest(unittest.TestCase):
    """Промт требует 3-5 групп «кому стоит читать» и отдельное «кому не подойдёт»."""

    @classmethod
    def setUpClass(cls):
        cls.analysis = analyze(load_channel(str(SAMPLE)))
        cls.md = render_markdown(cls.analysis)
        cls.section = cls.md.split("👀 **КОМУ СТОИТ ЧИТАТЬ**", 1)[1].split("━" * 18)[0]

    def test_between_three_and_five_groups(self):
        positives = [l for l in self.section.splitlines() if l.startswith("✅")]
        self.assertGreaterEqual(len(positives), 3, f"меньше трёх групп: {positives}")
        self.assertLessEqual(len(positives), 5, f"больше пяти групп: {positives}")

    def test_not_for_block_present(self):
        negatives = [l for l in self.section.splitlines() if l.startswith("❌")]
        self.assertGreaterEqual(len(negatives), 1)

    def test_rendered_groups_carry_a_post_count(self):
        # отчёт печатает не больше четырёх групп, поэтому проверяем именно их
        for name, cnt in self.analysis.audiences[:4]:
            self.assertGreaterEqual(cnt, 1)
            self.assertIn(f"✅ {name} — тема затрагивается в {cnt} постах", self.section)


class VerdictBranchesTest(unittest.TestCase):
    """Промт задаёт ровно четыре вердикта — каждый обязан быть достижимым."""

    def verdict(self, usefulness, expertise, trust, ad_share):
        from textforge.analyze import ChannelAnalysis

        analysis = ChannelAnalysis(channel=load_channel(str(SAMPLE)), posts=[])
        analysis.indices = {
            "usefulness": usefulness,
            "expertise": expertise,
            "trust": trust,
            "originality": 60.0,
            "depth": 60.0,
        }
        analysis.ad_share = ad_share
        return analysis._verdict()["label"]

    def test_green(self):
        self.assertEqual(self.verdict(75, 70, 65, 10.0), "🟢 СТОИТ ИЗУЧАТЬ")

    def test_yellow(self):
        self.assertEqual(self.verdict(55, 40, 40, 30.0), "🟡 ЕСТЬ ПОЛЕЗНЫЕ ИДЕИ")

    def test_orange(self):
        self.assertEqual(self.verdict(40, 40, 40, 30.0), "🟠 ПОЛЕЗНО, НО ПЕРЕОЦЕНЕНО")

    def test_red(self):
        self.assertEqual(self.verdict(20, 30, 40, 60.0), "🔴 МАЛО ПОЛЕЗНОГО КОНТЕНТА")

    def test_high_scores_with_heavy_ads_are_not_green(self):
        # популярность и продажи не должны давать зелёный вердикт сами по себе
        self.assertEqual(self.verdict(75, 70, 65, 50.0), "🟡 ЕСТЬ ПОЛЕЗНЫЕ ИДЕИ")


class RedFlagsTest(unittest.TestCase):
    """В промте 12 красных флагов — все должны быть представимы в отчёте."""

    PROMPT_FLAGS = {
        "обещания гарантированного результата": "обещания гарантированного результата",
        "«секретные» схемы заработка": "«секретные» схемы заработка",
        "демонстрация исключительно успехов": "демонстрация исключительно успехов",
        "отсутствие подтверждений громких результатов": "громкие результаты без подтверждений",
        "давление на аудиторию": "давление на аудиторию",
        "постоянные продажи": "постоянные продажи",
        "манипулятивный дефицит": "манипулятивный дефицит",
        "чрезмерный кликбейт": "чрезмерный кликбейт",
        "подмена доказательств статусом автора": "подмена доказательств статусом автора",
        "универсальные советы без учёта контекста": "универсальные советы без контекста",
        "псевдоэкспертность": "псевдоэкспертность",
        # считается отдельно от лексикона — по расхождению позиций между постами
        "противоречия между постами": "противоречия между постами",
    }

    def test_all_twelve_flags_are_implemented(self):
        from textforge import lexicons as LX

        self.assertEqual(len(self.PROMPT_FLAGS), 12)
        available = set(LX.RED_FLAGS) | {"противоречия между постами"}
        self.assertEqual(
            {v for v in self.PROMPT_FLAGS.values()} - available,
            set(),
            "не все флаги из промта реализованы",
        )

    def test_contradiction_becomes_a_red_flag(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contra.md"
            path.write_text(CONTRA_FIXTURE, encoding="utf-8")
            result = analyze(load_channel(str(path)))

        self.assertEqual(len(result.contradictions), 1)
        flag = result.red_flags["противоречия между постами"]
        self.assertEqual(flag["count"], 2)
        self.assertEqual([e["id"] for e in flag["examples"]], [1, 2])
        self.assertIn("прогрев", result.contradictions[0])

    def test_contradictions_render_in_report(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contra.md"
            path.write_text(CONTRA_FIXTURE, encoding="utf-8")
            md = render_markdown(analyze(load_channel(str(path))))

        section = md.split("🚩 **КРАСНЫЕ ФЛАГИ**", 1)[1].split("━" * 18)[0]
        self.assertIn("противоречия между постами", section)
        self.assertIn("Противоречия между постами:", section)


class DedupWiringTest(unittest.TestCase):
    """Дедупликация обязана влиять на оригинальность и попадать в слабые стороны."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "dup.md"
        self.path.write_text(DUP_FIXTURE, encoding="utf-8")
        self.result = analyze(load_channel(str(self.path)))

    def test_second_copy_is_marked_as_duplicate(self):
        by_id = {pa.doc.id: pa for pa in self.result.posts}
        self.assertFalse(by_id["1"].is_duplicate)
        self.assertTrue(by_id["2"].is_duplicate)
        self.assertEqual(by_id["2"].duplicate_of, "1")
        self.assertEqual(by_id["2"].dup_similarity, 1.0)
        self.assertFalse(by_id["3"].is_duplicate)

    def test_duplicate_caps_originality(self):
        by_id = {pa.doc.id: pa for pa in self.result.posts}
        self.assertLess(by_id["2"].originality, by_id["1"].originality)
        self.assertLessEqual(by_id["2"].originality, 22.0)

    def test_weakness_mentions_repetition_with_correct_agreement(self):
        line = [w for w in self.result.weaknesses if "повторя" in w]
        self.assertEqual(len(line), 1)
        self.assertIn("1 пост практически повторяет ранее опубликованный", line[0])

    def test_duplicates_are_in_to_dict(self):
        data = to_dict(self.result)
        self.assertEqual(data["duplicates"]["count"], 1)
        self.assertEqual(data["duplicates"]["posts"][0]["duplicate_of"], "1")
        self.assertEqual(
            [p["duplicate_of"] for p in data["posts"]], [None, "1", None]
        )

    def test_explain_reports_duplicate(self):
        self.assertEqual(
            self.result.explain("2")["duplicate"],
            {"is_duplicate": True, "duplicate_of": "1", "similarity": 1.0},
        )


class DeterminismTest(unittest.TestCase):
    """Отчёт обязан быть побайтово одинаковым между запусками (PYTHONHASHSEED случайный)."""

    def test_repeated_runs_are_identical_across_hash_seeds(self):
        import hashlib
        import os
        import subprocess
        import tempfile

        repo = Path(__file__).resolve().parent.parent
        digests = set()
        for seed in ("0", "1", "42"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "r.md"
                subprocess.run(
                    [sys.executable, "analyze_channel.py", str(SAMPLE), "--md", str(out), "--quiet"],
                    cwd=repo, env=env, check=True, capture_output=True,
                )
                digests.add(hashlib.md5(out.read_bytes()).hexdigest())
        self.assertEqual(len(digests), 1, f"отчёт недетерминирован: {digests}")

    def test_clients_sorted_by_count_then_name(self):
        ch = load_channel(str(SAMPLE))
        data = to_dict(analyze(ch))
        clients = data["author"]["clients_mentioned"]
        parsed = [(name, int(cnt.rstrip(")"))) for name, cnt in
                  (item.rsplit(" (", 1) for item in clients)]
        counts = [c for _, c in parsed]
        self.assertEqual(counts, sorted(counts, reverse=True))
        for (n1, c1), (n2, c2) in zip(parsed, parsed[1:]):
            if c1 == c2:
                self.assertLess(n1, n2, "равные счётчики должны идти по алфавиту")


class CliTest(unittest.TestCase):
    """Проверяем коды выхода и ключи CLI, включая ранее падавший случай с отсутствующим файлом."""

    NO_USERNAME = """# Exported channel history

Posts exported: 1

## Post 1
**Date:** 2024-05-01 10:00:00
**Views:** 1000

### Text

1. Сделай аудит.
2. Посчитай цифры.
Потому что иначе не поймёшь, сколько денег уходит в рекламу.
"""

    def setUp(self):
        import io
        import tempfile

        self._io = io
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _run(self, argv):
        from textforge import cli

        buf_out, buf_err = self._io.StringIO(), self._io.StringIO()
        real_out, real_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = buf_out, buf_err
        try:
            code = cli.main(argv)
        finally:
            sys.stdout, sys.stderr = real_out, real_err
        return code, buf_out.getvalue(), buf_err.getvalue()

    def test_missing_file_returns_1_without_traceback(self):
        code, _, err = self._run([str(self.root / "nope.md"), "--quiet"])
        self.assertEqual(code, 1)
        self.assertIn("Не удалось прочитать файл", err)
        self.assertNotIn("Traceback", err)

    def test_file_without_posts_returns_1(self):
        empty = self.root / "empty.md"
        empty.write_text("", encoding="utf-8")
        code, _, err = self._run([str(empty), "--quiet"])
        self.assertEqual(code, 1)
        self.assertIn("не найдено ни одного поста", err)

    def test_no_input_prints_help_and_returns_2(self):
        code, out, _ = self._run([])
        self.assertEqual(code, 2)
        self.assertIn("usage:", out.lower())

    def test_handle_fills_missing_username(self):
        src = self.root / "nouser.md"
        src.write_text(self.NO_USERNAME, encoding="utf-8")
        code, out, _ = self._run([str(src), "--handle", "@my_chan", "--quiet"])
        self.assertEqual(code, 0)
        self.assertIn("@my_chan", out)

    def test_md_and_json_are_written(self):
        src = self.root / "nouser.md"
        src.write_text(self.NO_USERNAME, encoding="utf-8")
        md, js = self.root / "sub" / "r.md", self.root / "r.json"
        code, _, _ = self._run([str(src), "--md", str(md), "--json", str(js), "--quiet"])
        self.assertEqual(code, 0)
        self.assertIn("📊 **СВОДКА ПО КАНАЛУ", md.read_text(encoding="utf-8"))
        data = json.loads(js.read_text(encoding="utf-8"))
        self.assertEqual(len(data["posts"]), 1)

    def test_explain_unknown_post_returns_1_without_traceback(self):
        code, _, err = self._run([str(SAMPLE), "--explain", "9999", "--quiet"])
        self.assertEqual(code, 1)
        self.assertIn("не найден", err)
        self.assertNotIn("Traceback", err)

    def test_explain_returns_json_for_post(self):
        src = self.root / "nouser.md"
        src.write_text(self.NO_USERNAME, encoding="utf-8")
        code, out, _ = self._run([str(src), "--explain", "1", "--quiet"])
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(data["id"], 1)
        self.assertIn("usefulness", data["scores"])


if __name__ == "__main__":
    unittest.main()
