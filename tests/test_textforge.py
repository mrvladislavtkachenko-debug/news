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
