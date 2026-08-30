"""Извлечение сущностей регулярными выражениями.

Каждый экстрактор возвращает список словарей `{"value": ..., ...}`. Словарь
EXTRACTORS используется конвейером по именам из конфига, поэтому добавление
нового типа сущности — это одна функция и одна строка в реестре.
"""

from __future__ import annotations

import re

from .text import Doc, normalize

MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "мая": 5, "май": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
MONTH_NAMES = "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"

MONTH_RE = re.compile(MONTH_NAMES, re.I)

DATE_NUM_RE = re.compile(
    r"\b(?P<d>[0-3]?\d)[./](?P<m>[01]?\d)[./](?P<y>\d{2,4})\b"
)
DATE_TEXT_RE = re.compile(
    rf"\b(?P<d>[0-3]?\d)\s+(?P<m>{MONTH_NAMES})(?:\s+(?P<y>(?:19|20)\d{{2}}))?", re.I
)
DATE_ISO_RE = re.compile(r"\b(?P<y>(?:19|20)\d{2})-(?P<m>[01]\d)-(?P<d>[0-3]\d)\b")
TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}")
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"'()\[\]]+", re.I)
HASHTAG_RE = re.compile(r"(?<![\w])#[\wа-яА-ЯёЁ]{2,}")

PHONE_RE = re.compile(
    r"(?:\+7|8)[\s\-.(]*(?:\d{3}|\(\d{3,5}\))[\s\-.)]*\d{2,3}[\s\-]*\d{2}[\s\-]*\d{2}"
)
PHONE_SHORT_RE = re.compile(r"\b\d{3}-\d{2}-\d{2}\b")

_MONEY_MULT = {"тыс": 1_000, "млн": 1_000_000, "млрд": 1_000_000_000, "трлн": 1_000_000_000_000}
MONEY_RE = re.compile(
    r"(?P<num>\d[\d\s]*(?:[.,]\d+)?)\s*"
    r"(?P<mult>тыс\.?|млн\.?|млрд\.?|трлн\.?)?\s*"
    r"(?P<cur>руб(?:лей|ля|лям|лями)?\.?|р\.?|₽|доллар[а-яё]*|usd|\$|евро|eur|€|юан[а-яё]*|cny|₸|тенге|гривн[а-яё]*|brl)",
    re.I,
)
MONEY_SYMBOL_RE = re.compile(r"(?P<cur>[$€£¥₹])\s*(?P<num>\d[\d\s]*(?:[.,]\d+)?)")

PERCENT_RE = re.compile(r"(?P<num>[+-]?\d[\d\s]*(?:[.,]\d+)?)\s*(?:%|процент[а-яё]*)", re.I)

NUM_UNIT_RE = re.compile(
    r"(?P<num>\d[\d\s]*(?:[.,]\d+)?)\s*"
    r"(?P<unit>тыс\.?|млн\.?|млрд\.?|трлн\.?|км|м2|м²|га|т|кг|г|чел\.?|человек[а-яё]*|раз|%"
    r"|кв\.?\s*м|м3|м³|л|мм|см|м|с|мин|ч\.?|час[а-яё]*|дн[а-яё]*|дней|лет|год[а-яё]*)\b",
    re.I,
)

LAW_RE = re.compile(
    r"\b(?:ст\.?\s*(?:\d{1,4}(?:\.\d+)?)|ч\.?\s*\d{1,2}\s+ст\.?\s*\d{1,4}|"
    r"Ф(?:К)?З[- ]?(?:№\s*)?\d{1,4}|УК\s*РФ|ГК\s*РФ|КоАП)\b",
    re.I,
)

QUOTE_RE = re.compile(r'"([^"\n]{15,400})"')

ORG_MARKERS = (
    "ОАО", "ООО", "ПАО", "АО", "ЗАО", "ГУП", "МУП", "ФГУП", "ОАО", "РЖД", "МВД",
    "СК", "ФСБ", "ФНС", "ЦБ", "Минфин", "Минэкономразвития", "Росстат", "НАТО",
    "ООН", "ЕС", "США", "ВТО", "ОПЕК", "ЕБРР", "МВФ", "ВОЗ", "ЮНЕСКО", "ФИФА",
)
ORG_MARKERS_LOWER = frozenset(x.lower() for x in ORG_MARKERS)
NON_NAME = frozenset(
    """
    понедельник вторник среда четверг пятница суббота воскресенье января февраля марта
    апреля мая июня июля августа сентября октября ноября декабря сегодня вчера завтра
    """.split()
)

# Заглавные слова, которые могут быть началом предложения.
CAP_RUN_RE = re.compile(r"(?:\b[А-ЯЁ][а-яё]{2,}\b(?:\s+[А-ЯЁ][а-яё]{2,}\b){0,2})")
LATIN_NAME_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,}){0,2}\b")

ID_RE = re.compile(r"\b(?:№|No\.?|номер)\s*(?P<num>\d{1,10})\b", re.I)


def _to_float(raw: str) -> float:
    return float(raw.replace("\u00a0", "").replace(" ", "").replace(",", "."))


def _positions(text: str, pattern: re.Pattern) -> list[int]:
    return [m.start() for m in pattern.finditer(text)]


# --------------------------------------------------------------------------------------
# Экстракторы
# --------------------------------------------------------------------------------------


def extract_dates(doc: Doc) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()

    for m in DATE_NUM_RE.finditer(doc.text):
        d, mo, y = m.group("d"), m.group("m"), m.group("y")
        y = int(y)
        if y < 100:
            y += 2000 if y < 70 else 1900
        try:
            key = f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
            if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31 and key not in seen:
                seen.add(key)
                out.append({"value": key, "raw": m.group(0), "at": m.start()})
        except ValueError:
            continue

    for m in DATE_TEXT_RE.finditer(doc.text):
        name = MONTH_RE.match(m.group("m"))
        mo = MONTHS.get((name.group(0)[:6]).lower(), None) if name else None
        if mo is None:
            continue
        y = int(m.group("y")) if m.group("y") else 0
        key = f"{y:04d}-{mo:02d}-{int(m.group('d')):02d}" if y else f"--{mo:02d}-{int(m.group('d')):02d}"
        if key in seen:
            continue
        seen.add(key)
        out.append({"value": key, "raw": m.group(0).strip(), "at": m.start()})

    for m in DATE_ISO_RE.finditer(doc.text):
        key = f"{m.group('y')}-{m.group('m')}-{m.group('d')}"
        if key not in seen:
            seen.add(key)
            out.append({"value": key, "raw": m.group(0), "at": m.start()})

    out.sort(key=lambda x: x.get("at", 0))
    return out


def extract_money(doc: Doc) -> list[dict]:
    out: list[dict] = []
    for m in MONEY_RE.finditer(doc.text):
        try:
            num = _to_float(m.group("num"))
        except ValueError:
            continue
        mult = (m.group("mult") or "").rstrip(".")
        value = num * _MONEY_MULT.get(mult, 1)
        out.append(
            {
                "value": round(value, 2),
                "raw": m.group(0).strip(),
                "currency": m.group("cur"),
                "at": m.start(),
            }
        )
    for m in MONEY_SYMBOL_RE.finditer(doc.text):
        if m.start() in [o["at"] for o in out]:
            continue
        try:
            out.append(
                {
                    "value": _to_float(m.group("num")),
                    "raw": m.group(0).strip(),
                    "currency": m.group("cur"),
                    "at": m.start(),
                }
            )
        except ValueError:
            continue
    return out


def extract_percent(doc: Doc) -> list[dict]:
    out = []
    for m in PERCENT_RE.finditer(doc.text):
        try:
            out.append({"value": _to_float(m.group("num")), "raw": m.group(0).strip(), "at": m.start()})
        except ValueError:
            continue
    return out


def extract_emails(doc: Doc) -> list[dict]:
    return [
        {"value": m.group(0).lower(), "raw": m.group(0), "at": m.start()}
        for m in EMAIL_RE.finditer(doc.text)
    ]


def extract_urls(doc: Doc) -> list[dict]:
    return [
        {"value": m.group(0).rstrip(".,;:"), "raw": m.group(0), "at": m.start()}
        for m in URL_RE.finditer(doc.text)
    ]


def extract_phones(doc: Doc) -> list[dict]:
    out = [{"value": re.sub(r"\D", "", m.group(0)), "raw": m.group(0).strip(), "at": m.start()}
           for m in PHONE_RE.finditer(doc.text)]
    return out


def extract_hashtags(doc: Doc) -> list[dict]:
    return [{"value": m.group(0), "at": m.start()} for m in HASHTAG_RE.finditer(doc.text)]


def extract_quotes(doc: Doc) -> list[dict]:
    return [{"value": " ".join(m.group(1).split()), "at": m.start()} for m in QUOTE_RE.finditer(doc.text)]


def extract_laws(doc: Doc) -> list[dict]:
    return [{"value": " ".join(m.group(0).split()), "at": m.start()} for m in LAW_RE.finditer(doc.text)]


def extract_numbers(doc: Doc) -> list[dict]:
    out = []
    for m in NUM_UNIT_RE.finditer(doc.text):
        try:
            num = _to_float(m.group("num"))
        except ValueError:
            continue
        unit = m.group("unit").rstrip(".").lower()
        mult = _MONEY_MULT.get(unit, 1)
        out.append({"value": num * mult, "unit": unit, "raw": m.group(0).strip(), "at": m.start()})
    return out


def extract_times(doc: Doc) -> list[dict]:
    return [{"value": m.group(0), "at": m.start()} for m in TIME_RE.finditer(doc.text)]


def extract_orgs(doc: Doc) -> list[dict]:
    out = []
    for m in re.finditer(r"\b(" + "|".join(re.escape(x) for x in ORG_MARKERS) + r")\b", doc.text):
        out.append({"value": m.group(1), "at": m.start()})
    return out


def extract_persons(doc: Doc) -> list[dict]:
    """Эвристика: заглавные слова не в начале предложения и не из служебного словаря."""
    out: list[dict] = []
    sent_starts = {0}
    for m in re.finditer(r"[.!?…]\s+", doc.text):
        sent_starts.add(m.end())
    for m in CAP_RUN_RE.finditer(doc.text):
        if m.start() in sent_starts:
            continue
        parts = m.group(0).split()
        if any(p.lower() in NON_NAME for p in parts):
            continue
        if len(parts) == 1 and len(parts[0]) < 4:
            continue
        out.append({"value": m.group(0), "at": m.start(), "confidence": "low"})
    return out


EXTRACTORS = {
    "date": extract_dates,
    "money": extract_money,
    "percent": extract_percent,
    "email": extract_emails,
    "url": extract_urls,
    "phone": extract_phones,
    "hashtag": extract_hashtags,
    "quote": extract_quotes,
    "law": extract_laws,
    "number": extract_numbers,
    "time": extract_times,
    "org": extract_orgs,
    "person": extract_persons,
}

DEFAULT_EXTRACTORS = ["date", "money", "percent", "email", "url", "phone", "law"]


def run_extractors(doc: Doc, names: list[str]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for name in names:
        fn = EXTRACTORS.get(name)
        if fn is None:
            continue
        try:
            result[name] = fn(doc)
        except Exception as exc:  # экстрактор не должен ронять конвейер
            result[name] = [{"error": f"{type(exc).__name__}: {exc}"}]
    return result


def values(entities: list[dict], key: str = "value") -> list:
    return [e[key] for e in entities if key in e and "error" not in e]
