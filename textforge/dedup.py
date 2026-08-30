"""Дедупликация: точные и почти-дубликаты.

Метод: shingling (k-граммы токенов) -> MinHash-подписи -> LSH-корзины для поиска
кандидатов -> точная проверка по Жаккару. Для новостных лент это закрывает 95% случаев
«одно и то же сообщение от пяти агентств с разным лидом».

Сложность: O(N * k * perm) на построение, поиск кандидатов — почти линейный.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .text import STOPWORDS, tokenize

_MERSENNE = (1 << 61) - 1
_WS = re.compile(r"\s+")


def _hash64(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


def shingles(tokens: list[str], k: int = 5) -> set[str]:
    """Множество k-грамм. Короткие тексты (< k токенов) хешируются целиком."""
    if not tokens:
        return set()
    if len(tokens) < k:
        return {"|".join(tokens)}
    return {"|".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


@dataclass
class DedupResult:
    exact: bool = False
    near: bool = False
    duplicate_of: str | None = None
    similarity: float = 0.0

    @property
    def is_duplicate(self) -> bool:
        return self.exact or self.near


@dataclass
class Deduper:
    k: int = 5
    num_perm: int = 64
    threshold: float = 0.85
    bands: int = 8
    drop_stopwords: bool = False
    min_tokens: int = 8

    _exact_seen: dict[str, str] = field(default_factory=dict, repr=False)
    _buckets: dict[tuple[int, bytes], list[str]] = field(default_factory=dict, repr=False)
    _shingles: dict[str, set[str]] = field(default_factory=dict, repr=False)
    _sigs: dict[str, list[int]] = field(default_factory=dict, repr=False)
    _perms: list[tuple[int, int]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.bands <= 0 or self.num_perm % self.bands:
            # число перестановок должно делиться на число лент
            self.bands = max(1, self.num_perm // max(1, self.bands)) or 1
            self.num_perm = self.bands * max(1, self.num_perm // self.bands)
        rows = self.num_perm // self.bands
        self.rows = rows
        # детерминированные коэффициенты — результат не зависит от запуска
        self._perms = [
            ((i * 2654435761 + 1) % _MERSENNE or 1, (i * 40503 + 7) % _MERSENNE)
            for i in range(self.num_perm)
        ]

    # ---------------------------------------------------------------- internals
    def _tokens(self, text: str) -> list[str]:
        toks = [t.lower() for t in tokenize(text)]
        if self.drop_stopwords:
            toks = [t for t in toks if t not in STOPWORDS]
        return toks

    def _signature(self, sh: set[str]) -> list[int]:
        if not sh:
            return [0] * self.num_perm
        hashes = [_hash64(s) for s in sh]
        return [min((a * h + b) % _MERSENNE for h in hashes) for a, b in self._perms]

    @staticmethod
    def jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = len(a & b)
        return inter / (len(a) + len(b) - inter)

    # ---------------------------------------------------------------- public
    def add(self, doc_id: str, text: str) -> DedupResult:
        res = DedupResult()
        key = hashlib.sha1(_WS.sub(" ", text.strip().lower()).encode("utf-8")).hexdigest()
        if key in self._exact_seen:
            res.exact = True
            res.duplicate_of = self._exact_seen[key]
            res.similarity = 1.0
            return res

        tokens = self._tokens(text)
        sh = shingles(tokens, self.k)
        if len(tokens) < self.min_tokens or not sh:
            self._exact_seen[key] = doc_id
            return res

        sig = self._signature(sh)
        candidates: set[str] = set()
        for b in range(self.bands):
            band = bytes(repr(sig[b * self.rows : (b + 1) * self.rows]), "utf-8")
            bucket = (b, hashlib.blake2b(band, digest_size=8).digest())
            for other in self._buckets.get(bucket, ()):
                candidates.add(other)
            self._buckets.setdefault(bucket, []).append(doc_id)

        best_id, best_sim = None, 0.0
        # sorted(): порядок обхода set зависит от PYTHONHASHSEED, а при равной
        # схожести победитель должен выбираться детерминированно
        for other in sorted(candidates):
            sim = self.jaccard(sh, self._shingles[other])
            if sim > best_sim:
                best_id, best_sim = other, sim

        self._shingles[doc_id] = sh
        self._sigs[doc_id] = sig
        self._exact_seen[key] = doc_id

        if best_id is not None and best_sim >= self.threshold:
            res.near = True
            res.duplicate_of = best_id
            res.similarity = round(best_sim, 4)
        return res

    def run(self, docs: list) -> dict[str, DedupResult]:
        """Принимает объекты с атрибутами `id` и `text`."""
        return {d.id: self.add(d.id, d.text) for d in docs}
