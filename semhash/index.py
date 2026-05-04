from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class SearchResult:
    """Single semantic-search result."""

    text: str
    score: float
    hamming_distance: int


class SemHashIndex:
    """In-memory semantic hash index with optional save/load support."""

    def __init__(self, bits: int = 1024, embedding_dim: int = 384) -> None:
        if bits <= 0 or bits % 64 != 0:
            raise ValueError("bits must be a positive multiple of 64")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.bits = bits
        self.embedding_dim = embedding_dim
        self._texts: List[str] = []
        self._hashes: List[int] = []

    def add_texts(self, texts: Sequence[str]) -> None:
        for text in texts:
            emb = _embed(text, self.embedding_dim)
            self._texts.append(text)
            self._hashes.append(_semantic_hash(emb, self.bits))

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        if not self._texts:
            return []

        query_hash = _semantic_hash(_embed(query, self.embedding_dim), self.bits)
        scored = []
        for text, fp in zip(self._texts, self._hashes):
            dist = (query_hash ^ fp).bit_count()
            score = 1.0 - (dist / self.bits)
            scored.append(SearchResult(text=text, score=score, hamming_distance=dist))
        scored.sort(key=lambda r: r.hamming_distance)
        return scored[: min(k, len(scored))]

    def save(self, path: str | Path) -> None:
        payload = {
            "bits": self.bits,
            "embedding_dim": self.embedding_dim,
            "texts": self._texts,
            "hashes": [format(h, f"0{self.bits}b") for h in self._hashes],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SemHashIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        idx = cls(bits=payload["bits"], embedding_dim=payload["embedding_dim"])
        idx._texts = list(payload["texts"])
        idx._hashes = [int(h, 2) for h in payload["hashes"]]
        return idx


def _embed(text: str, dim: int) -> List[float]:
    """Deterministic dense embedding via token hashing + normalization."""
    vec = [0.0] * dim
    for tok in text.lower().split():
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=16).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        magnitude = 1.0 + (digest[5] / 255.0)
        vec[bucket] += sign * magnitude

    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def _hyperplane_sign(bit_idx: int, dim_idx: int) -> float:
    seed = struct.pack(">II", bit_idx, dim_idx)
    d = hashlib.blake2b(seed, digest_size=8, person=b"semhash").digest()
    return 1.0 if d[0] & 1 else -1.0


def _semantic_hash(embedding: Iterable[float], bits: int) -> int:
    emb = list(embedding)
    out = 0
    for bit in range(bits):
        dot = 0.0
        for dim_idx, value in enumerate(emb):
            if value == 0:
                continue
            dot += value * _hyperplane_sign(bit, dim_idx)
        if dot >= 0.0:
            out |= 1 << bit
    return out
