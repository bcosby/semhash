_FIXED_SCALE = 1 << 15
_MASK64 = (1 << 64) - 1


class SearchResult:
    """Single semantic-search result."""

    __slots__ = ("text", "score", "hamming_distance")

    def __init__(self, text, score, hamming_distance):
        self.text = text
        self.score = score
        self.hamming_distance = hamming_distance


class SemHashIndex:
    """In-memory semantic hash index with optional save/load support."""

    def __init__(self, bits=1024, embedding_dim=384):
        if bits <= 0 or bits % 64 != 0:
            raise ValueError("bits must be a positive multiple of 64")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.bits = bits
        self.embedding_dim = embedding_dim
        self._texts = []
        self._hashes = []

    def add_texts(self, texts):
        for text in texts:
            emb = _embed_sparse_fixed(text, self.embedding_dim)
            self._texts.append(text)
            self._hashes.append(_semantic_hash_sparse_fixed(emb, self.bits))

    def search(self, query, k=5):
        if k <= 0:
            raise ValueError("k must be positive")
        if not self._texts:
            return []

        query_hash = _semantic_hash_sparse_fixed(_embed_sparse_fixed(query, self.embedding_dim), self.bits)
        scored = []
        for text, fp in zip(self._texts, self._hashes):
            dist = (query_hash ^ fp).bit_count()
            score = 1.0 - (dist / self.bits)
            scored.append(SearchResult(text=text, score=score, hamming_distance=dist))
        scored.sort(key=lambda r: r.hamming_distance)
        return scored[: min(k, len(scored))]

    def save(self, path):
        hashes = [format(h, "0" + str(self.bits) + "b") for h in self._hashes]
        payload = (
            '{\n'
            '  "bits": ' + str(self.bits) + ',\n'
            '  "embedding_dim": ' + str(self.embedding_dim) + ',\n'
            '  "texts": ' + _json_str_list(self._texts) + ',\n'
            '  "hashes": ' + _json_str_list(hashes) + '\n'
            '}\n'
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(payload)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        data = _parse_payload(raw)
        idx = cls(bits=data["bits"], embedding_dim=data["embedding_dim"])
        idx._texts = list(data["texts"])
        idx._hashes = [int(h, 2) for h in data["hashes"]]
        return idx


def _embed_sparse_fixed(text, dim):
    buckets = {}
    prev_tok = None
    for tok in text.lower().split():
        h = _fnv1a64(tok + "|u")
        bucket = (h >> 16) % dim
        sign = 1 if (h & 1) else -1
        magnitude = 256 + ((h >> 8) & 255)
        buckets[bucket] = buckets.get(bucket, 0) + sign * magnitude

        if prev_tok is not None:
            pair = prev_tok + "\x1f" + tok
            hb = _fnv1a64(pair + "|b")
            b2 = (hb >> 20) % dim
            s2 = 1 if (hb & 1) else -1
            m2 = 64 + ((hb >> 10) & 127)
            buckets[b2] = buckets.get(b2, 0) + s2 * m2
        prev_tok = tok

    if not buckets:
        return {}

    norm = (sum(v * v for v in buckets.values())) ** 0.5
    if norm == 0.0:
        return {}

    scale = _FIXED_SCALE / norm
    out = {}
    for k, v in buckets.items():
        q = int(round(v * scale))
        if q != 0:
            out[k] = q
    return out


def _fnv1a64(s):
    x = 1469598103934665603
    for b in s.encode("utf-8"):
        x ^= b
        x = (x * 1099511628211) & _MASK64
    return x


def _splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def _hyperplane_word(block_idx, dim_idx):
    seed = ((block_idx + 1) << 32) ^ (dim_idx + 0xD1B54A32D192ED03)
    return _splitmix64(seed)


def _semantic_hash_sparse_fixed(embedding, bits):
    if not embedding:
        return 0
    out = 0
    blocks = bits // 64
    for block_idx in range(blocks):
        acc = [0] * 64
        for dim_idx, value in embedding.items():
            signs = _hyperplane_word(block_idx, dim_idx)
            for b in range(64):
                acc[b] += value if (signs >> b) & 1 else -value
        word = 0
        for b, total in enumerate(acc):
            if total >= 0:
                word |= 1 << b
        out |= word << (block_idx * 64)
    return out


def _json_escape(s):
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')


def _json_str_list(items):
    return "[" + ", ".join('"' + _json_escape(x) + '"' for x in items) + "]"


def _parse_payload(raw):
    bits = _parse_int_field(raw, '"bits"')
    embedding_dim = _parse_int_field(raw, '"embedding_dim"')
    texts = _parse_string_list_field(raw, '"texts"')
    hashes = _parse_string_list_field(raw, '"hashes"')
    return {"bits": bits, "embedding_dim": embedding_dim, "texts": texts, "hashes": hashes}


def _parse_int_field(raw, key):
    i = raw.find(key)
    if i < 0:
        raise ValueError("invalid index file")
    j = raw.find(":", i)
    k = j + 1
    while k < len(raw) and raw[k] in " \n\t":
        k += 1
    e = k
    while e < len(raw) and raw[e] in "0123456789":
        e += 1
    return int(raw[k:e])


def _parse_string_list_field(raw, key):
    i = raw.find(key)
    if i < 0:
        raise ValueError("invalid index file")
    j = raw.find("[", i)
    k = raw.find("]", j)
    body = raw[j + 1:k].strip()
    if not body:
        return []
    parts = []
    cur = ""
    in_str = False
    esc = False
    for ch in body:
        if not in_str:
            if ch == '"':
                in_str = True
                cur = ""
            continue
        if esc:
            if ch == 'n':
                cur += "\n"
            elif ch == 'r':
                cur += "\r"
            elif ch == 't':
                cur += "\t"
            else:
                cur += ch
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            in_str = False
            parts.append(cur)
        else:
            cur += ch
    return parts
