# semhash

> Compress embeddings into binary semantic fingerprints and search millions of documents locally with Hamming distance.

**Semantic search at the speed of XOR.**

`semhash` is a lightweight Python library + CLI for local approximate semantic search without a vector database. It converts text into deterministic dense embeddings, compresses them into high-dimensional binary fingerprints, and ranks matches with Hamming distance (`XOR + popcount`).

## Install

```bash
pip install -e .
```

## Python API

```python
from semhash import SemHashIndex

index = SemHashIndex(bits=1024)
index.add_texts([
    "hello world",
    "refund policy",
    "database backup",
])
results = index.search("how do I get my money back?", k=3)

print(results)
```

## CLI

Create a text file with one document per line:

```text
hello world
refund policy
database backup
```

Build an index:

```bash
semhash index --input docs.txt --output index.json --bits 1024
```

Search it:

```bash
semhash search --index index.json --query "how do I get my money back?" -k 3
```

## How it works

1. **Embed**: map each text into a deterministic dense vector.
2. **Hash**: project the vector against many pseudo-random hyperplanes.
3. **Pack**: store signs as a compact bitset (semantic fingerprint).
4. **Search**: compute `hamming = popcount(query_hash XOR doc_hash)`.

## Why semhash

- No external service required.
- Indexes are tiny compared to float32 embeddings.
- Fast CPU search with bit operations.
- Easy to persist and ship as a JSON index.

## MVP limitations

- Embeddings are deterministic hash-based, not SOTA neural encoders.
- Exact scan over all fingerprints (no multi-index LSH yet).
- Best for local prototypes and medium-size corpora.

## License

MIT
