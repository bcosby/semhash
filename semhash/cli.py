from __future__ import annotations

import argparse
import json
from pathlib import Path

from .index import SemHashIndex


def _read_texts(path: str) -> list[str]:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        return [json.loads(line)["text"] for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(prog="semhash", description="Semantic search at the speed of XOR.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Build and save an index")
    p_index.add_argument("--input", required=True, help="Text file (one document per line) or JSONL with {'text': ...}")
    p_index.add_argument("--output", required=True, help="Output index file (.json)")
    p_index.add_argument("--bits", type=int, default=1024)
    p_index.add_argument("--embedding-dim", type=int, default=384)

    p_search = sub.add_parser("search", help="Run a semantic search")
    p_search.add_argument("--index", required=True, help="Path to saved index")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("-k", type=int, default=5)

    args = parser.parse_args()
    if args.cmd == "index":
        idx = SemHashIndex(bits=args.bits, embedding_dim=args.embedding_dim)
        idx.add_texts(_read_texts(args.input))
        idx.save(args.output)
        print(f"Indexed {len(idx._texts)} docs -> {args.output}")
        return

    idx = SemHashIndex.load(args.index)
    results = idx.search(args.query, k=args.k)
    for rank, result in enumerate(results, 1):
        print(f"{rank}. score={result.score:.4f} hamming={result.hamming_distance} :: {result.text}")


if __name__ == "__main__":
    main()
