#!/usr/bin/env python3
"""
Build or rebuild data/corpus/situations.jsonl from CounselBench-Eval.

Deterministic: pinned HF revision + sorted question_id → S000, S001, ...

Usage:
    python scripts/build_situations.py
    python scripts/build_situations.py --force
    python scripts/build_situations.py --config configs/corpus.yaml --force
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relate.data import load_situations
from relate.util import load_config, load_env, situations_path


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description="Build reproducible situations.jsonl")
    ap.add_argument("--config", default="configs/corpus.yaml")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and re-download from HuggingFace",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache = situations_path(cfg)
    rows = load_situations(cfg, force_reload=args.force)
    print(f"[build_situations] wrote {len(rows)} rows -> {cache}")
    meta = cache.replace(".jsonl", ".meta.json")
    print(f"[build_situations] metadata -> {meta}")
    if rows:
        print(f"[build_situations] S000 = {rows[0]['question_id']} ({rows[0]['topic']})")
        print(f"[build_situations] S{len(rows)-1:03d} = {rows[-1]['question_id']} ({rows[-1]['topic']})")


if __name__ == "__main__":
    main()
