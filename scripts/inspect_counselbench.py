#!/usr/bin/env python3
"""Inspect CounselBench-Eval (the only data source for RELATE situations).

Usage:
    python scripts/inspect_counselbench.py
"""
from datasets import load_dataset


def main():
    ds = load_dataset("izi-ano/CounselBench-Eval", split="test")
    print("=" * 60)
    print("CounselBench-Eval  (izi-ano/CounselBench-Eval, split=test)")
    print("=" * 60)
    print(f"rows: {len(ds)}")
    print(f"columns: {ds.column_names}")
    print()
    print("RELATE uses this dataset only: the test split, topic-filtered to 76 situations.")
    print()

    by_id: dict[str, dict] = {}
    for row in ds:
        qid = row.get("questionID")
        if qid and qid not in by_id:
            by_id[qid] = {
                "topic": row.get("topic"),
                "title": (row.get("questionTitle") or "")[:70],
                "text": (row.get("questionText") or "")[:100],
            }

    print(f"unique questionIDs: {len(by_id)}")
    topics: dict[str, int] = {}
    for v in by_id.values():
        t = v["topic"] or "(none)"
        topics[t] = topics.get(t, 0) + 1
    print(f"unique topics: {len(topics)}  (5 questions each)")
    print()
    print("topics:")
    for t, n in sorted(topics.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {n:3d}  {t}")

    print()
    print("default exclude_topics in configs/corpus.yaml:")
    print("  counseling-fundamentals, professional-ethics, legal-regulatory")
    print("  → 85 situations after filter")
    print()
    print("sample row:")
    r = ds[0]
    for k in ("questionID", "topic", "questionTitle", "questionText"):
        v = str(r.get(k) or "")
        if len(v) > 200:
            v = v[:200] + "..."
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
