"""Independent rebuild of sentences.parquet from the CANONICAL merged judgment
files (deepseek_judge/judgments.jsonl.gz, openai_judge/judgments.jsonl.gz)
instead of the raw per-shard files that build_table.py expects (those raw
shards -- outputs/pdf_prompt_v11_remaining75/*, shards1-3/* -- are not present
in this checkout).

This exists purely to cross-check analysis/data/sentences.parquet: if this
script's output matches the committed parquet row-for-row, the published
table is reproducible from the canonical judgment files that are actually
tracked in git, independent of build_table.py's own pivot/join logic.

  python analysis/build_table_from_canonical.py --verify
"""
import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data"

DEEPSEEK_FILE = ROOT / "deepseek_judge/judgments.jsonl.gz"
GPT4O_FILE = ROOT / "openai_judge/judgments.jsonl.gz"
SITUATIONS = ROOT / "data/corpus/situations.jsonl"

CLASS_COL = {
    "Bond Anchoring": "bond",
    "Intimate Dyad": "dyad",
    "Inner Life Claims": "inner",
    "Return Hooks": "hook",
    "Outward-Scaffolding": "os",
}
IF_COLS = ["bond", "dyad", "inner", "hook"]
ALL_COLS = IF_COLS + ["os"]

MODEL_LABEL = {
    "meta-llama-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama-Meta-Llama-3.1-70B-Instruct-Turbo": "Llama-3.1-70B",
    "mistralai-Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Qwen-Qwen3-14B": "Qwen3-14B",
    "Qwen-Qwen3-32B": "Qwen3-32B",
    "deepseek-ai-DeepSeek-V3-0324": "DeepSeek-V3",
    "claude-haiku-4-5-20251001": "Claude-Haiku-4.5",
}
MODEL_ORDER = ["Llama-3.1-8B", "Llama-3.1-70B", "Mistral-7B", "Qwen3-14B",
               "Qwen3-32B", "DeepSeek-V3", "Claude-Haiku-4.5"]
FAMILY = {"Llama-3.1-8B": "Meta", "Llama-3.1-70B": "Meta", "Mistral-7B": "Mistral AI",
          "Qwen3-14B": "Qwen", "Qwen3-32B": "Qwen", "DeepSeek-V3": "DeepSeek",
          "Claude-Haiku-4.5": "Anthropic"}
PERSONAS = ["P1", "P2", "P3"]


def parse_dialogue_id(dialogue_id):
    m = re.search(r"_(P\d+)__", dialogue_id)
    persona = m.group(1) if m else None
    target = dialogue_id.split("__", 1)[1] if "__" in dialogue_id else ""
    return persona, target


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def load_judge(path, judge):
    """Pivot the already-deduped canonical file to wide (no re-dedup needed,
    but we still guard against any stray duplicate key+class rows)."""
    seen = set()
    cells = defaultdict(dict)
    text = {}
    stats = Counter()
    with _open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_lines"] += 1
                continue
            col = CLASS_COL.get(rec.get("class"))
            if col is None:
                stats["unknown_class"] += 1
                continue
            key = (rec["dialogue_id"], str(rec["turn_id"]), str(rec["sentence_id"]))
            if (key, col) in seen:
                stats["dup_rows"] += 1
                continue
            seen.add((key, col))
            label = rec.get("label")
            if label == "MET":
                cells[key][col] = True
            elif label == "NOT_MET":
                cells[key][col] = False
            else:
                cells[key][col] = None
                stats["parse_fail"] += 1
            text.setdefault(key, rec.get("_target_sentence"))
            stats["rows"] += 1

    rows = []
    for (did, turn, sid), vals in cells.items():
        persona, target = parse_dialogue_id(did)
        row = {
            "judge": judge,
            "dialogue_id": did,
            "situation_id": did.split("_", 1)[0],
            "persona_id": persona,
            "target_raw": target,
            "target_model": MODEL_LABEL.get(target, target),
            "turn_id": int(turn),
            "sentence_id": int(sid),
            "sentence_text": text.get((did, turn, sid)),
            "n_criteria": len(vals),
        }
        row.update({c: vals.get(c) for c in ALL_COLS})
        rows.append(row)
    return pd.DataFrame(rows), stats


def attach_metadata(df):
    topic = {}
    with open(SITUATIONS) as fh:
        for line in fh:
            rec = json.loads(line)
            topic[rec["situation_id"]] = rec["topic"]
    df["topic"] = df["situation_id"].map(topic)
    df["family"] = df["target_model"].map(FAMILY)

    for col in ALL_COLS:
        df[col] = df[col].astype("boolean")
    inward = df[IF_COLS]
    df["if_any"] = inward.any(axis=1)
    df["n_if"] = inward.sum(axis=1)
    df["parse_fail"] = inward.isna().any(axis=1) | df["os"].isna()
    df["both"] = df["if_any"] & df["os"].fillna(False)
    df["neither"] = ~df["if_any"].fillna(False) & ~df["os"].fillna(False)

    df["target_model"] = pd.Categorical(df["target_model"], categories=MODEL_ORDER, ordered=True)
    df["persona_id"] = pd.Categorical(df["persona_id"], categories=PERSONAS, ordered=True)
    return df.sort_values(["judge", "dialogue_id", "turn_id", "sentence_id"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", default="sentences_from_canonical.parquet")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for judge, path in [("deepseek-r1-distill-qwen-32b", DEEPSEEK_FILE), ("gpt-4o", GPT4O_FILE)]:
        df, stats = load_judge(path, judge)
        print(f"{judge}: {len(df):,} sentences from {stats['rows']:,} rows "
              f"({stats['dup_rows']:,} dup, {stats['parse_fail']:,} parse-fail, "
              f"{stats['bad_lines']} bad lines)")
        frames.append(df)
    df = attach_metadata(pd.concat(frames, ignore_index=True))

    path = OUT / args.out
    df.to_parquet(path, index=False)
    print(f"wrote {path} ({len(df):,} rows)")

    if args.verify:
        ds = df[df.judge == "deepseek-r1-distill-qwen-32b"]
        g4 = df[df.judge == "gpt-4o"]
        print("\nverification:")
        print(f"  deepseek sentences: {len(ds):,} (paper claims 69,194)")
        print(f"  deepseek dialogues: {ds.dialogue_id.nunique():,} (paper claims 1,596)")
        print(f"  gpt-4o sentences:   {len(g4):,} (paper claims 6,827)")
        print(f"  gpt-4o dialogues:   {g4.dialogue_id.nunique():,} (paper claims 161)")


if __name__ == "__main__":
    main()
