"""Build the tidy sentence-level table used by every RELATE analysis.

One row per (judge, dialogue, turn, sentence); the five criterion jobs become
five boolean columns. PARSE_FAIL is kept as missing (NaN), not as NOT_MET.

  python analysis/build_table.py            # build
  python analysis/build_table.py --verify   # build + assert corpus invariants
"""
import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data"

# Primary judge: DeepSeek-R1-Distill-Qwen-32B, five shards.
# Order matters: the first occurrence of a duplicated key wins.
DEEPSEEK_FILES = [
    ROOT / "outputs/pdf_prompt_v11_sample25/judgments.jsonl",
    ROOT / "outputs/pdf_prompt_v11_remaining75/judgments_deduped_shard0.jsonl",
    ROOT / "shards1-3/judgments.shard1.jsonl",
    ROOT / "shards1-3/judgments.shard2.jsonl",
    ROOT / "shards1-3/judgments.shard3.jsonl",
]
GPT4O_FILE = ROOT / "outputs/pdf_prompt_v11_openai_batch/sample10/judgments.jsonl"
DIALOGUES = ROOT / "dialogues.jsonl"
SITUATIONS = ROOT / "data/corpus/situations.jsonl"

# Criterion name -> column. Matches INWARD/OUTWARD in
# scripts/llm_judge/export_if_met_excel.py.
CLASS_COL = {
    "Bond Anchoring": "bond",
    "Intimate Dyad": "dyad",
    "Inner Life Claims": "inner",
    "Return Hooks": "hook",
    "Outward-Scaffolding": "os",
}
IF_COLS = ["bond", "dyad", "inner", "hook"]
ALL_COLS = IF_COLS + ["os"]

# Display names, ordered by parameter count within family (paper Table 1).
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
    """persona, raw target-model slug. Same contract as export_if_met_excel.py."""
    m = re.search(r"_(P\d+)__", dialogue_id)
    persona = m.group(1) if m else None
    target = dialogue_id.split("__", 1)[1] if "__" in dialogue_id else ""
    return persona, target


def load_judge(paths, judge):
    """Dedupe on (dialogue_id, turn_id, sentence_id, class) and pivot to wide."""
    seen = set()
    cells = defaultdict(dict)
    text = {}
    stats = Counter()
    for path in paths:
        with open(path) as fh:
            for line in fh:
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
                else:  # PARSE_FAIL and anything unexpected -> missing
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


def coverage_table(df):
    """Judged vs expected dialogues, per model x persona. 76 situations each."""
    ds = df[df.judge == "deepseek-r1-distill-qwen-32b"]
    got = (ds.groupby(["target_model", "persona_id"], observed=True)
             .agg(dialogues=("dialogue_id", "nunique"),
                  sentences=("sentence_id", "size"),
                  turns=("turn_id", lambda s: len(set(zip(ds.loc[s.index, "dialogue_id"], s)))),
                  parse_fail=("parse_fail", "sum"))
             .reset_index())
    got["expected_dialogues"] = 76
    got["coverage_pct"] = 100 * got["dialogues"] / got["expected_dialogues"]
    got["parse_fail_pct"] = 100 * got["parse_fail"] / got["sentences"]
    return got


def verify(df):
    ds = df[df.judge == "deepseek-r1-distill-qwen-32b"]
    checks = []

    def chk(name, got, want, tol=0):
        ok = abs(got - want) <= tol if isinstance(want, (int, float)) else got == want
        checks.append((name, got, want, ok))

    chk("deepseek sentences", len(ds), 69194)
    chk("deepseek dialogues", ds.dialogue_id.nunique(), 1596)
    chk("duplicate keys", ds.duplicated(["dialogue_id", "turn_id", "sentence_id"]).sum(), 0)
    chk("sentences with all 5 criteria", (ds.n_criteria == 5).mean(), 1.0, tol=1e-4)
    chk("unmapped models", int(ds.target_model.isna().sum()), 0)
    chk("unmapped personas", int(ds.persona_id.isna().sum()), 0)
    chk("unmapped topics", int(ds.topic.isna().sum()), 0)
    chk("turn range", f"{ds.turn_id.min()}-{ds.turn_id.max()}", "1-6")

    g4 = df[df.judge == "gpt-4o"]
    chk("gpt-4o sentences", len(g4), 6827)
    chk("gpt-4o dialogues", g4.dialogue_id.nunique(), 161)

    width = max(len(c[0]) for c in checks)
    bad = 0
    for name, got, want, ok in checks:
        print(f"  [{'ok ' if ok else 'FAIL'}] {name:<{width}}  got={got!r}  want={want!r}")
        bad += not ok
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for judge, paths in [("deepseek-r1-distill-qwen-32b", DEEPSEEK_FILES), ("gpt-4o", [GPT4O_FILE])]:
        df, stats = load_judge(paths, judge)
        print(f"{judge}: {len(df):,} sentences from {stats['rows']:,} rows "
              f"({stats['dup_rows']:,} dup, {stats['parse_fail']:,} parse-fail, "
              f"{stats['bad_lines']} bad lines)")
        frames.append(df)
    df = attach_metadata(pd.concat(frames, ignore_index=True))

    path = OUT / "sentences.parquet"
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # pyarrow absent or a dtype it dislikes
        print(f"parquet write failed ({exc}); falling back to csv.gz", file=sys.stderr)
        path = OUT / "sentences.csv.gz"
        df.to_csv(path, index=False, compression="gzip")
    print(f"wrote {path} ({len(df):,} rows)")

    cov = coverage_table(df)
    cov.to_csv(OUT / "coverage.csv", index=False)
    print(f"wrote {OUT / 'coverage.csv'}")

    if args.verify:
        print("\nverification:")
        sys.exit(1 if verify(df) else 0)


if __name__ == "__main__":
    main()
