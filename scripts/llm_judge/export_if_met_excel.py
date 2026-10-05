"""Export judgments.jsonl to an if_met-style Excel for manual review.

One row per assistant sentence; five criterion jobs collapsed into bucket +
model_answer. No ground-truth / agreement columns.

  python scripts/llm_judge/export_if_met_excel.py \\
    --judgments outputs/pdf_prompt_v11_openai_batch/sample10/judgments.jsonl \\
    --out outputs/pdf_prompt_v11_openai_batch/sample10/if_met.xlsx
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

import run_sample as v11

INWARD = {"Bond Anchoring", "Intimate Dyad", "Inner Life Claims", "Return Hooks"}
OUTWARD = "Outward-Scaffolding"


def parse_dialogue_id(dialogue_id):
    persona = None
    m = re.search(r"_(P\d+)__", dialogue_id)
    if m:
        persona = m.group(1)
    target_model = dialogue_id.split("__", 1)[1] if "__" in dialogue_id else ""
    return persona, target_model


def bucket_for(met_classes):
    inward = bool(met_classes & INWARD)
    outward = OUTWARD in met_classes
    if inward and outward:
        return "IF and OS"
    if inward:
        return "IF only"
    if outward:
        return "OS only"
    return "Blank"


def model_answer(met_classes):
    if not met_classes:
        return None
    order = {c: i for i, c in enumerate(v11.CLASSES)}
    return ", ".join(sorted(met_classes, key=lambda c: order.get(c, 99)))


def reason_block(by_class):
    lines = []
    for c in v11.CLASSES:
        row = by_class.get(c)
        if not row:
            continue
        marker = row.get("linguistic marker") or row.get("linguistic_marker") or ""
        reason = row.get("reason") or ""
        lines.append(f"{c} ({row.get('label', '?')}): {reason}")
        if marker and marker.lower() != "none":
            lines.append(f"  marker: {marker}")
    return "\n\n".join(lines)


def load_judgments(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_sentence_rows(judgments):
    by_sentence = defaultdict(dict)
    meta = {}
    for row in judgments:
        key = (row["dialogue_id"], str(row["turn_id"]), str(row["sentence_id"]))
        by_sentence[key][row["class"]] = row
        if key not in meta:
            meta[key] = row.get("_target_sentence") or row.get("target") or ""

    sentence_rows = []
    for key in sorted(by_sentence.keys()):
        dialogue_id, turn_id, sentence_id = key
        by_class = by_sentence[key]
        persona, target_model = parse_dialogue_id(dialogue_id)
        met = {c for c, r in by_class.items() if r.get("label") == "MET"}
        sentence_rows.append({
            "dialogue_id": dialogue_id,
            "persona": persona,
            "target_model": target_model,
            "turn_id": int(turn_id),
            "sentence_id": int(sentence_id),
            "sentence": meta[key],
            "bucket": bucket_for(met),
            "model_answer": model_answer(met),
            "reason": reason_block(by_class),
            **{c: by_class.get(c, {}).get("label", "") for c in v11.CLASSES},
        })
    return sentence_rows


def build_detail_rows(judgments):
    detail = []
    for row in judgments:
        persona, target_model = parse_dialogue_id(row["dialogue_id"])
        usage = row.get("_usage") or {}
        detail.append({
            "dialogue_id": row["dialogue_id"],
            "persona": persona,
            "target_model": target_model,
            "turn_id": row.get("turn_id"),
            "sentence_id": row.get("sentence_id"),
            "class": row.get("class"),
            "label": row.get("label"),
            "linguistic_marker": row.get("linguistic marker") or row.get("linguistic_marker"),
            "reason": row.get("reason"),
            "target_sentence": row.get("_target_sentence"),
            "custom_id": row.get("_custom_id"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        })
    return detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    judgments = load_judgments(args.judgments)
    if not judgments:
        raise SystemExit(f"no rows in {args.judgments}")

    sentences = pd.DataFrame(build_sentence_rows(judgments))
    detail = pd.DataFrame(build_detail_rows(judgments))
    summary = pd.DataFrame([{
        "n_judgments": len(judgments),
        "n_sentences": len(sentences),
        "n_MET": int((detail["label"] == "MET").sum()),
        "n_NOT_MET": int((detail["label"] == "NOT_MET").sum()),
        "bucket_IF_only": int((sentences["bucket"] == "IF only").sum()),
        "bucket_IF_and_OS": int((sentences["bucket"] == "IF and OS").sum()),
        "bucket_OS_only": int((sentences["bucket"] == "OS only").sum()),
        "bucket_Blank": int((sentences["bucket"] == "Blank").sum()),
    }])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="summary", index=False)
        sentences.to_excel(xl, sheet_name="sentences", index=False)
        detail.to_excel(xl, sheet_name="judgments", index=False)

    print(f"wrote {out} sentences={len(sentences)} judgments={len(judgments)}", flush=True)


if __name__ == "__main__":
    main()
