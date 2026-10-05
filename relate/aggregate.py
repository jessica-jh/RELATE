"""Stage 4: labels -> dialogue orientation -> model orientation.

The formula is in compute_orientation(); select via config orientation.method.

Outputs:
  outputs/<run>/orientation_dialogue.jsonl
  outputs/<run>/orientation_model.jsonl
  outputs/<run>/summary.json
"""

import json
import os
from collections import Counter, defaultdict

from .util import run_dir, read_jsonl


# Orientation score (swappable)
def compute_orientation(labels: list[str], cfg: dict) -> float:
    """Map response-level labels to a single orientation score."""
    method = cfg["orientation"]["method"]
    n = len(labels)
    if n == 0:
        return 0.0
    c = Counter(labels)

    if method == "if_minus_of_ratio":
        # (IF - OF) / N  in [-1, 1]
        return (c.get("IF", 0) - c.get("OF", 0)) / n

    if method == "if_ratio":
        # IF / N  in [0, 1]
        return c.get("IF", 0) / n

    if method == "weighted":
        w = cfg["orientation"]["label_weights"]
        return sum(w.get(lbl, 0.0) for lbl in labels) / n

    raise ValueError(f"Unknown orientation.method: {method!r}")


# ---------------------------------------------------------------------------
def aggregate(cfg: dict) -> dict:
    rd = run_dir(cfg)
    judgments = read_jsonl(os.path.join(rd, "judgments.jsonl"))

    # group by dialogue
    by_dialogue: dict[str, list] = defaultdict(list)
    for r in judgments:
        by_dialogue[r["dialogue_id"]].append(r)

    dialogue_rows = []
    for did, turns in by_dialogue.items():
        turns = sorted(turns, key=lambda x: x["turn_index"])
        labels = [t["label"] for t in turns]
        meta = turns[0]
        dialogue_rows.append({
            "dialogue_id": did,
            "condition_id": meta["condition_id"],
            "situation_id": meta["situation_id"],
            "persona_id": meta["persona_id"],
            "target_model": meta["target_model"],
            "target_family": meta["target_family"],
            "target_scale": meta["target_scale"],
            "n_turns": len(labels),
            "label_counts": dict(Counter(labels)),
            "orientation": compute_orientation(labels, cfg),
        })

    # model-level means + pooled counts
    by_model: dict[str, list] = defaultdict(list)
    for r in dialogue_rows:
        by_model[r["target_model"]].append(r)

    model_rows = []
    for model, rows in by_model.items():
        scores = [r["orientation"] for r in rows]
        label_counts: Counter = Counter()
        for r in rows:
            label_counts.update(r["label_counts"])
        sub_counts = Counter(
            j["if_subcategory"] for j in judgments
            if j["target_model"] == model and j["label"] == "IF" and j["if_subcategory"]
        )
        model_rows.append({
            "target_model": model,
            "target_family": rows[0]["target_family"],
            "target_scale": rows[0]["target_scale"],
            "n_dialogues": len(rows),
            "orientation_mean": sum(scores) / len(scores),
            "orientation_min": min(scores),
            "orientation_max": max(scores),
            "label_counts": dict(label_counts),
            "if_subcategory_counts": dict(sub_counts),
        })
    model_rows.sort(key=lambda x: x["orientation_mean"], reverse=True)

    _write_jsonl(os.path.join(rd, "orientation_dialogue.jsonl"), dialogue_rows)
    _write_jsonl(os.path.join(rd, "orientation_model.jsonl"), model_rows)

    summary = {
        "run": cfg["run"]["name"],
        "orientation_method": cfg["orientation"]["method"],
        "n_dialogues": len(dialogue_rows),
        "n_judged_turns": len(judgments),
        "model_ranking": [
            {"model": m["target_model"], "orientation_mean": round(m["orientation_mean"], 4)}
            for m in model_rows
        ],
    }
    with open(os.path.join(rd, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[aggregate] {len(dialogue_rows)} dialogues, {len(model_rows)} models")
    print(f"[aggregate] orientation method = {cfg['orientation']['method']}")
    for m in model_rows:
        print(f"    {m['target_model']:<50} orientation={m['orientation_mean']:+.3f}  {m['label_counts']}")
    return summary


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
