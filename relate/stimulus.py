"""Stage 1: corpus construction (run once, then freeze).

situations x personas -> stimuli with vignette + fixed_turn1.

Output: data/corpus/<name>/situations.jsonl + stimuli.jsonl
Use configs/corpus/*.yaml, not experiment configs.
Resumable: skips condition_ids already present; writes incrementally.
"""

import os
import time

from .data import load_situations
from .prompts import (
    VIGNETTE_SYSTEM, vignette_prompt,
    TURN1_SYSTEM, turn1_prompt,
)
from .providers import get_provider
from .util import stimuli_path, read_jsonl, append_jsonl


def build_stimuli(cfg: dict) -> str:
    out_path = stimuli_path(cfg)

    situations = load_situations(cfg)
    if situations and "seed_question" not in situations[0]:
        raise RuntimeError(
            "The released situations file has IDs and topics only (CounselBench-Eval "
            "text is not redistributed). Rebuild it first: "
            "python scripts/build_situations.py --force"
        )
    personas = cfg["personas"]
    total = len(situations) * len(personas)

    existing = read_jsonl(out_path) if os.path.exists(out_path) else []
    done_ids = {r["condition_id"] for r in existing}
    expected_ids = {f"{sit['situation_id']}_{p['id']}" for sit in situations for p in personas}
    if expected_ids <= done_ids:
        print(f"[stimulus] already complete ({len(done_ids)}/{total}): {out_path}")
        return out_path
    print(f"[stimulus] {len(done_ids & expected_ids)}/{total} already done (resuming)")

    m = cfg["models"].get("stimulus_writer") or cfg["models"]["user_agent"]
    writer = get_provider(m["provider"], m["model"])

    t_start = time.time()
    n_new = 0
    vignette_cache: dict[str, str] = {}
    for sit in situations:
        sid = sit["situation_id"]
        needed = [p for p in personas if f"{sid}_{p['id']}" not in done_ids]
        if not needed:
            continue
        if sid not in vignette_cache:
            vignette_cache[sid] = writer.chat(
                [{"role": "user", "content": vignette_prompt(sit["seed_question"])}],
                system=VIGNETTE_SYSTEM, temperature=0.0,
            )
        vignette = vignette_cache[sid]

        for p in needed:
            turn1 = writer.chat(
                [{"role": "user", "content":
                  turn1_prompt(vignette, p["style"], p["spec"])}],
                system=TURN1_SYSTEM, temperature=0.0,
            )
            row = {
                "condition_id": f"{sid}_{p['id']}",
                "situation_id": sid,
                "persona_id": p["id"],
                "persona_style": p["style"],
                "persona_spec": p["spec"],
                "vignette": vignette,
                "fixed_turn1": turn1,
            }
            append_jsonl(out_path, row)
            n_new += 1
            done_so_far = len(done_ids) + n_new
            elapsed_min = (time.time() - t_start) / 60
            rate = n_new / elapsed_min if elapsed_min > 0 else 0
            eta_min = (total - done_so_far) / rate if rate > 0 else float("nan")
            print(f"[stimulus] {done_so_far}/{total} ({done_so_far/total:.0%}) "
                  f"done: {row['condition_id']}  | {rate:.1f}/min, ETA ~{eta_min:.0f} min")

    print(f"[stimulus] +{n_new} new, {len(done_ids) + n_new} total -> {out_path}")
    return out_path
