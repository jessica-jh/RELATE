"""Stage 2: multi-turn dialogue collection (target x stimulus).

turn 1: fixed user utterance
then: target reply -> user-agent next turn, until N assistant turns

Output: outputs/<run>/dialogues.jsonl
Resumable: skips dialogues already present.
"""
import hashlib
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .prompts import user_agent_system, build_target_system
from .providers import get_provider
from .util import run_dir, read_jsonl, append_jsonl, stimuli_path, filter_stimuli

# user-agent anti-repeat retry: if a generated user turn overlaps too much
# (word-level Jaccard) with any earlier user turn in the same dialogue,
# regenerate with an explicit nudge instead of keeping the near-duplicate.
_SIMILARITY_THRESHOLD = 0.6
_MAX_RETRIES = 2


def _dialogue_id(target_model: str, condition_id: str) -> str:
    return f"{condition_id}__{target_model.replace('/', '-')}"


def _stable_seed(run_seed: int, salt: str) -> int:
    """Deterministic per-call seed derived from the run seed + a salt string,
    so reruns are reproducible without every call collapsing onto the same
    sampled output."""
    h = hashlib.sha256(f"{run_seed}:{salt}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _jaccard(a: str, b: str) -> float:
    sa, sb = _word_set(a), _word_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _most_similar_prior(candidate: str, prior_turns: list[str]) -> tuple[str | None, float]:
    best_turn, best_score = None, 0.0
    for p in prior_turns:
        score = _jaccard(candidate, p)
        if score > best_score:
            best_turn, best_score = p, score
    return best_turn, best_score


def _generate_user_turn(user_agent, messages, ua_system, run_seed, did, t):
    """Generate the next user turn, retrying with a nudge + higher temperature
    if it's near-verbatim repeat of anything already in the transcript —
    either the user-agent repeating itself or echoing the assistant's
    last reply back verbatim."""
    prior_turns = [m["content"] for m in messages]

    candidate = user_agent.chat(
        messages, system=ua_system,
        temperature=0.7, max_tokens=256,
        seed=_stable_seed(run_seed, f"{did}:user:{t}"))

    attempt = 0
    while attempt < _MAX_RETRIES:
        similar_to, score = _most_similar_prior(candidate, prior_turns)
        if score < _SIMILARITY_THRESHOLD:
            break
        attempt += 1
        nudge = (
            f"{ua_system}\n\nInternal note, not visible to the user or "
            f"assistant: your previous attempt repeated, almost "
            f"word-for-word, something that already appeared in this "
            f"conversation: \"{similar_to}\". Silently write a better "
            "message — do not acknowledge this note, apologize, or mention "
            "that you repeated yourself or are trying again. Output ONLY "
            "the in-character message, with new wording, a new angle, or a "
            "different aspect of the same worry."
        )
        candidate = user_agent.chat(
            messages, system=nudge,
            temperature=min(0.7 + 0.15 * attempt, 1.0), max_tokens=256,
            seed=_stable_seed(run_seed, f"{did}:user:{t}:retry{attempt}"))
    return candidate


def _collect_dialogue(tgt, cond, target, user_agent, n_turns, temp, target_system,
                       run_seed, target_max_tokens):
    did = _dialogue_id(tgt["model"], cond["condition_id"])
    ua_system = user_agent_system(
        cond["vignette"], cond["persona_style"], cond["persona_spec"])
    target_system = build_target_system(target_system)

    # running transcript
    messages: list[dict[str, str]] = [
        {"role": "user", "content": cond["fixed_turn1"]}
    ]

    for t in range(n_turns):
        reply = target.chat(
            messages, system=target_system,
            temperature=temp, max_tokens=target_max_tokens,
            seed=_stable_seed(run_seed, f"{did}:target:{t}"))
        messages.append({"role": "assistant", "content": reply})

        # next user turn (skip after last)
        if t < n_turns - 1:
            # user-agent sees target replies as the turns to respond to
            next_user = _generate_user_turn(user_agent, messages, ua_system, run_seed, did, t)
            messages.append({"role": "user", "content": next_user})

    return {
        "dialogue_id": did,
        "condition_id": cond["condition_id"],
        "situation_id": cond["situation_id"],
        "persona_id": cond["persona_id"],
        "target_family": tgt["family"],
        "target_scale": tgt["scale"],
        "target_model": tgt["model"],
        "messages": messages,
    }


def run_dialogues(cfg: dict) -> str:
    rd = run_dir(cfg)
    sp = stimuli_path(cfg)
    if not os.path.exists(sp):
        raise FileNotFoundError(
            f"Frozen corpus not found: {sp}\n"
            "Build it first: python run.py --config configs/corpus.yaml --stage stimulus"
        )
    stimuli = filter_stimuli(read_jsonl(sp), cfg)
    print(f"[dialogue] using {len(stimuli)} stimulus conditions from {sp}")
    out_path = os.path.join(rd, "dialogues.jsonl")

    existing = {r["dialogue_id"] for r in read_jsonl(out_path)} if os.path.exists(out_path) else set()
    n_done_already = len(existing)

    n_turns = cfg["run"]["n_assistant_turns"]
    temp = cfg["run"]["temperature"]
    target_system = cfg["models"]["target_system_prompt"]
    run_seed = cfg["run"]["seed"]
    target_max_tokens = cfg["run"].get("target_max_tokens", 220)

    ua = cfg["models"]["user_agent"]
    user_agent = get_provider(ua["provider"], ua["model"])
    # one client per model (SDKs are thread-safe)
    _reserved = {"family", "scale", "provider", "model"}
    targets = {tgt["model"]: get_provider(tgt["provider"], tgt["model"],
                                           **{k: v for k, v in tgt.items() if k not in _reserved})
               for tgt in cfg["models"]["targets"]}

    max_workers = cfg["run"].get("max_concurrency", 4)

    total = len(cfg["models"]["targets"]) * len(stimuli)
    print(f"[dialogue] {n_done_already}/{total} already done (resuming, "
          f"concurrency={max_workers})")
    t_start = time.time()
    n_new = 0
    lock = threading.Lock()

    pending = [
        (tgt, cond)
        for tgt in cfg["models"]["targets"]
        for cond in stimuli
        if _dialogue_id(tgt["model"], cond["condition_id"]) not in existing
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _collect_dialogue, tgt, cond, targets[tgt["model"]], user_agent,
                n_turns, temp, target_system, run_seed, target_max_tokens,
            ): (tgt, cond)
            for tgt, cond in pending
        }
        for fut in as_completed(futures):
            row = fut.result()
            with lock:
                # serialize appends; write each dialogue as it finishes
                append_jsonl(out_path, row)
                n_new += 1
                done_so_far = n_done_already + n_new
                elapsed_min = (time.time() - t_start) / 60
                rate = n_new / elapsed_min if elapsed_min > 0 else 0
                eta_min = (total - done_so_far) / rate if rate > 0 else float("nan")
                print(f"[dialogue] {done_so_far}/{total} ({done_so_far/total:.0%}) "
                      f"done: {row['dialogue_id']}  | {rate:.1f}/min, ETA ~{eta_min:.0f} min")

    print(f"[dialogue] +{n_new} new, {n_done_already + n_new} total -> {out_path}")
    return out_path
