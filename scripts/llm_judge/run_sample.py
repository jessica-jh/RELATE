"""Sentence-level LLM judge with a quote-aware sentence split.

Random 25% of dialogues from `dialogues.jsonl` (seed 11).
If a quote opens mid-sentence, do not split on an inner period; end at
. / ? / ! plus a closing quote.

  python scripts/llm_judge/run_sample.py --print-sample
  python scripts/llm_judge/run_sample.py

Other models via --model/--base-url. phi-4-reasoning needs three extra
flags to work at all, found by trial and error on 2026-09-08:
  - its chat template discards a caller-supplied system message and
    substitutes its own fixed one -> use --system-in-user to fold the
    rubric prompt into the user turn instead
  - it loops forever at temperature=0 (greedy decoding) -> --temperature 0.8
    --repetition-penalty 1.1
  - it needs more than the 2048-token default to finish reasoning -> --max-tokens 4096

  python scripts/llm_judge/run_sample.py \
    --dialogues test1.json --sample-frac 1.0 \
    --model microsoft/Phi-4-reasoning --base-url http://127.0.0.1:8004/v1 \
    --concurrency 16 --temperature 0.8 --repetition-penalty 1.1 \
    --system-in-user --max-tokens 4096 \
    --out outputs/pdf_prompt_v11_phi4reasoning_test1
"""
import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = ROOT / "scripts/llm_judge/prompt.txt"
DIALOGUES_PATH = ROOT / "dialogues.jsonl"
OUT_DIR = ROOT / "outputs/pdf_prompt_v11_sample25"
OUT_PATH = OUT_DIR / "judgments.jsonl"
SAMPLE_PATH = OUT_DIR / "sampled_dialogue_ids.json"

BASE_URL = "http://127.0.0.1:8003/v1"
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
CONCURRENCY = 32
SAMPLE_FRAC = 0.25
SAMPLE_SEED = 11
CLASSES = [
    "Bond Anchoring",
    "Intimate Dyad",
    "Inner Life Claims",
    "Return Hooks",
    "Outward-Scaffolding",
]

OPEN_Q = "“\""
CLOSE_Q = "”\""
ENDP = ".?!"


def split_sentences(text: str) -> list[str]:
    sents = []
    buf = []
    in_quote = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        buf.append(ch)

        if not in_quote and ch in OPEN_Q:
            prev = buf[-2] if len(buf) > 1 else " "
            if ch == "“" or prev in " \t\n,;:—–-(":
                in_quote = True
            i += 1
            continue

        if in_quote and ch in CLOSE_Q:
            prev = buf[-2] if len(buf) > 1 else ""
            nxt = text[i + 1] if i + 1 < n else ""
            if prev in ENDP or nxt in ENDP:
                if nxt in ENDP:
                    buf.append(nxt)
                    i += 1
                in_quote = False
                sents.append("".join(buf).strip())
                buf = []
                i += 1
                if i < n and text[i] == " ":
                    i += 1
                continue
            in_quote = False
            i += 1
            continue

        if not in_quote and ch in ENDP:
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt in CLOSE_Q:
                i += 1
                continue
            if nxt in ("", " "):
                sents.append("".join(buf).strip())
                buf = []
                i += 1
                if i < n and text[i] == " ":
                    i += 1
                continue
        i += 1
    tail = "".join(buf).strip()
    if tail:
        sents.append(tail)
    return [s for s in sents if s]


def extract_json_obj(raw: str):
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{") :]
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return None, raw
    try:
        return json.loads(raw[start : end + 1]), raw
    except Exception:
        return None, raw


def load_dialogues(path: Path):
    if path.suffix == ".json":
        return json.loads(path.read_text())
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sample_dialogues(rows, frac: float, seed: int):
    ids = [d["dialogue_id"] for d in rows]
    if frac >= 1.0:
        return rows, sorted(ids)
    k = max(1, int(round(len(ids) * frac)))
    rng = random.Random(seed)
    chosen = sorted(rng.sample(ids, k))
    by_id = {d["dialogue_id"]: d for d in rows}
    return [by_id[i] for i in chosen], chosen


def load_jobs(dialogues):
    jobs = []
    for d in dialogues:
        asst_i = 0
        history = []
        for m in d["messages"]:
            if m["role"] != "assistant":
                history.append(m)
                continue
            asst_i += 1
            sents = split_sentences(m["content"])
            for sid, sent in enumerate(sents, 1):
                jobs.append({
                    "dialogue_id": d["dialogue_id"],
                    "turn_id": str(asst_i),
                    "sentence_id": str(sid),
                    "target": sent,
                    "history": list(history),
                    "earlier": sents[: sid - 1],
                })
            history.append(m)
    return jobs


def build_user(job, criterion):
    ctx = "\n".join(f"{m['role']}: {m['content']}" for m in job["history"]) or "(none)"
    if job["earlier"]:
        earlier = "\n".join(f"[{i}] {s}" for i, s in enumerate(job["earlier"], 1))
    else:
        earlier = "(none)"
    return (
        f"dialogue_id: {job['dialogue_id']}\n"
        f"turn_id: {job['turn_id']}\n"
        f"sentence_id: {job['sentence_id']}\n"
        f"class: {criterion}\n"
        f"criterion to evaluate: {criterion}\n\n"
        f"prior dialogue:\n{ctx}\n\n"
        f"earlier sentences in this assistant turn:\n{earlier}\n\n"
        f"target assistant sentence: {job['target']}"
    )


def already_done(out_path: Path):
    done = set()
    if not out_path.exists():
        return done
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        done.add((r["dialogue_id"], str(r["turn_id"]), str(r["sentence_id"]), r["class"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--print-sample", action="store_true")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only a random subset of N (job, criterion) pairs, for timing estimates. "
        "Writes to a separate output dir unless --out is given explicitly.",
    )
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument(
        "--dialogues",
        default=str(DIALOGUES_PATH),
        help="Path to a .jsonl (line-delimited) or .json (array) dialogues file.",
    )
    ap.add_argument("--sample-frac", type=float, default=SAMPLE_FRAC)
    ap.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="vLLM-specific sampling param, passed via extra_body. Some models "
        "(e.g. phi-4-reasoning) loop indefinitely at temperature=0 without it.",
    )
    ap.add_argument(
        "--system-in-user",
        action="store_true",
        help="Fold the system prompt into the user message instead of sending a "
        "separate system role. Some chat templates (e.g. phi-4-reasoning) discard "
        "a caller-supplied system message and substitute their own fixed one, "
        "silently dropping the rubric instructions.",
    )
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    system = PROMPT_PATH.read_text()
    dialogues_path = Path(args.dialogues)
    all_dlg = load_dialogues(dialogues_path)
    sampled, chosen_ids = sample_dialogues(all_dlg, args.sample_frac, args.sample_seed)
    jobs = load_jobs(sampled)

    if args.print_sample:
        sample = next((j for j in jobs if "“" in j["target"] or '"' in j["target"]), jobs[0])
        print("=== user ===")
        print(build_user(sample, "Outward-Scaffolding"))
        print(f"\ndialogues_all={len(all_dlg)} sampled={len(sampled)} sentences={len(jobs)} jobs={len(jobs)*5}")
        return

    out_dir = Path(args.out) if args.out != str(OUT_DIR) or args.limit is None else OUT_DIR.parent / f"{OUT_DIR.name}_timing_test"
    out_path = out_dir / "judgments.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(out_dir / "sampled_dialogue_ids.json").write_text(
        json.dumps(
            {"seed": args.sample_seed, "frac": args.sample_frac, "n": len(chosen_ids), "dialogue_ids": chosen_ids},
            indent=2,
        )
        + "\n"
    )
    done = already_done(out_path)
    work = []
    for job in jobs:
        for c in CLASSES:
            key = (job["dialogue_id"], job["turn_id"], job["sentence_id"], c)
            if key not in done:
                work.append((job, c))
    full_pending = len(work)
    if args.limit is not None and args.limit < len(work):
        work = random.Random(args.sample_seed).sample(work, args.limit)
    # Submit in dialogue/turn/sentence order (same order jobs were built in)
    # so calls sharing the longest common prompt prefix (system prompt +
    # growing history) land close together for vLLM's prefix cache.
    work.sort(key=lambda jc: (jc[0]["dialogue_id"], int(jc[0]["turn_id"]), int(jc[0]["sentence_id"])))
    print(
        f"prompt={PROMPT_PATH} dialogues={len(sampled)}/{len(all_dlg)} "
        f"sentences={len(jobs)} pending={len(work)}/{full_pending} already={len(done)} "
        f"concurrency={args.concurrency} model={args.model} base_url={args.base_url}",
        flush=True,
    )
    client = OpenAI(api_key="local-no-auth", base_url=args.base_url)
    extra_body = {"repetition_penalty": args.repetition_penalty} if args.repetition_penalty else {}

    def run_one(job, criterion):
        user_content = build_user(job, criterion)
        if args.system_in_user:
            messages = [{"role": "user", "content": system + "\n\n---\n\n" + user_content}]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ]
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            extra_body=extra_body,
        )
        raw = resp.choices[0].message.content
        parsed, _ = extract_json_obj(raw)
        row = parsed or {
            "dialogue_id": job["dialogue_id"],
            "turn_id": job["turn_id"],
            "Role": "assistant",
            "sentence_id": job["sentence_id"],
            "class": criterion,
            "label": "PARSE_FAIL",
            "linguistic marker": "",
            "reason": (raw or "")[:300],
        }
        row["_target_sentence"] = job["target"]
        row["class"] = criterion
        row["dialogue_id"] = job["dialogue_id"]
        row["turn_id"] = str(job["turn_id"])
        row["sentence_id"] = str(job["sentence_id"])
        return row

    ok = fail = 0
    start = time.monotonic()
    with out_path.open("a") as out, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_one, job, c): (job, c) for job, c in work}
        pbar = tqdm(total=len(work), unit="job", dynamic_ncols=True, mininterval=1.0)
        for fut in as_completed(futs):
            try:
                row = fut.result()
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                if row.get("label") == "PARSE_FAIL":
                    fail += 1
                else:
                    ok += 1
            except Exception as e:
                job, c = futs[fut]
                fail += 1
                err = {
                    "dialogue_id": job["dialogue_id"],
                    "turn_id": job["turn_id"],
                    "Role": "assistant",
                    "sentence_id": job["sentence_id"],
                    "class": c,
                    "label": "PARSE_FAIL",
                    "linguistic marker": "",
                    "reason": str(e),
                    "_target_sentence": job["target"],
                }
                out.write(json.dumps(err, ensure_ascii=False) + "\n")
                out.flush()
            pbar.set_postfix(ok=ok, fail=fail, refresh=False)
            pbar.update(1)
        pbar.close()

    elapsed = time.monotonic() - start
    per_job = elapsed / len(work) if work else 0.0
    print(
        f"wrote {out_path} | ran {len(work)} jobs in {elapsed / 60:.1f} min "
        f"({per_job:.2f} s/job wall, concurrency={args.concurrency})",
        flush=True,
    )
    if args.limit is not None:
        projected_hours = per_job * full_pending / 3600
        print(
            f"[timing test] extrapolated to the full remaining run "
            f"({full_pending} jobs at this concurrency/throughput): "
            f"~{projected_hours:.1f} hours",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
