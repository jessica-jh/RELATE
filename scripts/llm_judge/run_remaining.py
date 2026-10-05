"""v11 remaining 75% + retries from the 25% sample.

Does not change `run_sample.py`. Builds the job list as:

  1. every dialogue not in the seed-11 25% sample (the leftover 75%)
  2. the 25% dialogues that never wrote a row (interrupted)
  3. PARSE_FAIL jobs from the 25% jsonl (JSON never parsed)

Successful MET / NOT_MET rows from the 25% run are skipped, not redone.
Then leftover+retry dialogues are assigned by sorted index:

    GPU k gets dialogues[i] where i % n_shards == k

A whole dialogue stays on one GPU so vLLM's prefix cache stays hot.
Each replica writes its own jsonl.

Dry run (no GPU):
  python scripts/llm_judge/run_remaining.py --print-shards

On each GPU host (client + vLLM on the same machine), one shard per host.
Write --out to persistent storage:
  python scripts/llm_judge/run_remaining.py \\
    --shard 0 --base-url http://127.0.0.1:8000/v1 \\
    --out /workspace/outputs/pdf_prompt_v11_remaining75

When all four finish, copy the shard files off the volume, then:
  cat judgments.shard{0,1,2,3}.jsonl > judgments.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sample as v11

EXCLUDE_PATH = v11.ROOT / "outputs/pdf_prompt_v11_sample25/sampled_dialogue_ids.json"
PRIOR_PATH = v11.ROOT / "outputs/pdf_prompt_v11_sample25/judgments.jsonl"
OUT_DIR = v11.ROOT / "outputs/pdf_prompt_v11_remaining75"
N_SHARDS = 4
CONCURRENCY = 16


def load_id_file(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "dialogue_ids" in data:
        return list(data["dialogue_ids"])
    if isinstance(data, list):
        return [str(x) for x in data]
    raise ValueError(f"{path} must be a JSON list or an object with dialogue_ids")


def remaining_dialogues(all_dlg, exclude_ids):
    excl = set(exclude_ids)
    kept = [d for d in all_dlg if d["dialogue_id"] not in excl]
    kept.sort(key=lambda d: d["dialogue_id"])
    return kept


def job_key(row_or_job, criterion=None):
    if criterion is None:
        return (
            row_or_job["dialogue_id"],
            str(row_or_job["turn_id"]),
            str(row_or_job["sentence_id"]),
            row_or_job["class"],
        )
    return (
        row_or_job["dialogue_id"],
        str(row_or_job["turn_id"]),
        str(row_or_job["sentence_id"]),
        criterion,
    )


def already_done_success(out_path: Path):
    """Job keys with a non-PARSE_FAIL row in this shard file (last row wins)."""
    done = {}
    if not out_path.exists():
        return set()
    with out_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (
                r["dialogue_id"],
                str(r["turn_id"]),
                str(r["sentence_id"]),
                r["class"],
            )
            if r.get("label") != "PARSE_FAIL":
                done[key] = True
    return set(done)


def load_prior_status(path: Path):
    """Return successful keys, PARSE_FAIL keys, and dialogue IDs that appear at all."""
    success, fails, seen = set(), set(), set()
    if not path.exists():
        return success, fails, seen
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            seen.add(r["dialogue_id"])
            key = job_key(r)
            if r.get("label") == "PARSE_FAIL":
                fails.add(key)
            else:
                success.add(key)
    return success, fails, seen


def dialogues_to_run(all_dlg, leftover, exclude_ids, prior_seen, fail_keys):
    """Leftover 75% plus unfinished 25% dialogues plus any dialogue with PARSE_FAIL."""
    by_id = {d["dialogue_id"]: d for d in all_dlg}
    unfinished = [by_id[i] for i in sorted(set(exclude_ids) - prior_seen) if i in by_id]
    fail_ids = sorted({k[0] for k in fail_keys})
    fail_dlgs = [by_id[i] for i in fail_ids if i in by_id]
    merged = {d["dialogue_id"]: d for d in leftover + unfinished + fail_dlgs}
    return sorted(merged.values(), key=lambda d: d["dialogue_id"]), unfinished, fail_dlgs


def shard_dialogues(dialogues, shard: int, n_shards: int):
    if n_shards < 1:
        raise ValueError("--n-shards must be >= 1")
    if not (0 <= shard < n_shards):
        raise ValueError(f"--shard must be in [0, {n_shards})")
    if n_shards == 1:
        return dialogues
    return [d for i, d in enumerate(dialogues) if i % n_shards == shard]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--print-shards", action="store_true", help="Print per-GPU counts and exit.")
    ap.add_argument("--print-sample", action="store_true")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=v11.MODEL)
    ap.add_argument("--base-url", default=v11.BASE_URL)
    ap.add_argument("--dialogues", default=str(v11.DIALOGUES_PATH))
    ap.add_argument(
        "--exclude-ids",
        default=str(EXCLUDE_PATH),
        help="JSON of dialogue_ids already judged (the 25%% sample).",
    )
    ap.add_argument(
        "--prior-judgments",
        default=str(PRIOR_PATH),
        help="25%% sample jsonl. PARSE_FAIL and unfinished dialogues are retried.",
    )
    ap.add_argument(
        "--no-retry-failures",
        action="store_true",
        help="Do not retry PARSE_FAIL or unfinished 25%% dialogues.",
    )
    ap.add_argument(
        "--retry-parse-fail",
        action="store_true",
        help="Re-run (job, criterion) rows whose last row in this shard jsonl is PARSE_FAIL.",
    )
    ap.add_argument(
        "--only-parse-fail",
        action="store_true",
        help="Run only dialogues with PARSE_FAIL in --prior-judgments (skip leftover 75%%).",
    )
    ap.add_argument(
        "--shard",
        type=int,
        default=None,
        help="This replica's shard index (required unless --print-shards).",
    )
    ap.add_argument("--n-shards", type=int, default=N_SHARDS)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=None)
    ap.add_argument("--system-in-user", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY") or "local-no-auth",
    )
    args = ap.parse_args()

    all_dlg = v11.load_dialogues(Path(args.dialogues))
    exclude_ids = load_id_file(Path(args.exclude_ids)) if args.exclude_ids else []
    leftover = remaining_dialogues(all_dlg, exclude_ids)
    if args.no_retry_failures:
        success_keys, fail_keys, prior_seen = set(), set(), set()
        to_run, unfinished, fail_dlgs = leftover, [], []
    else:
        success_keys, fail_keys, prior_seen = load_prior_status(Path(args.prior_judgments))
        if args.only_parse_fail:
            by_id = {d["dialogue_id"]: d for d in all_dlg}
            fail_ids = sorted({k[0] for k in fail_keys})
            fail_dlgs = [by_id[i] for i in fail_ids if i in by_id]
            to_run = sorted(fail_dlgs, key=lambda d: d["dialogue_id"])
            unfinished = []
        else:
            to_run, unfinished, fail_dlgs = dialogues_to_run(
                all_dlg, leftover, exclude_ids, prior_seen, fail_keys
            )

    def pending_jobs(dialogues):
        jobs = v11.load_jobs(dialogues)
        n = 0
        for job in jobs:
            for c in v11.CLASSES:
                if job_key(job, c) not in success_keys:
                    n += 1
        return len(jobs), n

    if args.print_shards:
        n_sent_all, n_jobs_all = pending_jobs(to_run)
        print(
            f"total={len(all_dlg)} excluded={len(set(exclude_ids))} "
            f"remaining75={len(leftover)} unfinished25={len(unfinished)} "
            f"parse_fail_jobs={len(fail_keys)} parse_fail_dialogues={len(fail_dlgs)} "
            f"to_run={len(to_run)} jobs={n_jobs_all} n_shards={args.n_shards}"
        )
        for s in range(args.n_shards):
            shard_dlgs = shard_dialogues(to_run, s, args.n_shards)
            n_sent, n_jobs = pending_jobs(shard_dlgs)
            print(
                f"shard {s}/{args.n_shards}: dialogues={len(shard_dlgs)} "
                f"sentences={n_sent} jobs={n_jobs} "
                f"out=judgments.shard{s}.jsonl"
            )
        return

    if args.shard is None:
        ap.error("--shard is required (or pass --print-shards)")

    sampled = shard_dialogues(to_run, args.shard, args.n_shards)
    chosen_ids = [d["dialogue_id"] for d in sampled]
    jobs = v11.load_jobs(sampled)
    system = v11.PROMPT_PATH.read_text()

    if args.print_sample:
        sample = next((j for j in jobs if "“" in j["target"] or '"' in j["target"]), jobs[0])
        print("=== user ===")
        print(v11.build_user(sample, "Outward-Scaffolding"))
        print(
            f"\ndialogues_all={len(all_dlg)} excluded={len(set(exclude_ids))} "
            f"remaining75={len(leftover)} unfinished25={len(unfinished)} "
            f"parse_fail={len(fail_keys)} shard={args.shard}/{args.n_shards} "
            f"this_shard={len(sampled)} sentences={len(jobs)} jobs={len(jobs) * 5}"
        )
        return

    out_dir = Path(args.out)
    out_path = out_dir / f"judgments.shard{args.shard}.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(out_dir / f"dialogue_ids.shard{args.shard}.json").write_text(
        json.dumps(
            {
                "exclude_ids": args.exclude_ids,
                "prior_judgments": args.prior_judgments,
                "n_excluded": len(set(exclude_ids)),
                "n_remaining75": len(leftover),
                "n_unfinished25": len(unfinished),
                "n_parse_fail": len(fail_keys),
                "shard": args.shard,
                "n_shards": args.n_shards,
                "n": len(chosen_ids),
                "dialogue_ids": chosen_ids,
            },
            indent=2,
        )
        + "\n"
    )
    if args.shard == 0:
        Path(out_dir / "remaining_dialogue_ids.json").write_text(
            json.dumps(
                {
                    "exclude_ids": args.exclude_ids,
                    "prior_judgments": args.prior_judgments,
                    "n_excluded": len(set(exclude_ids)),
                    "n_remaining75": len(leftover),
                    "n_unfinished25": len(unfinished),
                    "n_parse_fail": len(fail_keys),
                    "n": len(to_run),
                    "dialogue_ids": [d["dialogue_id"] for d in to_run],
                },
                indent=2,
            )
            + "\n"
        )

    if args.retry_parse_fail:
        done = already_done_success(out_path)
    else:
        done = v11.already_done(out_path)
    skip = success_keys | done
    work = []
    for job in jobs:
        for c in v11.CLASSES:
            key = job_key(job, c)
            if args.only_parse_fail:
                if key not in fail_keys or key in skip:
                    continue
            elif key in skip:
                continue
            work.append((job, c))
    full_pending = len(work)
    if args.limit is not None and args.limit < len(work):
        work = random.Random(11).sample(work, args.limit)
    work.sort(key=lambda jc: (jc[0]["dialogue_id"], int(jc[0]["turn_id"]), int(jc[0]["sentence_id"])))
    print(
        f"prompt={v11.PROMPT_PATH} remaining75={len(leftover)}/{len(all_dlg)} "
        f"unfinished25={len(unfinished)} parse_fail={len(fail_keys)} "
        f"shard={args.shard}/{args.n_shards} this_shard={len(sampled)} "
        f"sentences={len(jobs)} pending={len(work)}/{full_pending} already={len(done)} "
        f"retry_parse_fail={args.retry_parse_fail} "
        f"skipped_ok={len(success_keys)} concurrency={args.concurrency} "
        f"model={args.model} base_url={args.base_url}",
        flush=True,
    )
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    extra_body = {"repetition_penalty": args.repetition_penalty} if args.repetition_penalty else None

    def run_one(job, criterion):
        user_content = v11.build_user(job, criterion)
        if args.system_in_user:
            messages = [{"role": "user", "content": system + "\n\n---\n\n" + user_content}]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ]
        kwargs = dict(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        last_err = None
        for attempt in range(8):
            try:
                resp = client.chat.completions.create(**kwargs)
                last_err = None
                break
            except Exception as e:
                last_err = e
                msg = str(e)
                if "429" not in msg and "rate limit" not in msg.lower():
                    raise
                wait = min(20.0, 3.0 * (attempt + 1))
                m = re.search(r"try again in ([0-9.]+)\s*s", msg, re.I)
                if m:
                    wait = max(wait, float(m.group(1)) + 0.4)
                time.sleep(wait)
        else:
            raise last_err
        raw = resp.choices[0].message.content
        parsed, _ = v11.extract_json_obj(raw)
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
            f"[timing test] this shard ({full_pending} jobs): ~{projected_hours:.1f} hours",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
