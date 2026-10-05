"""Build OpenAI Batch JSONLs for the v11 judge.

  smoke   — one full dialogue
  10pct   — random 10% of dialogues within each target_model
  collect — download finished batches → judgments.jsonl + failures.jsonl
  submit  — loop requests.jsonl in --batch-size chunks (default 40)
  retry   — same loop over failures.jsonl

  python scripts/llm_judge/build_openai_batch.py --mode smoke
  python scripts/llm_judge/build_openai_batch.py --mode 10pct
  python scripts/llm_judge/build_openai_batch.py --mode submit --work-dir .../sample10 --wait
  python scripts/llm_judge/build_openai_batch.py --mode collect
  python scripts/llm_judge/build_openai_batch.py --mode retry --submit --wait
  python scripts/llm_judge/build_openai_batch.py --mode retry --submit --parallel --wait

All jobs live in one requests.jsonl. Submit packs pending rows into upload chunks
(stored under chunks/chunk_NNNNN.jsonl) staying under --max-enqueued-tokens.
Default (--wait, no --parallel): upload each chunk serially, collect after each.
With --parallel: upload all chunks concurrently, poll once, collect once.
"""
import argparse
import json
import math
import os
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import tiktoken

    _TOKENCODER = tiktoken.encoding_for_model("gpt-4o")
except Exception:
    _TOKENCODER = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batch as batch
import run_sample as v11

ROOT = v11.ROOT
DIALOGUES_PATH = ROOT / "dialogues (1).jsonl"
OUT_ROOT = ROOT / "outputs/pdf_prompt_v11_openai_batch"
DEFAULT_MAX_ENQUEUED_TOKENS = 85_000
REQUESTS_NAME = "requests.jsonl"
CHUNKS_DIR = "chunks"
_META_LOCK = threading.Lock()


def group_by_model(dialogues):
    by = defaultdict(list)
    for d in dialogues:
        by[d["target_model"]].append(d)
    for model in by:
        by[model].sort(key=lambda d: d["dialogue_id"])
    return dict(by)


def shuffled_ids(dialogues, seed):
    """Per-model frozen order: sort, then shuffle with one RNG (seed)."""
    by = group_by_model(dialogues)
    rng = random.Random(seed)
    order = {}
    for model in sorted(by):
        ids = [d["dialogue_id"] for d in by[model]]
        rng.shuffle(ids)
        order[model] = ids
    return order


def slice_ids(order, start_frac, end_frac):
    chosen = {}
    for model, ids in order.items():
        n = len(ids)
        start = int(round(n * start_frac))
        end = int(round(n * end_frac))
        if end < start:
            raise ValueError("--end-frac must be >= --start-frac")
        chosen[model] = ids[start:end]
    return chosen


def pick_smoke_dialogue(dialogues, seed):
    ids = sorted(d["dialogue_id"] for d in dialogues)
    rng = random.Random(seed)
    rng.shuffle(ids)
    by_id = {d["dialogue_id"]: d for d in dialogues}
    return by_id[ids[0]]


def items_from_dialogues(dialogues):
    jobs = v11.load_jobs(dialogues)
    items = []
    for i, job in enumerate(jobs):
        for c in v11.CLASSES:
            idx = len(items)
            items.append({
                "custom_id": batch.custom_id(job, c, idx),
                "dialogue_id": job["dialogue_id"],
                "turn_id": str(job["turn_id"]),
                "sentence_id": str(job["sentence_id"]),
                "class": c,
                "target": job["target"],
                "user": v11.build_user(job, c),
            })
    return items


def slim(items):
    return [{k: it[k] for k in (
        "custom_id", "dialogue_id", "turn_id", "sentence_id", "class", "target"
    )} for it in items]


def read_jsonl(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_requests(out_dir, items, system, model, max_tokens):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REQUESTS_NAME
    batch.write_openai_jsonl(path, items, system, model, max_tokens)
    mb = path.stat().st_size / 1e6
    print(f"  wrote {path} jobs={len(items)} mb={mb:.1f}", flush=True)
    return path


def consolidate_requests(work_dir):
    """Merge legacy part*.jsonl into requests.jsonl if needed."""
    work_dir = Path(work_dir)
    req = work_dir / REQUESTS_NAME
    if req.exists():
        return req
    parts = sorted(work_dir.glob("part*.jsonl"))
    if not parts:
        raise RuntimeError(f"no {REQUESTS_NAME} or part*.jsonl in {work_dir}")
    rows = []
    for part in parts:
        rows.extend(read_jsonl(part))
    write_jsonl(req, rows)
    mb = req.stat().st_size / 1e6
    print(f"  consolidated {len(parts)} parts -> {req} jobs={len(rows)} mb={mb:.1f}", flush=True)
    return req


def request_input_tokens(rec):
    msgs = rec["body"]["messages"]
    text = "\n".join(m["content"] for m in msgs)
    if _TOKENCODER is not None:
        return len(_TOKENCODER.encode(text))
    return sum(len(m["content"]) for m in msgs) // 4


def pack_token_chunks(pending, max_tokens=DEFAULT_MAX_ENQUEUED_TOKENS):
    """Group request lines so each chunk stays under the enqueued-token cap.

    Walk jobs in order. If the next job would push the running token sum over
    max_tokens, close the current chunk (without that job) and start a new one.
    The job is always included — in the next chunk, never dropped.
    Chunk size varies; there is no fixed jobs-per-batch target.
    """
    chunks = []
    current = []
    tok_sum = 0
    for rec in pending:
        tok = request_input_tokens(rec)
        if tok > max_tokens:
            raise RuntimeError(
                f"request {rec.get('custom_id')} has {tok} input tokens; "
                f"exceeds --max-enqueued-tokens {max_tokens}"
            )
        if current and tok_sum + tok > max_tokens:
            chunks.append(current)
            current = []
            tok_sum = 0
        current.append(rec)
        tok_sum += tok
    if current:
        chunks.append(current)
    return chunks


def write_manifest(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def openai_client():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    from openai import OpenAI

    return OpenAI(api_key=key)


def work_dir_from_meta(meta):
    if meta.get("work_dir"):
        return Path(meta["work_dir"])
    req = Path(meta["requests_path"])
    if req.parent.name == CHUNKS_DIR:
        return req.parent.parent
    return req.parent


def save_meta(work_dir, meta):
    work_dir = Path(work_dir)
    with _META_LOCK:
        by_id = {m["batch_id"]: m for m in load_metas(work_dir)}
        by_id[meta["batch_id"]] = meta
        stem = Path(meta["requests_path"]).name.replace(".jsonl", "")
        (work_dir / f"batch_meta.{stem}.json").write_text(json.dumps(meta, indent=2) + "\n")
        all_metas = list(by_id.values())
        (work_dir / "batch_meta.json").write_text(json.dumps(all_metas, indent=2) + "\n")
    return meta


def submit_one(work_dir, path, model, tag):
    client = openai_client()
    work_dir = Path(work_dir)
    path = Path(path)
    with path.open("rb") as fh:
        uploaded = client.files.create(file=fh, purpose="batch")
    b = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"v11 {tag} {path.name}"},
    )
    meta = {
        "provider": "openai",
        "model": model,
        "batch_id": b.id,
        "input_file_id": uploaded.id,
        "n": sum(1 for line in path.read_text().splitlines() if line.strip()),
        "requests_path": str(path),
        "work_dir": str(work_dir),
    }
    print(f"submitted {b.id} {path.name} file={uploaded.id}", flush=True)
    save_meta(work_dir, meta)
    return meta


def submit_files(work_dir, paths, model, tag, wait=False):
    client = openai_client()
    metas = []
    for path in paths:
        meta = submit_one(work_dir, path, model, tag)
        metas.append(meta)
        if wait:
            poll_until_done(client, meta)
            collect_all(work_dir)
    return metas


def submit_files_parallel(work_dir, paths, model, tag, concurrency=10):
    paths = list(paths)
    if not paths:
        return []
    concurrency = max(1, min(concurrency, len(paths)))
    metas = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(submit_one, work_dir, path, model, tag): path
            for path in paths
        }
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                metas.append(fut.result())
            except Exception as exc:
                raise RuntimeError(f"upload failed for {path.name}: {exc}") from exc
    metas.sort(key=lambda m: m["requests_path"])
    return metas


def load_metas(work_dir):
    work_dir = Path(work_dir)
    metas = []
    seen = set()
    search_dirs = [work_dir]
    chunks = work_dir / CHUNKS_DIR
    if chunks.is_dir():
        search_dirs.append(chunks)
    for base in search_dirs:
        for path in sorted(base.glob("batch_meta*.json")):
            if path.name == "batch_meta.json":
                continue
            data = json.loads(path.read_text())
            rows = data if isinstance(data, list) else [data]
            for meta in rows:
                bid = meta.get("batch_id")
                if not bid or bid in seen:
                    continue
                seen.add(bid)
                meta.setdefault("work_dir", str(work_dir))
                metas.append(meta)
    agg = work_dir / "batch_meta.json"
    if agg.exists():
        data = json.loads(agg.read_text())
        for meta in (data if isinstance(data, list) else [data]):
            bid = meta.get("batch_id")
            if not bid or bid in seen:
                continue
            seen.add(bid)
            meta.setdefault("work_dir", str(work_dir))
            metas.append(meta)
    return metas


def index_request_lines(out_dir):
    out_dir = Path(out_dir)
    paths = []
    req = out_dir / REQUESTS_NAME
    if req.exists():
        paths.append(req)
    paths.extend(sorted(out_dir.glob("part*.jsonl")))
    paths.extend(sorted((out_dir / CHUNKS_DIR).glob("chunk_*.jsonl")))
    by_id = {}
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        for rec in read_jsonl(path):
            cid = rec.get("custom_id")
            if cid:
                by_id[cid] = rec
    return by_id


def load_job_map(out_dir):
    path = Path(out_dir) / "jobs.json"
    if not path.exists():
        return {}
    return {it["custom_id"]: it for it in json.loads(path.read_text())}


def load_judgments(out_dir):
    path = Path(out_dir) / "judgments.jsonl"
    by = {}
    if not path.exists():
        return by
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row.get("_custom_id")
        if cid:
            by[cid] = row
    return by


def load_failures(out_dir):
    path = Path(out_dir) / "failures.jsonl"
    by = {}
    if not path.exists():
        return by
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid = rec.get("custom_id")
        if cid:
            by[cid] = rec
    return by


def write_judgments(out_dir, judgments):
    path = Path(out_dir) / "judgments.jsonl"
    with path.open("w") as f:
        for cid in sorted(judgments):
            f.write(json.dumps(judgments[cid], ensure_ascii=False) + "\n")
    return path


def write_failures(out_dir, failures):
    path = Path(out_dir) / "failures.jsonl"
    with path.open("w") as f:
        for cid in sorted(failures):
            f.write(json.dumps(failures[cid], ensure_ascii=False) + "\n")
    return path


def is_success(rec):
    if rec.get("error"):
        return False
    resp = rec.get("response") or {}
    return resp.get("status_code") == 200 and bool(resp.get("body"))


def judgment_from_output(rec, job):
    body = (rec.get("response") or {}).get("body") or {}
    raw = (((body.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
    row = batch.row_from_text(job, raw)
    row["_custom_id"] = job["custom_id"]
    if body.get("usage"):
        row["_usage"] = body["usage"]
    return row


def merge_batch(meta, job_map, req_by_id, judgments, failures, client):
    b = client.batches.retrieve(meta["batch_id"])
    status = b.status
    print(
        f"  {meta['batch_id']} status={status} "
        f"completed={getattr(b.request_counts, 'completed', '?')} "
        f"failed={getattr(b.request_counts, 'failed', '?')} "
        f"total={getattr(b.request_counts, 'total', '?')}",
        flush=True,
    )
    if status in {"validating", "in_progress", "finalizing"}:
        print(f"  skip {meta['batch_id']}: still {status}", flush=True)
        return

    req_path = Path(meta["requests_path"])
    work_dir = work_dir_from_meta(meta)
    stem = req_path.stem
    artifact_dir = work_dir / CHUNKS_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    req_for_batch = {}
    if req_path.exists():
        for rec in read_jsonl(req_path):
            if rec.get("custom_id"):
                req_for_batch[rec["custom_id"]] = rec

    seen = set()

    if b.output_file_id:
        text = client.files.content(b.output_file_id).text
        (artifact_dir / f"results.{stem}.jsonl").write_text(text)
        for line in text.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec.get("custom_id")
            job = job_map.get(cid)
            if not cid or job is None:
                continue
            seen.add(cid)
            if is_success(rec):
                judgments[cid] = judgment_from_output(rec, job)
                failures.pop(cid, None)
            else:
                req = req_for_batch.get(cid) or req_by_id.get(cid)
                if req:
                    failures[cid] = req

    if b.error_file_id:
        text = client.files.content(b.error_file_id).text
        (artifact_dir / f"errors.{stem}.jsonl").write_text(text)
        for line in text.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec.get("custom_id")
            if not cid or cid in judgments:
                continue
            seen.add(cid)
            req = req_for_batch.get(cid) or req_by_id.get(cid) or failures.get(cid)
            if req:
                failures[cid] = req

    if status in {"failed", "expired", "cancelled"}:
        for cid, req in req_for_batch.items():
            if cid not in seen and cid not in judgments:
                failures[cid] = req


def collect_all(out_dir):
    out_dir = Path(out_dir)
    metas = load_metas(out_dir)
    if not metas:
        raise RuntimeError(f"no batch_meta.json in {out_dir}; submit first")
    client = openai_client()
    job_map = load_job_map(out_dir)
    req_by_id = index_request_lines(out_dir)
    judgments = load_judgments(out_dir)
    failures = load_failures(out_dir)
    print(
        f"collect dir={out_dir} batches={len(metas)} "
        f"judgments={len(judgments)} failures={len(failures)}",
        flush=True,
    )
    for meta in metas:
        merge_batch(meta, job_map, req_by_id, judgments, failures, client)
    jpath = write_judgments(out_dir, judgments)
    fpath = write_failures(out_dir, failures)
    pending = max(0, len(job_map) - len(judgments) - len(failures))
    print(
        f"  wrote {jpath} n={len(judgments)} | {fpath} n={len(failures)} | pending={pending}",
        flush=True,
    )


def poll_until_done(client, meta, interval=20):
    while True:
        b = client.batches.retrieve(meta["batch_id"])
        print(
            f"  poll {meta['batch_id']} status={b.status} "
            f"{getattr(b.request_counts, 'completed', '?')}/"
            f"{getattr(b.request_counts, 'total', '?')}",
            flush=True,
        )
        if b.status in batch.DONE_STATES_OAI:
            return b
        time.sleep(interval)


def poll_all_until_done(work_dir, metas=None, interval=20):
    client = openai_client()
    all_metas = list(metas) if metas is not None else load_metas(work_dir)
    if not all_metas:
        print("  no batches to poll", flush=True)
        return
    done_ids = set()
    while True:
        in_flight = 0
        completed_jobs = 0
        total_jobs = 0
        for meta in all_metas:
            bid = meta["batch_id"]
            if bid in done_ids:
                continue
            b = client.batches.retrieve(bid)
            counts = b.request_counts
            completed_jobs += getattr(counts, "completed", 0) or 0
            total_jobs += getattr(counts, "total", 0) or 0
            if b.status in batch.DONE_STATES_OAI:
                done_ids.add(bid)
            else:
                in_flight += 1
        print(
            f"  poll batches done={len(done_ids)}/{len(all_metas)} "
            f"jobs={completed_jobs}/{total_jobs} in_flight={in_flight}",
            flush=True,
        )
        if in_flight == 0:
            return
        time.sleep(interval)


def pending_requests(work_dir, source=REQUESTS_NAME):
    work_dir = Path(work_dir)
    if source == REQUESTS_NAME:
        src = consolidate_requests(work_dir)
    else:
        src = work_dir / source
        if not src.exists():
            raise RuntimeError(f"missing {src}")
    judgments = load_judgments(work_dir)
    failures = load_failures(work_dir)
    pending = []
    for rec in read_jsonl(src):
        cid = rec.get("custom_id")
        if not cid or cid in judgments:
            continue
        if source == REQUESTS_NAME and cid in failures:
            continue
        pending.append(rec)
    return pending


def write_chunk(work_dir, chunk_idx, rows):
    path = Path(work_dir) / CHUNKS_DIR / f"chunk_{chunk_idx:05d}.jsonl"
    write_jsonl(path, rows)
    return path


def chunk_token_total(rows):
    return sum(request_input_tokens(r) for r in rows)


def next_chunk_idx(work_dir):
    existing = sorted((Path(work_dir) / CHUNKS_DIR).glob("chunk_*.jsonl"))
    if not existing:
        return 0
    return max(int(p.stem.split("_")[1]) for p in existing) + 1


def submit_loop(
    work_dir,
    model,
    max_tokens=DEFAULT_MAX_ENQUEUED_TOKENS,
    wait=False,
    parallel=False,
    upload_concurrency=10,
    poll_interval=20,
    source=REQUESTS_NAME,
    tag="submit",
):
    work_dir = Path(work_dir)
    pending = pending_requests(work_dir, source=source)
    chunks = pack_token_chunks(pending, max_tokens=max_tokens)
    print(
        f"mode={tag} dir={work_dir} source={source} "
        f"max_tokens={max_tokens} pending={len(pending)} chunks={len(chunks)} "
        f"parallel={parallel}",
        flush=True,
    )
    if not chunks:
        print("  nothing to submit", flush=True)
        return []
    if not wait and not parallel:
        print("pass --wait for serial upload, or --parallel to upload all chunks at once", flush=True)
        return []

    chunk_idx = next_chunk_idx(work_dir)
    paths = []
    for i, chunk in enumerate(chunks):
        path = write_chunk(work_dir, chunk_idx, chunk)
        tok = chunk_token_total(chunk)
        print(
            f"chunk {i + 1}/{len(chunks)} jobs={len(chunk)} tokens={tok:,} -> {path.name}",
            flush=True,
        )
        paths.append(path)
        chunk_idx += 1

    if parallel:
        print(
            f"uploading {len(paths)} chunks with concurrency={upload_concurrency}",
            flush=True,
        )
        metas = submit_files_parallel(
            work_dir, paths, model, tag, concurrency=upload_concurrency
        )
        if wait:
            print(
                f"waiting for {len(metas)} batches (poll every {poll_interval}s)...",
                flush=True,
            )
            poll_all_until_done(work_dir, metas=metas, interval=poll_interval)
            collect_all(work_dir)
        else:
            print(
                f"submitted {len(metas)} batches; "
                f"run --mode poll --work-dir {work_dir} when ready",
                flush=True,
            )
        return metas

    for path in paths:
        submit_files(work_dir, [path], model, tag, wait=True)
    return []


def build_retry(
    retry_dir,
    model,
    submit,
    wait=False,
    parallel=False,
    upload_concurrency=10,
    poll_interval=20,
    max_tokens=DEFAULT_MAX_ENQUEUED_TOKENS,
):
    retry_dir = Path(retry_dir)
    print(f"mode=retry dir={retry_dir}", flush=True)
    collect_all(retry_dir)
    n = len(load_failures(retry_dir))
    if n == 0:
        print("no failures.jsonl rows to retry", flush=True)
        return
    print(f"  retry failures.jsonl jobs={n}", flush=True)
    if submit:
        submit_loop(
            retry_dir,
            model,
            max_tokens=max_tokens,
            wait=wait,
            parallel=parallel,
            upload_concurrency=upload_concurrency,
            poll_interval=poll_interval,
            source="failures.jsonl",
            tag="retry",
        )
    else:
        print("pass --submit to upload failures.jsonl", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        required=True,
        choices=["smoke", "10pct", "submit", "retry", "collect", "poll"],
    )
    ap.add_argument("--work-dir", default=None, help="smoke/ or sample10/ output directory")
    ap.add_argument("--dialogues", default=str(DIALOGUES_PATH))
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--openai-model", default="gpt-4o")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--wait", action="store_true", help="after --submit, poll then collect")
    ap.add_argument(
        "--parallel",
        action="store_true",
        help="upload all chunks concurrently, then poll/collect once (not one-at-a-time)",
    )
    ap.add_argument(
        "--upload-concurrency",
        type=int,
        default=10,
        help="max parallel file uploads when --parallel (default 10)",
    )
    ap.add_argument(
        "--poll-interval",
        type=int,
        default=20,
        help="seconds between batch status polls (default 20)",
    )
    ap.add_argument(
        "--max-enqueued-tokens",
        type=int,
        default=DEFAULT_MAX_ENQUEUED_TOKENS,
        help="close batch when adding the next job would exceed this input-token sum (default 85000)",
    )
    args = ap.parse_args()
    submit_kw = {
        "parallel": args.parallel,
        "upload_concurrency": args.upload_concurrency,
        "poll_interval": args.poll_interval,
    }

    if args.work_dir:
        work_dir = Path(args.work_dir)
    elif args.mode == "smoke":
        work_dir = Path(args.out) / "smoke"
    else:
        work_dir = Path(args.out) / "sample10"
    if args.mode == "collect":
        collect_all(work_dir)
        return
    if args.mode == "poll":
        poll_all_until_done(work_dir, interval=args.poll_interval)
        collect_all(work_dir)
        return
    if args.mode == "submit":
        submit_loop(
            work_dir,
            args.openai_model,
            max_tokens=args.max_enqueued_tokens,
            wait=args.wait,
            **submit_kw,
        )
        return
    if args.mode == "retry":
        build_retry(
            work_dir,
            args.openai_model,
            args.submit,
            wait=args.wait,
            max_tokens=args.max_enqueued_tokens,
            **submit_kw,
        )
        return

    all_dlg = v11.load_dialogues(Path(args.dialogues))
    by_id = {d["dialogue_id"]: d for d in all_dlg}
    system = v11.PROMPT_PATH.read_text()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "smoke":
        d = pick_smoke_dialogue(all_dlg, args.seed)
        sampled = [d]
        chosen = [d["dialogue_id"]]
        per_model = {d["target_model"]: chosen}
        out_dir = out_root / "smoke"
        print(
            f"mode=smoke dialogue={d['dialogue_id']} model={d['target_model']}",
            flush=True,
        )
    else:
        order = shuffled_ids(all_dlg, args.seed)
        per_model = slice_ids(order, 0.0, 0.10)
        chosen = [i for m in sorted(per_model) for i in per_model[m]]
        sampled = [by_id[i] for i in chosen]
        out_dir = out_root / "sample10"
        print(
            f"mode=10pct dialogues={len(sampled)}/{len(all_dlg)} models={len(per_model)}",
            flush=True,
        )
        for m, ids in per_model.items():
            print(f"  {len(ids):3}  {m}", flush=True)

    items = items_from_dialogues(sampled)
    print(f"jobs={len(items)} max_tokens={args.max_tokens} model={args.openai_model}", flush=True)

    write_manifest(out_dir / "sampled_dialogue_ids.json", {
        "mode": args.mode,
        "seed": args.seed,
        "frac": 0.10 if args.mode == "10pct" else None,
        "n_dialogues": len(chosen),
        "n_jobs": len(items),
        "per_model": {m: {"n": len(ids), "dialogue_ids": ids} for m, ids in per_model.items()},
        "dialogue_ids": chosen,
    })
    write_manifest(out_dir / "jobs.json", slim(items))
    write_requests(out_dir, items, system, args.openai_model, args.max_tokens)

    if args.submit:
        submit_loop(
            out_dir,
            args.openai_model,
            max_tokens=args.max_enqueued_tokens,
            wait=args.wait,
            **submit_kw,
        )
    else:
        print("wrote files only (pass --submit --parallel --wait to upload)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
