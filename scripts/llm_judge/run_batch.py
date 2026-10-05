"""Submit the v11 judge through OpenAI / Gemini Batch APIs.

Same prompt, same one-(sentence, criterion) job as run_sample.py.
Only the transport changes: upload JSONL, wait, download.

  # small smoke test (25 jobs = first 5 sentences x 5 classes)
  python scripts/llm_judge/run_batch.py \
    --provider both --dialogues test1.json --sample-frac 1.0 --limit 25 \
    --out outputs/pdf_prompt_v11_batch_smoke

  # collect later if you submitted without --wait
  python scripts/llm_judge/run_batch.py \
    --collect --out outputs/pdf_prompt_v11_batch_smoke
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sample as v11

ROOT = v11.ROOT
PROMPT_PATH = v11.PROMPT_PATH
CLASSES = v11.CLASSES
DONE_STATES_OAI = {"completed", "failed", "cancelled", "expired"}
DONE_STATES_GEM = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def custom_id(job, criterion, idx):
    raw = f"{job['dialogue_id']}|{job['turn_id']}|{job['sentence_id']}|{criterion}"
    return raw if len(raw) <= 64 else f"j{idx:06d}"


def build_work(args):
    system = PROMPT_PATH.read_text()
    dialogues_path = Path(args.dialogues)
    all_dlg = v11.load_dialogues(dialogues_path)
    sampled, chosen_ids = v11.sample_dialogues(all_dlg, args.sample_frac, args.sample_seed)
    jobs = v11.load_jobs(sampled)
    work = []
    for job in jobs:
        for c in CLASSES:
            work.append((job, c))
    work.sort(key=lambda jc: (jc[0]["dialogue_id"], int(jc[0]["turn_id"]), int(jc[0]["sentence_id"]), jc[1]))
    full = len(work)
    if args.limit is not None and args.limit < len(work):
        work = work[: args.limit]
    items = []
    for i, (job, c) in enumerate(work):
        cid = custom_id(job, c, i)
        items.append({
            "custom_id": cid,
            "dialogue_id": job["dialogue_id"],
            "turn_id": str(job["turn_id"]),
            "sentence_id": str(job["sentence_id"]),
            "class": c,
            "target": job["target"],
            "user": v11.build_user(job, c),
        })
    return system, sampled, chosen_ids, items, full


def write_openai_jsonl(path, items, system, model, max_tokens):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps({
                "custom_id": it["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": it["user"]},
                    ],
                },
            }, ensure_ascii=False) + "\n")


def write_gemini_jsonl(path, items, system, thinking_level, max_tokens):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps({
                "key": it["custom_id"],
                "request": {
                    "system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": it["user"]}]}],
                    "generation_config": {
                        "max_output_tokens": max_tokens,
                        "thinking_config": {"thinking_level": thinking_level},
                    },
                },
            }, ensure_ascii=False) + "\n")


def gemini_inline_requests(items, system, thinking_level, max_tokens):
    reqs = []
    for it in items:
        reqs.append({
            "contents": [{"role": "user", "parts": [{"text": it["user"]}]}],
            "metadata": {"key": it["custom_id"]},
            "config": {
                "system_instruction": system,
                "max_output_tokens": max_tokens,
                "thinking_config": {"thinking_level": thinking_level},
            },
        })
    return reqs


def submit_openai(out_dir, items, system, model, max_tokens):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    req_path = out_dir / "requests.jsonl"
    write_openai_jsonl(req_path, items, system, model, max_tokens)
    client = OpenAI(api_key=key)
    uploaded = client.files.create(file=req_path.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"v11 smoke {out_dir.name}"},
    )
    meta = {
        "provider": "openai",
        "model": model,
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "n": len(items),
        "requests_path": str(req_path),
    }
    (out_dir / "batch_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"openai submitted {batch.id} n={len(items)} file={uploaded.id}", flush=True)
    return meta


def submit_gemini(out_dir, items, system, model, thinking_level, max_tokens):
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY not set")
    from google import genai

    req_path = out_dir / "requests.jsonl"
    write_gemini_jsonl(req_path, items, system, thinking_level, max_tokens)
    client = genai.Client(api_key=key)
    # small jobs: inline (avoids File API mime issues). larger: JSONL upload.
    if len(items) <= 80:
        src = gemini_inline_requests(items, system, thinking_level, max_tokens)
        src_kind = "inline"
    else:
        uploaded = client.files.upload(
            file=str(req_path),
            config={"display_name": out_dir.name, "mime_type": "application/json"},
        )
        src = uploaded.name
        src_kind = "file"
    job = client.batches.create(
        model=model,
        src=src,
        config={"display_name": f"v11-{out_dir.name}"},
    )
    meta = {
        "provider": "gemini",
        "model": model,
        "thinking_level": thinking_level,
        "batch_id": job.name,
        "src_kind": src_kind,
        "n": len(items),
        "requests_path": str(req_path),
    }
    (out_dir / "batch_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"gemini submitted {job.name} n={len(items)} src={src_kind}", flush=True)
    return meta


def poll_openai(meta, interval):
    client = OpenAI()
    bid = meta["batch_id"]
    while True:
        b = client.batches.retrieve(bid)
        counts = b.request_counts
        print(
            f"openai {bid} status={b.status} "
            f"completed={getattr(counts, 'completed', '?')}/"
            f"{getattr(counts, 'total', '?')} failed={getattr(counts, 'failed', '?')}",
            flush=True,
        )
        if b.status in DONE_STATES_OAI:
            return b
        time.sleep(interval)


def poll_gemini(meta, interval):
    from google import genai

    client = genai.Client()
    name = meta["batch_id"]
    while True:
        job = client.batches.get(name=name)
        state = job.state.name if job.state else "?"
        print(f"gemini {name} state={state}", flush=True)
        if state in DONE_STATES_GEM:
            return job
        time.sleep(interval)


def row_from_text(it, raw):
    parsed, _ = v11.extract_json_obj(raw)
    row = parsed or {
        "dialogue_id": it["dialogue_id"],
        "turn_id": it["turn_id"],
        "Role": "assistant",
        "sentence_id": it["sentence_id"],
        "class": it["class"],
        "label": "PARSE_FAIL",
        "linguistic marker": "",
        "reason": (raw or "")[:300],
    }
    row["_target_sentence"] = it["target"]
    row["class"] = it["class"]
    row["dialogue_id"] = it["dialogue_id"]
    row["turn_id"] = str(it["turn_id"])
    row["sentence_id"] = str(it["sentence_id"])
    return row


def collect_openai(out_dir, items):
    meta = json.loads((out_dir / "batch_meta.json").read_text())
    client = OpenAI()
    b = client.batches.retrieve(meta["batch_id"])
    if b.status != "completed":
        raise RuntimeError(f"openai batch not completed: {b.status}")
    by_id = {it["custom_id"]: it for it in items}
    raw_path = out_dir / "results.raw.jsonl"
    text = client.files.content(b.output_file_id).text
    raw_path.write_text(text)
    if b.error_file_id:
        (out_dir / "errors.raw.jsonl").write_text(client.files.content(b.error_file_id).text)
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        it = by_id.get(rec.get("custom_id"))
        if it is None:
            continue
        err = rec.get("error")
        body = (rec.get("response") or {}).get("body") or {}
        if err or not body:
            raw = json.dumps(err or rec)[:300]
        else:
            raw = (((body.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
        row = row_from_text(it, raw)
        if body.get("usage"):
            row["_usage"] = body["usage"]
        rows.append(row)
    return rows, b.status


def _gemini_part_text(resp_obj):
    if resp_obj is None:
        return ""
    if hasattr(resp_obj, "text") and resp_obj.text:
        return resp_obj.text
    if isinstance(resp_obj, dict):
        cands = resp_obj.get("candidates") or []
        texts = []
        for c in cands:
            parts = ((c.get("content") or {}).get("parts")) or []
            for p in parts:
                if p.get("thought"):
                    continue
                if p.get("text"):
                    texts.append(p["text"])
        return "".join(texts)
    cands = getattr(resp_obj, "candidates", None) or []
    texts = []
    for c in cands:
        content = getattr(c, "content", None)
        parts = getattr(content, "parts", None) or []
        for p in parts:
            if getattr(p, "thought", False):
                continue
            t = getattr(p, "text", None)
            if t:
                texts.append(t)
    return "".join(texts)


def collect_gemini(out_dir, items):
    from google import genai

    meta = json.loads((out_dir / "batch_meta.json").read_text())
    client = genai.Client()
    job = client.batches.get(name=meta["batch_id"])
    state = job.state.name if job.state else "?"
    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"gemini batch not succeeded: {state} error={getattr(job, 'error', None)}")
    by_id = {it["custom_id"]: it for it in items}
    rows = []
    dest = job.dest
    records = []
    if dest and getattr(dest, "file_name", None):
        blob = client.files.download(file=dest.file_name)
        text = blob.decode("utf-8") if isinstance(blob, (bytes, bytearray)) else str(blob)
        (out_dir / "results.raw.jsonl").write_text(text)
        for line in text.splitlines():
            if line.strip():
                records.append(json.loads(line))
    elif dest and getattr(dest, "inlined_responses", None):
        raw_dump = []
        for i, ir in enumerate(dest.inlined_responses):
            key = None
            md = getattr(ir, "metadata", None)
            if isinstance(md, dict):
                key = md.get("key")
            elif md is not None:
                key = getattr(md, "key", None)
            if key is None and i < len(items):
                key = items[i]["custom_id"]
            rec = {"key": key, "error": None, "response": None}
            if getattr(ir, "error", None):
                rec["error"] = str(ir.error)
            rec["_text"] = _gemini_part_text(getattr(ir, "response", None))
            raw_dump.append(rec)
            records.append(rec)
        (out_dir / "results.raw.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in raw_dump) + "\n"
        )
    else:
        raise RuntimeError("gemini batch succeeded but dest has no file or inline responses")

    for rec in records:
        key = rec.get("key") or rec.get("custom_id")
        it = by_id.get(key)
        if it is None:
            continue
        if rec.get("error") and not rec.get("response") and not rec.get("_text"):
            raw = str(rec.get("error"))[:300]
        else:
            raw = rec.get("_text") or _gemini_part_text(rec.get("response"))
        rows.append(row_from_text(it, raw))
    return rows, state


def write_judgments(out_dir, rows):
    path = out_dir / "judgments.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    ok = sum(1 for r in rows if r.get("label") not in (None, "PARSE_FAIL"))
    fail = len(rows) - ok
    met = sum(1 for r in rows if r.get("label") == "MET")
    print(f"wrote {path} n={len(rows)} ok={ok} parse_fail={fail} met={met}", flush=True)
    return path


def load_items(out_dir):
    return json.loads((out_dir / "jobs.json").read_text())


def providers_from_args(args):
    if args.provider == "both":
        return ["openai", "gemini"]
    return [args.provider]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", choices=["openai", "gemini", "both"], default="both")
    ap.add_argument("--dialogues", default=str(ROOT / "test1.json"))
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--sample-seed", type=int, default=11)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--openai-model", default="gpt-4o")
    ap.add_argument("--gemini-model", default="gemini-3.8-flash")
    ap.add_argument("--thinking-level", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--poll-interval", type=int, default=20)
    ap.add_argument("--wait", action="store_true", default=True)
    ap.add_argument("--no-wait", action="store_false", dest="wait")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    providers = providers_from_args(args)

    if args.collect:
        system, sampled, chosen_ids, items, full = None, None, None, None, None
        # prefer saved jobs.json per provider
        for p in providers:
            pdir = out_root / p
            items = load_items(pdir)
            print(f"collecting {p} n={len(items)}", flush=True)
            if p == "openai":
                rows, status = collect_openai(pdir, items)
            else:
                rows, status = collect_gemini(pdir, items)
            write_judgments(pdir, rows)
            print(f"{p} collect status={status}", flush=True)
        return

    system, sampled, chosen_ids, items, full = build_work(args)
    print(
        f"prompt={PROMPT_PATH} dialogues={len(sampled)} jobs={len(items)}/{full} "
        f"providers={providers} thinking={args.thinking_level}",
        flush=True,
    )

    metas = {}
    for p in providers:
        pdir = out_root / p
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "jobs.json").write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n")
        (pdir / "sampled_dialogue_ids.json").write_text(
            json.dumps(
                {
                    "seed": args.sample_seed,
                    "frac": args.sample_frac,
                    "n_dialogues": len(chosen_ids),
                    "n_jobs": len(items),
                    "n_jobs_full": full,
                    "dialogue_ids": chosen_ids,
                },
                indent=2,
            )
            + "\n"
        )
        try:
            if p == "openai":
                metas[p] = submit_openai(pdir, items, system, args.openai_model, args.max_tokens)
            else:
                metas[p] = submit_gemini(
                    pdir, items, system, args.gemini_model, args.thinking_level, args.max_tokens
                )
        except RuntimeError as e:
            print(f"skip {p}: {e}", flush=True)

    if not metas:
        print("nothing submitted", flush=True)
        sys.exit(1)

    if not args.wait:
        print("submitted; rerun with --collect after the batches finish", flush=True)
        return

    for p in providers:
        if p not in metas:
            continue
        pdir = out_root / p
        print(f"polling {p}...", flush=True)
        if p == "openai":
            b = poll_openai(metas[p], args.poll_interval)
            if b.status != "completed":
                print(f"openai finished with status={b.status}", flush=True)
                continue
            rows, status = collect_openai(pdir, items)
        else:
            job = poll_gemini(metas[p], args.poll_interval)
            state = job.state.name if job.state else "?"
            if state != "JOB_STATE_SUCCEEDED":
                print(f"gemini finished with state={state} error={getattr(job, 'error', None)}", flush=True)
                continue
            rows, status = collect_gemini(pdir, items)
        write_judgments(pdir, rows)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
