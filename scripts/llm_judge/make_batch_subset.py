"""Slice an existing batch dir (requests.jsonl + jobs.json) into a smaller test dir.

  python scripts/llm_judge/make_batch_subset.py \\
    --src outputs/pdf_prompt_v11_openai_batch/sample10 \\
    --dst outputs/pdf_prompt_v11_openai_batch/sample10_test \\
    --n-jobs 4
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_openai_batch as bob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    req = bob.consolidate_requests(src)
    lines = [l for l in req.read_text().splitlines() if l.strip()][: args.n_jobs]
    (dst / bob.REQUESTS_NAME).write_text("\n".join(lines) + ("\n" if lines else ""))
    cids = [json.loads(line)["custom_id"] for line in lines]
    print(f"  {bob.REQUESTS_NAME} -> {len(lines)} jobs", flush=True)

    jobs_path = src / "jobs.json"
    if jobs_path.exists() and cids:
        by = {j["custom_id"]: j for j in json.loads(jobs_path.read_text())}
        missing = [c for c in cids if c not in by]
        if missing:
            raise RuntimeError(f"{len(missing)} custom_ids missing from jobs.json, e.g. {missing[0]}")
        (dst / "jobs.json").write_text(
            json.dumps([by[c] for c in cids], indent=2, ensure_ascii=False) + "\n"
        )

    if (src / "sampled_dialogue_ids.json").exists():
        shutil.copy(src / "sampled_dialogue_ids.json", dst / "sampled_dialogue_ids.json")

    meta = {
        "subset_of": str(src),
        "n_jobs": len(cids),
        "custom_ids": cids,
    }
    (dst / "subset_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {dst} n_jobs={len(cids)}", flush=True)


if __name__ == "__main__":
    main()
