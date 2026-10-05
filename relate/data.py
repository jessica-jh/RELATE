"""
Load situation seeds from CounselBench-Eval (HuggingFace), with a local JSONL cache.

Reproducible rebuild (sorted by question_id, pinned HF revision):
    python scripts/build_situations.py --force

Config keys (under data:):
  hf_revision    pin dataset snapshot on HuggingFace
  force_reload   if true, ignore cache and re-download
  dedupe_key     questionID (default) | questionText
  question_field column for seed text (default questionText)
  exclude_topics / include_topics
  n_situations   cap after filter; -1 = all
"""

import json
import os
import re
from datetime import datetime, timezone

from .util import read_jsonl, situations_path, write_jsonl

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def _clean_text(text: str) -> str:
    return _BR_RE.sub(" ", text).strip()


def _meta_path(cache: str) -> str:
    base, _ = os.path.splitext(cache)
    return f"{base}.meta.json"


def _passes_topic_filter(topic: str, dcfg: dict) -> bool:
    include = dcfg.get("include_topics")
    if include:
        return topic in include
    exclude = set(dcfg.get("exclude_topics") or [])
    return topic not in exclude


def _passes_keyword_filter(text: str, dcfg: dict) -> bool:
    """Drop situations whose content centers on an existing counseling/
    therapy relationship (independent of topic tag — CounselBench topic
    labels miss this, e.g. a question tagged "anxiety" that's actually
    about the user's own therapist)."""
    keywords = dcfg.get("exclude_keywords")
    if not keywords:
        return True
    pattern = r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b"
    return not re.search(pattern, text, re.IGNORECASE)


def _fetch_situations_from_hf(cfg: dict) -> list[dict]:
    """Download CounselBench-Eval and build deterministic situation rows."""
    from datasets import load_dataset

    dcfg = cfg["data"]
    kwargs: dict = {}
    if rev := dcfg.get("hf_revision"):
        kwargs["revision"] = rev

    ds = load_dataset(
        dcfg["hf_dataset"],
        split=dcfg["hf_split"],
        **kwargs,
    )
    field = dcfg.get("question_field", "questionText")
    dedupe_key = dcfg.get("dedupe_key", "questionID")

    seen: dict[str, dict] = {}
    for row in ds:
        topic = (row.get("topic") or "").strip()
        if not _passes_topic_filter(topic, dcfg):
            continue

        key = (row.get(dedupe_key) or row.get("questionID") or "").strip()
        q = _clean_text(row.get(field) or "")
        title = _clean_text(row.get("questionTitle") or "")
        if not q:
            q = title
        if not key or not q:
            continue
        if key in seen:
            continue
        if not _passes_keyword_filter(f"{title} {q}", dcfg):
            continue

        seen[key] = {
            "question_id": key,
            "topic": topic,
            "question_title": _clean_text(row.get("questionTitle") or ""),
            "seed_question": q,
        }

    rows = sorted(seen.values(), key=lambda r: r["question_id"])
    for i, row in enumerate(rows):
        row["situation_id"] = f"S{i:03d}"

    n = dcfg.get("n_situations", -1)
    return rows if n < 0 else rows[:n]


def _write_situations_cache(cache: str, rows: list[dict], cfg: dict) -> None:
    write_jsonl(cache, rows)
    dcfg = cfg["data"]
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "hf_dataset": dcfg["hf_dataset"],
        "hf_revision": dcfg.get("hf_revision"),
        "hf_split": dcfg["hf_split"],
        "dedupe_key": dcfg.get("dedupe_key", "questionID"),
        "question_field": dcfg.get("question_field", "questionText"),
        "exclude_topics": dcfg.get("exclude_topics") or [],
        "include_topics": dcfg.get("include_topics"),
        "exclude_keywords": dcfg.get("exclude_keywords") or [],
        "n_situations": dcfg.get("n_situations", -1),
        "situation_id_order": "sorted_by_question_id",
        "count": len(rows),
    }
    with open(_meta_path(cache), "w") as f:
        json.dump(meta, f, indent=2)


def load_situations(cfg: dict, *, force_reload: bool = False) -> list[dict]:
    """Return situation rows. Uses cache unless force_reload is set."""
    dcfg = cfg["data"]
    cache = situations_path(cfg)
    reload = force_reload or bool(dcfg.get("force_reload"))

    if os.path.exists(cache) and not reload:
        rows = read_jsonl(cache)
        n = dcfg.get("n_situations", -1)
        return rows if n < 0 else rows[:n]

    rows = _fetch_situations_from_hf(cfg)
    _write_situations_cache(cache, rows, cfg)
    return rows
