import json
import os

import yaml


def load_env() -> None:
    """API keys come from the process environment. No repo env file is read."""
    return


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_dir(cfg: dict) -> str:
    d = os.path.join(cfg["run"]["output_dir"], cfg["run"]["name"])
    os.makedirs(d, exist_ok=True)
    return d


def situations_path(cfg: dict) -> str:
    """Frozen situations cache (corpus build)."""
    if path := cfg.get("corpus", {}).get("situations_file"):
        return path
    return cfg["data"]["situations_cache"]


def stimuli_path(cfg: dict) -> str:
    """Frozen stimulus corpus (vignette + fixed_turn1)."""
    if path := cfg.get("corpus", {}).get("stimuli_file"):
        return path
    return os.path.join(run_dir(cfg), "stimuli.jsonl")


def filter_stimuli(stimuli: list[dict], cfg: dict) -> list[dict]:
    """Subset frozen corpus for pilot runs.

    Prefer experiment.situation_ids if set; else first N by situation_id
    via experiment.n_situations.
    """
    ecfg = cfg.get("experiment", {})
    if explicit := ecfg.get("situation_ids"):
        allowed = set(explicit)
        return [s for s in stimuli if s["situation_id"] in allowed]

    n = ecfg.get("n_situations")
    if n is None or n < 0:
        return stimuli
    sids = sorted({s["situation_id"] for s in stimuli})[:n]
    allowed = set(sids)
    return [s for s in stimuli if s["situation_id"] in allowed]


_RESERVED_MODEL_KEYS = {"family", "scale", "provider", "model"}


def _models_for_stage(cfg: dict, stage: str) -> dict[tuple[str, str], dict]:
    """Collect (provider, model) -> extra kwargs (e.g. base_url, extra_body)
    a stage will actually call."""
    m = cfg.get("models", {})
    pairs: dict[tuple[str, str], dict] = {}

    def add(entry):
        if entry and entry.get("provider") and entry.get("model"):
            extra = {k: v for k, v in entry.items() if k not in _RESERVED_MODEL_KEYS}
            pairs[(entry["provider"], entry["model"])] = extra

    if stage == "stimulus":
        add(m.get("stimulus_writer") or m.get("user_agent"))
    elif stage == "dialogue":
        add(m.get("user_agent"))
        for t in m.get("targets", []):
            add(t)
    elif stage == "judge":
        add(m.get("judge"))
        add(m.get("validation_judge"))
    return pairs


def preflight_check(cfg: dict, stages: list[str]) -> None:
    """One trivial call per (provider, model) before the real run."""
    from .providers import get_provider

    pairs: dict[tuple[str, str], dict] = {}
    for s in stages:
        pairs.update(_models_for_stage(cfg, s))
    pairs = {k: v for k, v in pairs.items() if k[0] != "mock"}
    if not pairs:
        return

    print(f"[preflight] checking {len(pairs)} provider/model pair(s)...")
    failures = []
    for provider, model in sorted(pairs):
        extra = pairs[(provider, model)]
        try:
            reply = get_provider(provider, model, **extra).chat(
                [{"role": "user", "content": "Reply with the single word OK."}],
                temperature=0.0, max_tokens=5,
            )
            print(f"  ok   {provider:<10} {model:<50} -> {reply[:20]!r}")
        except Exception as e:
            print(f"  FAIL {provider:<10} {model:<50} -> {e}")
            failures.append((provider, model, str(e)))

    if failures:
        lines = "\n".join(f"  - {p}/{m}: {err}" for p, m, err in failures)
        raise RuntimeError(
            f"[preflight] {len(failures)} model(s) unreachable — fix before running "
            f"the pipeline (nothing billed for the failing stage yet):\n{lines}"
        )


def stages_for_config(cfg: dict, stage: str) -> list[str]:
    """Return runnable stages for this config type (corpus vs experiment)."""
    available: list[str] = []
    if cfg.get("data") and cfg.get("personas"):
        available.append("stimulus")
    if cfg.get("models", {}).get("targets"):
        available.extend(["dialogue", "judge", "aggregate", "report"])
    if not available:
        raise ValueError("Config defines no runnable stages.")
    if stage == "all":
        return available
    if stage not in available:
        raise ValueError(
            f"Stage {stage!r} not available for this config. "
            f"Choose from: {available}"
        )
    return [stage]


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str, row: dict) -> None:
    """Append one row and fsync immediately (crash-safe for long runs)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
