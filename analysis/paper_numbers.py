"""Reproduce every number that appears in the RELATE paper's results and
appendix, and nothing else.

This is a trimmed companion to analyze.py: that script builds ~35 tables and
10 figures for the full analysis dossier, most of which never made it into
the paper. This script builds only the tables actually cited in the draft,
in the order they appear, so a reader (or co-author) can check exactly where
each number in the LaTeX came from.

  python analysis/paper_numbers.py

Tables produced (name -> LaTeX label):
  1. table_models          -> tab:models              (main body)
  2. table_inward_delta    -> tab:inward-delta         (appendix)
  3. table_persona_if_delta-> tab:persona-if-delta     (appendix)
  4. table_high_stakes     -> tab:high-stakes          (appendix)
  5. table_criterion_turns -> tab:app-criterion        (appendix)
  6. table_mixed_moves     -> tab:mixed                (appendix)
  7. table_judge_agreement -> tab:judge-agreement       (appendix)
  8. table_judge_replication -> tab:judge-replication  (appendix)

Figures (fig_depth, fig_persona, fig_model_trajectories, fig_topic) are built
separately by analysis/make_paper_figures.py -- this script is numbers only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import statlib as S  # noqa: E402

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

DEEPSEEK = "deepseek-r1-distill-qwen-32b"
GPT4O = "gpt-4o"
IF_COLS = ["bond", "dyad", "inner", "hook"]
CRIT_LABEL = {"bond": "Bond Anchoring", "dyad": "Intimate Dyad",
              "inner": "Inner Life Claims", "hook": "Return Hooks", "os": "Outward-Scaffolding"}
MODEL_ORDER = ["Llama-3.1-8B", "Llama-3.1-70B", "Qwen3-14B", "Qwen3-32B",
               "DeepSeek-V3", "Claude-Haiku-4.5", "Mistral-7B"]
PERSONAS = ["P1", "P2", "P3"]
HIGH_STAKES = {"domestic-violence", "trauma", "substance-abuse",
               "eating-disorders", "depression"}


def load():
    df = pd.read_parquet(HERE / "data/sentences.parquet")
    ds = df[df.judge == DEEPSEEK].copy()
    ds["target_model"] = pd.Categorical(ds["target_model"], categories=MODEL_ORDER, ordered=True)
    return df, ds


def paired_deltas(sub, col, by=None):
    """Per-dialogue turn6 - turn1 share of sentences meeting `col`."""
    t = (sub[sub.turn_id.isin([1, 6])]
         .groupby(["dialogue_id", "turn_id"] + (by or []), observed=True)[col]
         .mean().unstack("turn_id"))
    t = t.dropna(subset=[1, 6])
    return (t[6] - t[1]).rename("delta").reset_index()


# ---------------------------------------------------------------- 1. tab:models
def table_models(ds):
    rows = []
    for m in MODEL_ORDER:
        sub = ds[ds.target_model == m]
        if_pct = 100 * sub.if_any.mean()
        os_pct = 100 * sub.os.mean()
        rows.append({"Model": m, "Dialogues": sub.dialogue_id.nunique(),
                     "Sentences": len(sub), "IF (%)": if_pct, "OS (%)": os_pct,
                     "Net": if_pct - os_pct})
    return pd.DataFrame(rows).round(1)


# ---------------------------------------------------------- 2. tab:inward-delta
def table_inward_delta(ds, n_boot=10_000):
    rows = []
    for m in ["ALL"] + MODEL_ORDER:
        sub = ds if m == "ALL" else ds[ds.target_model == m]
        row = {"Model": m, "Dialogues": sub.dialogue_id.nunique()}
        for col, name in [("if_any", "IF"), ("os", "OS")]:
            d = paired_deltas(sub, col)
            mu, lo, hi, _, _ = S.paired_cluster_boot(d["delta"].to_numpy() * 100, n_boot=n_boot)
            row[f"{name} delta"] = mu
            row[f"{name} CI"] = f"[{lo:.1f}, {hi:.1f}]"
        rows.append(row)
    return pd.DataFrame(rows).round(1)


# --------------------------------------------------- 3. tab:persona-if-delta
def table_persona_if_delta(ds, n_boot=10_000):
    rows = []
    for m in MODEL_ORDER:
        row = {"Model": m}
        for p_ in PERSONAS:
            sub = ds[(ds.target_model == m) & (ds.persona_id == p_)]
            d = paired_deltas(sub, "if_any")
            mu, _, _, _, _ = S.paired_cluster_boot(d["delta"].to_numpy() * 100, n_boot=n_boot)
            row[p_] = mu
        row["P1 - P3"] = row["P1"] - row["P3"]
        rows.append(row)
    out = pd.DataFrame(rows).round(1)
    print(f"  P3 lowest of the three in {(out[PERSONAS].idxmin(axis=1) == 'P3').sum()}/7 models")
    print(f"  monotone P1>P2>P3 in {((out.P1 > out.P2) & (out.P2 > out.P3)).sum()}/7 models")
    return out


# --------------------------------------------------------- 4. tab:high-stakes
def table_high_stakes(ds, n_boot=10_000):
    """Pre-specified high-stakes topics vs. the rest, pooled across models.
    Matches analyze.py module G exactly: one dialogue-clustered bootstrap
    resampling all dialogues jointly (not two independent group-wise
    resamples), via statlib.cluster_boot_diff.
    """
    hs = ds[ds.topic.isin(HIGH_STAKES)]
    rest = ds[~ds.topic.isin(HIGH_STAKES)]
    rows = []
    for col, name in [("if_any", "IF"), ("os", "OS")]:
        pooled = pd.concat([
            pd.DataFrame({"v": hs[col].astype(float), "g": 1, "d": hs.dialogue_id}),
            pd.DataFrame({"v": rest[col].astype(float), "g": 0, "d": rest.dialogue_id})]).dropna()
        d, lo, hi, p = S.cluster_boot_diff(
            (pooled.v * (pooled.g == 1)).to_numpy(), (pooled.g == 1).to_numpy().astype(float),
            (pooled.v * (pooled.g == 0)).to_numpy(), (pooled.g == 0).to_numpy().astype(float),
            pooled.d.to_numpy(), n_boot=n_boot)
        rows.append({"Dimension": name, "High-stakes (%)": 100 * hs[col].mean(),
                     "Other (%)": 100 * rest[col].mean(), "delta_pp": 100 * d,
                     "95% CI": f"[{100*lo:.1f}, {100*hi:.1f}]", "p_raw": p})
    out = pd.DataFrame(rows)
    out["p (Holm)"] = S.holm(out["p_raw"].values)
    out["delta_pp"] = out["delta_pp"].round(2)
    out[["High-stakes (%)", "Other (%)"]] = out[["High-stakes (%)", "Other (%)"]].round(1)
    return out[["Dimension", "High-stakes (%)", "Other (%)", "delta_pp", "95% CI", "p (Holm)"]]


# ------------------------------------------------------ 5. tab:app-criterion
def table_criterion_turns(ds):
    turns = range(1, 7)
    rows = []
    for c in IF_COLS + ["os"]:
        row = {"Criterion": CRIT_LABEL[c]}
        for t in turns:
            row[f"Turn {t}"] = 100 * ds.loc[ds.turn_id == t, c].mean()
        row["Pooled"] = 100 * ds[c].mean()
        rows.append(row)
    row = {"Criterion": "Inward-facing (any)"}
    for t in turns:
        row[f"Turn {t}"] = 100 * ds.loc[ds.turn_id == t, "if_any"].mean()
    row["Pooled"] = 100 * ds["if_any"].mean()
    rows.insert(4, row)  # after the four inward criteria, before OS
    return pd.DataFrame(rows).round(1)


# ---------------------------------------------------------- 6. tab:mixed
def table_mixed_moves(ds):
    rows = []
    for m in MODEL_ORDER:
        sub = ds[ds.target_model == m]
        ifb = sub.if_any.fillna(False); osb = sub.os.fillna(False)
        if_only = (ifb & ~osb).mean(); os_only = (~ifb & osb).mean()
        both = (ifb & osb).mean(); neither = (~ifb & ~osb).mean()
        hedged = (ifb & osb).sum() / max(osb.sum(), 1)
        rows.append({"Model": m, "IF only": 100 * if_only, "OS only": 100 * os_only,
                     "Both": 100 * both, "Neither": 100 * neither,
                     "Hedged (% of OS)": 100 * hedged})
    return pd.DataFrame(rows).round(1)


# ------------------------------------------------------ 7. tab:judge-agreement
def table_judge_agreement(df):
    ds = df[df.judge == DEEPSEEK].set_index(["dialogue_id", "turn_id", "sentence_id"])
    g4 = df[df.judge == GPT4O].set_index(["dialogue_id", "turn_id", "sentence_id"])
    common = ds.index.intersection(g4.index)
    a, b = ds.loc[common], g4.loc[common]

    rows = []
    for c in IF_COLS + ["os"]:
        mask = a[c].notna() & b[c].notna()
        x = a.loc[mask, c].astype(bool).to_numpy()
        y = b.loc[mask, c].astype(bool).to_numpy()
        _, _, p = S.mcnemar(x, y)
        rows.append({"Criterion": CRIT_LABEL[c], "n": int(mask.sum()),
                     "DS MET (%)": 100 * x.mean(), "GPT-4o MET (%)": 100 * y.mean(),
                     "Agreement (%)": 100 * (x == y).mean(),
                     "Cohen kappa": S.cohen_kappa(x, y), "Gwet AC1": S.gwet_ac1(x, y),
                     "McNemar p": p})
    mask = a["if_any"].notna() & b["if_any"].notna()
    x = a.loc[mask, "if_any"].astype(bool).to_numpy()
    y = b.loc[mask, "if_any"].astype(bool).to_numpy()
    _, _, p = S.mcnemar(x, y)
    rows.append({"Criterion": "Any inward (IF)", "n": int(mask.sum()),
                 "DS MET (%)": 100 * x.mean(), "GPT-4o MET (%)": 100 * y.mean(),
                 "Agreement (%)": 100 * (x == y).mean(),
                 "Cohen kappa": S.cohen_kappa(x, y), "Gwet AC1": S.gwet_ac1(x, y),
                 "McNemar p": p})
    out = pd.DataFrame(rows)
    out["McNemar p (Holm)"] = S.holm(out["McNemar p"].values)
    return out.round(3)


# ---------------------------------------------------- 8. tab:judge-replication
def table_judge_replication(df):
    ds = df[df.judge == DEEPSEEK].set_index(["dialogue_id", "turn_id", "sentence_id"])
    g4 = df[df.judge == GPT4O].set_index(["dialogue_id", "turn_id", "sentence_id"])
    common = ds.index.intersection(g4.index)
    sub_ids = g4.loc[common].index.get_level_values(0).unique()

    dsx = df[(df.judge == DEEPSEEK) & (df.dialogue_id.isin(sub_ids))]
    g4x = df[df.judge == GPT4O]

    repl = []
    for judge, frame in [("DeepSeek", dsx), ("GPT-4o", g4x)]:
        by_m = frame.groupby("target_model", observed=True)[["if_any", "os"]].mean().mul(100)
        by_m["net"] = by_m["if_any"] - by_m["os"]
        for m in MODEL_ORDER:
            if m in by_m.index:
                repl.append({"judge": judge, "level": "Model", "group": m,
                             "IF": by_m.loc[m, "if_any"], "OS": by_m.loc[m, "os"], "net": by_m.loc[m, "net"]})
        by_p = frame.groupby("persona_id", observed=True)[["if_any", "os"]].mean().mul(100)
        for p_ in PERSONAS:
            if p_ in by_p.index:
                repl.append({"judge": judge, "level": "Persona", "group": p_,
                             "IF": by_p.loc[p_, "if_any"], "OS": by_p.loc[p_, "os"],
                             "net": by_p.loc[p_, "if_any"] - by_p.loc[p_, "os"]})
        by_t = frame.groupby("turn_id")[["if_any", "os"]].mean().mul(100)
        for t in sorted(by_t.index):
            repl.append({"judge": judge, "level": "Turn", "group": str(t),
                         "IF": by_t.loc[t, "if_any"], "OS": by_t.loc[t, "os"],
                         "net": by_t.loc[t, "if_any"] - by_t.loc[t, "os"]})
    repl = pd.DataFrame(repl)

    rows = []
    for level in ["Model", "Persona", "Turn"]:
        sl = repl[repl.level == level]
        d_ = sl[sl.judge == "DeepSeek"].set_index("group")
        g_ = sl[sl.judge == "GPT-4o"].set_index("group")
        idx = d_.index.intersection(g_.index)
        for metric, label in [("IF", "Inward-facing"), ("OS", "Outward-scaffolding"), ("net", "Net orientation")]:
            r = sps.spearmanr(d_.loc[idx, metric], g_.loc[idx, metric])
            rows.append({"Level": level, "Metric": label, "n": len(idx),
                         "Spearman rho": round(float(r.statistic), 2), "p": round(float(r.pvalue), 3)})
    return pd.DataFrame(rows)


def main():
    df, ds = load()
    print(f"loaded {len(ds):,} DeepSeek-judged sentences, {len(df[df.judge==GPT4O]):,} GPT-4o sentences\n")

    print("=" * 70, "\n1. tab:models\n", "=" * 70, sep="")
    print(table_models(ds).to_string(index=False))

    print("\n" + "=" * 70, "\n2. tab:inward-delta\n", "=" * 70, sep="")
    print(table_inward_delta(ds).to_string(index=False))

    print("\n" + "=" * 70, "\n3. tab:persona-if-delta\n", "=" * 70, sep="")
    print(table_persona_if_delta(ds).to_string(index=False))

    print("\n" + "=" * 70, "\n4. tab:high-stakes\n", "=" * 70, sep="")
    print(table_high_stakes(ds).to_string(index=False))

    print("\n" + "=" * 70, "\n5. tab:app-criterion\n", "=" * 70, sep="")
    print(table_criterion_turns(ds).to_string(index=False))

    print("\n" + "=" * 70, "\n6. tab:mixed\n", "=" * 70, sep="")
    print(table_mixed_moves(ds).to_string(index=False))

    print("\n" + "=" * 70, "\n7. tab:judge-agreement\n", "=" * 70, sep="")
    print(table_judge_agreement(df).to_string(index=False))

    print("\n" + "=" * 70, "\n8. tab:judge-replication\n", "=" * 70, sep="")
    print(table_judge_replication(df).to_string(index=False))


if __name__ == "__main__":
    main()
