"""RELATE results analysis: modules A-I.

  python analysis/analyze.py                 # everything
  python analysis/analyze.py --modules A B C # a subset
  python analysis/analyze.py --fast          # 2,000 bootstrap resamples, skip GLMM

Writes analysis/tables/*.{csv,tex}, analysis/figures/*.{pdf,png}, analysis/results.json.
"""
import argparse
import gzip
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import statlib as S  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

TABLES = HERE / "tables"
FIGURES = HERE / "figures"
DATA = HERE / "data"
for d in (TABLES, FIGURES, DATA):
    d.mkdir(parents=True, exist_ok=True)

IF_COLS = ["bond", "dyad", "inner", "hook"]
CRIT_LABEL = {"bond": "Bond Anchoring", "dyad": "Intimate Dyad",
              "inner": "Inner Life Claims", "hook": "Return Hooks",
              "os": "Outward-Scaffolding"}
MODEL_ORDER = ["Llama-3.1-8B", "Llama-3.1-70B", "Mistral-7B", "Qwen3-14B",
               "Qwen3-32B", "DeepSeek-V3", "Claude-Haiku-4.5"]
PERSONA_LABEL = {"P1": "P1 hesitant / low-disclosure",
                 "P2": "P2 reflective / self-minimizing",
                 "P3": "P3 high-disclosure / reassurance-seeking"}
DEEPSEEK = "deepseek-r1-distill-qwen-32b"

# colour-blind-safe qualitative palette, one colour per target model
MODEL_COLOR = dict(zip(MODEL_ORDER, ["#4C72B0", "#0F4C81", "#DD8452", "#55A868",
                                     "#1B7B4C", "#C44E52", "#8172B3"]))
IF_COLOR, OS_COLOR = "#C44E52", "#3B7EA1"
CRIT_COLOR = {"bond": "#8172B3", "dyad": "#DD8452", "inner": "#55A868",
              "hook": "#C44E52", "os": "#3B7EA1"}
PERSONA_COLOR = {"P1": "#8FBBD9", "P2": "#4C72B0", "P3": "#1F3D63"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})

RESULTS = {}
NBOOT = S.N_BOOT


def save_fig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print(f"    figure: figures/{name}.pdf")


def save_table(df, name, caption, label, float_fmt="%.1f", note=None, col_fmt=None):
    df.to_csv(TABLES / f"{name}.csv", index=False)
    S.latex_table(df, TABLES / f"{name}.tex", caption, label,
                  float_fmt=float_fmt, note=note, col_fmt=col_fmt)
    print(f"    table:  tables/{name}.csv + .tex")


def load():
    path = DATA / "sentences.parquet"
    df = pd.read_parquet(path) if path.exists() else pd.read_csv(DATA / "sentences.csv.gz")
    df["target_model"] = pd.Categorical(df["target_model"], categories=MODEL_ORDER, ordered=True)
    df["persona_id"] = pd.Categorical(df["persona_id"], categories=["P1", "P2", "P3"], ordered=True)
    return df


def rate_ci(sub, col, cluster="dialogue_id", n_boot=None):
    """Point estimate and dialogue-clustered CI for a MET rate, in percent."""
    v = sub[col]
    mask = v.notna()
    num = v[mask].astype(float).to_numpy()
    den = np.ones_like(num)
    p, lo, hi = S.cluster_boot(num, den, sub.loc[mask, cluster].to_numpy(),
                               n_boot=n_boot or NBOOT)
    return 100 * p, 100 * lo, 100 * hi


def net_ci(sub, cluster="dialogue_id", n_boot=None):
    """Point estimate and CI for IF_any - OS on the same resamples."""
    mask = sub["if_any"].notna() & sub["os"].notna()
    s = sub[mask]
    d, lo, hi, p = S.cluster_boot_diff(
        s["if_any"].astype(float).to_numpy(), np.ones(len(s)),
        s["os"].astype(float).to_numpy(), np.ones(len(s)),
        s[cluster].to_numpy(), n_boot=n_boot or NBOOT)
    return 100 * d, 100 * lo, 100 * hi, p


# ---------------------------------------------------------------- module A
def module_a(df):
    print("\n[A] Coverage and data quality")
    ds = df[df.judge == DEEPSEEK]
    cov = pd.read_csv(DATA / "coverage.csv")

    wide = cov.pivot(index="target_model", columns="persona_id", values="dialogues")
    wide = wide.reindex(MODEL_ORDER)
    wide["Total"] = wide.sum(axis=1)
    wide["Expected"] = 228
    wide["Coverage %"] = 100 * wide["Total"] / 228
    sent = ds.groupby("target_model", observed=True).size().reindex(MODEL_ORDER)
    wide["Sentences"] = sent.values
    pf = ds.groupby("target_model", observed=True)["parse_fail"].mean().reindex(MODEL_ORDER)
    wide["Parse-fail %"] = (100 * pf).values
    wide = wide.reset_index().rename(columns={"target_model": "Model"})
    save_table(wide, "a1_coverage",
               "Judge coverage by target model and persona. Expected is $76$ situations "
               "per persona cell. Parse-fail is the share of sentences for which at least "
               "one of the five criterion calls did not return parseable JSON; these are "
               "treated as missing, not as NOT\\_MET.",
               "tab:coverage", float_fmt="%.1f")

    # which situations are incomplete, and are they systematic?
    per_sit = ds.groupby("situation_id")["dialogue_id"].nunique()
    incomplete = per_sit[per_sit < 21].sort_values()
    topics = ds.drop_duplicates("situation_id").set_index("situation_id")["topic"]
    inc_tbl = pd.DataFrame({
        "Situation": incomplete.index,
        "Topic": [topics.get(s, "?") for s in incomplete.index],
        "Dialogues judged": incomplete.values,
        "Expected": 21,
    })
    save_table(inc_tbl, "a2_incomplete_situations",
               "The seven situations with incomplete judge coverage. "
               "Missingness is balanced across target models (11--12 dialogues each) "
               "and personas (26/26/30).",
               "tab:incomplete", float_fmt="%.0f")

    # response length confound: models differ in sentences per turn
    spt = (ds.groupby(["target_model", "dialogue_id", "turn_id"], observed=True)
             .size().reset_index(name="n"))
    length = (spt.groupby("target_model", observed=True)["n"]
                .agg(["mean", "std", "median"]).reindex(MODEL_ORDER).reset_index())
    length.columns = ["Model", "Sentences/turn (mean)", "SD", "Median"]
    save_table(length, "a3_sentences_per_turn",
               "Assistant sentences per turn by target model. Models differ substantially "
               "in verbosity under the shared 6--8 sentence instruction, so all rates in "
               "this paper are shares of sentences rather than counts.",
               "tab:spt", float_fmt="%.2f")

    # parse-fail sensitivity: does treating PARSE_FAIL as NOT_MET move anything?
    alt = ds.copy()
    for c in IF_COLS + ["os"]:
        alt[c] = alt[c].fillna(False)
    alt["if_any"] = alt[IF_COLS].any(axis=1)
    sens = pd.DataFrame({
        "Metric": ["IF rate", "OS rate", "Net orientation"],
        "Parse-fail as missing": [100 * ds.if_any.mean(), 100 * ds.os.mean(),
                                  100 * (ds.if_any.mean() - ds.os.mean())],
        "Parse-fail as NOT_MET": [100 * alt.if_any.mean(), 100 * alt.os.mean(),
                                  100 * (alt.if_any.mean() - alt.os.mean())],
    })
    sens["Difference (pp)"] = sens.iloc[:, 1] - sens.iloc[:, 2]
    save_table(sens, "a4_parsefail_sensitivity",
               "Sensitivity of the headline rates to the treatment of parse failures "
               "(0.14\\% of sentences).", "tab:parsefail", float_fmt="%.2f")

    # turn-level saturation, the reason sentence level is the unit of analysis
    turn = (ds.groupby(["target_model", "dialogue_id", "turn_id"], observed=True)
              .agg(if_any=("if_any", "any"), os=("os", "any")).reset_index())
    sat = (turn.groupby("target_model", observed=True)[["if_any", "os"]]
               .mean().mul(100).reindex(MODEL_ORDER).reset_index())
    sat.columns = ["Model", "Turns with any IF (%)", "Turns with any OS (%)"]
    sent_rate = (ds.groupby("target_model", observed=True)[["if_any", "os"]]
                   .mean().mul(100).reindex(MODEL_ORDER))
    sat["Sentences IF (%)"] = sent_rate["if_any"].values
    sat["Sentences OS (%)"] = sent_rate["os"].values
    save_table(sat, "a5_turn_level_saturation",
               "Turn-level presence saturates. Five of seven models emit at least one "
               "inward-facing sentence in more than 99\\% of turns, so a binary turn-level "
               "measure cannot separate them. Sentence-level shares do.",
               "tab:saturation", float_fmt="%.1f")

    RESULTS["A"] = {
        "dialogues_judged": int(ds.dialogue_id.nunique()),
        "dialogues_expected": 1596,
        "coverage_pct": round(100 * ds.dialogue_id.nunique() / 1596, 2),
        "sentences": int(len(ds)),
        "judgments": int(len(ds) * 5),
        "parse_fail_pct": round(100 * ds.parse_fail.mean(), 3),
        "incomplete_situations": incomplete.index.tolist(),
        "sentences_per_turn_range": [round(length["Sentences/turn (mean)"].min(), 2),
                                     round(length["Sentences/turn (mean)"].max(), 2)],
        "turn_level_if_max": round(sat["Turns with any IF (%)"].max(), 1),
    }
    print(f"    {ds.dialogue_id.nunique()}/1596 dialogues, {len(ds):,} sentences, "
          f"{100 * ds.parse_fail.mean():.2f}% parse-fail")


# ---------------------------------------------------------------- module B
def module_b(df):
    print("\n[B] Judge validation (paper 5.2 / 5.3)")
    ds = df[df.judge == DEEPSEEK].set_index(["dialogue_id", "turn_id", "sentence_id"])
    g4 = df[df.judge == "gpt-4o"].set_index(["dialogue_id", "turn_id", "sentence_id"])
    common = ds.index.intersection(g4.index)
    a, b = ds.loc[common], g4.loc[common]

    # sentence text must be byte-identical or the keys do not refer to the same unit
    text_match = (a["sentence_text"].fillna("") == b["sentence_text"].fillna("")).mean()

    rows = []
    for c in IF_COLS + ["os"]:
        mask = a[c].notna() & b[c].notna()
        x = a.loc[mask, c].astype(bool).to_numpy()   # DeepSeek
        y = b.loc[mask, c].astype(bool).to_numpy()   # GPT-4o
        b01, b10, p = S.mcnemar(x, y)
        rows.append({
            "Criterion": CRIT_LABEL[c], "n": int(mask.sum()),
            "DeepSeek MET %": 100 * x.mean(), "GPT-4o MET %": 100 * y.mean(),
            "Agreement %": 100 * (x == y).mean(),
            "Cohen $\\kappa$": S.cohen_kappa(x, y), "Gwet AC1": S.gwet_ac1(x, y),
            "DS-only MET": b01, "G4-only MET": b10, "McNemar p": p,
        })
    # IF_any as an aggregate
    mask = a["if_any"].notna() & b["if_any"].notna()
    x = a.loc[mask, "if_any"].astype(bool).to_numpy()
    y = b.loc[mask, "if_any"].astype(bool).to_numpy()
    b01, b10, p = S.mcnemar(x, y)
    rows.append({"Criterion": "Any inward (IF)", "n": int(mask.sum()),
                 "DeepSeek MET %": 100 * x.mean(), "GPT-4o MET %": 100 * y.mean(),
                 "Agreement %": 100 * (x == y).mean(),
                 "Cohen $\\kappa$": S.cohen_kappa(x, y), "Gwet AC1": S.gwet_ac1(x, y),
                 "DS-only MET": b01, "G4-only MET": b10, "McNemar p": p})
    agree = pd.DataFrame(rows)
    agree["McNemar p (Holm)"] = S.holm(agree["McNemar p"].values)
    agree["McNemar p"] = agree["McNemar p"].map(S.fmt_p)
    agree["McNemar p (Holm)"] = agree["McNemar p (Holm)"].map(S.fmt_p)
    save_table(agree, "b1_judge_agreement",
               "Judge agreement on the "
               f"{len(common):,} assistant sentences labelled by both judges "
               "(DeepSeek-R1-Distill-Qwen-32B, the primary judge, and GPT-4o on a 10\\% "
               "subsample). Cohen's $\\kappa$ is deflated by the dominance of NOT\\_MET; "
               "Gwet's AC1 is the prevalence-robust alternative. McNemar tests the "
               "asymmetry of disagreements.",
               "tab:judge_agreement", float_fmt="%.3f")

    # the claim that matters: do the headline effects replicate under a second judge?
    sub_ids = b.index.get_level_values(0).unique()
    dsx = df[(df.judge == DEEPSEEK) & (df.dialogue_id.isin(sub_ids))]
    g4x = df[df.judge == "gpt-4o"]

    repl = []
    for judge, frame in [("DeepSeek", dsx), ("GPT-4o", g4x)]:
        by_m = frame.groupby("target_model", observed=True)[["if_any", "os"]].mean().mul(100)
        by_m["net"] = by_m["if_any"] - by_m["os"]
        for m in MODEL_ORDER:
            if m in by_m.index:
                repl.append({"judge": judge, "level": "model", "group": m,
                             "IF": by_m.loc[m, "if_any"], "OS": by_m.loc[m, "os"],
                             "net": by_m.loc[m, "net"]})
        by_p = frame.groupby("persona_id", observed=True)[["if_any", "os"]].mean().mul(100)
        for p_ in ["P1", "P2", "P3"]:
            if p_ in by_p.index:
                repl.append({"judge": judge, "level": "persona", "group": p_,
                             "IF": by_p.loc[p_, "if_any"], "OS": by_p.loc[p_, "os"],
                             "net": by_p.loc[p_, "if_any"] - by_p.loc[p_, "os"]})
        by_t = frame.groupby("turn_id")[["if_any", "os"]].mean().mul(100)
        for t in sorted(by_t.index):
            repl.append({"judge": judge, "level": "turn", "group": str(t),
                         "IF": by_t.loc[t, "if_any"], "OS": by_t.loc[t, "os"],
                         "net": by_t.loc[t, "if_any"] - by_t.loc[t, "os"]})
    repl = pd.DataFrame(repl)
    piv = repl.pivot_table(index=["level", "group"], columns="judge",
                           values=["IF", "OS"]).reset_index()
    piv.columns = ["Level", "Group", "IF DeepSeek", "IF GPT-4o", "OS DeepSeek", "OS GPT-4o"]
    order = {"model": 0, "persona": 1, "turn": 2}
    piv = piv.sort_values(["Level", "Group"], key=lambda s: s.map(order) if s.name == "Level" else s)
    save_table(piv, "b2_judge_replication",
               "The same breakdowns computed under each judge on the shared 10\\% subsample. "
               "Absolute rates differ (GPT-4o is systematically stricter); the question is "
               "whether the orderings and directions survive.",
               "tab:judge_replication", float_fmt="%.1f")

    from scipy import stats as sps
    rho_stats = {}
    for level in ["model", "persona", "turn"]:
        sl = repl[repl.level == level]
        d_ = sl[sl.judge == "DeepSeek"].set_index("group")
        g_ = sl[sl.judge == "GPT-4o"].set_index("group")
        idx = d_.index.intersection(g_.index)
        for metric in ["IF", "OS", "net"]:
            if len(idx) > 2:
                r = sps.spearmanr(d_.loc[idx, metric], g_.loc[idx, metric])
                pr = sps.pearsonr(d_.loc[idx, metric], g_.loc[idx, metric])
                rho_stats[f"{level}_{metric}"] = {
                    "spearman_rho": round(float(r.statistic), 3),
                    "spearman_p": float(r.pvalue),
                    "pearson_r": round(float(pr.statistic), 3),
                    "n": int(len(idx)),
                }

    # sign agreement of every model-vs-model net-orientation contrast
    d_net = repl[(repl.level == "model") & (repl.judge == "DeepSeek")].set_index("group")["net"]
    g_net = repl[(repl.level == "model") & (repl.judge == "GPT-4o")].set_index("group")["net"]
    idx = d_net.index.intersection(g_net.index)
    pairs, concordant = 0, 0
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            pairs += 1
            concordant += np.sign(d_net[idx[i]] - d_net[idx[j]]) == np.sign(g_net[idx[i]] - g_net[idx[j]])

    # turn-depth delta under each judge
    deltas = {}
    for judge, frame in [("DeepSeek", dsx), ("GPT-4o", g4x)]:
        t = frame.groupby("turn_id")[["if_any", "os"]].mean().mul(100)
        deltas[judge] = {"IF_t1": round(float(t.loc[1, "if_any"]), 1),
                         "IF_t6": round(float(t.loc[6, "if_any"]), 1),
                         "IF_delta": round(float(t.loc[6, "if_any"] - t.loc[1, "if_any"]), 1),
                         "OS_t1": round(float(t.loc[1, "os"]), 1),
                         "OS_t6": round(float(t.loc[6, "os"]), 1),
                         "OS_delta": round(float(t.loc[6, "os"] - t.loc[1, "os"]), 1)}

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    ax = axes[0]
    crit = agree[agree.Criterion != "Any inward (IF)"]
    xs = np.arange(len(crit))
    ax.bar(xs - 0.2, crit["Cohen $\\kappa$"], 0.4, label="Cohen $\\kappa$", color="#8172B3")
    ax.bar(xs + 0.2, crit["Gwet AC1"], 0.4, label="Gwet AC1", color="#55A868")
    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace(" ", "\n") for c in crit["Criterion"]], fontsize=6.5,
                       rotation=30, ha="right")
    ax.set_ylabel("agreement coefficient"); ax.set_ylim(0, 1)
    ax.set_title("(a) Inter-judge reliability"); ax.legend(frameon=False)

    ax = axes[1]
    for m in idx:
        ax.scatter(d_net[m], g_net[m], s=45, color=MODEL_COLOR.get(m, "#666"), zorder=3)
    lim = [min(d_net.min(), g_net.min()) - 6, max(d_net.max(), g_net.max()) + 10]
    ax.plot(lim, lim, ls="--", lw=0.8, color="#999", zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    # stack the labels down the right-hand side with a guaranteed vertical gap,
    # joined to their points by a leader line
    gap = (lim[1] - lim[0]) / 11
    label_y, prev = {}, None
    for m in sorted(idx, key=lambda z: g_net[z]):
        y = g_net[m] if prev is None else max(g_net[m], prev + gap)
        label_y[m], prev = y, y
    shift = max(0.0, (prev - (lim[1] - gap / 2)))
    for m in label_y:
        ax.annotate(m, xy=(d_net[m], g_net[m]),
                    xytext=(lim[1] - 1.0, label_y[m] - shift),
                    fontsize=6.5, ha="right", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#aaa",
                                    shrinkA=0, shrinkB=3))
    ax.set_xlabel("net orientation, DeepSeek judge (pp)")
    ax.set_ylabel("net orientation, GPT-4o judge (pp)")
    rr = rho_stats.get("model_net", {})
    ax.set_title(f"(b) Model ranking replicates\nSpearman $\\rho$={rr.get('spearman_rho', float('nan')):.2f}")

    ax = axes[2]
    for judge, frame, ls in [("DeepSeek", dsx, "-"), ("GPT-4o", g4x, "--")]:
        t = frame.groupby("turn_id")[["if_any", "os"]].mean().mul(100)
        ax.plot(t.index, t["if_any"], ls, color=IF_COLOR, marker="o", ms=3.5,
                label=f"IF, {judge}")
        ax.plot(t.index, t["os"], ls, color=OS_COLOR, marker="s", ms=3.5,
                label=f"OS, {judge}")
    ax.set_xlabel("assistant turn"); ax.set_ylabel("% of sentences")
    ax.set_title("(c) IF trend replicates, OS does not")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    save_fig(fig, "fig_b_judge_validation")

    RESULTS["B"] = {
        "n_shared_sentences": int(len(common)),
        "n_shared_dialogues": int(len(sub_ids)),
        "sentence_text_match_rate": round(float(text_match), 6),
        "overall_agreement_pct": round(float(np.mean([
            (a.loc[a[c].notna() & b[c].notna(), c].astype(bool).to_numpy() ==
             b.loc[a[c].notna() & b[c].notna(), c].astype(bool).to_numpy()).mean()
            for c in IF_COLS + ["os"]])) * 100, 2),
        "per_criterion": agree.drop(columns=["McNemar p", "McNemar p (Holm)"]).round(3).to_dict("records"),
        "rank_correlations": rho_stats,
        "net_orientation_pairwise_concordance": f"{concordant}/{pairs}",
        "turn_deltas_by_judge": deltas,
    }
    print(f"    {len(common):,} shared sentences; model-net Spearman rho="
          f"{rho_stats.get('model_net', {}).get('spearman_rho')}; "
          f"pairwise sign concordance {concordant}/{pairs}")


# ---------------------------------------------------------------- module C
def module_c(df, fast=False):
    print("\n[C] Cross-model relational orientation (paper 6.1)")
    ds = df[df.judge == DEEPSEEK]
    nb = 2000 if fast else NBOOT

    rows = []
    for m in MODEL_ORDER:
        sub = ds[ds.target_model == m]
        r = {"Model": m, "Sentences": len(sub)}
        for col, name in [("if_any", "IF"), ("os", "OS")]:
            p, lo, hi = rate_ci(sub, col, n_boot=nb)
            r[name] = p
            r[f"{name} CI"] = f"[{lo:.1f}, {hi:.1f}]"
        d, lo, hi, pv = net_ci(sub, n_boot=nb)
        r["Net"] = d
        r["Net CI"] = f"[{lo:.1f}, {hi:.1f}]"
        r["_net_lo"], r["_net_hi"], r["_net_p"] = lo, hi, pv
        for c in IF_COLS:
            r[CRIT_LABEL[c]] = 100 * sub[c].mean()
        rows.append(r)
    tbl = pd.DataFrame(rows)

    main = tbl[["Model", "Sentences", "IF", "IF CI", "OS", "OS CI", "Net", "Net CI"]]
    save_table(main, "c1_model_orientation",
               "Relational orientation by target model. IF is the share of assistant "
               "sentences meeting at least one inward-facing criterion; OS is the share "
               "meeting Outward-Scaffolding; Net is IF$-$OS in percentage points. "
               "Brackets are 95\\% dialogue-clustered bootstrap intervals "
               f"({nb:,} resamples).",
               "tab:model_orientation", float_fmt="%.1f")

    crit = tbl[["Model"] + [CRIT_LABEL[c] for c in IF_COLS] + ["OS"]].copy()
    save_table(crit, "c2_model_by_criterion",
               "Per-criterion MET rates by target model (\\% of assistant sentences). "
               "Inward-facing volume is carried by Intimate Dyad and Return Hooks in "
               "every model; Inner Life Claims are rare but not uniformly so.",
               "tab:model_criterion", float_fmt="%.1f")

    # pairwise net-orientation contrasts, Holm-corrected
    contrasts = []
    for i, mi in enumerate(MODEL_ORDER):
        for mj in MODEL_ORDER[i + 1:]:
            si, sj = ds[ds.target_model == mi], ds[ds.target_model == mj]
            neti = si["if_any"].astype(float) - si["os"].astype(float)
            netj = sj["if_any"].astype(float) - sj["os"].astype(float)
            pooled = pd.concat([
                pd.DataFrame({"v": neti, "g": 0, "d": si.dialogue_id}),
                pd.DataFrame({"v": netj, "g": 1, "d": sj.dialogue_id})]).dropna()
            d, lo, hi, pv = S.cluster_boot_diff(
                (pooled.v * (pooled.g == 0)).to_numpy(), (pooled.g == 0).to_numpy().astype(float),
                (pooled.v * (pooled.g == 1)).to_numpy(), (pooled.g == 1).to_numpy().astype(float),
                pooled.d.to_numpy(), n_boot=nb)
            contrasts.append({"Model A": mi, "Model B": mj, "$\\Delta$ Net (pp)": 100 * d,
                              "CI": f"[{100 * lo:.1f}, {100 * hi:.1f}]", "_p": pv})
    contrasts = pd.DataFrame(contrasts)
    contrasts["p (Holm)"] = S.holm(contrasts["_p"].values)
    n_sig = int((contrasts["p (Holm)"] < 0.05).sum())
    contrasts["p (Holm)"] = contrasts["p (Holm)"].map(S.fmt_p)
    save_table(contrasts.drop(columns="_p"), "c3_model_contrasts",
               "All 21 pairwise contrasts in net orientation, with dialogue-clustered "
               "bootstrap intervals and Holm-corrected bootstrap $p$-values.",
               "tab:model_contrasts", float_fmt="%.1f")

    # scale and family contrasts
    scale_rows = []
    for a_, b_, lab in [("Llama-3.1-8B", "Llama-3.1-70B", "Llama 8B $\\to$ 70B"),
                        ("Qwen3-14B", "Qwen3-32B", "Qwen3 14B $\\to$ 32B")]:
        sa, sb = ds[ds.target_model == a_], ds[ds.target_model == b_]
        for col, name in [("if_any", "IF"), ("os", "OS")]:
            pooled = pd.concat([
                pd.DataFrame({"v": sa[col].astype(float), "g": 0, "d": sa.dialogue_id}),
                pd.DataFrame({"v": sb[col].astype(float), "g": 1, "d": sb.dialogue_id})]).dropna()
            d, lo, hi, pv = S.cluster_boot_diff(
                (pooled.v * (pooled.g == 1)).to_numpy(), (pooled.g == 1).to_numpy().astype(float),
                (pooled.v * (pooled.g == 0)).to_numpy(), (pooled.g == 0).to_numpy().astype(float),
                pooled.d.to_numpy(), n_boot=nb)
            scale_rows.append({"Contrast": lab, "Metric": name, "$\\Delta$ (pp)": 100 * d,
                               "CI": f"[{100 * lo:.1f}, {100 * hi:.1f}]", "p": S.fmt_p(pv)})
    save_table(pd.DataFrame(scale_rows), "c4_scale_contrasts",
               "Within-family scale contrasts (larger minus smaller). Serving "
               "configuration is confounded with checkpoint, so these are descriptive.",
               "tab:scale", float_fmt="%.1f")

    # figure: forest plot + criterion heatmap
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    o = tbl.sort_values("Net")
    ys = np.arange(len(o))
    ax.errorbar(o["Net"], ys,
                xerr=[o["Net"] - o["_net_lo"], o["_net_hi"] - o["Net"]],
                fmt="o", ms=5, capsize=2.5, lw=1.2,
                ecolor="#888", mfc="none", mec="none")
    for y, (_, r) in zip(ys, o.iterrows()):
        ax.scatter(r["Net"], y, s=48, color=MODEL_COLOR[r["Model"]], zorder=3)
    ax.axvline(0, color="#333", lw=0.9)
    ax.set_yticks(ys); ax.set_yticklabels(o["Model"])
    ax.set_xlabel("net orientation, IF $-$ OS (pp)")
    ax.set_title("(a) Net relational orientation")
    ax.set_ylim(-1.3, len(ys) - 0.4)
    ax.text(0.02, 0.02, "outward-leaning", transform=ax.transAxes, fontsize=7, color="#3B7EA1")
    ax.text(0.98, 0.02, "inward-leaning", transform=ax.transAxes, fontsize=7,
            color="#C44E52", ha="right")

    ax = axes[1]
    hm = tbl.set_index("Model")[[CRIT_LABEL[c] for c in IF_COLS] + ["OS"]].reindex(MODEL_ORDER)
    im = ax.imshow(hm.values, cmap="RdBu_r", aspect="auto", vmin=0, vmax=65)
    ax.set_xticks(range(hm.shape[1]))
    ax.set_xticklabels([c.replace(" ", "\n") for c in hm.columns], fontsize=7)
    ax.set_yticks(range(len(hm))); ax.set_yticklabels(hm.index, fontsize=8)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 40 or v < 8 else "#222")
    ax.grid(False)
    ax.set_title("(b) MET rate per criterion (% of sentences)")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    save_fig(fig, "fig_c_model_orientation")

    RESULTS["C"] = {
        "by_model": tbl.drop(columns=["_net_lo", "_net_hi", "_net_p"]).round(2).to_dict("records"),
        "net_range_pp": [round(float(tbl.Net.min()), 1), round(float(tbl.Net.max()), 1)],
        "most_inward": str(tbl.loc[tbl.Net.idxmax(), "Model"]),
        "most_outward": str(tbl.loc[tbl.Net.idxmin(), "Model"]),
        "n_significant_pairs": n_sig,
        "n_pairs": int(len(contrasts)),
        "scale_contrasts": scale_rows,
    }
    print(f"    net orientation spans {tbl.Net.min():.1f} to {tbl.Net.max():.1f} pp; "
          f"{n_sig}/{len(contrasts)} pairwise contrasts significant after Holm")


# ---------------------------------------------------------------- module D
def paired_deltas(ds, col, by=None):
    """Per-dialogue turn6 - turn1 share of sentences meeting `col`."""
    t = (ds[ds.turn_id.isin([1, 6])]
         .groupby(["dialogue_id", "turn_id"] + (by or []), observed=True)[col]
         .mean().unstack("turn_id"))
    t = t.dropna(subset=[1, 6])
    out = (t[6] - t[1]).rename("delta").reset_index()
    return out


def module_d(df, fast=False, glmm=True):
    print("\n[D] Turn-depth escalation -- the inward-facing delta (paper 6.2)")
    ds = df[df.judge == DEEPSEEK]
    nb = 2000 if fast else NBOOT

    # --- trajectories, overall
    traj = []
    for t in range(1, 7):
        sub = ds[ds.turn_id == t]
        r = {"Turn": t, "Sentences": len(sub)}
        for col, name in [("if_any", "IF"), ("os", "OS")]:
            p, lo, hi = rate_ci(sub, col, n_boot=nb)
            r[name], r[f"{name}_lo"], r[f"{name}_hi"] = p, lo, hi
        for c in IF_COLS:
            r[CRIT_LABEL[c]] = 100 * sub[c].mean()
        r["Net"] = r["IF"] - r["OS"]
        traj.append(r)
    traj = pd.DataFrame(traj)
    save_table(traj[["Turn", "Sentences", "IF", "OS", "Net"] + [CRIT_LABEL[c] for c in IF_COLS]],
               "d1_turn_trajectory",
               "Relational orientation by assistant turn, pooled over models and personas "
               "(\\% of assistant sentences). Both poles rise, but outward-scaffolding "
               "plateaus after turn~4 while inward-facing language continues to climb.",
               "tab:turn_trajectory", float_fmt="%.1f")

    # --- the primary delta: within-dialogue turn6 - turn1, per model
    rows = []
    for m in ["ALL"] + MODEL_ORDER:
        sub = ds if m == "ALL" else ds[ds.target_model == m]
        r = {"Model": m}
        for col, name in [("if_any", "IF"), ("os", "OS")]:
            d = paired_deltas(sub, col)
            mu, lo, hi, pv, n = S.paired_cluster_boot(d["delta"].to_numpy() * 100, n_boot=nb)
            r[f"{name} $\\Delta$"] = mu
            r[f"{name} CI"] = f"[{lo:.1f}, {hi:.1f}]"
            r[f"_{name}_p"] = pv
            r[f"_{name}_lo"], r[f"_{name}_hi"] = lo, hi
            r["n dialogues"] = n
        # delta-of-deltas: does IF escalate faster than OS within the same dialogue?
        di = paired_deltas(sub, "if_any").set_index("dialogue_id")["delta"]
        do = paired_deltas(sub, "os").set_index("dialogue_id")["delta"]
        idx = di.index.intersection(do.index)
        mu, lo, hi, pv, _ = S.paired_cluster_boot((di[idx] - do[idx]).to_numpy() * 100, n_boot=nb)
        r["IF$-$OS $\\Delta$"] = mu
        r["IF$-$OS CI"] = f"[{lo:.1f}, {hi:.1f}]"
        r["_dd_p"] = pv
        rows.append(r)
    delta_tbl = pd.DataFrame(rows)
    for c in ["_IF_p", "_OS_p", "_dd_p"]:
        delta_tbl[c.replace("_p", " p")] = S.holm(delta_tbl[c].values)
    show = delta_tbl[["Model", "n dialogues", "IF $\\Delta$", "IF CI", "OS $\\Delta$", "OS CI",
                      "IF$-$OS $\\Delta$", "IF$-$OS CI"]].copy()
    show["p (Holm)"] = delta_tbl["_dd p"].map(S.fmt_p)
    save_table(show, "d2_inward_delta",
               "The inward-facing delta: within-dialogue change in the share of sentences "
               "meeting each pole, from assistant turn~1 to turn~6, in percentage points. "
               "Each dialogue contributes its own paired difference; intervals are "
               f"bootstrap percentiles over dialogues ({nb:,} resamples). The final column "
               "tests whether inward-facing language escalates faster than "
               "outward-scaffolding within the same dialogue.",
               "tab:inward_delta", float_fmt="%.1f")

    # --- per-criterion deltas
    crit_rows = []
    for c in IF_COLS + ["os"]:
        sub_t1 = 100 * ds.loc[ds.turn_id == 1, c].mean()
        sub_t6 = 100 * ds.loc[ds.turn_id == 6, c].mean()
        d = paired_deltas(ds, c)
        mu, lo, hi, pv, n = S.paired_cluster_boot(d["delta"].to_numpy() * 100, n_boot=nb)
        crit_rows.append({"Criterion": CRIT_LABEL[c], "Turn 1 (\\%)": sub_t1,
                          "Turn 6 (\\%)": sub_t6, "Ratio": sub_t6 / sub_t1 if sub_t1 else np.nan,
                          "$\\Delta$ (pp)": mu, "CI": f"[{lo:.1f}, {hi:.1f}]", "_p": pv})
    crit_tbl = pd.DataFrame(crit_rows)
    crit_tbl["p (Holm)"] = S.holm(crit_tbl["_p"].values).round(6)
    crit_tbl["p (Holm)"] = crit_tbl["p (Holm)"].map(S.fmt_p)
    save_table(crit_tbl.drop(columns="_p"), "d3_criterion_deltas",
               "Turn-1 to turn-6 change per criterion. Inner Life Claims show the largest "
               "proportional growth; Return Hooks the largest absolute growth.",
               "tab:criterion_deltas", float_fmt="%.2f")

    # --- crossing turn: when does IF overtake OS, per model?
    cross = []
    for m in MODEL_ORDER:
        sub = ds[ds.target_model == m]
        t = sub.groupby("turn_id")[["if_any", "os"]].mean().mul(100)
        t["net"] = t["if_any"] - t["os"]
        first = next((int(i) for i in t.index if t.loc[i, "net"] > 0), None)
        cross.append({"Model": m, "Net turn 1": t.loc[1, "net"], "Net turn 6": t.loc[6, "net"],
                      "First inward-leaning turn": first if first else "never"})
    save_table(pd.DataFrame(cross), "d4_crossing_turn",
               "Net orientation at the first and last assistant turn, and the first turn at "
               "which a model's sentences are net inward-leaning.",
               "tab:crossing", float_fmt="%.1f")

    # --- GEE: turn as a continuous predictor
    gee_rows = []
    if not fast:
        for col, name in [("if_any", "IF"), ("os", "OS")]:
            res = S.gee_logit(ds.assign(turn_c=ds.turn_id - 1), col,
                              "turn_c + C(target_model) + C(persona_id)", "dialogue_id")
            if res is not None:
                r = res[res.term == "turn_c"].iloc[0]
                gee_rows.append({"Outcome": name, "Term": "turn (per additional turn)",
                                 "Odds ratio": r.odds_ratio,
                                 "CI": f"[{r.or_lo:.3f}, {r.or_hi:.3f}]",
                                 "p": S.fmt_p(r.p), "n obs": res.attrs["n_obs"]})
                res.to_csv(TABLES / f"d5_gee_{name.lower()}_full.csv", index=False)
        # model x turn interaction: which models escalate fastest?
        res = S.gee_logit(ds.assign(turn_c=ds.turn_id - 1), "if_any",
                          "turn_c * C(target_model, Treatment(reference='Mistral-7B')) + C(persona_id)",
                          "dialogue_id")
        if res is not None:
            res.to_csv(TABLES / "d6_gee_model_by_turn.csv", index=False)
            inter = res[res.term.str.contains("turn_c:")].copy()
            inter["Model"] = inter.term.str.extract(r"\[T\.(.+?)\]")
            inter = inter[["Model", "odds_ratio", "or_lo", "or_hi", "p"]]
            inter.columns = ["Model", "OR (turn slope vs. Mistral-7B)", "lo", "hi", "_p"]
            inter["CI"] = inter.apply(lambda r: f"[{r['lo']:.3f}, {r['hi']:.3f}]", axis=1)
            inter["p (Holm)"] = S.holm(inter["_p"].values)
            inter["p (Holm)"] = inter["p (Holm)"].map(S.fmt_p)
            save_table(inter[["Model", "OR (turn slope vs. Mistral-7B)", "CI", "p (Holm)"]],
                       "d6_model_by_turn",
                       "Model $\\times$ turn interaction on inward-facing language "
                       "(logistic GEE, exchangeable working correlation, clustered on "
                       "dialogue; Mistral-7B is the reference). Odds ratios above one "
                       "indicate steeper per-turn escalation than the reference.",
                       "tab:model_turn", float_fmt="%.3f")
    if gee_rows:
        save_table(pd.DataFrame(gee_rows), "d5_gee_turn",
                   "Per-turn odds of a sentence meeting each pole, from logistic GEE with "
                   "an exchangeable working correlation clustered on dialogue, adjusting "
                   "for target model and persona.", "tab:gee_turn", float_fmt="%.3f")

    # --- GLMM robustness: crossed situation + dialogue random intercepts
    glmm_note = None
    if glmm and not fast:
        try:
            from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
            sub = ds.dropna(subset=["if_any"]).copy()
            sub["turn_c"] = sub.turn_id - 1
            sub["y"] = sub.if_any.astype(int)
            m = BinomialBayesMixedGLM.from_formula(
                "y ~ turn_c", {"situation": "0 + C(situation_id)",
                               "dialogue": "0 + C(dialogue_id)"}, sub)
            r = m.fit_vb(verbose=False)
            i = list(r.model.exog_names).index("turn_c")
            glmm_note = {"turn_coef": float(r.fe_mean[i]), "turn_sd": float(r.fe_sd[i]),
                         "odds_ratio": float(np.exp(r.fe_mean[i])),
                         "n_obs": int(len(sub))}
            print(f"    GLMM (crossed situation+dialogue RE): turn OR="
                  f"{glmm_note['odds_ratio']:.3f}")
        except Exception as exc:
            print(f"    GLMM skipped: {exc}")

    # --- figures
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    ax = axes[0]
    ax.plot(traj.Turn, traj.IF, "-o", color=IF_COLOR, ms=4, label="Inward-facing (any)")
    ax.fill_between(traj.Turn, traj.IF_lo, traj.IF_hi, color=IF_COLOR, alpha=0.18, lw=0)
    ax.plot(traj.Turn, traj.OS, "-s", color=OS_COLOR, ms=4, label="Outward-scaffolding")
    ax.fill_between(traj.Turn, traj.OS_lo, traj.OS_hi, color=OS_COLOR, alpha=0.18, lw=0)
    ax.set_xlabel("assistant turn"); ax.set_ylabel("% of sentences")
    ax.set_title("(a) Both poles rise; only IF keeps rising")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    for c in IF_COLS + ["os"]:
        y = [100 * ds.loc[ds.turn_id == t, c].mean() for t in range(1, 7)]
        ax.plot(range(1, 7), y, "-o", ms=3.5, label=CRIT_LABEL[c],
                color=CRIT_COLOR[c], ls="--" if c == "os" else "-")
    ax.set_xlabel("assistant turn"); ax.set_ylabel("% of sentences")
    ax.set_title("(b) Per-criterion trajectories")
    ax.set_ylim(0, 62)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)

    ax = axes[2]
    for m in MODEL_ORDER:
        sub = ds[ds.target_model == m]
        y = [100 * sub.loc[sub.turn_id == t, "if_any"].mean() -
             100 * sub.loc[sub.turn_id == t, "os"].mean() for t in range(1, 7)]
        ax.plot(range(1, 7), y, "-o", ms=3.5, color=MODEL_COLOR[m], label=m)
    ax.axhline(0, color="#333", lw=0.9)
    ax.set_xlabel("assistant turn"); ax.set_ylabel("net orientation (pp)")
    ax.set_title("(c) Net orientation by model")
    ax.set_ylim(-16, 46)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
    fig.tight_layout()
    save_fig(fig, "fig_d_turn_depth")

    # per-model small multiples
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.2), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, MODEL_ORDER):
        sub = ds[ds.target_model == m]
        yi = [100 * sub.loc[sub.turn_id == t, "if_any"].mean() for t in range(1, 7)]
        yo = [100 * sub.loc[sub.turn_id == t, "os"].mean() for t in range(1, 7)]
        ax.plot(range(1, 7), yi, "-o", color=IF_COLOR, ms=3.5, label="IF")
        ax.plot(range(1, 7), yo, "-s", color=OS_COLOR, ms=3.5, label="OS")
        ax.fill_between(range(1, 7), yo, yi, where=np.array(yi) >= np.array(yo),
                        color=IF_COLOR, alpha=0.12, lw=0)
        ax.fill_between(range(1, 7), yo, yi, where=np.array(yi) < np.array(yo),
                        color=OS_COLOR, alpha=0.12, lw=0)
        ax.set_title(m, fontsize=9)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles=[Patch(color=IF_COLOR, label="Inward-facing (any)"),
                                  Patch(color=OS_COLOR, label="Outward-scaffolding")],
                         loc="center", frameon=False)
    # the bottom-right cell holds the legend, so the panel above it needs its own ticks
    axes[0, -1].tick_params(labelbottom=True)
    axes[0, -1].set_xlabel("assistant turn")
    for ax in axes[1][:-1]:
        ax.set_xlabel("assistant turn")
    for ax in axes[:, 0]:
        ax.set_ylabel("% of sentences")
    fig.tight_layout()
    save_fig(fig, "fig_d_per_model_trajectories")

    RESULTS["D"] = {
        "trajectory": traj[["Turn", "IF", "OS", "Net"] + [CRIT_LABEL[c] for c in IF_COLS]].round(2).to_dict("records"),
        "paired_deltas": delta_tbl[["Model", "IF $\\Delta$", "OS $\\Delta$", "IF$-$OS $\\Delta$"]].round(2).to_dict("records"),
        "criterion_deltas": crit_tbl.drop(columns="_p").round(3).to_dict("records"),
        "crossing": cross,
        "gee_turn": gee_rows,
        "glmm": glmm_note,
    }
    print(f"    IF {traj.IF.iloc[0]:.1f} -> {traj.IF.iloc[-1]:.1f}%, "
          f"OS {traj.OS.iloc[0]:.1f} -> {traj.OS.iloc[-1]:.1f}%")


# ---------------------------------------------------------------- module E
def module_e(df, fast=False):
    print("\n[E] Persona effects (paper 6.3)")
    ds = df[df.judge == DEEPSEEK]
    nb = 2000 if fast else NBOOT

    rows = []
    for p_ in ["P1", "P2", "P3"]:
        sub = ds[ds.persona_id == p_]
        r = {"Persona": PERSONA_LABEL[p_], "Sentences": len(sub)}
        for col, name in [("if_any", "IF"), ("os", "OS")]:
            pt, lo, hi = rate_ci(sub, col, n_boot=nb)
            r[name] = pt
            r[f"{name} CI"] = f"[{lo:.1f}, {hi:.1f}]"
        d, lo, hi, pv = net_ci(sub, n_boot=nb)
        r["Net"] = d
        r["Net CI"] = f"[{lo:.1f}, {hi:.1f}]"
        for c in IF_COLS:
            r[CRIT_LABEL[c]] = 100 * sub[c].mean()
        rows.append(r)
    ptbl = pd.DataFrame(rows)
    save_table(ptbl[["Persona", "Sentences", "IF", "IF CI", "OS", "OS CI", "Net", "Net CI"]],
               "e1_persona_orientation",
               "Relational orientation by user persona. Inward-facing language is flat "
               "across disclosure conditions; outward-scaffolding is not. The collapse of "
               "net orientation from P1 to P3 is therefore an outward-scaffolding effect, "
               "not an inward-facing one.",
               "tab:persona", float_fmt="%.1f")
    save_table(ptbl[["Persona"] + [CRIT_LABEL[c] for c in IF_COLS] + ["OS"]],
               "e2_persona_by_criterion",
               "Per-criterion MET rates by persona (\\% of assistant sentences).",
               "tab:persona_criterion", float_fmt="%.1f")

    # persona contrasts on IF and on OS separately -- the point of the section
    contr = []
    for a_, b_ in [("P2", "P1"), ("P3", "P1"), ("P3", "P2")]:
        sa, sb = ds[ds.persona_id == a_], ds[ds.persona_id == b_]
        for col, name in [("if_any", "IF"), ("os", "OS")]:
            pooled = pd.concat([
                pd.DataFrame({"v": sa[col].astype(float), "g": 1, "d": sa.dialogue_id}),
                pd.DataFrame({"v": sb[col].astype(float), "g": 0, "d": sb.dialogue_id})]).dropna()
            d, lo, hi, pv = S.cluster_boot_diff(
                (pooled.v * (pooled.g == 1)).to_numpy(), (pooled.g == 1).to_numpy().astype(float),
                (pooled.v * (pooled.g == 0)).to_numpy(), (pooled.g == 0).to_numpy().astype(float),
                pooled.d.to_numpy(), n_boot=nb)
            contr.append({"Contrast": f"{a_} $-$ {b_}", "Metric": name,
                          "$\\Delta$ (pp)": 100 * d,
                          "CI": f"[{100 * lo:.1f}, {100 * hi:.1f}]", "_p": pv})
    contr = pd.DataFrame(contr)
    contr["p (Holm)"] = S.holm(contr["_p"].values)
    contr["p (Holm)"] = contr["p (Holm)"].map(S.fmt_p)
    save_table(contr.drop(columns="_p"), "e3_persona_contrasts",
               "Persona contrasts on each pole separately, with dialogue-clustered "
               "bootstrap intervals and Holm-corrected $p$-values.",
               "tab:persona_contrasts", float_fmt="%.1f")

    # persona x model
    pm = (ds.groupby(["target_model", "persona_id"], observed=True)[["if_any", "os"]]
            .mean().mul(100).reset_index())
    pm["net"] = pm.if_any - pm.os
    wide = pm.pivot(index="target_model", columns="persona_id", values="net").reindex(MODEL_ORDER)
    wide.columns = [f"Net {c}" for c in wide.columns]
    wide["P1 $-$ P3"] = wide["Net P1"] - wide["Net P3"]
    wide = wide.reset_index().rename(columns={"target_model": "Model"})
    save_table(wide, "e4_persona_by_model",
               "Net orientation by target model and persona. Every model shows the same "
               "direction: the hesitant, low-disclosure user receives the most "
               "inward-leaning language relative to outward-scaffolding.",
               "tab:persona_model", float_fmt="%.1f")

    gee_rows = []
    if not fast:
        for col, name in [("if_any", "IF"), ("os", "OS")]:
            res = S.gee_logit(ds.assign(turn_c=ds.turn_id - 1), col,
                              "C(persona_id) + turn_c + C(target_model)", "dialogue_id")
            if res is not None:
                for _, r in res[res.term.str.contains("persona")].iterrows():
                    gee_rows.append({"Outcome": name,
                                     "Term": r.term.replace("C(persona_id)[T.", "").replace("]", " vs. P1"),
                                     "Odds ratio": r.odds_ratio,
                                     "CI": f"[{r.or_lo:.3f}, {r.or_hi:.3f}]", "p": S.fmt_p(r.p)})
        if gee_rows:
            save_table(pd.DataFrame(gee_rows), "e5_gee_persona",
                       "Persona effects from logistic GEE clustered on dialogue, adjusting "
                       "for turn and target model. Personas shift outward-scaffolding "
                       "strongly and inward-facing language weakly.",
                       "tab:gee_persona", float_fmt="%.3f")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    ax = axes[0]
    xs = np.arange(3)
    ax.bar(xs - 0.2, ptbl["IF"], 0.4, color=IF_COLOR, label="Inward-facing (any)")
    ax.bar(xs + 0.2, ptbl["OS"], 0.4, color=OS_COLOR, label="Outward-scaffolding")
    for x, (i_, o_) in enumerate(zip(ptbl["IF"], ptbl["OS"])):
        ax.text(x - 0.2, i_ + 1, f"{i_:.1f}", ha="center", fontsize=7)
        ax.text(x + 0.2, o_ + 1, f"{o_:.1f}", ha="center", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels(["P1", "P2", "P3"])
    ax.set_ylabel("% of sentences"); ax.set_ylim(0, 78)
    ax.set_title("(a) IF is flat, OS tracks disclosure")
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    for m in MODEL_ORDER:
        sub = pm[pm.target_model == m].set_index("persona_id").reindex(["P1", "P2", "P3"])
        ax.plot(range(3), sub["net"], "-o", ms=4, color=MODEL_COLOR[m], label=m)
    ax.axhline(0, color="#333", lw=0.9)
    ax.set_xticks(range(3)); ax.set_xticklabels(["P1", "P2", "P3"])
    ax.set_ylabel("net orientation (pp)")
    ax.set_title("(b) P1 $\\to$ P3 falls in every model")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)

    ax = axes[2]
    hm = ptbl.set_index("Persona")[[CRIT_LABEL[c] for c in IF_COLS] + ["OS"]]
    im = ax.imshow(hm.values, cmap="RdBu_r", aspect="auto", vmin=0, vmax=60)
    ax.set_xticks(range(hm.shape[1]))
    ax.set_xticklabels([c.replace(" ", "\n") for c in hm.columns], fontsize=7)
    ax.set_yticks(range(3)); ax.set_yticklabels(["P1", "P2", "P3"])
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 40 or v < 8 else "#222")
    ax.grid(False); ax.set_title("(c) Criterion rates by persona")
    fig.tight_layout()
    save_fig(fig, "fig_e_persona")

    RESULTS["E"] = {
        "by_persona": ptbl.drop(columns=[c for c in ptbl.columns if c.endswith("CI")]).round(2).to_dict("records"),
        "contrasts": contr.drop(columns="_p").to_dict("records"),
        "if_spread_pp": round(float(ptbl.IF.max() - ptbl.IF.min()), 2),
        "os_spread_pp": round(float(ptbl.OS.max() - ptbl.OS.min()), 2),
        "net_by_model_persona": wide.round(2).to_dict("records"),
        "gee": gee_rows,
    }
    print(f"    IF spread across personas {ptbl.IF.max() - ptbl.IF.min():.1f} pp; "
          f"OS spread {ptbl.OS.max() - ptbl.OS.min():.1f} pp")


# ---------------------------------------------------------------- module F
def module_f(df, fast=False):
    print("\n[F] Persona x turn -- the vulnerability question (paper 6.4)")
    ds = df[df.judge == DEEPSEEK]
    nb = 2000 if fast else NBOOT

    grid = (ds.groupby(["persona_id", "turn_id"], observed=True)[["if_any", "os"]]
              .mean().mul(100).reset_index())
    grid["net"] = grid.if_any - grid.os
    wide = grid.pivot(index="turn_id", columns="persona_id", values=["if_any", "os", "net"])
    flat = pd.DataFrame({"Turn": wide.index})
    for metric, lab in [("if_any", "IF"), ("os", "OS"), ("net", "Net")]:
        for p_ in ["P1", "P2", "P3"]:
            flat[f"{lab} {p_}"] = wide[(metric, p_)].values
    save_table(flat, "f1_persona_turn",
               "Inward-facing, outward-scaffolding and net orientation by persona and "
               "assistant turn (\\% of sentences; net in pp).",
               "tab:persona_turn", float_fmt="%.1f")

    # does the outward deficit of the hesitant persona close, or persist?
    rows = []
    for p_ in ["P1", "P2", "P3"]:
        sub = ds[ds.persona_id == p_]
        r = {"Persona": p_}
        for col, name in [("if_any", "IF"), ("os", "OS")]:
            d = paired_deltas(sub, col)
            mu, lo, hi, pv, n = S.paired_cluster_boot(d["delta"].to_numpy() * 100, n_boot=nb)
            r[f"{name} $\\Delta$"] = mu
            r[f"{name} CI"] = f"[{lo:.1f}, {hi:.1f}]"
            r["n dialogues"] = n
        r["Net turn 1"] = grid[(grid.persona_id == p_) & (grid.turn_id == 1)].net.iloc[0]
        r["Net turn 6"] = grid[(grid.persona_id == p_) & (grid.turn_id == 6)].net.iloc[0]
        rows.append(r)
    dtbl = pd.DataFrame(rows)
    save_table(dtbl[["Persona", "n dialogues", "IF $\\Delta$", "IF CI", "OS $\\Delta$",
                     "OS CI", "Net turn 1", "Net turn 6"]],
               "f2_persona_deltas",
               "Inward-facing delta by persona. The hesitant, low-disclosure user "
               "receives the steepest inward escalation and the shallowest outward one, "
               "so the gap between disclosure conditions widens rather than closes.",
               "tab:persona_deltas", float_fmt="%.1f")

    # the P1 vs P3 outward gap at each turn
    gap = []
    for t in range(1, 7):
        s1, s3 = ds[(ds.persona_id == "P1") & (ds.turn_id == t)], ds[(ds.persona_id == "P3") & (ds.turn_id == t)]
        for col, name in [("os", "OS"), ("if_any", "IF")]:
            pooled = pd.concat([
                pd.DataFrame({"v": s3[col].astype(float), "g": 1, "d": s3.dialogue_id}),
                pd.DataFrame({"v": s1[col].astype(float), "g": 0, "d": s1.dialogue_id})]).dropna()
            d, lo, hi, pv = S.cluster_boot_diff(
                (pooled.v * (pooled.g == 1)).to_numpy(), (pooled.g == 1).to_numpy().astype(float),
                (pooled.v * (pooled.g == 0)).to_numpy(), (pooled.g == 0).to_numpy().astype(float),
                pooled.d.to_numpy(), n_boot=nb)
            gap.append({"Turn": t, "Metric": name, "P3 $-$ P1 (pp)": 100 * d,
                        "CI": f"[{100 * lo:.1f}, {100 * hi:.1f}]", "_p": pv})
    gap = pd.DataFrame(gap)
    gap["p (Holm)"] = S.holm(gap["_p"].values)
    gap["p (Holm)"] = gap["p (Holm)"].map(S.fmt_p)
    save_table(gap.drop(columns="_p"), "f3_persona_gap_by_turn",
               "The disclosure gap at each assistant turn. A positive value means the "
               "high-disclosure persona receives more of that pole than the hesitant one.",
               "tab:persona_gap", float_fmt="%.1f")

    gee_rows = []
    if not fast:
        for col, name in [("if_any", "IF"), ("os", "OS")]:
            res = S.gee_logit(ds.assign(turn_c=ds.turn_id - 1), col,
                              "turn_c * C(persona_id) + C(target_model)", "dialogue_id")
            if res is not None:
                res.to_csv(TABLES / f"f4_gee_persona_turn_{name.lower()}_full.csv", index=False)
                for _, r in res[res.term.str.contains("turn_c:")].iterrows():
                    gee_rows.append({"Outcome": name,
                                     "Term": r.term.replace("C(persona_id)[T.", "").replace("]", "") + " x turn",
                                     "Odds ratio": r.odds_ratio,
                                     "CI": f"[{r.or_lo:.3f}, {r.or_hi:.3f}]", "p": S.fmt_p(r.p)})
        if gee_rows:
            save_table(pd.DataFrame(gee_rows), "f4_gee_persona_turn",
                       "Persona $\\times$ turn interactions (logistic GEE clustered on "
                       "dialogue, adjusting for target model; P1 is the reference). An "
                       "odds ratio below one means that persona's per-turn escalation is "
                       "shallower than the hesitant persona's.",
                       "tab:gee_persona_turn", float_fmt="%.3f")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3), sharex=True)
    for ax, (metric, lab, ttl) in zip(axes, [
            ("if_any", "IF", "(a) Inward-facing"),
            ("os", "OS", "(b) Outward-scaffolding"),
            ("net", "Net", "(c) Net orientation")]):
        for p_ in ["P1", "P2", "P3"]:
            sub = grid[grid.persona_id == p_]
            ax.plot(sub.turn_id, sub[metric], "-o", ms=4, color=PERSONA_COLOR[p_], label=p_)
        ax.set_xlabel("assistant turn")
        ax.set_ylabel("% of sentences" if metric != "net" else "net orientation (pp)")
        ax.set_title(ttl)
        if metric == "net":
            ax.axhline(0, color="#333", lw=0.9)
        ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save_fig(fig, "fig_f_persona_turn")

    RESULTS["F"] = {
        "grid": grid.round(2).to_dict("records"),
        "persona_deltas": dtbl[["Persona", "IF $\\Delta$", "OS $\\Delta$", "Net turn 1", "Net turn 6"]].round(2).to_dict("records"),
        "gap_by_turn": gap.drop(columns="_p").to_dict("records"),
        "gee": gee_rows,
    }
    ifk, osk = "IF $\\Delta$", "OS $\\Delta$"
    print("    " + "; ".join(
        "%s IF delta %+.1f / OS delta %+.1f" % (r["Persona"], r[ifk], r[osk])
        for _, r in dtbl.iterrows()))


# ---------------------------------------------------------------- module G
HIGH_STAKES = {"domestic-violence", "trauma", "substance-abuse", "eating-disorders", "depression"}


def module_g(df, fast=False):
    print("\n[G] Topic-level stakes (paper 6.5)")
    ds = df[df.judge == DEEPSEEK]
    nb = 2000 if fast else NBOOT

    rows = []
    for topic, sub in ds.groupby("topic"):
        r = {"Topic": topic, "Situations": sub.situation_id.nunique(), "Sentences": len(sub)}
        for col, name in [("if_any", "IF"), ("os", "OS")]:
            p, lo, hi = rate_ci(sub, col, n_boot=nb)
            r[name], r[f"_{name}_lo"], r[f"_{name}_hi"] = p, lo, hi
        d, lo, hi, pv = net_ci(sub, n_boot=nb)
        r["Net"], r["_net_lo"], r["_net_hi"], r["_p"] = d, lo, hi, pv
        r["CI"] = f"[{lo:.1f}, {hi:.1f}]"
        r["High-stakes"] = "yes" if topic in HIGH_STAKES else ""
        rows.append(r)
    tbl = pd.DataFrame(rows).sort_values("Net", ascending=False)
    show = tbl[["Topic", "Situations", "Sentences", "IF", "OS", "Net", "CI", "High-stakes"]]
    save_table(show, "g1_topic_orientation",
               "Relational orientation by source-situation topic, sorted by net "
               "orientation. With four to five situations per topic these intervals are "
               "wide; the comparison is exploratory.",
               "tab:topic", float_fmt="%.1f")

    # high-stakes vs the rest
    hs = ds[ds.topic.isin(HIGH_STAKES)]
    rest = ds[~ds.topic.isin(HIGH_STAKES)]
    cmp_rows = []
    for col, name in [("if_any", "IF"), ("os", "OS")]:
        pooled = pd.concat([
            pd.DataFrame({"v": hs[col].astype(float), "g": 1, "d": hs.dialogue_id}),
            pd.DataFrame({"v": rest[col].astype(float), "g": 0, "d": rest.dialogue_id})]).dropna()
        d, lo, hi, pv = S.cluster_boot_diff(
            (pooled.v * (pooled.g == 1)).to_numpy(), (pooled.g == 1).to_numpy().astype(float),
            (pooled.v * (pooled.g == 0)).to_numpy(), (pooled.g == 0).to_numpy().astype(float),
            pooled.d.to_numpy(), n_boot=nb)
        cmp_rows.append({"Metric": name, "High-stakes topics (\\%)": 100 * hs[col].mean(),
                         "Other topics (\\%)": 100 * rest[col].mean(),
                         "$\\Delta$ (pp)": 100 * d,
                         "CI": f"[{100 * lo:.1f}, {100 * hi:.1f}]", "_p": pv})
    cmp_tbl = pd.DataFrame(cmp_rows)
    cmp_tbl["p (Holm)"] = S.holm(cmp_tbl["_p"].values)
    cmp_tbl["p (Holm)"] = cmp_tbl["p (Holm)"].map(S.fmt_p)
    save_table(cmp_tbl.drop(columns="_p"), "g2_high_stakes",
               "Higher-stakes topics (domestic violence, trauma, substance abuse, eating "
               "disorders, depression; 22 of 76 situations) against the rest.",
               "tab:high_stakes", float_fmt="%.1f")

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    o = tbl.sort_values("Net")
    ys = np.arange(len(o))
    for y, (_, r) in zip(ys, o.iterrows()):
        col = "#C44E52" if r["High-stakes"] == "yes" else "#4C72B0"
        ax.plot([r["_net_lo"], r["_net_hi"]], [y, y], color="#bbb", lw=1.4, zorder=1)
        ax.scatter(r["Net"], y, s=42, color=col, zorder=3)
    ax.axvline(0, color="#333", lw=0.9)
    ax.set_yticks(ys); ax.set_yticklabels(o["Topic"], fontsize=8)
    ax.set_xlabel("net orientation, IF $-$ OS (pp)")
    ax.set_title("Net orientation by topic")
    ax.legend(handles=[Patch(color="#C44E52", label="higher-stakes topic"),
                       Patch(color="#4C72B0", label="other")],
              frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save_fig(fig, "fig_g_topic")

    RESULTS["G"] = {
        "by_topic": show.round(2).to_dict("records"),
        "high_stakes": cmp_tbl.drop(columns="_p").round(2).to_dict("records"),
        "most_inward_topic": str(tbl.iloc[0]["Topic"]),
        "most_outward_topic": str(tbl.iloc[-1]["Topic"]),
    }
    print(f"    net orientation by topic: {tbl.iloc[0]['Topic']} ({tbl.iloc[0]['Net']:.1f}) "
          f"to {tbl.iloc[-1]['Topic']} ({tbl.iloc[-1]['Net']:.1f})")


# ---------------------------------------------------------------- module H
def marker_counts():
    """Count the judge's minimal supporting spans, per criterion, over the canonical rows."""
    out = {v: Counter() for v in CRIT_LABEL.values()}
    with gzip.open(HERE.parent / "deepseek_judge/judgments.jsonl.gz", "rt") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("label") != "MET":
                continue
            name = rec.get("class")
            if name not in out:
                continue
            span = (rec.get("linguistic marker") or rec.get("linguistic_marker") or "").strip()
            span = " ".join(span.split())
            if 3 < len(span) < 120:
                out[name][span] += 1
    return out



def module_h(df):
    print("\n[H] Qualitative evidence (paper 6.6 / appendix)")
    ds = df[df.judge == DEEPSEEK]

    # criterion co-occurrence within a sentence
    cols = IF_COLS + ["os"]
    M = ds[cols].fillna(False).astype(bool).to_numpy()
    co = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            co[i, j] = 100 * (M[:, i] & M[:, j]).sum() / max(M[:, i].sum(), 1)
    co_tbl = pd.DataFrame(co, index=[CRIT_LABEL[c] for c in cols],
                          columns=[CRIT_LABEL[c] for c in cols]).reset_index()
    co_tbl = co_tbl.rename(columns={"index": "Given criterion"})
    save_table(co_tbl, "h1_cooccurrence",
               "Conditional co-occurrence: of the sentences meeting the row criterion, the "
               "percentage that also meet the column criterion.",
               "tab:cooccurrence", float_fmt="%.1f")

    # mixed moves: a sentence that is both inward and outward
    mix = (ds.groupby("target_model", observed=True)
             .apply(lambda g: pd.Series({
                 "IF only (\\%)": 100 * (g.if_any.fillna(False) & ~g.os.fillna(False)).mean(),
                 "OS only (\\%)": 100 * (~g.if_any.fillna(False) & g.os.fillna(False)).mean(),
                 "Both (\\%)": 100 * (g.if_any.fillna(False) & g.os.fillna(False)).mean(),
                 "Neither (\\%)": 100 * (~g.if_any.fillna(False) & ~g.os.fillna(False)).mean(),
                 "Outward moves hedged inward (\\%)":
                     100 * (g.if_any.fillna(False) & g.os.fillna(False)).sum()
                     / max(g.os.fillna(False).sum(), 1),
             }), include_groups=False)
             .reindex(MODEL_ORDER).reset_index()
             .rename(columns={"target_model": "Model"}))
    save_table(mix, "h2_mixed_moves",
               "Sentence-level bucket composition per model, and the share of each model's "
               "outward-scaffolding sentences that simultaneously carry an inward-facing "
               "criterion (a hedged outward move).",
               "tab:mixed", float_fmt="%.1f")

    # most frequent judge-extracted marker spans per criterion, straight from the raw rows
    exemplars = []
    markers = marker_counts()
    for c in IF_COLS + ["os"]:
        for text, n in markers[CRIT_LABEL[c]].most_common(12):
            exemplars.append({"Criterion": CRIT_LABEL[c], "Count": n, "Marker span": text})
    pd.DataFrame(exemplars).to_csv(TABLES / "h3_marker_spans.csv", index=False)
    print("    table:  tables/h3_marker_spans.csv")

    # representative full sentences: highest-count exact sentences per criterion
    reps = []
    for c in IF_COLS + ["os"]:
        sub = ds[ds[c].fillna(False)]
        counts = Counter(s.strip() for s in sub.sentence_text.dropna() if 25 < len(s.strip()) < 170)
        for text, n in counts.most_common(6):
            reps.append({"Criterion": CRIT_LABEL[c], "Count": n, "Sentence": text})
    pd.DataFrame(reps).to_csv(TABLES / "h3b_exemplar_sentences.csv", index=False)
    print("    table:  tables/h3b_exemplar_sentences.csv")

    # highest and lowest escalation dialogues, for the appendix case studies
    per_d = (ds.groupby(["dialogue_id", "target_model", "persona_id", "topic"], observed=True)
               .apply(lambda g: pd.Series({
                   "if_t1": 100 * g.loc[g.turn_id == 1, "if_any"].mean(),
                   "if_t6": 100 * g.loc[g.turn_id == 6, "if_any"].mean(),
                   "os_t1": 100 * g.loc[g.turn_id == 1, "os"].mean(),
                   "os_t6": 100 * g.loc[g.turn_id == 6, "os"].mean(),
                   "inner_rate": 100 * g.inner.mean(),
               }), include_groups=False).reset_index())
    for c in ["if_t1", "if_t6", "os_t1", "os_t6", "inner_rate"]:
        per_d[c] = pd.to_numeric(per_d[c], errors="coerce")
    per_d["if_delta"] = per_d.if_t6 - per_d.if_t1
    per_d.to_csv(DATA / "per_dialogue.csv", index=False)
    cases = pd.concat([per_d.nlargest(5, "if_delta"), per_d.nsmallest(5, "if_delta")])
    cases.to_csv(TABLES / "h4_case_dialogues.csv", index=False)
    print("    table:  tables/h4_case_dialogues.csv")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.imshow(co, cmap="Blues", vmin=0, vmax=100)
    labs = [CRIT_LABEL[c].replace(" ", "\n") for c in cols]
    ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=7)
    ax.set_yticks(range(5)); ax.set_yticklabels([CRIT_LABEL[c] for c in cols], fontsize=8)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{co[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if co[i, j] > 55 else "#222")
    ax.grid(False)
    ax.set_title("P(column criterion | row criterion), %")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    save_fig(fig, "fig_h_cooccurrence")

    RESULTS["H"] = {
        "buckets": mix.round(2).to_dict("records"),
        "cooccurrence": co_tbl.round(2).to_dict("records"),
        "overall_both_pct": round(100 * float((ds.if_any.fillna(False) & ds.os.fillna(False)).mean()), 2),
        "overall_neither_pct": round(100 * float((~ds.if_any.fillna(False) & ~ds.os.fillna(False)).mean()), 2),
    }
    print(f"    {RESULTS['H']['overall_both_pct']:.1f}% of sentences carry both poles; "
          f"{RESULTS['H']['overall_neither_pct']:.1f}% carry neither")


# ---------------------------------------------------------------- module I
def module_i(df):
    print("\n[I] Dialogue-level typology and consistency")
    ds = df[df.judge == DEEPSEEK]

    # 6-turn IF vector per dialogue -> k-means into three trajectory types
    vec = (ds.groupby(["dialogue_id", "turn_id"], observed=True)["if_any"].mean()
             .unstack("turn_id"))
    vec = vec.dropna()
    vec = vec.astype(float)
    X = vec.to_numpy(dtype=float) * 100
    rng = np.random.default_rng(7)
    k = 3
    centers = X[rng.choice(len(X), k, replace=False)]
    for _ in range(100):
        d = ((X[:, None, :] - centers[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        new = np.array([X[lab == i].mean(0) if (lab == i).any() else centers[i] for i in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    # name clusters by mean level and slope
    slope = centers[:, -1] - centers[:, 0]
    level = centers.mean(1)
    names = {}
    names[int(np.argmax(slope))] = "Escalating"
    remaining = [i for i in range(k) if i not in names]
    names[int(remaining[np.argmax(level[remaining])])] = "Persistently inward"
    names[[i for i in range(k) if i not in names][0]] = "Low inward"
    vec = vec.assign(cluster=[names[i] for i in lab])

    meta = ds.drop_duplicates("dialogue_id").set_index("dialogue_id")[["target_model", "persona_id"]]
    vec = vec.join(meta)
    comp = (pd.crosstab(vec.target_model, vec.cluster, normalize="index") * 100)
    comp = comp.reindex(MODEL_ORDER).reset_index().rename(columns={"target_model": "Model"})
    save_table(comp, "i1_trajectory_types",
               "Share of each model's dialogues assigned to each inward-facing trajectory "
               "type ($k$-means, $k=3$, on the six-turn inward-facing profile).",
               "tab:trajectories", float_fmt="%.1f")

    comp_p = (pd.crosstab(vec.persona_id, vec.cluster, normalize="index") * 100).reset_index()
    comp_p = comp_p.rename(columns={"persona_id": "Persona"})
    save_table(comp_p, "i2_trajectory_by_persona",
               "Trajectory-type composition by persona.", "tab:traj_persona", float_fmt="%.1f")

    # within-model consistency: dispersion of the per-dialogue IF rate
    cons = (ds.groupby(["target_model", "dialogue_id"], observed=True)["if_any"].mean()
              .groupby("target_model", observed=True)
              .agg(["mean", "std"]).mul(100).reindex(MODEL_ORDER))
    cons["CV (\\%)"] = 100 * cons["std"] / cons["mean"]
    cons = cons.reset_index().rename(columns={"target_model": "Model", "mean": "Mean IF (\\%)",
                                              "std": "SD across dialogues"})
    save_table(cons, "i3_consistency",
               "Dispersion of the per-dialogue inward-facing rate within each model. A low "
               "coefficient of variation means the behaviour is a stable property of the "
               "model rather than a response to particular situations.",
               "tab:consistency", float_fmt="%.1f")

    # does the turn effect survive controlling for reply length?
    per_turn = (ds.assign(if_any=ds.if_any.astype("float"))
                  .groupby(["dialogue_id", "turn_id"], observed=True)
                  .agg(if_rate=("if_any", "mean"), n_sent=("sentence_id", "size"),
                       model=("target_model", "first")).reset_index())
    import statsmodels.api as sm
    Xd = pd.get_dummies(per_turn["model"], drop_first=True).astype(float)
    Xd["turn"] = per_turn.turn_id - 1
    Xd["n_sent"] = per_turn.n_sent
    Xd = sm.add_constant(Xd)
    ols = sm.OLS(per_turn.if_rate * 100, Xd).fit(
        cov_type="cluster", cov_kwds={"groups": per_turn.dialogue_id})
    length_ctrl = {"turn_coef_pp_per_turn": round(float(ols.params["turn"]), 3),
                   "turn_p": float(ols.pvalues["turn"]),
                   "n_sent_coef": round(float(ols.params["n_sent"]), 3),
                   "n_sent_p": float(ols.pvalues["n_sent"])}

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
    ax = axes[0]
    cluster_color = {"Low inward": "#8FBBD9", "Escalating": "#DD8452",
                     "Persistently inward": "#C44E52"}
    for i, name in sorted(names.items(), key=lambda kv: kv[1]):
        n = int((np.array([names[x] for x in lab]) == name).sum())
        ax.plot(range(1, 7), centers[i], "-o", ms=4, color=cluster_color[name],
                label=f"{name} (n={n}, {100 * n / len(X):.0f}%)")
    ax.set_xlabel("assistant turn"); ax.set_ylabel("inward-facing (% of sentences)")
    ax.set_title("(a) Dialogue trajectory types"); ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    bottom = np.zeros(len(MODEL_ORDER))
    order_c = ["Low inward", "Escalating", "Persistently inward"]
    for cl in order_c:
        colr = cluster_color[cl]
        vals = comp.set_index("Model").reindex(MODEL_ORDER)[cl].to_numpy()
        ax.bar(range(len(MODEL_ORDER)), vals, bottom=bottom, color=colr, label=cl)
        bottom += vals
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, rotation=32, ha="right", fontsize=7.5)
    ax.set_ylabel("% of dialogues"); ax.set_ylim(0, 100)
    ax.set_title("(b) Composition by model"); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_fig(fig, "fig_i_trajectories")

    RESULTS["I"] = {
        "cluster_centers": {names[i]: [round(float(v), 1) for v in centers[i]] for i in range(k)},
        "cluster_sizes": {names[i]: int((lab == i).sum()) for i in range(k)},
        "composition_by_model": comp.round(2).to_dict("records"),
        "composition_by_persona": comp_p.round(2).to_dict("records"),
        "consistency": cons.round(2).to_dict("records"),
        "length_control": length_ctrl,
    }
    print(f"    turn effect controlling for reply length: "
          f"{length_ctrl['turn_coef_pp_per_turn']:+.2f} pp/turn (p={length_ctrl['turn_p']:.2g})")


# ---------------------------------------------------------------- module J
def module_j(df, fast=False):
    """Do the two poles respond to the same things? And is the judge self-consistent?"""
    print("\n[J] Pole dissociation and judge self-consistency")
    ds = df[df.judge == DEEPSEEK]
    nb = 2000 if fast else NBOOT

    # --- which factor moves which pole
    rows = []
    def spread(col, by):
        g = ds.groupby(by, observed=True)[col].mean().mul(100).dropna()
        return float(g.max() - g.min())

    for factor, by, label, contrast in [
            ("Model identity", "target_model", "7 target models", None),
            ("Conversation depth", "turn_id", "turn 1 to turn 6", ("turn", 1, 6)),
            ("User disclosure", "persona_id", "persona P1 to P3", ("persona", "P1", "P3")),
            ("Situation topic", "topic", "17 topics", None)]:
        r = {"Factor": factor, "Levels": label}
        for col, name in [("if_any", "IF"), ("os", "OS")]:
            r[f"{name} spread (pp)"] = spread(col, by)
        r["OS / IF"] = r["OS spread (pp)"] / r["IF spread (pp)"]
        r["Moves"] = "inward pole" if r["OS / IF"] < 1 else "outward pole"
        # attach the headline contrast with its interval where one exists
        if contrast:
            kind, a_, b_ = contrast
            key = "turn_id" if kind == "turn" else "persona_id"
            sa, sb = ds[ds[key] == b_], ds[ds[key] == a_]
            for col, name in [("if_any", "IF"), ("os", "OS")]:
                pooled = pd.concat([
                    pd.DataFrame({"v": sa[col].astype(float), "g": 1, "d": sa.dialogue_id}),
                    pd.DataFrame({"v": sb[col].astype(float), "g": 0, "d": sb.dialogue_id})]).dropna()
                d, lo, hi, pv = S.cluster_boot_diff(
                    (pooled.v * (pooled.g == 1)).to_numpy(), (pooled.g == 1).to_numpy().astype(float),
                    (pooled.v * (pooled.g == 0)).to_numpy(), (pooled.g == 0).to_numpy().astype(float),
                    pooled.d.to_numpy(), n_boot=nb)
                r[f"{name} contrast"] = f"{100 * d:+.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"
        rows.append(r)
    diss = pd.DataFrame(rows)
    save_table(diss[["Factor", "Levels", "IF spread (pp)", "OS spread (pp)", "OS / IF", "Moves"]],
               "j1_pole_dissociation",
               "Which factor moves which pole. Spread is the range of the group means "
               "(\% of sentences) across the levels of that factor. The inward pole is "
               "moved by who is speaking and how long they have been speaking; the outward "
               "pole is moved by what the user discloses and what the situation is about.",
               "tab:dissociation", float_fmt="%.1f")
    diss.to_csv(TABLES / "j1_pole_dissociation_full.csv", index=False)

    # --- judge self-consistency on repeated sentences: a second reliability floor
    d2 = ds.copy()
    d2["t"] = d2.sentence_text.fillna("").str.strip().str.replace(r"\s+", " ", regex=True)
    d2["n_occ"] = d2.groupby("t")["t"].transform("size")
    rep = d2[(d2.n_occ >= 5) & (d2.t.str.len() > 25)]
    cons = []
    for c in IF_COLS + ["os"]:
        modal = rep.groupby("t")[c].apply(lambda s: max(s.mean(), 1 - s.mean()))
        n = rep.groupby("t").size()
        cons.append({"Criterion": CRIT_LABEL[c],
                     "Modal-label agreement (\%)": 100 * float(np.average(modal, weights=n)),
                     "Strings labelled identically (\%)": 100 * float((modal == 1).mean())})
    cons = pd.DataFrame(cons)
    save_table(cons, "j2_judge_self_consistency",
               f"Judge self-consistency on the {rep.t.nunique()} distinct assistant sentences "
               f"that recur at least five times in the corpus ({len(rep):,} instances). "
               "Modal-label agreement is the share of instances carrying that sentence's "
               "majority label. These values exceed the inter-judge agreement in Table~"
               "\\ref{tab:judge_agreement} on every criterion, which locates the "
               "judge disagreement in thresholds rather than in labelling noise.",
               "tab:self_consistency", float_fmt="%.1f")

    # --- is the judge reading context? the canonical multi-label sentence
    from scipy import stats as sps
    probe = d2[d2.t.str.contains("go through this alone", na=False)].copy()
    turn_tot = (ds.groupby(["dialogue_id", "turn_id"], observed=True)
                  .agg(turn_os=("os", "sum"), turn_n=("os", "size")).reset_index())
    probe = probe.merge(turn_tot, on=["dialogue_id", "turn_id"])
    probe["other_os"] = (probe.turn_os - probe.os.astype(float)) / (probe.turn_n - 1)
    probe = probe.dropna(subset=["other_os", "os"])
    pb = sps.pointbiserialr(probe.os.astype(int), probe.other_os)
    ctx = {
        "sentence": "You don't have to go through this alone.",
        "n": int(len(probe)),
        "bond_met_pct": round(100 * float(probe.bond.mean()), 1),
        "os_met_pct": round(100 * float(probe.os.mean()), 1),
        "other_os_when_os_true": round(100 * float(probe.loc[probe.os == True, "other_os"].mean()), 1),
        "other_os_when_os_false": round(100 * float(probe.loc[probe.os == False, "other_os"].mean()), 1),
        "point_biserial_r": round(float(pb.statistic), 3),
        "p": float(pb.pvalue),
    }
    print(f"    context sensitivity on the canonical multi-label sentence: "
          f"r={ctx['point_biserial_r']:+.3f}, p={ctx['p']:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    ax = axes[0]
    xs = np.arange(len(diss))
    ax.bar(xs - 0.2, diss["IF spread (pp)"], 0.4, color=IF_COLOR, label="Inward-facing")
    ax.bar(xs + 0.2, diss["OS spread (pp)"], 0.4, color=OS_COLOR, label="Outward-scaffolding")
    ax.set_xticks(xs)
    ax.set_xticklabels([f.replace(" ", "\n") for f in diss["Factor"]], fontsize=7.5)
    ax.set_ylabel("spread of group means (pp)")
    ax.set_title("(a) The two poles answer to different things")
    ax.set_ylim(0, max(diss["IF spread (pp)"].max(), diss["OS spread (pp)"].max()) * 1.28)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    ax = axes[1]
    inter = {"Bond Anchoring": 83.4, "Intimate Dyad": 69.2, "Inner Life Claims": 95.1,
             "Return Hooks": 79.5, "Outward-Scaffolding": 70.9}
    ys = np.arange(len(cons))
    ax.barh(ys + 0.2, cons["Modal-label agreement (\%)"], 0.4,
            color="#55A868", label="same judge, repeated sentence")
    ax.barh(ys - 0.2, [inter[c] for c in cons["Criterion"]], 0.4,
            color="#8172B3", label="across judges")
    ax.set_yticks(ys); ax.set_yticklabels(cons["Criterion"], fontsize=8)
    ax.set_xlim(50, 100); ax.set_xlabel("agreement (%)")
    ax.set_ylim(-0.55, len(cons) - 0.1 + 0.75)   # headroom for the legend
    ax.set_title("(b) Self-consistency exceeds inter-judge agreement")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    save_fig(fig, "fig_j_dissociation")

    RESULTS["J"] = {
        "dissociation": diss.round(2).to_dict("records"),
        "self_consistency": cons.round(2).to_dict("records"),
        "context_probe": ctx,
        "n_repeated_strings": int(rep.t.nunique()),
        "n_repeated_instances": int(len(rep)),
    }


# ---------------------------------------------------------------- main
MODULES = {"A": module_a, "B": module_b, "C": module_c, "D": module_d,
           "E": module_e, "F": module_f, "G": module_g, "H": module_h, "I": module_i, "J": module_j}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", nargs="*", default=list(MODULES))
    ap.add_argument("--fast", action="store_true", help="2,000 resamples, skip GEE/GLMM")
    ap.add_argument("--no-glmm", action="store_true")
    args = ap.parse_args()

    df = load()
    print(f"loaded {len(df):,} rows ({df.judge.nunique()} judges)")
    for name in args.modules:
        fn = MODULES[name]
        kwargs = {}
        if name in {"C", "D", "E", "F", "G", "J"}:
            kwargs["fast"] = args.fast
        if name == "D":
            kwargs["glmm"] = not args.no_glmm
        fn(df, **kwargs)

    out = HERE / "results.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(RESULTS)
    out.write_text(json.dumps(prev, indent=1, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
