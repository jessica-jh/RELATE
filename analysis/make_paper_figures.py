"""Build every figure that appears in the RELATE paper (main results and
appendix), and nothing else. The numbers-only companion is
analysis/paper_numbers.py; between the two, this pair reproduces everything
cited in the paper without touching analyze.py's larger internal dossier
(~35 tables, 10 figures, most of which never made it into the paper).

The analysis figures in analysis/figures/ are sized for screen reading. These are
sized for the 5.5in ICLR text block so they are placed at scale=1 with no shrink,
and are typeset in a serif face to sit with the body text.

  python analysis/make_paper_figures.py

Figures produced (function -> LaTeX label -> where it's used):
  fig_depth               -> fig:depth               (main body, Sec. 6.2)
  fig_persona              -> fig:persona             (main body, Sec. 6.3)
  fig_model_trajectories   -> fig:model-trajectories  (appendix)
  fig_topic                -> fig:topic               (appendix)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import statlib as S  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

OUT = HERE / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

TEXTWIDTH = 5.5          # ICLR \textwidth, inches
DEEPSEEK = "deepseek-r1-distill-qwen-32b"
IF_COLS = ["bond", "dyad", "inner", "hook"]
CRIT = {"bond": "Bond Anchoring", "dyad": "Intimate Dyad", "inner": "Inner Life Claims",
        "hook": "Return Hooks", "os": "Outward-Scaffolding"}
# inward = blue, outward = red, held constant across every figure in the paper
IF_COLOR, OS_COLOR = "#C44E52", "#3B7EA1"
# per-persona shade ramps, light (P1) -> dark (P3), one per dimension's hue
PERSONA_COLOR_IF = {"P1": "#F4A6A0", "P2": "#D9534F", "P3": "#7B241C"}
PERSONA_COLOR_OS = {"P1": "#9CC3DC", "P2": "#4C72B0", "P3": "#16324F"}
# family order, matching Table 1: Llama (asc.), Qwen (asc.), DeepSeek, Claude, Mistral
MODEL_ORDER = ["Llama-3.1-8B", "Llama-3.1-70B", "Qwen3-14B", "Qwen3-32B",
               "DeepSeek-V3", "Claude-Haiku-4.5", "Mistral-7B"]
# checkpoint names as cited; keys stay the short ids used in the sentence table
MODEL_LABEL = {
    "Llama-3.1-8B": "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B": "Llama-3.1-70B-Instruct-Turbo",
    "Qwen3-14B": "Qwen3-14B",
    "Qwen3-32B": "Qwen3-32B",
    "DeepSeek-V3": "DeepSeek-V3-0324",
    "Claude-Haiku-4.5": "Claude-Haiku-4.5",
    "Mistral-7B": "Mistral-7B-Instruct-v0.3",
}
# same hue within a family, darker = larger scale; singletons get their own hue,
# chosen to stay clear of the reserved inward-blue / outward-red pair above
MODEL_COLOR = {"Llama-3.1-8B": "#81C784", "Llama-3.1-70B": "#2E7D32",
               "Qwen3-14B": "#B39DDB", "Qwen3-32B": "#5E35B1",
               "DeepSeek-V3": "#EF6C00", "Claude-Haiku-4.5": "#D81B60",
               "Mistral-7B": "#795548"}
# darker, higher-contrast variants of MODEL_COLOR for label text only -- the
# light shades above (8B, 14B) read fine as lines but disappear as text
MODEL_TEXT_COLOR = {"Llama-3.1-8B": "#2E7D32", "Llama-3.1-70B": "#1B5E20",
                     "Qwen3-14B": "#5E35B1", "Qwen3-32B": "#311B92",
                     "DeepSeek-V3": "#E65100", "Claude-Haiku-4.5": "#AD1457",
                     "Mistral-7B": "#4E342E"}
# four inward criteria as a blue ramp (light -> dark = weak -> strong signal);
# Inner Life Claims gets the darkest shade and a bolder line since it is the
# one that keeps accelerating
CRIT_COLOR = {"bond": "#FC9272", "dyad": "#EF3B2C", "hook": "#A50F15",
              "inner": "#67000D", "os": OS_COLOR}
# distinct dash pattern + marker per line so panel (b) reads in grayscale too
CRIT_STYLE = {"bond":  dict(ls=(0, (1, 1)),       marker="o"),
              "dyad":  dict(ls=(0, (5, 2)),       marker="s"),
              "hook":  dict(ls=(0, (3, 1, 1, 1)), marker="^"),
              "inner": dict(ls="-",               marker="D"),
              "os":    dict(ls=(0, (6, 2)),       marker="x")}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "legend.fontsize": 6.2,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "figure.dpi": 400,
})


def load():
    p = HERE / "data/sentences.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(HERE / "data/sentences.csv.gz")
    return df[df.judge == DEEPSEEK].copy()


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  analysis/paper_figures/{name}.pdf  ({fig.get_size_inches()[0]:.2f} x "
          f"{fig.get_size_inches()[1]:.2f} in)")


def fig_depth(ds):
    """One figure for the turn-depth story: models diverge, and only
    inner-life claims keep accelerating once the other criteria plateau."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.62))
    turns = np.arange(1, 7)

    ax = axes[0]
    # diverging background: above zero reads as net inward (red), below as
    # net outward (blue) -- a very light wash so it reads at a glance without
    # competing with the per-model lines on top
    ax.axhspan(0, 44, color=IF_COLOR, alpha=0.05, lw=0, zorder=0)
    ax.axhspan(-24, 0, color=OS_COLOR, alpha=0.05, lw=0, zorder=0)

    end_y = {}
    for m in MODEL_ORDER:
        sub = ds[ds.target_model == m]
        y = [100 * sub.loc[sub.turn_id == t, "if_any"].mean()
             - 100 * sub.loc[sub.turn_id == t, "os"].mean() for t in turns]
        ax.plot(turns, y, "-", marker="o", ms=1.9, color=MODEL_COLOR[m], lw=1.1, zorder=3)
        end_y[m] = y[-1]
    ax.axhline(0, color="#333", lw=0.7, zorder=1)

    # stack the turn-6 labels vertically so they do not overlap, preserving
    # each model's relative order (highest net orientation on top)
    order = sorted(end_y, key=lambda m: end_y[m], reverse=True)
    min_gap = 5.4
    placed = []
    for m in order:
        y = end_y[m]
        if placed and y > placed[-1] - min_gap:
            y = placed[-1] - min_gap
        placed.append(y)
        ax.annotate(m, (6, end_y[m]), xytext=(4, y - end_y[m]),
                    textcoords="offset points", fontsize=5.6, fontweight="bold",
                    color=MODEL_TEXT_COLOR[m], va="center", ha="left",
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])

    ax.set_xlabel("assistant turn"); ax.set_ylabel("net orientation (pp)")
    ax.set_title("(a) Net orientation by model")
    ax.set_xticks(turns); ax.set_xlim(0.7, 9.0)
    ax.set_ylim(-24, 44)

    ax = axes[1]
    # absolute rates differ by an order of magnitude and hide the shapes;
    # growth relative to turn 1 is what the claim is about
    handles, labels = [], []
    for c in ["os"] + IF_COLS:
        y = np.array([ds.loc[ds.turn_id == t, c].mean() for t in turns], dtype=float)
        st = CRIT_STYLE[c]
        line, = ax.plot(turns, y / y[0], color=CRIT_COLOR[c], ls=st["ls"], marker=st["marker"],
                        ms=2.4 if c != "inner" else 2.9, lw=1.5 if c == "inner" else 1.0,
                        mfc="none" if c == "os" else CRIT_COLOR[c],
                        label=CRIT[c].replace("Outward-Scaffolding", "Outward-scaff."))
        handles.append(line); labels.append(line.get_label())
    ax.axhline(1, color="#999", lw=0.6, zorder=0)
    ax.set_xlabel("assistant turn"); ax.set_ylabel(r"rate $\div$ turn-1 rate")
    ax.set_title("(b) Criterion growth relative to turn 1")
    ax.set_ylim(0.8, 3.6); ax.set_xticks(turns)
    ax.legend(handles, labels, frameon=False, loc="upper left", ncol=1,
              handlelength=1.6, borderpad=0.1, labelspacing=0.22)

    fig.tight_layout(pad=0.28, w_pad=1.1)
    save(fig, "fig_depth")


def fig_persona(ds):
    """One figure for disclosure: inward converges, outward stays stratified."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.48), sharex=True)
    turns = np.arange(1, 7)
    grid = (ds.groupby(["persona_id", "turn_id"], observed=True)[["if_any", "os"]]
              .mean().mul(100).reset_index())
    labels = {"P1": "P1 hesitant", "P2": "P2 self-minimizing", "P3": "P3 high-disclosure"}

    for ax, col, ramp, ttl in [(axes[0], "if_any", PERSONA_COLOR_IF, "(a) Inward-facing by persona"),
                               (axes[1], "os", PERSONA_COLOR_OS, "(b) Outward-scaffolding by persona")]:
        for p_ in ["P1", "P2", "P3"]:
            sub = grid[grid.persona_id == p_]
            ax.plot(sub.turn_id, sub[col], "-", marker="o", ms=2.8,
                    color=ramp[p_], label=labels[p_])
        ax.set_xlabel("assistant turn")
        ax.set_ylabel("% of sentences")
        ax.set_title(ttl)
        ax.set_xticks(turns)
    # one shared scale so "converges" and "stays stratified" are directly comparable,
    # wide enough to contain both panels' turn-1 values
    lo = min(grid.if_any.min(), grid.os.min()) - 3
    hi = max(grid.if_any.max(), grid.os.max()) + 4
    axes[0].set_ylim(lo, hi); axes[1].set_ylim(lo, hi)
    # each panel gets its own legend -- the two panels no longer share a color
    # ramp (red shades for inward, blue shades for outward), so one legend
    # would not identify the lines in the other panel
    axes[0].legend(frameon=False, loc="lower right", handlelength=1.4, borderpad=0.1)
    axes[1].legend(frameon=False, loc="lower right", handlelength=1.4, borderpad=0.1)

    # annotate the P3-P1 gap at turn 6 on both panels, so the reader compares
    # two numbers -- near-zero vs. 13 pp -- instead of eyeballing line spacing.
    # A vertical bracket at x=6.1 shows the actual span, but only when it is
    # large enough to read as a bracket (>2 pp); below that it collapses into
    # an illegible knot, so we drop it and keep just the offset text + a
    # single leader line to the midpoint.
    for ax, col in [(axes[0], "if_any"), (axes[1], "os")]:
        g1 = grid[(grid.persona_id == "P1") & (grid.turn_id == 6)][col].iloc[0]
        g3 = grid[(grid.persona_id == "P3") & (grid.turn_id == 6)][col].iloc[0]
        mid = (g1 + g3) / 2
        if abs(g3 - g1) > 2:
            ax.annotate("", xy=(6.1, g1), xytext=(6.1, g3),
                        arrowprops=dict(arrowstyle="<->", lw=0.7, color="#333",
                                         shrinkA=0, shrinkB=0))
            leader = dict(arrowstyle="-", lw=0.5, color="#333", shrinkA=2, shrinkB=3)
        else:
            leader = dict(arrowstyle="-", lw=0.6, color="#333", shrinkA=0, shrinkB=3)
        ax.annotate(rf"$\Delta$ = {g3 - g1:+.1f} pp", xy=(6.1, mid), xytext=(16, 0),
                    textcoords="offset points", fontsize=6, va="center", ha="left",
                    arrowprops=leader)
        ax.set_xlim(0.7, 7.4)

    fig.tight_layout(pad=0.28, w_pad=1.3)
    save(fig, "fig_persona")


HIGH_STAKES = {"domestic-violence", "trauma", "substance-abuse",
               "eating-disorders", "depression"}


def fig_topic(ds, n_boot=4000):
    """Appendix figure: net orientation by source-situation topic.

    The point is that the higher-stakes topics are spread across the whole range
    rather than clustered at the outward end, so they are drawn in a second colour.
    """
    rows = []
    for topic, sub in ds.groupby("topic"):
        m = sub.if_any.notna() & sub.os.notna()
        t = sub[m]
        d, lo, hi, _ = S.cluster_boot_diff(
            t.if_any.astype(float).to_numpy(), np.ones(len(t)),
            t.os.astype(float).to_numpy(), np.ones(len(t)),
            t.dialogue_id.to_numpy(), n_boot=n_boot)
        rows.append({"topic": topic.replace("-", " "), "net": 100 * d,
                     "lo": 100 * lo, "hi": 100 * hi,
                     "stakes": topic in HIGH_STAKES})
    tbl = pd.DataFrame(rows).sort_values("net").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.55))
    ys = np.arange(len(tbl))
    for y, r in tbl.iterrows():
        ax.plot([r.lo, r.hi], [y, y], color="#C3C8CF", lw=1.1, zorder=1,
                solid_capstyle="round")
        ax.scatter(r.net, y, s=17, zorder=3,
                   color="#C44E52" if r.stakes else "#4C72B0")
    ax.axvline(0, color="#333", lw=0.7)
    ax.set_yticks(ys); ax.set_yticklabels(tbl.topic, fontsize=6.2)
    ax.set_ylim(-0.8, len(tbl) - 0.2)
    ax.set_xlabel("net orientation, IF $-$ OS (pp)")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", alpha=0.12)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=3.6, color="#C44E52",
                          label="higher-stakes topic"),
               plt.Line2D([], [], marker="o", ls="", ms=3.6, color="#4C72B0",
                          label="other")]
    ax.legend(handles=handles, frameon=False, loc="lower right", handletextpad=0.3,
              borderpad=0.1)
    # the pre-specified stakes test, stated on the figure so the null is
    # legible without cross-referencing the appendix prose -- placed top-left,
    # clear of both the lower-right legend and the topic error bars
    ax.text(0.02, 0.90,
            r"high-stakes vs.\ other: $\Delta$IF $=-0.04$~pp, $\Delta$OS $=-0.60$~pp (both n.s.)",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.6,
            color="#666", style="italic")
    fig.tight_layout(pad=0.28)
    save(fig, "fig_topic")


def fig_model_trajectories(ds):
    """Small multiples: IF and OS by turn, one panel per model.

    Ordered to match Table 1 -- grouped by family, ascending scale within
    family, Mistral last. The shaded gap is net orientation; a model that
    crosses zero has the shading flip from red-on-top to blue-on-top.
    """
    order = ["Llama-3.1-8B", "Llama-3.1-70B", "Qwen3-14B", "Qwen3-32B",
             "DeepSeek-V3", "Claude-Haiku-4.5", "Mistral-7B"]
    turns = np.arange(1, 7)

    fig, axes = plt.subplots(2, 4, figsize=(TEXTWIDTH, 2.55), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, m in zip(axes, order):
        sub = ds[ds.target_model == m]
        y_if = np.array([100 * sub.loc[sub.turn_id == t, "if_any"].mean() for t in turns])
        y_os = np.array([100 * sub.loc[sub.turn_id == t, "os"].mean() for t in turns])
        ax.fill_between(turns, y_if, y_os, where=(y_if >= y_os),
                        color=IF_COLOR, alpha=0.15, lw=0, interpolate=True)
        ax.fill_between(turns, y_if, y_os, where=(y_if < y_os),
                        color=OS_COLOR, alpha=0.15, lw=0, interpolate=True)
        ax.plot(turns, y_if, "-", marker="o", ms=2.2, color=IF_COLOR, label="Inward-facing")
        ax.plot(turns, y_os, "-", marker="s", ms=2.2, color=OS_COLOR, label="Outward-scaff.")
        ax.set_title(MODEL_LABEL[m], fontsize=5.7, pad=2)
        ax.set_xticks(turns)
        ax.set_ylim(15, 85)

    axes[-1].axis("off")
    axes[-1].legend(*axes[0].get_legend_handles_labels(), loc="center",
                    frameon=False, handlelength=1.4, fontsize=6.5)

    for ax in axes[[0, 4]]:
        ax.set_ylabel("% of sentences")
    for ax in axes[4:7]:
        ax.set_xlabel("assistant turn")

    fig.tight_layout(pad=0.28, w_pad=0.6, h_pad=0.7)
    save(fig, "fig_model_trajectories")


if __name__ == "__main__":
    ds = load()
    print("writing ICLR-width figures:")
    fig_depth(ds)
    fig_persona(ds)
    fig_topic(ds)
    fig_model_trajectories(ds)
