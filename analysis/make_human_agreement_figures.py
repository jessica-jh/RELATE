"""Figures for the human annotation agreement analysis (Appendix F).

Two figures:
  fig_human_agreement  -- how much the raters agree, and what they mark
  fig_human_depth      -- how IF and OS move with conversational depth

    python analysis/make_human_agreement_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import figstyle
import statlib as S
from human_agreement import (DS, G4, HUMAN_A, HUMAN_B, PAIRS, RATER_ORDER,
                             DEEPSEEK_FILE, GPT4O_FILE, ANNOTATOR_1, ANNOTATOR_2,
                             composition, judge_frame, load_human, load_judge)

HERE = Path(__file__).resolve().parent
OUT = HERE / "paper_figures"

IF_COLOR, OS_COLOR = "#C44E52", "#3B7EA1"
RATER_COLOR = {HUMAN_A: "#1B5E20", HUMAN_B: "#66A05F", DS: "#C44E52", G4: "#3B7EA1"}
RATER_MARKER = {HUMAN_A: "o", HUMAN_B: "v", DS: "s", G4: "^"}
RATER_LS = {HUMAN_A: "-", HUMAN_B: "-", DS: "--", G4: "-."}
COMP_COLOR = {"IF only": IF_COLOR, "OS only": OS_COLOR,
              "Both": "#8E6C8A", "Neither": "#C9C9C9"}

# Landis and Koch bands, drawn behind the kappa bars so the numbers can be read
# as strengths rather than as bare decimals.
BANDS = [(0.0, 0.2, "slight"), (0.2, 0.4, "fair"), (0.4, 0.6, "moderate"),
         (0.6, 0.8, "substantial"), (0.8, 1.0, "almost perfect")]
BAND_SHADE = ["#FBEAEA", "#FCF1E3", "#FAF8E0", "#EDF5E6", "#E2F0E4"]


def short(pair):
    return pair.replace("Annotator ", "A")


def load_all():
    hum_a, hum_b = load_human(ANNOTATOR_1), load_human(ANNOTATOR_2)
    meta = hum_a[["Dialogue", "Turn", "dialogue_id"]].copy()
    keys_in_order = [(r.dialogue_id, str(int(r.Turn)), str(int(r.Sentence)))
                     for r in hum_a.itertuples(index=False)]
    keys = set(keys_in_order)
    ds = judge_frame(load_judge(DEEPSEEK_FILE, keys), keys_in_order)
    g4 = judge_frame(load_judge(GPT4O_FILE, keys), keys_in_order)
    rater = {
        HUMAN_A: {"if": hum_a["if_any"].to_numpy(), "os": hum_a["os"].to_numpy()},
        HUMAN_B: {"if": hum_b["if_any"].to_numpy(), "os": hum_b["os"].to_numpy()},
        DS: {"if": ds["if_any"].to_numpy(bool), "os": ds["os"].to_numpy(bool)},
        G4: {"if": g4["if_any"].to_numpy(bool), "os": g4["os"].to_numpy(bool)},
    }
    return rater, meta


def fig_agreement(rater, meta, n):
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.TEXTWIDTH, 2.45),
                             gridspec_kw={"width_ratios": [1.32, 1]})

    # -- (a) Cohen's kappa for IF and OS, every pair -----------------------
    ax = axes[0]
    labels = [short(f"{a} vs {b}") for a, b in PAIRS]
    y = np.arange(len(PAIRS))[::-1]
    h = 0.34
    for lo, hi, shade in zip([b[0] for b in BANDS], [b[1] for b in BANDS], BAND_SHADE):
        ax.axvspan(lo, hi, color=shade, lw=0, zorder=0)
    for dx, (measure, col, colour) in enumerate([("IF", "if", IF_COLOR),
                                                 ("OS", "os", OS_COLOR)]):
        vals = [S.cohen_kappa(rater[a][col], rater[b][col]) for a, b in PAIRS]
        ax.barh(y + (h / 2 if dx == 0 else -h / 2), vals, height=h, color=colour,
                label=measure, zorder=3, edgecolor="white", lw=0.4)
        for yi, v in zip(y, vals):
            ax.text(v + 0.015, yi + (h / 2 if dx == 0 else -h / 2), f"{v:.2f}",
                    va="center", ha="left", fontsize=figstyle.ANNOT,
                    color=figstyle.darken_for_text(colour))
    for lo, hi, name in BANDS:
        ax.text((lo + hi) / 2, len(PAIRS) - 0.28, name, ha="center", va="bottom",
                fontsize=figstyle.ANNOT - 0.5, color=figstyle.MUTED, rotation=0)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.7, len(PAIRS) - 0.05)
    ax.set_xlabel("Cohen's $\\kappa$")
    ax.set_title("(a) Agreement, chance-corrected", fontsize=figstyle.PANEL_TITLE)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False, handlelength=1.1, borderpad=0.2)

    # -- (b) what each rater actually marks --------------------------------
    ax = axes[1]
    names = RATER_ORDER
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    for cat in ["IF only", "OS only", "Both", "Neither"]:
        vals = np.array([100 * (composition(rater[nm]["if"], rater[nm]["os"]) == cat).mean()
                         for nm in names])
        ax.bar(x, vals, 0.62, bottom=bottom, color=COMP_COLOR[cat], label=cat,
               edgecolor="white", lw=0.5)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 6:
                ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center",
                        fontsize=figstyle.ANNOT,
                        color="white" if cat != "Neither" else figstyle.TEXT)
        bottom += vals
    ax.set_xticks(x, [nm.replace("Annotator ", "A") for nm in names])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of sentences")
    ax.set_title("(b) What each rater marks", fontsize=figstyle.PANEL_TITLE)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.42), frameon=False,
              handlelength=0.9, columnspacing=0.8, handletextpad=0.4)

    fig.suptitle(f"Human and judge agreement on {n:,} doubly annotated assistant sentences",
                 fontsize=figstyle.TITLE)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_human_agreement.{ext}")
    plt.close(fig)
    print(f"    figure: {OUT.name}/fig_human_agreement.pdf + .png")


def fig_depth(rater, meta):
    turn = meta["Turn"].to_numpy()
    turns = np.array(sorted(set(turn)))
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.TEXTWIDTH, 2.1),
                             gridspec_kw={"width_ratios": [1, 1, 1.18]})

    for ax, (measure, col) in zip(axes[:2], [("IF", "if"), ("OS", "os")]):
        for nm in RATER_ORDER:
            vals = [100 * rater[nm][col][turn == t].mean() for t in turns]
            ax.plot(turns, vals, marker=RATER_MARKER[nm], ls=RATER_LS[nm], ms=2.6,
                    color=RATER_COLOR[nm], label=nm.replace("Annotator ", "A"))
        ax.set_xlabel("assistant turn")
        ax.set_ylim(0, 80)
        ax.set_xticks(turns)
        ax.set_title(f"({'a' if measure == 'IF' else 'b'}) {measure} rate by turn",
                     fontsize=figstyle.PANEL_TITLE)
    axes[0].set_ylabel("% of sentences")
    axes[0].legend(frameon=False, loc="lower left", handlelength=1.4, ncol=2,
                   labelspacing=0.2, columnspacing=0.8, borderpad=0.15,
                   handletextpad=0.4)

    # -- (c) within-dialogue turn 6 minus turn 1, with bootstrap CI --------
    ax = axes[2]
    rows = []
    for measure, col in [("IF", "if"), ("OS", "os")]:
        for nm in RATER_ORDER:
            per_dialogue = []
            for _, idx in meta.groupby("dialogue_id").groups.items():
                pos = meta.index.get_indexer(idx)
                first = rater[nm][col][pos][turn[pos] == 1]
                last = rater[nm][col][pos][turn[pos] == 6]
                if len(first) and len(last):
                    per_dialogue.append(100 * (last.mean() - first.mean()))
            mean, lo, hi, _, _ = S.paired_cluster_boot(per_dialogue)
            rows.append((measure, nm, mean, lo, hi))
    ch = pd.DataFrame(rows, columns=["measure", "rater", "mean", "lo", "hi"])
    y = np.arange(len(ch))[::-1]
    ax.axvline(0, color=figstyle.MUTED, lw=0.6, zorder=1)
    for yi, r in zip(y, ch.itertuples(index=False)):
        colour = IF_COLOR if r.measure == "IF" else OS_COLOR
        ax.plot([r.lo, r.hi], [yi, yi], color=colour, lw=1.0, zorder=2,
                solid_capstyle="butt")
        ax.plot([r.mean], [yi], marker=RATER_MARKER[r.rater], ms=3.2, color=colour,
                zorder=3, mec="white", mew=0.4)
    ax.set_yticks(y, [f"{r.measure}  {r.rater.replace('Annotator ', 'A')}"
                      for r in ch.itertuples(index=False)])
    ax.set_xlabel("turn 6 $-$ turn 1 (pp)")
    span = ch["hi"].max() - ch["lo"].min()
    ax.set_xlim(ch["lo"].min() - 0.08 * span, ch["hi"].max() + 0.08 * span)
    ax.set_title("(c) Within-dialogue change", fontsize=figstyle.PANEL_TITLE)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)

    fig.suptitle("Conversational depth: the judges show IF rising, the annotators do not",
                 fontsize=figstyle.TITLE)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_human_depth.{ext}")
    plt.close(fig)
    print(f"    figure: {OUT.name}/fig_human_depth.pdf + .png")


def main():
    figstyle.apply()
    rater, meta = load_all()
    n = len(meta)
    print(f"[H] figures from {n:,} sentences, {meta.dialogue_id.nunique()} dialogues")
    fig_agreement(rater, meta, n)
    fig_depth(rater, meta)


if __name__ == "__main__":
    main()
