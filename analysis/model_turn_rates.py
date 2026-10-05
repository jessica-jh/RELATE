"""Generate the appendix table of IF and OS rates by target model and turn
(tab:model-turn-rates), the values plotted in fig_model_trajectories.

Rates are sentence-weighted within each model and turn. Each dimension uses
its own non-missing denominator, as in the rest of the paper.

  python analysis/model_turn_rates.py
"""
from pathlib import Path

import pandas as pd

from paper_numbers import MODEL_ORDER, load

TURNS = range(1, 7)


def main():
    _, ds = load()
    assert len(ds) == 69194 and ds.dialogue_id.nunique() == 1596
    rows = []
    for model in MODEL_ORDER:
        for measure, col in [("IF", "if_any"), ("OS", "os")]:
            for t in TURNS:
                v = ds.loc[(ds.target_model == model) & (ds.turn_id == t), col]
                valid = int(v.notna().sum())
                positive = int(v.fillna(False).astype(bool).sum())
                rows.append(dict(model=model, measure=measure, turn=t,
                                 positive_sentences=positive, valid_sentences=valid,
                                 rate_percent=100 * positive / valid))
    table = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent / "tables"
    table.to_csv(out / "model_turn_rates.csv", index=False)

    wide = table.pivot_table(index=["model", "measure"], columns="turn",
                             values="rate_percent", sort=False)
    blocks = []
    for model in MODEL_ORDER:
        lines = []
        for measure in ["IF", "OS"]:
            vals = " & ".join(f"{wide.loc[(model, measure), t]:.1f}" for t in TURNS)
            lines.append(f"{model if measure == 'IF' else ''} & {measure} & {vals} " + r"\\")
        blocks.append("\n".join(lines))
    tex = r"""\begin{table}[htbp]
\centering
\small
\caption{IF and OS rates (\%) by target model and assistant turn,
corresponding to Figure~\ref{fig:model-trajectories}.
Rates pool sentences within each model and turn and use
DeepSeek judge annotations. Values are rounded to one decimal place.}
\label{tab:model-turn-rates}
\begin{tabular}{llrrrrrr}
\toprule
Model & Measure & Turn 1 & Turn 2 & Turn 3 & Turn 4 & Turn 5 & Turn 6 \\
\midrule
""" + "\n\\addlinespace\n".join(blocks) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (out / "model_turn_rates.tex").write_text(tex)
    print(wide.round(1).to_string())


if __name__ == "__main__":
    main()
