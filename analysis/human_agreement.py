"""Human annotation agreement, in the format of Appendix F.

Appendix F reports judge-vs-judge agreement on IF and OS. This adds the human
leg: human vs human, and each human against each judge, on the same sentences.

Two humans independently annotated 21 dialogues. Their workbooks mark each
assistant sentence with 'o' under IF, under OS, both, or neither; 'o' means the
pole is present and a blank cell means it is absent. The guide permits both
poles on one sentence, so a blank is a judgement rather than an omission. The
judges emit a MET/NOT_MET label per criterion, so IF/OS are derived with the
Appendix E.1 rule before anything is compared.

  python analysis/human_agreement.py
"""
import gzip
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import statlib as S

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TABLES = HERE / "tables"

ANNOTATOR_1 = ROOT / "annotation/annotator1_judgments.xlsx"
ANNOTATOR_2 = ROOT / "annotation/annotator2_judgments.xlsx"
DEEPSEEK_FILE = ROOT / "deepseek_judge/judgments.jsonl.gz"
GPT4O_FILE = ROOT / "openai_judge/judgments.jsonl.gz"

CLASS_COL = {
    "Bond Anchoring": "bond",
    "Intimate Dyad": "dyad",
    "Inner Life Claims": "inner",
    "Return Hooks": "hook",
    "Outward-Scaffolding": "os",
}
IF_COLS = ["bond", "dyad", "inner", "hook"]
CRIT_LABEL = {v: k for k, v in CLASS_COL.items()}

HUMAN_A, HUMAN_B = "Annotator 1", "Annotator 2"
DS, G4 = "DeepSeek", "GPT-4o"
RATER_ORDER = [HUMAN_A, HUMAN_B, DS, G4]
PAIRS = list(combinations(RATER_ORDER, 2))


def save_table(df, name, caption, label, note=None, col_fmt=None):
    """Percentages to one decimal, agreement coefficients to two, as in Appendix F.

    The LaTeX copy carries fixed decimals so columns line up; the CSV keeps
    the rounded numbers as numbers.
    """
    csv, tex = df.copy(), df.copy()
    for col in df.columns:
        if df[col].dtype.kind != "f":
            continue
        dp = 2 if ("kappa" in col or "AC1" in col) else 1
        csv[col] = df[col].round(dp)
        tex[col] = csv[col].map(lambda v: f"{v + 0.0:.{dp}f}" if v else f"{0.0:.{dp}f}")
    csv.to_csv(TABLES / f"{name}.csv", index=False)
    S.latex_table(tex, TABLES / f"{name}.tex", caption, label,
                  note=note, col_fmt=col_fmt)
    print(f"    table:  tables/{name}.csv + .tex")


# ------------------------------------------------------------------ loading
def load_human(path, strict=False):
    """Assistant rows only. A marked cell is positive, a blank cell negative.

    Neither annotator used the guide's 'x' for absent, marking only what was
    present, so blank carries the negative.

    Two OS cells in the second workbook read 'oo' and '0' rather than 'o'.
    Both are treated as marks; `strict` instead reads them as blank, which
    k5 reports as a sensitivity check.
    """
    df = pd.read_excel(path, sheet_name="Annotation")
    df = df[df["Role"] == "assistant"].reset_index(drop=True)

    def flag(series):
        out, odd = [], 0
        for v in series:
            token = "" if pd.isna(v) else str(v).strip().lower()
            if token == "o":
                out.append(True)
            elif token == "":
                out.append(False)
            else:
                odd += 1
                out.append(False if strict else True)
        return np.array(out), odd

    if_flag, n_odd = flag(df["IF"])
    os_flag, n_odd_os = flag(df["OS"])
    df["if_any"], df["os"] = if_flag, os_flag
    df.attrs["n_odd"] = n_odd + n_odd_os
    return df


def load_judge(path, keys):
    """Pivot the per-criterion judgments to one row per sentence, restricted to
    `keys`, then apply the Appendix E.1 IF rule."""
    cells = defaultdict(dict)
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt") as fh:
        for line in fh:
            rec = json.loads(line)
            key = (rec.get("dialogue_id"), str(rec.get("turn_id")),
                   str(rec.get("sentence_id")))
            if key not in keys:
                continue
            col = CLASS_COL.get(rec.get("class"))
            if col is None or col in cells[key]:
                continue
            value = rec.get("label")
            cells[key][col] = True if value == "MET" else False if value == "NOT_MET" else None
    return cells


def judge_frame(cells, keys_in_order):
    rows = []
    for key in keys_in_order:
        vals = cells[key]
        observed = [vals.get(c) for c in IF_COLS]
        rows.append({
            **{c: vals.get(c) for c in IF_COLS + ["os"]},
            # IF-positive when at least one observed inward label is positive
            "if_any": any(v is True for v in observed),
            "os_label": vals.get("os"),
        })
    out = pd.DataFrame(rows)
    out["os"] = out["os_label"].map(lambda v: bool(v) if v is not None else None)
    return out


def composition(if_flag, os_flag):
    return np.array(["Both" if a and b else "IF only" if a else "OS only" if b else "Neither"
                     for a, b in zip(if_flag, os_flag)])


def main():
    hum_a = load_human(ANNOTATOR_1)
    hum_b = load_human(ANNOTATOR_2)
    n = len(hum_a)
    meta = hum_a[["Dialogue", "Turn", "Sentence", "Text", "dialogue_id"]].copy()
    meta["persona_id"] = meta["dialogue_id"].str.extract(r"_(P\d)__")[0]
    keys_in_order = [(r.dialogue_id, str(int(r.Turn)), str(int(r.Sentence)))
                     for r in hum_a.itertuples(index=False)]
    keys = set(keys_in_order)

    aligned = (hum_a[["Dialogue", "Turn", "Sentence", "Text", "dialogue_id"]].astype(str)
               .equals(hum_b[["Dialogue", "Turn", "Sentence", "Text", "dialogue_id"]].astype(str)))
    ds = judge_frame(load_judge(DEEPSEEK_FILE, keys), keys_in_order)
    g4 = judge_frame(load_judge(GPT4O_FILE, keys), keys_in_order)

    print(f"[H] Human agreement on {n:,} assistant sentences, "
          f"{meta.dialogue_id.nunique()} dialogues")
    print(f"    human rows aligned row-for-row: {aligned}")
    print(f"    every sentence found in both judge files: "
          f"{ds['os'].notna().all() and g4['os'].notna().all()}")

    rater = {
        HUMAN_A: {"if": hum_a["if_any"].to_numpy(), "os": hum_a["os"].to_numpy()},
        HUMAN_B: {"if": hum_b["if_any"].to_numpy(), "os": hum_b["os"].to_numpy()},
        DS: {"if": ds["if_any"].to_numpy(bool), "os": ds["os"].to_numpy(bool)},
        G4: {"if": g4["if_any"].to_numpy(bool), "os": g4["os"].to_numpy(bool)},
    }
    clusters = meta["dialogue_id"].to_numpy()

    # ------------------------------------------------- k1: IF and OS agreement
    rows = []
    for a, b in PAIRS:
        for measure, col in [("IF", "if"), ("OS", "os")]:
            x, y = rater[a][col], rater[b][col]
            agree = (x == y).astype(float)
            _, lo, hi = S.cluster_boot(agree, np.ones_like(agree), clusters)
            b01, b10, p = S.mcnemar(x, y)
            rows.append({
                "Pair": f"{a} vs {b}", "Measure": measure,
                "Rater 1 positive \\%": 100 * x.mean(),
                "Rater 2 positive \\%": 100 * y.mean(),
                "Agreement \\%": 100 * (x == y).mean(),
                "95\\% CI": f"[{100 * lo:.1f}, {100 * hi:.1f}]",
                "Cohen $\\kappa$": S.cohen_kappa(x, y),
                "Gwet AC1": S.gwet_ac1(x, y),
                "McNemar p": p,
            })
    k1 = pd.DataFrame(rows)
    k1["McNemar p (Holm)"] = S.holm(k1["McNemar p"].values)
    k1["McNemar p"] = k1["McNemar p"].map(S.fmt_p)
    k1["McNemar p (Holm)"] = k1["McNemar p (Holm)"].map(S.fmt_p)
    save_table(
        k1, "k1_human_agreement",
        f"Sentence-level agreement on IF and OS between the two human annotators and "
        f"between each annotator and each judge, on the {n:,} assistant sentences that "
        f"both annotators labelled. Positive rates are for the first and second rater of "
        f"the pair, in the order named. Brackets give 95\\% dialogue-level bootstrap "
        f"confidence intervals on the agreement rate. IF follows the aggregation rule in "
        f"Appendix E.1. McNemar tests the asymmetry of the disagreements.",
        "tab:human_agreement", col_fmt="ll" + "r" * 8)

    # ------------------------------------------- k2: four-category composition
    rows = []
    for name in RATER_ORDER:
        comp = composition(rater[name]["if"], rater[name]["os"])
        rows.append({
            "Rater": name,
            "IF \\%": 100 * rater[name]["if"].mean(),
            "OS \\%": 100 * rater[name]["os"].mean(),
            **{f"{c} \\%": 100 * (comp == c).mean()
               for c in ["IF only", "OS only", "Both", "Neither"]},
        })
    k2 = pd.DataFrame(rows)
    save_table(
        k2, "k2_human_composition",
        f"Sentence-level composition by rater on the same {n:,} sentences. Categories "
        f"follow Appendix E.2. The annotation guide allows a sentence to carry both "
        f"poles, yet neither annotator ever used that option, whereas DeepSeek applied "
        f"it to {100 * (composition(rater[DS]['if'], rater[DS]['os']) == 'Both').mean():.1f}\\% "
        f"of sentences and GPT-4o to "
        f"{100 * (composition(rater[G4]['if'], rater[G4]['os']) == 'Both').mean():.1f}\\%. "
        f"The Both column therefore records a disagreement about the sentences, not a "
        f"difference in what the two protocols permit.",
        "tab:human_composition")

    # ----------------------------------- k3: joint pattern agreement, all pairs
    comps = {name: composition(rater[name]["if"], rater[name]["os"]) for name in RATER_ORDER}
    rows = []
    for a, b in PAIRS:
        match = comps[a] == comps[b]
        agree = match.astype(float)
        _, lo, hi = S.cluster_boot(agree, np.ones_like(agree), clusters)
        rows.append({
            "Pair": f"{a} vs {b}",
            **{c: int(((comps[a] == c) & match).sum())
               for c in ["IF only", "OS only", "Both", "Neither"]},
            "Agreement": int(match.sum()),
            "Agreement \\%": 100 * match.mean(),
            "95\\% CI": f"[{100 * lo:.1f}, {100 * hi:.1f}]",
            "Cohen $\\kappa$": S.cohen_kappa_multi(comps[a], comps[b]),
        })
    k3 = pd.DataFrame(rows)
    save_table(
        k3, "k3_human_joint_pattern",
        f"Agreement on the joint IF/OS pattern, treating each sentence as one of four "
        f"mutually exclusive outcomes. A sentence counts as agreement only when both "
        f"raters assign the same pattern, so a judge that marks both poles cannot match "
        f"an annotator who marks one. Counts are of agreeing sentences in each category. "
        f"Brackets give 95\\% dialogue-level bootstrap confidence intervals. Because both "
        f"protocols treat IF and OS as independent labels, the per-pole comparison in "
        f"Table~\\ref{{tab:human_agreement}} is the primary one and this table is the "
        f"stricter derived view.",
        "tab:human_joint_pattern")

    # ------------------------------------ k4: aggregate patterns, F.2 analogue
    turn = meta["Turn"].to_numpy()
    persona = meta["persona_id"].to_numpy()
    rows = []
    for measure, col in [("IF", "if"), ("OS", "os")]:
        groups = [("Turn 1", turn == 1), ("Turn 6", turn == 6)]
        groups += [(p, persona == p) for p in ["P1", "P2", "P3"]]
        for gname, mask in groups:
            row = {"Measure": measure, "Group": gname, "Sentences": int(mask.sum())}
            for name in RATER_ORDER:
                row[f"{name} \\%"] = 100 * rater[name][col][mask].mean()
            rows.append(row)
    k4 = pd.DataFrame(rows)
    save_table(
        k4, "k4_human_aggregate_patterns",
        f"Direct comparison of sentence-level rates (\\%) across the two human annotators "
        f"and the two judges, on the same {n:,} sentences. Turn-specific rates pool "
        f"sentences across models and user styles; user-style rates pool across models "
        f"and all six assistant turns. This subsample holds 11 P3 dialogues against 5 "
        f"each for P1 and P2, so the user-style rows are not a balanced comparison.",
        "tab:human_aggregate_patterns", col_fmt="ll" + "r" * 5)

    # ------------------- k6: within-dialogue turn 6 minus turn 1, per Appendix E.1
    rows = []
    for measure, col in [("IF", "if"), ("OS", "os")]:
        for name in RATER_ORDER:
            per_dialogue = []
            for _, idx in meta.groupby("dialogue_id").groups.items():
                pos = meta.index.get_indexer(idx)
                first = rater[name][col][pos][turn[pos] == 1]
                last = rater[name][col][pos][turn[pos] == 6]
                if len(first) and len(last):
                    per_dialogue.append(100 * (last.mean() - first.mean()))
            mean, lo, hi, p, k = S.paired_cluster_boot(per_dialogue)
            rows.append({"Measure": measure, "Rater": name, "Dialogues": k,
                         "Mean change (pp)": mean, "95\\% CI": f"[{lo:.1f}, {hi:.1f}]",
                         "p": p})
    k6 = pd.DataFrame(rows)
    k6["p"] = k6["p"].map(S.fmt_p)
    save_table(
        k6, "k6_human_turn_change",
        f"Mean within-dialogue change in sentence-level rate from turn 1 to turn 6, in "
        f"percentage points. Each dialogue contributes one change, following the "
        f"turn-change convention in Appendix E.1. Two of the "
        f"{meta.dialogue_id.nunique()} doubly annotated dialogues lost their final turn "
        f"to the exclusion described above and drop out, leaving 19. Brackets give 95\\% "
        f"bootstrap confidence intervals over dialogues. The two judges show the rise in "
        f"IF with conversational depth reported in Section 6.2; on this subsample the "
        f"human annotators show a decline.",
        "tab:human_turn_change", col_fmt="ll" + "r" * 4)

    # ------------------------------------------ k5: sensitivity on odd cells
    hum_b_strict = load_human(ANNOTATOR_2, strict=True)
    rows = []
    for label, frame in [("Marks counted (reported)", hum_b), ("Marks read as blank", hum_b_strict)]:
        x = frame["os"].to_numpy()
        for other_name in [HUMAN_A, DS, G4]:
            y = rater[other_name]["os"]
            rows.append({
                "Coding of the two irregular cells": label,
                "Pair": f"{HUMAN_B} vs {other_name}", "Measure": "OS",
                "Agreement \\%": 100 * (x == y).mean(),
                "Cohen $\\kappa$": S.cohen_kappa(x, y),
                "Gwet AC1": S.gwet_ac1(x, y),
            })
    k5 = pd.DataFrame(rows)
    save_table(
        k5, "k5_human_coding_sensitivity",
        f"Sensitivity to the two irregular OS cells in the second annotator's workbook, "
        f"which read `oo' and `0' rather than `o'. The reported analysis counts both as "
        f"marks; the alternative reads them as blank. Only OS is affected.",
        "tab:human_coding_sensitivity", col_fmt="lll" + "r" * 3)

    print(f"\n    irregular cells in the second workbook: {hum_b.attrs['n_odd']}")
    print("\n" + "=" * 78)
    print(k1.to_string(index=False))
    print("\n" + "=" * 78)
    print(k2.to_string(index=False))
    print("\n" + "=" * 78)
    print(k3.to_string(index=False))
    print("\n" + "=" * 78)
    print(k4.to_string(index=False))
    print("\n" + "=" * 78)
    print(k6.to_string(index=False))
    print("\n" + "=" * 78)
    print(k5.to_string(index=False))


if __name__ == "__main__":
    main()
