"""Statistics helpers for the RELATE analysis.

Sentences are nested in turns, in dialogues, in situations. Every confidence
interval here therefore resamples *dialogues*, not sentences.
"""
import re

import numpy as np
import pandas as pd
from scipy import stats as sps

N_BOOT = 10_000
SEED = 20260913


def _codes(labels):
    cat = pd.Categorical(labels)
    return cat.codes.astype(np.int64), len(cat.categories)


def cluster_boot(num, den, clusters, n_boot=N_BOOT, seed=SEED, chunk=1000):
    """Percentile CI for sum(num)/sum(den), resampling clusters with replacement.

    num/den may be 1-D (n_rows,) or 2-D (n_rows, n_stats) to bootstrap many
    statistics against one shared set of resamples.
    """
    codes, k = _codes(clusters)
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    two_d = num.ndim == 2
    if not two_d:
        num, den = num[:, None], den[:, None]

    # collapse to one row per cluster, then resample cluster rows
    cn = np.zeros((k, num.shape[1]))
    cd = np.zeros((k, den.shape[1]))
    np.add.at(cn, codes, num)
    np.add.at(cd, codes, den)

    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, num.shape[1]))
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, k, size=(b, k))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[done:done + b] = cn[idx].sum(1) / cd[idx].sum(1)
        done += b

    point = cn.sum(0) / cd.sum(0)
    lo, hi = np.nanpercentile(out, [2.5, 97.5], axis=0)
    if not two_d:
        return float(point[0]), float(lo[0]), float(hi[0])
    return point, lo, hi


def cluster_boot_diff(num_a, den_a, num_b, den_b, clusters, n_boot=N_BOOT, seed=SEED, chunk=1000):
    """CI and two-sided bootstrap p for rate(a) - rate(b) under one resampling."""
    codes, k = _codes(clusters)
    stacked_n = np.column_stack([num_a, num_b]).astype(float)
    stacked_d = np.column_stack([den_a, den_b]).astype(float)
    cn = np.zeros((k, 2))
    cd = np.zeros((k, 2))
    np.add.at(cn, codes, stacked_n)
    np.add.at(cd, codes, stacked_d)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, k, size=(b, k))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = cn[idx].sum(1) / cd[idx].sum(1)
        diffs[done:done + b] = r[:, 0] - r[:, 1]
        done += b

    tot_n, tot_d = cn.sum(0), cd.sum(0)
    point = tot_n[0] / tot_d[0] - tot_n[1] / tot_d[1]
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    # p = 2 x smaller tail mass at 0, floored at the bootstrap resolution
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(point), float(lo), float(hi), float(max(p, 1.0 / n_boot))


def paired_cluster_boot(values, clusters=None, n_boot=N_BOOT, seed=SEED):
    """CI and p for the mean of a per-dialogue paired quantity (e.g. turn6 - turn1)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = v[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(v.mean()), float(lo), float(hi), float(max(p, 1.0 / n_boot)), len(v)


def cohen_kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    if pe == 1:
        return float("nan")
    return float((po - pe) / (1 - pe))


def cohen_kappa_multi(a, b):
    """Cohen's kappa for labels with more than two categories."""
    a, b = np.asarray(a), np.asarray(b)
    cats = np.unique(np.concatenate([a, b]))
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    if pe == 1:
        return float("nan")
    return float((po - pe) / (1 - pe))


def gwet_ac1(a, b):
    """Gwet's AC1: chance agreement from the pooled marginal, so it does not
    collapse when one category dominates the way Cohen's kappa does."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pi = (a.mean() + b.mean()) / 2
    pe = 2 * pi * (1 - pi)
    if pe == 1:
        return float("nan")
    return float((po - pe) / (1 - pe))


def mcnemar(a, b):
    """Exact McNemar for a systematic labelling difference between two judges."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    b01 = int((a & ~b).sum())
    b10 = int((~a & b).sum())
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    p = float(min(1.0, 2 * sps.binom.cdf(min(b01, b10), n, 0.5)))
    return b01, b10, p


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def gee_logit(df, y, formula_rhs, groups, maxiter=100):
    """Logistic GEE with an exchangeable working correlation, clustered on `groups`.

    Returns a tidy frame of coefficients with cluster-robust CIs, or None if the
    fit does not converge.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    data = df.dropna(subset=[y]).copy()
    data[y] = data[y].astype(int)
    try:
        model = smf.gee(f"{y} ~ {formula_rhs}", groups=groups, data=data,
                        family=sm.families.Binomial(),
                        cov_struct=sm.cov_struct.Exchangeable())
        res = model.fit(maxiter=maxiter)
    except Exception as exc:
        print(f"    GEE failed for {y} ~ {formula_rhs}: {exc}")
        return None
    ci = res.conf_int()
    out = pd.DataFrame({
        "term": res.params.index,
        "coef": res.params.values,
        "se": res.bse.values,
        "z": res.tvalues.values,
        "p": res.pvalues.values,
        "lo": ci[0].values,
        "hi": ci[1].values,
    })
    out["odds_ratio"] = np.exp(out["coef"])
    out["or_lo"] = np.exp(out["lo"])
    out["or_hi"] = np.exp(out["hi"])
    out.attrs["n_obs"] = int(res.nobs)
    out.attrs["n_groups"] = int(data[groups if isinstance(groups, str) else groups.name].nunique())
    return out


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if p < 1e-4:
        return "$<$0.0001"
    return f"{p:.4f}"


_ESC = {"&": r"\&", "#": r"\#"}


def _tex_escape(s):
    """Escape only what LaTeX would choke on, leaving deliberate math/markup alone."""
    s = str(s)
    for ch, rep in _ESC.items():
        s = s.replace(ch, rep)
    # bare _ and % that are not already escaped, and not inside math mode
    s = re.sub(r"(?<!\\)_", r"\\_", s)
    s = re.sub(r"(?<!\\)%", r"\\%", s)
    return s


def latex_table(df, path, caption, label, float_fmt="%.1f", col_fmt=None, note=None):
    """booktabs table, ICLR-ready. Content is emitted verbatim apart from minimal escaping,
    so deliberate math such as $\\Delta$ in a column name survives."""
    def cell(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "--"
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else ""
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        if isinstance(v, (float, np.floating)):
            return float_fmt % v
        return _tex_escape(v)

    header = " & ".join(_tex_escape(c) for c in df.columns) + r" \\"
    body = [" & ".join(cell(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\begin{tabular}{%s}" % (col_fmt or ("l" + "r" * (df.shape[1] - 1))),
        r"\toprule", header, r"\midrule", *body, r"\bottomrule",
        r"\end{tabular}",
    ]
    if note:
        lines.append(f"\\vspace{{2pt}}\\par\\footnotesize {note}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n")
