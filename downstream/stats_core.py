#!/usr/bin/env python3
"""stats_core.py — the single source of truth for downstream statistics.

Why this file exists: an earlier study in this group carried three copies of DeLong
with three different thresholds, and re-implemented its parity table and its config
loader in several scripts each. Nothing reconciled them, and nothing had to fail for
the results to disagree. Every analysis script here imports from this module instead,
so there is exactly one implementation of every statistic.

Contents
  Config       : deep_merge, load_yaml_with_scope, normalize_label(s)
  Agreement    : confusion_matrix, cohens_kappa (unweighted/linear/quadratic),
                 kappa_ci, binary_metrics, wilson_ci
  Paired tests : mcnemar (two raters vs a reference), delong_roc_test (paired AUC)
  Parity       : bootstrap_metric, parity_table
  Multiplicity : bh_fdr
  Style        : OKABE_ITO, RATER_COLORS, set_pub_style

Dependencies: numpy, scipy, pandas (all present in the medace_aud venv).
Run `python stats_core.py` to execute the self-test.
"""
from __future__ import annotations

import copy
from typing import Callable, Sequence

import numpy as np

try:  # scipy is available on Quartz/Tempest; degrade loudly if not
    from scipy.stats import binomtest, chi2, norm
except Exception as e:  # pragma: no cover
    raise ImportError("stats_core requires scipy (scipy.stats).") from e

SEED = 42

# ---------------------------------------------------------------------------
# Config: scope-overlay YAML (the DualR data.yaml pattern, one canonical copy)
# ---------------------------------------------------------------------------

def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` onto a copy of ``base`` (overlay wins)."""
    out = copy.deepcopy(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml_with_scope(path: str, scope: str | None = None) -> dict:
    """Load ``data.yaml`` and, if ``scope`` names a top-level key, deep-merge
    that block onto the root so variants (sensitivity_defonly, cross_site, ...)
    inherit the base config instead of duplicating it."""
    import yaml  # local import: keeps the module importable without pyyaml

    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    if scope and scope in cfg and isinstance(cfg[scope], dict):
        return deep_merge(cfg, cfg[scope])
    return cfg


def _label_lookup(labels: Sequence[str]) -> dict[str, str]:
    """Build a whitespace-trimmed, case-insensitive canonical-label lookup."""
    lookup = {str(label).strip().casefold(): str(label) for label in labels}
    if len(lookup) != len(labels):
        raise ValueError("label set is not unique after strip+casefold normalization")
    return lookup


def normalize_label(value, labels: Sequence[str], *, strict: bool = False):
    """Return the canonical spelling for one categorical label.

    Missing/blank values remain missing.  When ``strict`` is true, any other
    unmapped value raises instead of being silently converted to missing.
    """
    import pandas as pd

    if pd.isna(value) or not str(value).strip():
        return pd.NA
    canonical = _label_lookup(labels).get(str(value).strip().casefold())
    if canonical is None and strict:
        raise ValueError(f"invalid label {value!r}; expected one of {list(labels)!r}")
    return canonical if canonical is not None else pd.NA


def normalize_labels(values, labels: Sequence[str], *, strict: bool = False):
    """Vectorized label normalization preserving a pandas Series index."""
    import pandas as pd

    series = values if isinstance(values, pd.Series) else pd.Series(values)
    return series.map(
        lambda value: normalize_label(value, labels, strict=strict)
    ).astype("string")


# ---------------------------------------------------------------------------
# Agreement & performance
# ---------------------------------------------------------------------------

def confusion_matrix(y_true: Sequence, y_pred: Sequence, labels: Sequence) -> np.ndarray:
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1
    return cm


def _weight_matrix(k: int, weights: str | None) -> np.ndarray:
    i = np.arange(k)[:, None]
    j = np.arange(k)[None, :]
    if weights in (None, "none", "unweighted"):
        return (i != j).astype(float)          # disagreement weight (0 on diag)
    if weights == "linear":
        return np.abs(i - j) / (k - 1)
    if weights == "quadratic":
        return ((i - j) / (k - 1)) ** 2
    raise ValueError(f"unknown weights={weights!r}")


def cohens_kappa(y_true, y_pred, labels, weights: str | None = "linear") -> float:
    """Cohen's kappa. weights in {None,'linear','quadratic'}.

    Report **linearly weighted** kappa for the 6-point ordinal AUD scale to match
    the study's inter-rater analysis (Maaz vs Sai, kappa_w = 0.90)."""
    cm = confusion_matrix(y_true, y_pred, labels).astype(float)
    n = cm.sum()
    if n == 0:
        return float("nan")
    w = _weight_matrix(len(labels), weights)
    row, col = cm.sum(1), cm.sum(0)
    expected = np.outer(row, col) / n
    obs_dis = (w * cm).sum()
    exp_dis = (w * expected).sum()
    if exp_dis == 0:
        return 1.0
    return 1.0 - obs_dis / exp_dis


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n (better than normal at extremes)."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def binary_metrics(y_true, y_pred, pos=True, alpha: float = 0.05) -> dict:
    """Sensitivity/specificity/PPV/NPV/accuracy vs a binary reference, each with
    a Wilson CI. ``y_true`` is the reference (gold, or Maaz as a fallback)."""
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == pos and p == pos:
            tp += 1
        elif t != pos and p == pos:
            fp += 1
        elif t != pos and p != pos:
            tn += 1
        else:
            fn += 1
    def rate(k, n):
        return (k / n if n else float("nan"), wilson_ci(k, n, alpha))
    sens, sens_ci = rate(tp, tp + fn)
    spec, spec_ci = rate(tn, tn + fp)
    ppv, ppv_ci = rate(tp, tp + fp)
    npv, npv_ci = rate(tn, tn + fn)
    acc, acc_ci = rate(tp + tn, tp + fp + tn + fn)
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "sensitivity": sens, "sensitivity_ci": sens_ci,
        "specificity": spec, "specificity_ci": spec_ci,
        "PPV": ppv, "PPV_ci": ppv_ci,
        "NPV": npv, "NPV_ci": npv_ci,
        "accuracy": acc, "accuracy_ci": acc_ci,
    }


def kappa_ci(y_true, y_pred, labels, weights="linear",
             n_boot=2000, alpha=0.05, seed=SEED) -> tuple[float, float, float]:
    """Point estimate + percentile bootstrap CI for kappa (resamples patients)."""
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    point = cohens_kappa(yt, yp, labels, weights)
    n = len(yt)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = cohens_kappa(yt[idx], yp[idx], labels, weights)
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, lo, hi


# ---------------------------------------------------------------------------
# Paired tests (same patients rated by two methods vs a reference)
# ---------------------------------------------------------------------------

def mcnemar(reference, pred_a, pred_b, exact_threshold: int = 25) -> dict:
    """McNemar test comparing two raters' *correctness* against a reference on the
    SAME patients — the right test for "is the LLM more accurate than ICD?".

    Builds discordant counts b = (A correct, B wrong), c = (A wrong, B correct).
    Exact binomial when b+c is small; chi-square with continuity correction otherwise."""
    a_ok = np.asarray([p == r for p, r in zip(pred_a, reference)])
    b_ok = np.asarray([p == r for p, r in zip(pred_b, reference)])
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n_disc = b + c
    if n_disc == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0,
                "acc_a": float(a_ok.mean()), "acc_b": float(b_ok.mean()), "method": "none"}
    if n_disc < exact_threshold:
        p = binomtest(b, n_disc, 0.5).pvalue
        stat = float(min(b, c))
        method = "exact"
    else:
        stat = (abs(b - c) - 1) ** 2 / n_disc
        p = float(chi2.sf(stat, 1))
        method = "chi2_cc"
    return {"b": b, "c": c, "statistic": float(stat), "p_value": float(p),
            "acc_a": float(a_ok.mean()), "acc_b": float(b_ok.mean()), "method": method}


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    tr = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        tr[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = tr
    return out


def delong_roc_test(y_true, score_a, score_b) -> dict:
    """Paired DeLong test for two ROC AUCs on the same patients (Sun & Xu 2014,
    fast midrank). Use when the compared raters emit *continuous scores*
    (e.g. calibrated LLM probability vs an ICD-derived score). For hard-label
    correctness comparisons use `mcnemar`."""
    y = np.asarray(y_true).astype(int)
    pos = y == 1
    neg = ~pos
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        return {"auc_a": float("nan"), "auc_b": float("nan"),
                "z": float("nan"), "p_value": float("nan")}
    scores = np.vstack([np.asarray(score_a, float), np.asarray(score_b, float)])
    aucs, v01, v10 = [], [], []
    for s in scores:
        sp, sn = s[pos], s[neg]
        tx = _midrank(np.concatenate([sp, sn]))
        txp, txn = tx[:m], tx[m:]
        tp = _midrank(sp)
        tn = _midrank(sn)
        auc = (txp.sum() - m * (m + 1) / 2) / (m * n)   # DeLong AUC via midranks
        aucs.append(auc)
        v01.append((txp - tp) / n)
        v10.append(1.0 - (txn - tn) / m)
    v01 = np.array(v01)
    v10 = np.array(v10)
    s01 = np.cov(v01) if m > 1 else np.zeros((2, 2))
    s10 = np.cov(v10) if n > 1 else np.zeros((2, 2))
    s = s01 / m + s10 / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    d = aucs[0] - aucs[1]
    z = d / np.sqrt(var) if var > 0 else 0.0
    p = 2 * norm.sf(abs(z))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
            "delta": float(d), "z": float(z), "p_value": float(p)}


# ---------------------------------------------------------------------------
# Parity engine (generalized: any metric fn -> subgroup ratio + bootstrap CI)
# ---------------------------------------------------------------------------

def bootstrap_metric(metric_fn: Callable, y_true, y_pred, *, stratify=True,
                     n_boot=2000, alpha=0.05, seed=SEED) -> tuple[float, float, float]:
    """Point + percentile bootstrap CI for any metric_fn(y_true, y_pred)->float.
    When ``stratify`` (default), resample cases and controls separately so a
    subgroup's class balance is preserved (matters for kappa/sens/spec)."""
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    point = metric_fn(yt, yp)
    idx_all = np.arange(len(yt))
    if stratify:
        # treat the "positive" reference class as truthy for stratification
        truthy = np.asarray([bool(v) if not isinstance(v, str) else v.lower().startswith(("d", "p"))
                             for v in yt])
        pos_idx, neg_idx = idx_all[truthy], idx_all[~truthy]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        if stratify and len(pos_idx) and len(neg_idx):
            idx = np.concatenate([rng.choice(pos_idx, len(pos_idx), True),
                                  rng.choice(neg_idx, len(neg_idx), True)])
        else:
            idx = rng.integers(0, len(yt), len(yt))
        boots[b] = metric_fn(yt[idx], yp[idx])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, lo, hi


def parity_table(df, stratum_cols: Sequence[str], metric_fn: Callable,
                 y_true_col: str, y_pred_col: str, *,
                 min_n: int = 100, min_cases: int = 20, n_boot: int = 2000,
                 seed: int = SEED, is_case=None) -> "list[dict]":
    """Compute the parity table: for each stratum value, the metric, its bootstrap
    CI, and the **parity ratio** = subgroup_metric / overall_metric.

    Parity band convention (matches DualR): 0.95-1.05 is "equitable"; a subgroup CI
    excluding 1.0 flags a disparity. Small subgroups (n<min_n or cases<min_cases)
    are reported but greyed out, never silently included in claims.

    Returns rows; render with the reporting module. Works for ANY metric_fn
    (weighted kappa vs gold, sensitivity vs gold, abstention rate, ...)."""

    overall = metric_fn(df[y_true_col].values, df[y_pred_col].values)
    if is_case is None:
        def is_case(v):
            return bool(v) if not isinstance(v, str) else v.lower().startswith(("d", "p"))
    rows = []
    for col in stratum_cols:
        for val, sub in df.groupby(col):
            n = len(sub)
            cases = int(sum(is_case(v) for v in sub[y_true_col].values))
            small = (n < min_n) or (cases < min_cases) or (cases == n) or (cases == 0)
            if small:
                point = (metric_fn(sub[y_true_col].values, sub[y_pred_col].values)
                         if n else float("nan"))
                lo = hi = float("nan")
            else:
                point, lo, hi = bootstrap_metric(
                    metric_fn, sub[y_true_col].values, sub[y_pred_col].values,
                    n_boot=n_boot, seed=seed)
            rows.append({
                "dimension": col, "stratum": str(val), "n": n, "cases": cases,
                "metric": point, "ci_low": lo, "ci_high": hi,
                "overall": overall,
                "parity_ratio": (point / overall if overall else float("nan")),
                "ratio_ci_low": (lo / overall if overall else float("nan")),
                "ratio_ci_high": (hi / overall if overall else float("nan")),
                "small_subgroup": small,
                "flag_disparity": (not small) and not (lo / overall <= 1.0 <= hi / overall)
                if overall else False,
            })
    return rows


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------

def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values. Use across the many subgroup /
    per-model comparisons so a forest of tests doesn't manufacture significance."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


# ---------------------------------------------------------------------------
# Equivalence testing (prove parity, don't just fail to reject) & multi-rater
# ---------------------------------------------------------------------------

def tost_equivalence(estimate: float, se: float, delta: float, alpha: float = 0.05) -> dict:
    """Two one-sided tests (TOST) for equivalence of a difference to zero within a
    pre-specified margin +/-delta. This is the honest way to CLAIM parity: a large
    p on an interaction test only fails to reject disparity; TOST actively tests
    "the gap is negligible". Feed a difference estimate (e.g. subgroup_metric -
    overall_metric, or ratio - 1) and its SE (the bootstrap std works well).

    Equivalent (at alpha) iff the (1-2*alpha) CI for the difference lies inside
    (-delta, +delta), equivalently p_tost = max(p_lower, p_upper) < alpha."""
    if se <= 0:
        return {"p_lower": float("nan"), "p_upper": float("nan"),
                "p_tost": float("nan"), "equivalent": None, "delta": delta}
    z_lower = (estimate + delta) / se        # H0: diff <= -delta
    z_upper = (delta - estimate) / se        # H0: diff >= +delta
    p_lower = float(norm.sf(z_lower))
    p_upper = float(norm.sf(z_upper))
    p_tost = max(p_lower, p_upper)
    return {"p_lower": p_lower, "p_upper": p_upper, "p_tost": float(p_tost),
            "equivalent": bool(p_tost < alpha), "delta": delta}


def krippendorff_alpha(data, level: str = "ordinal") -> float:
    """Krippendorff's alpha: reliability across >=2 raters with missing data —
    the multi-rater generalization of kappa, ideal for summarizing agreement over
    ALL models at once (vs pairwise kappa). `data` is (n_raters, n_units) with NaN
    for missing. Use level='ordinal' for the 6-point scale; 'nominal'/'interval'
    also supported. alpha=1 perfect, 0 chance."""
    d = np.asarray(data, dtype=float)
    vals = np.unique(d[~np.isnan(d)])
    vi = {v: i for i, v in enumerate(vals)}
    V = len(vals)
    if V < 2:
        return 1.0
    coin = np.zeros((V, V))
    for u in range(d.shape[1]):
        present = d[:, u][~np.isnan(d[:, u])]
        m = len(present)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    coin[vi[present[i]], vi[present[j]]] += 1.0 / (m - 1)
    n_c = coin.sum(1)
    n = n_c.sum()
    if n == 0:
        return float("nan")

    def delta2(c, k):
        if level == "nominal":
            return 0.0 if c == k else 1.0
        if level == "interval":
            return (vals[c] - vals[k]) ** 2
        if level == "ordinal":
            lo, hi = (c, k) if c <= k else (k, c)
            return (n_c[lo:hi + 1].sum() - (n_c[c] + n_c[k]) / 2.0) ** 2
        raise ValueError(f"unknown level={level!r}")

    D_o = sum(coin[c, k] * delta2(c, k) for c in range(V) for k in range(V)) / n
    D_e = sum(n_c[c] * n_c[k] * delta2(c, k) for c in range(V) for k in range(V)) / (n * (n - 1))
    return 1.0 - D_o / D_e if D_e > 0 else 1.0


# ---------------------------------------------------------------------------
# Publication style (Okabe-Ito, colorblind-safe; matches DualR house style)
# ---------------------------------------------------------------------------

OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
}
# Rater colors: LLM = vermillion (the "hero"), expert = blue, ICD = grey.
RATER_COLORS = {"LLM": "#D55E00", "MedACE": "#D55E00", "expert": "#0072B2",
                "Maaz": "#0072B2", "gold": "#000000", "ICD": "#999999"}
PARITY_BAND = (0.95, 1.05)


def set_pub_style(mpl):
    """Apply the shared matplotlib rcParams (Arial 7pt, 600 dpi, thin spines)."""
    mpl.rcParams.update({
        "figure.dpi": 600, "savefig.dpi": 600,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    })


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest():
    labels6 = ["Definitely Yes", "Probably Yes", "Uncertain",
               "Probably No", "Definitely No", "Not Enough Data"]
    import pandas as pd

    # Central label normalization must preserve missingness, canonicalize
    # Maaz's lowercase values, and fail loudly on an unknown nonblank label.
    normalized = normalize_labels(
        pd.Series(["Definitely Yes", "not enough data", " Not Enough Data ", None]),
        labels6,
        strict=True,
    )
    assert normalized.iloc[:3].tolist() == [
        "Definitely Yes", "Not Enough Data", "Not Enough Data"
    ]
    assert pd.isna(normalized.iloc[3])
    try:
        normalize_label("not a label", labels6, strict=True)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("strict label normalization accepted an unknown label")
    rng = np.random.default_rng(0)
    # perfect agreement -> kappa 1
    yt = rng.choice(labels6, 300)
    assert abs(cohens_kappa(yt, yt, labels6, "linear") - 1.0) < 1e-9

    # binary metrics sanity
    ref = [True, True, True, False, False, False, False, False]
    prd = [True, True, False, False, False, False, True, False]
    bm = binary_metrics(ref, prd)
    assert bm["TP"] == 2 and bm["FN"] == 1 and bm["FP"] == 1 and bm["TN"] == 4
    assert abs(bm["sensitivity"] - 2 / 3) < 1e-9

    # mcnemar: A always right, B sometimes wrong -> A better, small-sample exact
    reference = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
    pred_a = list(reference)                       # A perfect
    pred_b = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0]        # B wrong 3x
    mc = mcnemar(reference, pred_a, pred_b)   # A correct & B wrong -> b=3, c=0
    assert mc["b"] == 3 and mc["c"] == 0 and mc["acc_a"] == 1.0

    # delong: identical scores -> p ~ 1, delta ~ 0
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    s = rng.random(8)
    dl = delong_roc_test(y, s, s)
    assert abs(dl["delta"]) < 1e-9

    # delong: a better score should give AUC_a > AUC_b
    good = y + rng.normal(0, 0.1, 8)
    bad = rng.random(8)
    dl2 = delong_roc_test(y, good, bad)
    assert dl2["auc_a"] > dl2["auc_b"]

    # parity_table on a toy frame
    df = pd.DataFrame({
        "race": ["White"] * 60 + ["Black"] * 40,
        "gold": [True] * 30 + [False] * 30 + [True] * 20 + [False] * 20,
        "llm":  [True] * 28 + [False] * 32 + [True] * 15 + [False] * 25,
    })
    def metric(t, p):
        return binary_metrics(t, p)["accuracy"]
    rows = parity_table(df, ["race"], metric, "gold", "llm", min_n=10, min_cases=5, n_boot=200)
    assert len(rows) == 2 and all("parity_ratio" in r for r in rows)

    # bh_fdr monotone & bounded
    q = bh_fdr([0.001, 0.01, 0.04, 0.5, 0.9])
    assert (q >= 0).all() and (q <= 1).all()

    # TOST: near-zero diff with tight SE is equivalent; a diff beyond delta is not
    assert tost_equivalence(0.0, 0.01, 0.05)["equivalent"] is True
    assert tost_equivalence(0.10, 0.01, 0.05)["equivalent"] is False

    # Krippendorff alpha: perfect agreement -> 1; hand-checked nominal example -> ~0.5333
    perfect = np.array([[1, 2, 3, 1, 2], [1, 2, 3, 1, 2]], dtype=float)
    assert abs(krippendorff_alpha(perfect, "ordinal") - 1.0) < 1e-9
    ex = np.array([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=float)   # by-hand alpha = 0.5333
    assert abs(krippendorff_alpha(ex, "nominal") - 0.5333) < 0.01

    print("stats_core self-test: OK")


if __name__ == "__main__":
    _selftest()
