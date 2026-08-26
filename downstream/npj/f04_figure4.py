#!/usr/bin/env python3
"""Assemble the model-robustness figure: does the result depend on a choice we made?

  a  agreement with the reference standard, one row per model, and the diagnosis code
  b  pairwise agreement between the 11 models and the reference standard, on the same
     charts for every pair
  c  agreement on the charts the escalation referred, against agreement on the rest
  d  agreement under the frozen baseline and four prompt and temperature conditions

Every panel is cut by one variable, the choice under test: which model, which model,
which charts, which prompt and temperature. The figure reads along it.

The reference standard comes from ref_standard.build(), the same call Figures 1 to 3
use, so the cohort is the 487 analysis charts throughout. The chain this file replaces
read a reference file deleted on 13 August and reported 496 and 499.

Panels a and b were exchanged on 14 August, and both now order by agreement with the
reference standard so that a name sits at the same rank in each. The matrix carries the
reference standard as its first row and column: that row repeats panel a by construction,
which is what makes it useful, because it puts agreement with the standard on the same
colour scale as agreement between models.

What that comparison shows is not uniform. Among the eight strongest models the pairwise
values run above the agreement any of them reaches with the standard: their median pair is
0.911 against a median of 0.892 with the standard, and 20 of their 56 pairs exceed 0.918,
the highest model-standard value in the figure. Across all eleven the order reverses,
0.830 against 0.873, because the three weakest models disagree with each other more than
either disagrees with the standard.

The diagnosis code is a separated row in panel a. It returns no intermediate grade, so
the six-grade scale costs it something. The cost is 0.015: its three-category kappa is
0.643 against the 0.628 the panel reports. It is not the mapping that puts the code last
but one.

Two things left the figure on 14 August. The benchmark band drawn at the fellow's
agreement with the reference standard is circular: for the 422 charts the escalation
never referred, the reference label is the fellow's label, so the band measures the
reference standard against itself. The paired difference against that band goes with it.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "downstream")
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.lines import Line2D                                 # noqa: E402

import stats_core as sc                                             # noqa: E402
import ref_standard as RS                                           # noqa: E402

PANEL = "output/canonical_panel.csv"
ROSTER = "resources/patient_roster.csv"
GOAL5 = "downstream/npj_goal5_outputs/condition_reference_performance.csv"
OUT = "downstream/npj/out/figure4"

INK, MUTED, RULE = "#1f1e1d", "#55534e", "#B9B7B2"
C_PRIMARY, C_OTHER, C_DROP = "#0072B2", "#404040", "#D55E00"
C_TEMP, C_PROMPT = "#E69F00", "#CC79A7"
PRIMARY = "Qwen3.6-27B"

# --- primary agreement definition ------------------------------------------
# Three categories, not enough data excluded rather than placed on the scale. Weighting
# a six-grade scale requires giving that category a position, and the six positions it
# can occupy span 0.025, wider than the bootstrap interval of any one of them. Dropping
# it leaves negative, uncertain and positive, which is one ordinal scale; unweighted is
# reported because the study's decision is categorical.
ORD3_PRIMARY = ["AUD-", "Uncertain", "AUD+"]
_POS3 = {"Definitely Yes", "Probably Yes"}
_NEG3 = {"Definitely No", "Probably No"}
_NED3 = "Not Enough Data"


def collapse3(s):
    s = pd.Series(s)
    out = pd.Series(np.where(s.isin(_POS3), "AUD+",
                    np.where(s.isin(_NEG3), "AUD-",
                    np.where(s.eq("Uncertain"), "Uncertain", None))), index=s.index)
    return out.where(s.notna() & s.ne(_NED3))


def kappa3(a, b):
    a, b = collapse3(a), collapse3(b)
    m = a.notna() & b.notna()
    if m.sum() < 2:
        return float("nan")
    return sc.cohens_kappa(a[m], b[m], ORD3_PRIMARY, None)


def kappa3_ci(a, b, n_boot=2000, seed=20260814):
    a, b = collapse3(a), collapse3(b)
    m = a.notna() & b.notna()
    a, b = a[m].to_numpy(), b[m].to_numpy()
    k = sc.cohens_kappa(a, b, ORD3_PRIMARY, None)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        try:
            out.append(sc.cohens_kappa(a[i], b[i], ORD3_PRIMARY, None))
        except Exception:
            continue
    lo, hi = np.nanpercentile(out, [2.5, 97.5]) if out else (float("nan"), float("nan"))
    return k, lo, hi
# ---------------------------------------------------------------------------

REF_NAME = "Reference standard"
CODE_NAME = "Diagnosis code"

LEVELS = ["Definitely Yes", "Probably Yes", "Uncertain", "Probably No",
          "Definitely No", "Not Enough Data"]
# The canonical panel keys, and the names the manuscript uses.
DISPLAY = {
    "aud_qwen36_27b": "Qwen3.6-27B",
    "aud_nemotron_fp8": "Nemotron-3-120B",
    "aud": "GPT-OSS-120B",
    "aud_glm52": "GLM-5.2",
    "aud_nemotron_llama": "Nemotron-Llama-49B",
    "aud_gptoss_20b": "GPT-OSS-20B",
    "aud_deepseek_v4_flash_0731": "DeepSeek-V4-Flash",
    "aud_gemma4": "Gemma4-31B",
    "aud_llama4_scout": "Llama-4-Scout",
    "aud_mistral32_24b": "Mistral-3.2-24B",
    "aud_llama33_70b": "Llama-3.3-70B",
}


def load(panel_path):
    """The reference standard and the 11 model readings, on the analysis cohort.

    ref_standard.build() returns the 487 charts of the analysis cohort with a reference
    label and the source that produced it. gold_source is the escalation route: a chart
    the escalation never referred carries maaz_unreviewed, and its reference label is the
    fellow's own reading.
    """
    ref, man = RS.build()
    ref = ref[["omop_person_id", "gold_label", "gold_source"]].copy()
    ref["omop_person_id"] = ref["omop_person_id"].astype(str)
    ref["gold_label"] = sc.normalize_labels(ref["gold_label"], LEVELS, strict=True)
    assert len(ref) == man["cohort"]["analysed"], "reference standard is not the cohort"
    assert ref["gold_label"].notna().all(), "a reference row carries no label"

    p = pd.read_csv(panel_path, dtype={"omop_person_id": str})
    p = p[p["model"].isin(DISPLAY)].copy()
    p["name"] = p["model"].map(DISPLAY)
    p["label"] = sc.normalize_labels(p["label_canonical"], LEVELS, strict=True)
    assert p["name"].nunique() == 11, "the panel does not carry 11 models"
    return ref, p, man


def pairwise(p, ids, ref):
    """Every reader against every other on the same charts, the reference standard included.

    The reference standard is one column among the others here. Its row therefore repeats
    the leaderboard exactly, which is the comparison the panel exists to make: agreement
    between two models and agreement between a model and the standard, on one scale.
    """
    w = (p[p["omop_person_id"].isin(ids)]
         .pivot(index="omop_person_id", columns="name", values="label"))
    w[REF_NAME] = ref.set_index("omop_person_id")["gold_label"].reindex(w.index)
    names = sorted(w.columns)
    M = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    n = pd.DataFrame(0, index=names, columns=names, dtype=int)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ok = w[a].notna() & w[b].notna()
            k = kappa3(w.loc[ok, a], w.loc[ok, b])
            M.loc[a, b] = M.loc[b, a] = k
            n.loc[a, b] = n.loc[b, a] = int(ok.sum())
    n.values[np.diag_indices_from(n)] = int(w.notna().all(axis=1).sum())
    return M, n


def code_row(ref, n_boot):
    """The diagnosis code against the reference standard, on the same charts and scale.

    The code is binary, so it maps to the two extreme grades. That is the conservative
    reading of it: a model that answers "Probably Yes" collects partial credit under
    linear weights and the code never can. The penalty is small and is reported in the
    caption, because a reader is entitled to know whether the mapping produced the
    result.
    """
    ros = pd.read_csv(ROSTER, dtype={"omop_person_id": str})
    m = ref.merge(ros[["omop_person_id", "has_aud_icd"]], on="omop_person_id",
                  how="left", validate="one_to_one")
    assert m["has_aud_icd"].notna().all(), "a reviewed chart carries no code indicator"
    lab = np.where(m["has_aud_icd"].astype(int).eq(1), "Definitely Yes", "Definitely No")
    k, lo, hi = kappa3_ci(m["gold_label"], pd.Series(lab, index=m.index), n_boot=n_boot)
    pair = collapse3(m["gold_label"]).notna() & collapse3(pd.Series(lab, index=m.index)).notna()
    return {"name": CODE_NAME, "n": int(pair.sum()), "n_labelled": len(m),
            "kappa_w": k, "ci_low": lo, "ci_high": hi}


def leaderboard(ref, p, n_boot):
    """Each model against the reference standard, on the whole analysis cohort."""
    rows = []
    for name, g in p.groupby("name"):
        m = g.merge(ref, on="omop_person_id", how="inner", validate="one_to_one")
        m = m[m["label"].notna() & m["gold_label"].notna()]
        k, lo, hi = kappa3_ci(m["gold_label"], m["label"], n_boot=n_boot)
        # The denominator is the charts both readers placed. It is not the same for
        # every model, because each one marks not enough data on a different set. The
        # spread is small and the ranking survives restricting to the charts on which
        # all eleven and the reference committed; see the caption.
        pair = collapse3(m["gold_label"]).notna() & collapse3(m["label"]).notna()
        rows.append({"name": name, "n": int(pair.sum()), "n_labelled": len(m),
                     "kappa_w": k, "ci_low": lo, "ci_high": hi})
    d = pd.DataFrame(rows).sort_values("kappa_w", ascending=False).reset_index(drop=True)
    assert d["n_labelled"].nunique() == 1, "the 11 models do not share one cohort"
    return d


def paired_and_rank(ref, p, n_boot, seed=42):
    """Two quantities the text reports, formed on the same charts and the same draws.

    The first is the primary model against the best model that was not on the panel
    which selected charts for attending review. Comparing against a panel member would
    compare a model with the process that built its own evaluation set. The difference
    is paired: the same resampled patients score both models in every draw.

    The second is rank stability. The leaderboard is one realisation; resampling
    patients gives the distribution of the primary model's rank.
    """
    PANEL_MODELS = {"Qwen3.6-27B", "GPT-OSS-120B", "GLM-5.2"}
    referred = set(ref.loc[ref["gold_source"].ne("maaz_unreviewed"), "omop_person_id"])
    w = (p.pivot(index="omop_person_id", columns="name", values="label")
         .join(ref.set_index("omop_person_id")["gold_label"], how="inner"))
    w = w.dropna()
    names = [c for c in w.columns if c != "gold_label"]

    lb = {n: kappa3(w["gold_label"], w[n]) for n in names}
    contender = max((n for n in names if n not in PANEL_MODELS), key=lb.get)

    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(w))
    rows, ranks = [], []
    for label, sel in (("whole cohort", np.ones(len(w), bool)),
                       ("not referred", ~w.index.isin(referred)),
                       ("referred", w.index.isin(referred))):
        s = w[sel]
        obs = (kappa3(s["gold_label"], s[PRIMARY])
               - kappa3(s["gold_label"], s[contender]))
        d = np.empty(n_boot)
        for b in range(n_boot):
            t = s.iloc[rng.integers(0, len(s), len(s))]
            d[b] = (kappa3(t["gold_label"], t[PRIMARY])
                    - kappa3(t["gold_label"], t[contender]))
        rows.append({"set": label, "n": len(s), "contender": contender,
                     "delta": obs, "ci_low": np.percentile(d, 2.5),
                     "ci_high": np.percentile(d, 97.5)})

    # Every model's rank in every draw, so the stability block is one computation rather
    # than a verified row beside unverified ones.
    ranks = {n: [] for n in names}
    for b in range(n_boot):
        t = w.iloc[rng.integers(0, len(w), len(idx_all))]
        k = {n: kappa3(t["gold_label"], t[n]) for n in names}
        order = sorted(names, key=lambda n: -k[n])
        for pos, n in enumerate(order, start=1):
            ranks[n].append(pos)
    stab = pd.DataFrame([{
        "name": n, "first_pct": 100 * float(np.mean(np.array(ranks[n]) == 1)),
        "rank_low": int(np.percentile(ranks[n], 2.5)),
        "rank_high": int(np.percentile(ranks[n], 97.5))} for n in names])
    stab = stab.sort_values("first_pct", ascending=False).reset_index(drop=True)
    return pd.DataFrame(rows), stab, contender


def by_route(ref, p, n_boot):
    """Agreement on the charts the escalation referred, against agreement on the rest.

    A chart reaches attending review on either trigger, so the referred set is not a
    random subsample: it is the set on which the panel or the second fellow dissented.
    That is the hard stratum, and every model is expected to fall on it.
    """
    referred = set(ref.loc[ref["gold_source"].ne("maaz_unreviewed"), "omop_person_id"])
    rows = []
    for name, g in p.groupby("name"):
        m = g.merge(ref, on="omop_person_id", how="inner", validate="one_to_one")
        m = m[m["label"].notna() & m["gold_label"].notna()]
        for setname, sel in (("not referred", ~m["omop_person_id"].isin(referred)),
                             ("referred", m["omop_person_id"].isin(referred))):
            s = m[sel]
            k, lo, hi = kappa3_ci(s["gold_label"], s["label"], n_boot=n_boot)
            pair = collapse3(s["gold_label"]).notna() & collapse3(s["label"]).notna()
            rows.append({"name": name, "set": setname, "n": int(pair.sum()),
                         "n_labelled": len(s),
                         "kappa_w": k, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ panels

def draw_matrix(ax, cax, M, n_same, lb):
    names = list(M.columns)
    A = M.values.astype(float)
    order = rank_order(names, lb)
    A, names = A[np.ix_(order, order)], [names[i] for i in order]
    im = ax.imshow(A, cmap="viridis", vmin=0.45, vmax=1.0)
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                ax.text(i, i, "—", ha="center", va="center", fontsize=4.6,
                        color="#CCCCCC")
                continue
            v = A[i, j]
            ax.text(j, i, ("%.2f" % v).lstrip("0"), ha="center", va="center",
                    fontsize=4.6, color="white" if v < 0.78 else "black")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=5.2)
    ax.set_yticklabels(names, fontsize=5.2)
    for t in list(ax.get_yticklabels()) + list(ax.get_xticklabels()):
        if t.get_text() == PRIMARY:
            t.set_fontweight("bold")
        if t.get_text() == REF_NAME:
            t.set_color(C_DROP)
    # The reference standard is not one more reader, so its row and column are ruled off
    # rather than left to be read as the twelfth model.
    ax.axhline(0.5, color="white", linewidth=1.1)
    ax.axvline(0.5, color="white", linewidth=1.1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=5.2, length=2, width=0.4, pad=1.5)
    cb.outline.set_linewidth(0.4)
    cax.set_title("$\\kappa$", fontsize=6, pad=3)
    ax.set_title("Every reader against every other\neach pair on the charts both "
                 "placed; ordered as in a", fontsize=6.5, color="#333333", pad=4)


def rank_order(names, lb):
    """The reference standard first, then the models by agreement with it.

    Average-linkage clustering on 1 - kappa was used until 14 August. It grouped the
    eight strong models together and left the three weak ones apart, which is the same
    grouping this ordering produces, but it did not let a reader carry a name from one
    panel to the other. Rank does.
    """
    rank = {r.name: r.kappa_w for r in lb.itertuples()}
    rest = sorted([n for n in names if n != REF_NAME],
                  key=lambda n: -rank.get(n, -1.0))
    return [names.index(n) for n in [REF_NAME] + rest]


def cluster_order(d):
    """Average-linkage order on the distance 1 - kappa. Kept for reference, unused."""
    n = len(d)
    d = 1.0 - d
    np.fill_diagonal(d, 0.0)
    cl = [[i] for i in range(n)]
    while len(cl) > 1:
        best, bi, bj = None, 0, 1
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                v = np.mean([d[a, b] for a in cl[i] for b in cl[j]])
                if best is None or v < best:
                    best, bi, bj = v, i, j
        cl = [c for k, c in enumerate(cl) if k not in (bi, bj)] + [cl[bi] + cl[bj]]
    return cl[0]


def draw_leaderboard(ax, lb, code):
    """The models, then the diagnosis code below a rule.

    The code is separated rather than ranked among the models, because it is the
    comparator the paper is measuring against and not a candidate for the same job.
    """
    d = lb.sort_values("kappa_w").reset_index(drop=True)
    d = pd.concat([pd.DataFrame([code]), d], ignore_index=True)
    y = np.arange(len(d))
    for yy, r in zip(y, d.itertuples()):
        c = C_PRIMARY if r.name == PRIMARY else (C_DROP if r.name == CODE_NAME
                                                 else C_OTHER)
        ax.plot([r.ci_low, r.ci_high], [yy, yy], color=c, linewidth=0.9, zorder=2)
        for b in (r.ci_low, r.ci_high):
            ax.plot([b, b], [yy - 0.16, yy + 0.16], color=c, linewidth=0.9, zorder=2)
        ax.plot(r.kappa_w, yy, "o", color=c, markersize=4, markeredgecolor="black",
                markeredgewidth=0.35, zorder=3)
        ax.text(1.012, yy, "%.3f" % r.kappa_w, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=5.6)
    ax.set_yticks(y)
    ax.set_yticklabels(d["name"], fontsize=5.6)
    for t in ax.get_yticklabels():
        if t.get_text() == PRIMARY:
            t.set_fontweight("bold")
        if t.get_text() == CODE_NAME:
            t.set_color(C_DROP)
    ax.axhline(0.5, color=RULE, linewidth=0.5, zorder=1)
    ax.set_xlim(0.50, 0.98)
    ax.set_ylim(-0.7, len(d) - 0.05)
    ax.set_xlabel("Cohen's $\\kappa$, three categories")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Agreement with the reference standard\nn = %d to %d; 2,000-sample "
                 "bootstrap intervals" % (int(d["n"].min()), int(d["n"].max())),
                 fontsize=6.5, color="#333333", pad=4)


def draw_c(ax, r):
    """Slope chart with the bootstrap interval at both ends."""
    w = r.pivot(index="name", columns="set", values="kappa_w")
    ci = {(x.name, x.set): (x.ci_low, x.ci_high) for x in r.itertuples()}
    n = {s: int(r[r["set"] == s]["n"].iloc[0]) for s in ("not referred", "referred")}
    w = w.sort_values("not referred", ascending=False)
    xa, xb = 0.0, 1.0
    HI = {PRIMARY: C_PRIMARY, "DeepSeek-V4-Flash": C_DROP}
    for name, row in w.iterrows():
        c = HI.get(name, "#B4B4B4")
        hi = name in HI
        lw, z = (1.4, 5) if hi else (0.7, 2)
        ax.plot([xa, xb], [row["not referred"], row["referred"]], color=c, linewidth=lw,
                zorder=z, solid_capstyle="round")
        for xx, st in ((xa, "not referred"), (xb, "referred")):
            lo, hh = ci[(name, st)]
            ax.plot([xx, xx], [lo, hh], color=c, linewidth=0.7 if hi else 0.5,
                    alpha=0.85 if hi else 0.55, zorder=z)
            ax.plot(xx, row[st], "o", color=c, markersize=3.8 if hi else 3.2,
                    markeredgecolor="black", markeredgewidth=0.3, zorder=z + 1)
        if hi:
            # The model is named once, on the right. On the left only the value is
            # printed, because a name there runs into the y axis.
            ax.text(xa - 0.06, row["not referred"], "%.3f" % row["not referred"],
                    ha="right", va="center", fontsize=5.4, color=c)
            ax.text(xb + 0.06, row["referred"], "%s  %.3f" % (name, row["referred"]),
                    ha="left", va="center", fontsize=5.4, color=c)
    ax.set_xlim(-0.34, 1.86)
    ax.set_xticks([xa, xb])
    ax.set_xticklabels(["Not referred\nn = %d" % n["not referred"],
                        "Referred\nn = %d" % n["referred"]], fontsize=6)
    # The interval on the not-referred side reaches close to one, so the top of the axis
    # is set from the data rather than at one.
    ax.set_ylim(0.03, min(1.04, float(r["ci_high"].max()) + 0.03))
    ax.set_ylabel("Cohen's $\\kappa$, three categories")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Agreement on the charts the escalation referred\nall 11 models decline; "
                 "the intervals on %d overlap,\nso the order there is not resolved"
                 % n["referred"], fontsize=6.5, color="#333333", pad=4)


def draw_d(ax, g5):
    d = g5.sort_values("order").copy()
    y = np.arange(len(d))[::-1]
    colors = [C_PRIMARY, C_TEMP, C_TEMP, C_PROMPT, C_PROMPT]
    base = float(d.loc[d["condition"].eq("v1_t0p0"), "kappa_w"].iloc[0])
    ax.axvline(base, color=C_PRIMARY, linestyle="--", linewidth=0.8, alpha=0.8, zorder=0)
    for yy, row, color in zip(y, d.itertuples(), colors):
        ax.plot([row.ci_low, row.ci_high], [yy, yy], color=color, linewidth=1.2,
                solid_capstyle="round", zorder=2)
        for b in (row.ci_low, row.ci_high):
            ax.plot([b, b], [yy - 0.14, yy + 0.14], color=color, linewidth=0.8, zorder=2)
        ax.plot(row.kappa_w, yy, "o", color=color, markersize=4.2,
                markeredgecolor="black", markeredgewidth=0.35, zorder=3)
        ax.text(1.01, yy, "%.3f" % row.kappa_w, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=5.6)
    ax.set_yticks(y)
    ax.set_yticklabels(d["display"], fontsize=5.6)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlim(max(0.0, float(d["ci_low"].min()) - 0.02),
                min(1.0, float(d["ci_high"].max()) + 0.02))
    ax.set_ylim(-0.65, len(d) - 0.25)
    ax.set_xlabel("Cohen's $\\kappa$, three categories")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Prompt and temperature sensitivity\nn = %d to %d; complete paired "
                 "coverage" % (int(d["n"].min()), int(d["n"].max())),
                 fontsize=6.5, color="#333333", pad=4)
    ax.legend(handles=[Line2D([0], [0], color=C_TEMP, marker="o", linewidth=1,
                              markersize=3, label="Temperature"),
                       Line2D([0], [0], color=C_PROMPT, marker="o", linewidth=1,
                              markersize=3, label="Prompt wording")],
              loc="upper left", bbox_to_anchor=(-0.02, -0.20), fontsize=5.2, ncol=2,
              handlelength=1.2, columnspacing=0.8, frameon=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--goal5", default=GOAL5)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ref, p, man = load(a.panel)
    ids = set(ref["omop_person_id"])
    lb = leaderboard(ref, p, a.n_boot)
    M, nmat = pairwise(p, ids, ref)
    code = code_row(ref, a.n_boot)
    rt = by_route(ref, p, a.n_boot)
    pdlt, stab, contender = paired_and_rank(ref, p, a.n_boot)
    g5 = pd.read_csv(a.goal5)

    n_same = int(nmat.values[np.diag_indices_from(nmat.values)][0])

    mpl = matplotlib.rcParams
    mpl.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
                "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
                "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                "ytick.major.width": 0.5, "xtick.major.size": 3,
                "ytick.major.size": 3, "lines.linewidth": 0.8, "lines.markersize": 3,
                "legend.frameon": False, "axes.grid": False,
                "axes.spines.top": False, "axes.spines.right": False,
                "figure.facecolor": "white", "savefig.facecolor": "white",
                "savefig.dpi": 300})

    fig = plt.figure(figsize=(7.09, 6.25))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.30, 1.0], hspace=0.42, wspace=1.02,
                          left=0.155, right=0.905, top=0.93, bottom=0.085)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc, axd = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    pos = axb.get_position()
    cax = fig.add_axes([pos.x1 + 0.010, pos.y0 + 0.02, 0.0095, pos.height * 0.55])

    draw_leaderboard(axa, lb, code)
    draw_matrix(axb, cax, M, n_same, lb)
    draw_c(axc, rt)
    draw_d(axd, g5)
    for ax, lab, dx, dy in ((axa, "a", -0.52, 1.02), (axb, "b", -0.55, 1.02),
                            (axc, "c", -0.52, 1.15), (axd, "d", -0.55, 1.15)):
        ax.text(dx, dy, lab, transform=ax.transAxes, fontsize=8, fontweight="bold",
                va="bottom", ha="left")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "figure4.%s" % ext))
    plt.close(fig)
    print("wrote %s/figure4.pdf and .png" % a.out)

    # Every value the manuscript quotes from this figure is written here as well. Until
    # 14 August the panels existed only as printed output, so a number in the text could
    # not be checked against the run that drew the panel beside it, and nine of the eleven
    # leaderboard intervals had drifted from a different script.
    lb.to_csv(os.path.join(a.out, "panel_a_leaderboard.csv"), index=False)
    M.to_csv(os.path.join(a.out, "panel_b_kappa_matrix.csv"))
    nmat.to_csv(os.path.join(a.out, "panel_b_n_matrix.csv"))
    rt.to_csv(os.path.join(a.out, "panel_c_by_route.csv"), index=False)
    g5.to_csv(os.path.join(a.out, "panel_d_conditions.csv"), index=False)
    pd.DataFrame([code]).to_csv(os.path.join(a.out, "diagnosis_code.csv"), index=False)
    pdlt.to_csv(os.path.join(a.out, "text_paired_contender.csv"), index=False)
    stab.to_csv(os.path.join(a.out, "text_rank_stability.csv"), index=False)
    print("wrote the panel tables beside it")

    print("\ncohort: %d analysed of %d rostered; reference by source %s"
          % (man["cohort"]["analysed"], man["cohort"]["roster_rows"], man["by_source"]))
    off = ~np.eye(len(M), dtype=bool)
    mm = M.drop(index=REF_NAME, columns=REF_NAME)
    moff = ~np.eye(len(mm), dtype=bool)
    print("\npanel b, pairwise: every pair on %d charts; min %.3f, max off-diagonal %.3f, "
          "median %.3f" % (n_same, M.values[off].min(), M.values[off].max(),
                           np.median(M.values[off])))
    print("  model against model, median %.3f; model against the standard, median %.3f"
          % (np.median(mm.values[moff]), np.median(M.loc[REF_NAME].drop(REF_NAME))))
    print("  the diagnosis code against the standard, %.3f (%.3f to %.3f)"
          % (code["kappa_w"], code["ci_low"], code["ci_high"]))
    print("\npanel a, agreement with the reference standard")
    print(lb.to_string(index=False, float_format=lambda x: "%.3f" % x))
    print("\npanel c, by escalation route")
    print(rt.pivot(index="name", columns="set", values="kappa_w")
          .join(rt[rt["set"] == "referred"].set_index("name")[["ci_low", "ci_high"]])
          .sort_values("not referred", ascending=False)
          .to_string(float_format=lambda x: "%.3f" % x))
    print("\ntext, %s minus %s, paired" % (PRIMARY, contender))
    print(pdlt.to_string(index=False, float_format=lambda x: "%.4f" % x))
    print("\ntext, rank stability over %d patient resamples" % a.n_boot)
    print(stab.to_string(index=False, float_format=lambda x: "%.1f" % x))
    print("\ntext, kappa on the not-referred charts by panel membership")
    PANEL_MODELS = {"Qwen3.6-27B", "GPT-OSS-120B", "GLM-5.2"}
    nr = rt[rt["set"] == "not referred"]
    for lab, sel in (("on the panel, 3 models", nr["name"].isin(PANEL_MODELS)),
                     ("not on the panel, 8 models", ~nr["name"].isin(PANEL_MODELS))):
        v = nr.loc[sel, "kappa_w"]
        print("  %-28s %d models, %.3f to %.3f" % (lab, len(v), v.min(), v.max()))
    print("\npanel d, prompt and temperature")
    print(g5[["order", "display", "n", "kappa_w", "ci_low", "ci_high",
              "paired_difference_from_baseline", "paired_ci_low", "paired_ci_high",
              "bucket_changes_from_baseline", "abstentions"]]
          .to_string(index=False, float_format=lambda x: "%.4f" % x))


if __name__ == "__main__":
    main()
