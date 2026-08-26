#!/usr/bin/env python3
"""Findings 01: how closely the model reproduces the clinician reference standard,
and how the diagnosis code compares on the same patients.

Three things are reported, in the order they should be read.

  1  The full six-category agreement, before anything is collapsed. A model that misses
     by one grade is not doing what a model that inverts the answer is doing, and only the
     six-category table shows the difference.

  2  Accuracy against the reference standard, in the three-category collapse, with the
     diagnosis code scored on exactly the same patients by exactly the same rule. The code
     comparison is the point of the study, so it is not a secondary analysis: the model and
     the code are two candidate readings of one record and they are scored side by side.

  3  The same quantities weighted back to the registry. Charts were sampled in roughly
     equal numbers with and without an AUD diagnosis code, so the sample holds about twice
     the coded fraction the registry does. Every proportion computed on the sample
     therefore describes the sample. That distorts the model's numbers a little and the
     code's numbers a great deal, because the code is the variable the sample was drawn on,
     and the sensitivity of a diagnosis code is a statement about coded patients.

Patients whose reference standard is indeterminate carry no label to score against and are
excluded from every accuracy figure here; they are counted, not silently dropped. Where the
model itself returns an indeterminate category the patient stays in the denominator, so a
model cannot raise its accuracy by declining to commit.

Outputs, under --out:
    confusion_six.csv        reference by model, six categories
    confusion_three.csv      the same collapsed to three
    accuracy.csv             every metric, as sampled and weighted, with intervals
    accuracy_by_site.csv     the same split by hospital system
    disagreements.csv        every patient the model and the reference disagree on
    accuracy.pdf, .png       the confusion matrices and the model-versus-code comparison

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "downstream")
sys.path.insert(0, str(HERE))
import fig6_bayes_strata as B  # noqa: E402
import ref_standard as R  # noqa: E402
import stats_core as sc  # noqa: E402

OUT = "downstream/npj/out/f01_accuracy"
MODEL = "aud_qwen36_27b"
MODEL_NAME = "LLM (Qwen3.6-27B)"
L6 = ["Definitely Yes", "Probably Yes", "Uncertain", "Probably No", "Definitely No",
      "Not Enough Data"]
SHORT6 = ["Def. yes", "Prob. yes", "Uncertain", "Prob. no", "Def. no", "No data"]
L3 = ["AUD+", "Indeterminate", "AUD-"]
NB = 2000
SEED = 20260811


# ----------------------------------------------------------------- point estimates
def wprop(mask, w):
    return float(w[mask].sum() / w.sum()) if w.sum() else float("nan")


def wrate(s, num, den):
    """Weighted proportion of num among den, both predicates over the rows of s.

    The predicates run on whatever frame arrives rather than on arrays fixed in advance,
    so a bootstrap replicate derives its own numerator and its own denominator.
    """
    keep = den(s).to_numpy()
    if not keep.any():
        return float("nan")
    return wprop((num(s).to_numpy() & keep)[keep], s["w"].to_numpy()[keep])


def boot(s, metric, rng, nb=NB):
    """Percentile interval from a patient-level resample within the sampling stratum.

    The resample covers the whole reviewed set rather than the denominator group alone.
    Confining it to the denominator would hold the number of denominator patients per
    stratum fixed, and any metric whose numerator is a function of the sampling variable
    would then be identical in every replicate. That applies to the diagnosis code, which
    is the variable the sample was drawn on, so its interval would collapse onto its point
    estimate and look far more certain than it is.
    """
    u = s["u"].to_numpy()
    strata = [np.flatnonzero(u == lv) for lv in np.unique(u)]
    out = []
    for _ in range(nb):
        take = np.concatenate([rng.choice(v, len(v), replace=True) for v in strata])
        out.append(metric(s.iloc[take]))
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def wilson(k, n):
    if not n:
        return float("nan"), float("nan")
    z, p = 1.959964, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def wkappa(s, pred):
    """Cohen's kappa on the two determinate categories, using weighted marginals."""
    w = s["w"].to_numpy()
    obs = (s[pred] == s["ref3"]).to_numpy()
    po = wprop(obs, w)
    pe = sum(wprop((s["ref3"] == c).to_numpy(), w) * wprop((s[pred] == c).to_numpy(), w)
             for c in ("AUD+", "AUD-"))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


# ----------------------------------------------------------------------- assembly
def load(cohort):
    g, man = R.build(cohort=cohort)
    r = pd.read_csv("output/raters/rater_long.csv", dtype=str)
    w3 = r.pivot_table(index="omop_person_id", columns="rater", values="label_3",
                       aggfunc="first")
    w6 = r.pivot_table(index="omop_person_id", columns="rater", values="label_6",
                       aggfunc="first")
    ros = pd.read_csv("resources/patient_roster.csv", dtype={"omop_person_id": str})
    inst = pd.read_csv("output/fairness/patient_institution.csv")
    inst["omop_person_id"] = inst["person_id"].astype(str)

    d = g[["omop_person_id", "review_id", "gold_label", "gold_three",
           "gold_provisional"]].copy()
    d["model6"] = sc.normalize_labels(d["omop_person_id"].map(w6[MODEL]), L6, strict=False)
    d["ref6"] = sc.normalize_labels(d["gold_label"], L6, strict=False)
    d["model3"] = d["omop_person_id"].map(w3[MODEL])
    d["ref3"] = d["gold_three"]
    d = d.merge(ros[["omop_person_id", "has_aud_icd"]], on="omop_person_id")
    d = d.merge(inst[["omop_person_id", "institution"]], on="omop_person_id", how="left")
    d["code3"] = d["has_aud_icd"].map({1: "AUD+", 0: "AUD-"})
    d["u"] = d["has_aud_icd"].astype(int)
    if d["model3"].isna().any():
        raise SystemExit("%d cohort patients have no model result; the index test was "
                         "reported as complete" % int(d["model3"].isna().sum()))
    return d, man


def weights(d):
    """Horvitz-Thompson weights, registry stratum size over sampled size."""
    reg = B.build()
    N = reg["aud"].notna().astype(int).value_counts()
    scored = d[d["ref3"].isin(["AUD+", "AUD-"])]
    n = scored.groupby("u").size()
    w = {u: N.get(u, 0) / n.get(u, 1) for u in (0, 1)}
    d["w"] = d["u"].map(w)
    return d, {"registry_coded": int(N.get(1, 0)), "registry_uncoded": int(N.get(0, 0)),
               "scored_coded": int(n.get(1, 0)), "scored_uncoded": int(n.get(0, 0)),
               "weight_coded": w[1], "weight_uncoded": w[0]}


def metrics_block(s, pred, label, rng):
    """Every metric for one predictor on one subset, as sampled and weighted."""
    pos = lambda t: t["ref3"] == "AUD+"          # noqa: E731
    neg = lambda t: t["ref3"] == "AUD-"          # noqa: E731
    said_pos = lambda t: t[pred] == "AUD+"       # noqa: E731
    said_neg = lambda t: t[pred] == "AUD-"       # noqa: E731
    rows = []
    for name, num, den in [
            ("sensitivity", said_pos, pos),
            ("specificity", said_neg, neg),
            ("positive predictive value", pos, said_pos),
            ("negative predictive value", neg, said_neg),
            ("accuracy", lambda t: t[pred] == t["ref3"], lambda t: t["ref3"].notna())]:
        keep = den(s).to_numpy()
        hit = num(s).to_numpy() & keep
        k, n = int(hit.sum()), int(keep.sum())
        raw = k / n if n else float("nan")
        lo_r, hi_r = wilson(k, n)
        m = partial(wrate, num=num, den=den)
        wt = m(s)
        lo, hi = boot(s, m, rng)
        rows.append({"subset": label, "predictor": pred, "metric": name, "k": k, "n": n,
                     "as_sampled": raw, "sampled_lo": lo_r, "sampled_hi": hi_r,
                     "weighted": wt, "weighted_lo": lo, "weighted_hi": hi})
    both = s[s[pred].isin(["AUD+", "AUD-"])]
    rk = sc.cohens_kappa(both["ref3"], both[pred], ["AUD+", "AUD-"], weights=None)
    rows.append({"subset": label, "predictor": pred, "metric": "kappa",
                 "k": int((both[pred] == both["ref3"]).sum()), "n": len(both),
                 "as_sampled": rk, "sampled_lo": np.nan, "sampled_hi": np.nan,
                 "weighted": wkappa(both, pred),
                 "weighted_lo": np.nan, "weighted_hi": np.nan})
    return rows


# ------------------------------------------------------------------------- figure
# One colour vocabulary for the whole paper. The index test and its comparator are two
# categories, so they take two hues; the reference standard is the yardstick and stays
# black; grey means no information, either a reader who could not decide or a null
# expectation. Colour never encodes how large a number is. Each series also carries a
# marker shape, so the figure survives being printed in grey.
C_MODEL = "#0072B2"      # Okabe-Ito blue
C_CODE = "#D55E00"       # Okabe-Ito vermillion
C_REF = "#000000"
C_INDET = "#BDC3C7"
C_NULL = "#8A8A8A"
M_MODEL, M_CODE = "o", "s"


def figure(c6, c3, acc, site, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from nature_style import apply_nature_style, add_panel_labels, save_nature_figure

    apply_nature_style()
    fig = plt.figure(figsize=(7.205, 4.7))
    # constrained_layout is on by default in the house style and does not combine with
    # hand-set spacing; the panels here need fixed spacing, so the engine is switched off.
    fig.set_layout_engine("none")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.30], height_ratios=[0.86, 1.0],
                          wspace=0.34, hspace=0.55,
                          left=0.075, right=0.985, top=0.93, bottom=0.10)

    # -- a. what the readers were asked, and how six answers become three ------------
    # The collapse is shown by filling each grade with its class colour rather than by
    # drawing brackets. Indeterminate is not contiguous on the scale, since Uncertain
    # sits in the middle and Not enough data is off it entirely, and a bracket would
    # imply an adjacency that does not exist.
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-0.05, 6.05)
    ax.set_ylim(-2.05, 1.5)
    ax.axis("off")
    klass = [("AUD+", C_MODEL), ("AUD+", C_MODEL), ("Indeterminate", C_INDET),
             ("AUD−", C_CODE), ("AUD−", C_CODE), ("Indeterminate", C_INDET)]
    tint = {C_MODEL: "#CCE3F0", C_CODE: "#F7DCCC", C_INDET: "#EDEDEA"}
    for i, (lab, (_, col)) in enumerate(zip(SHORT6, klass)):
        ax.add_patch(Rectangle((i + 0.06, 0.0), 0.88, 0.72, fc=tint[col], ec=col,
                               lw=0.6))
        ax.text(i + 0.5, 0.36, lab, ha="center", va="center", fontsize=5.4)
    ax.annotate("", xy=(5.0, -0.30), xytext=(0.06, -0.30),
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#55534e"))
    ax.text(2.53, -0.62, "ordered, most to least certain", ha="center", fontsize=5.2,
            color="#55534e")
    ax.text(5.5, -0.62, "off the\nscale", ha="center", va="top", fontsize=5.2,
            color="#55534e")
    for k, (name, col) in enumerate([("AUD+", C_MODEL), ("Indeterminate", C_INDET),
                                     ("AUD−", C_CODE)]):
        x = 0.15 + k * 2.1
        ax.add_patch(Rectangle((x, -1.75), 0.34, 0.34, fc=tint[col], ec=col, lw=0.6))
        ax.text(x + 0.48, -1.58, name, ha="left", va="center", fontsize=5.6)
    ax.text(3.0, 1.28, "Every reader, human or model, returned one of six grades",
            ha="center", fontsize=5.4, color="#55534e")
    ax.text(3.0, -2.02, "the two indeterminate grades differ in kind and are counted "
            "apart", ha="center", va="bottom", fontsize=5.0, color="#55534e")

    # -- b. agreement, six categories -------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    m = c6.to_numpy(float)
    grp = [C_MODEL, C_MODEL, C_INDET, C_CODE, C_CODE, C_INDET]
    for i in range(6):
        for j in range(6):
            # A cell is tinted when the collapse treats the two grades as agreeing, and
            # it takes the colour of the class they agree on. The tint therefore carries
            # the same meaning it has in panel a, and it says nothing about how many
            # patients the cell holds.
            same = grp[i] == grp[j]
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   fc=tint[grp[i]] if same else "white",
                                   ec="#E3E1DC", lw=0.3))
            if m[i, j]:
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=5.5,
                        color=C_REF, fontweight="bold" if i == j else "normal")
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(5.5, -0.5)
    ax.set_xticks(range(6))
    ax.set_xticklabels(SHORT6, rotation=45, ha="right")
    ax.set_yticks(range(6))
    ax.set_yticklabels(SHORT6)
    ax.set_xlabel(MODEL_NAME)
    ax.set_ylabel("Reference standard")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

    # -- c. what weighting to the registry does ---------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    show = ["sensitivity", "specificity"]
    a = acc[acc["subset"] == "all"].set_index(["predictor", "metric"])
    xs = [0, 1]
    for pred, col, mk, lab in ((MODEL, C_MODEL, M_MODEL, MODEL_NAME),
                               ("code3", C_CODE, M_CODE, "Diagnosis code")):
        for k, met in enumerate(show):
            r = a.loc[(pred, met)]
            ax.plot([k - 0.16, k + 0.16], [r["as_sampled"], r["weighted"]],
                    lw=0.8, color=col, zorder=1)
            ax.plot(k - 0.16, r["as_sampled"], mk, ms=3.4, mfc="white", mec=col,
                    mew=0.9, zorder=2)
            ax.errorbar(k + 0.16, r["weighted"],
                        yerr=[[r["weighted"] - r["weighted_lo"]],
                              [r["weighted_hi"] - r["weighted"]]],
                        fmt=mk, ms=4, color=col, lw=0.8, capsize=1.6, zorder=2,
                        label=lab if k == 0 else None)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Sensitivity", "Specificity"])
    ax.set_ylim(0.3, 1.03)
    ax.set_ylabel("Estimate")
    ax.text(0.02, 0.06, "open marker, in the sample\nfilled, weighted to the registry",
            transform=ax.transAxes, fontsize=5, color="#55534e", va="bottom")
    ax.legend(loc="lower right", fontsize=5.5)

    # -- d. the gradient across hospital systems --------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    order = ["academic health system", "other hospitals", "safety-net hospital"]
    short = ["Academic", "Other", "Safety-net"]
    s = site[site["metric"] == "sensitivity"].set_index(["subset", "predictor"])
    x = np.arange(len(order))
    for pred, col, mk, lab in ((MODEL, C_MODEL, M_MODEL, MODEL_NAME),
                               ("code3", C_CODE, M_CODE, "Diagnosis code")):
        v = [s.loc[(o, pred), "weighted"] for o in order]
        lo = [s.loc[(o, pred), "weighted_lo"] for o in order]
        hi = [s.loc[(o, pred), "weighted_hi"] for o in order]
        ax.plot(x, v, "-", lw=0.8, color=col, zorder=1)
        ax.errorbar(x, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt=mk, ms=4,
                    color=col, lw=0.8, capsize=1.6, label=lab, zorder=2)
    ax.axhline(0.5, ls=":", lw=0.6, color=C_NULL, zorder=0)
    ax.text(2.42, 0.5, "half of\ncases", fontsize=5, color=C_NULL, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_xlim(-0.35, 2.75)
    ax.set_ylim(0.3, 1.04)
    ax.set_ylabel("Sensitivity, weighted")
    ax.legend(loc="lower left", fontsize=5.5)

    add_panel_labels(fig.axes, x=-0.10, y=1.12)
    save_nature_figure(fig, os.path.join(out, "accuracy"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--cohort", default=R.COHORT, choices=["era_a", "all"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    d, man = load(a.cohort)
    d, wsum = weights(d)
    scored = d[d["ref3"].isin(["AUD+", "AUD-"])].copy()

    print("cohort %s, %d patients; %d carry a determinate reference and are scored"
          % (man["cohort"]["name"], len(d), len(scored)))
    print("  registry %d coded and %d uncoded; each scored patient stands for %.1f "
          "coded or %.1f uncoded"
          % (wsum["registry_coded"], wsum["registry_uncoded"],
             wsum["weight_coded"], wsum["weight_uncoded"]))
    print("  excluded as indeterminate: %d" % int((d["ref3"] == "Indeterminate").sum()))

    print()
    print("=" * 92)
    print("SIX CATEGORIES, REFERENCE BY MODEL")
    print("=" * 92)
    det = d[d["ref6"].notna()]
    c6 = pd.crosstab(pd.Categorical(det["ref6"], L6), pd.Categorical(det["model6"], L6),
                     dropna=False)
    c6.index.name, c6.columns.name = "reference", MODEL_NAME
    print(c6.to_string())
    diag = float(np.trace(c6.to_numpy())) / c6.to_numpy().sum()
    off1 = sum(c6.to_numpy()[i, j] for i in range(6) for j in range(6)
               if abs(i - j) == 1 and i < 5 and j < 5)
    print("  on the diagonal %.1f%%, one grade away a further %.1f%%"
          % (100 * diag, 100 * off1 / c6.to_numpy().sum()))
    c6.to_csv(os.path.join(a.out, "confusion_six.csv"))

    print()
    print("=" * 92)
    print("THREE CATEGORIES")
    print("=" * 92)
    c3 = pd.crosstab(pd.Categorical(d["ref3"], L3), pd.Categorical(d["model3"], L3),
                     dropna=False)
    c3.index.name, c3.columns.name = "reference", MODEL_NAME
    print(c3.to_string())
    c3.to_csv(os.path.join(a.out, "confusion_three.csv"))

    print()
    print("=" * 92)
    print("ACCURACY AGAINST THE REFERENCE STANDARD")
    print("=" * 92)
    rows = []
    for pred, lab in ((MODEL, MODEL_NAME), ("code3", "diagnosis code")):
        rows += metrics_block(scored, pred if pred == "code3" else "model3", "all", rng)
    acc = pd.DataFrame(rows)
    acc["predictor"] = acc["predictor"].map({"model3": MODEL, "code3": "code3"})
    print("  %-26s %-26s %10s %22s %22s"
          % ("predictor", "metric", "k / n", "as sampled", "weighted"))
    for _, r in acc.iterrows():
        nm = MODEL_NAME if r["predictor"] == MODEL else "Diagnosis code"
        s_ci = ("" if np.isnan(r["sampled_lo"]) else
                " (%.3f-%.3f)" % (r["sampled_lo"], r["sampled_hi"]))
        w_ci = ("" if np.isnan(r["weighted_lo"]) else
                " (%.3f-%.3f)" % (r["weighted_lo"], r["weighted_hi"]))
        print("  %-26s %-26s %10s %22s %22s"
              % (nm, r["metric"], "%d/%d" % (r["k"], r["n"]),
                 "%.3f%s" % (r["as_sampled"], s_ci), "%.3f%s" % (r["weighted"], w_ci)))

    print()
    print("=" * 92)
    print("BY HOSPITAL SYSTEM")
    print("=" * 92)
    site_rows = []
    for site, lab in (("IUH", "academic health system"),
                      ("Eskenazi", "safety-net hospital"),
                      ("Other", "other hospitals")):
        s = scored[scored["institution"] == site]
        if len(s) < 20:
            print("  %-24s n=%d, too few to report" % (lab, len(s)))
            continue
        for pred in ("model3", "code3"):
            site_rows += metrics_block(s, pred, lab, rng)
    site = pd.DataFrame(site_rows)
    if len(site):
        site["predictor"] = site["predictor"].map({"model3": MODEL, "code3": "code3"})
        for _, r in site[site["metric"].isin(["sensitivity", "specificity"])].iterrows():
            nm = MODEL_NAME if r["predictor"] == MODEL else "Diagnosis code"
            print("  %-24s %-16s %-12s %d/%-4d  sampled %.3f  weighted %.3f (%.3f-%.3f)"
                  % (r["subset"], nm, r["metric"], r["k"], r["n"], r["as_sampled"],
                     r["weighted"], r["weighted_lo"], r["weighted_hi"]))
        site.to_csv(os.path.join(a.out, "accuracy_by_site.csv"), index=False)

    dis = scored[scored["model3"] != scored["ref3"]][
        ["omop_person_id", "review_id", "institution", "has_aud_icd",
         "ref6", "model6", "ref3", "model3", "gold_provisional"]]
    dis.to_csv(os.path.join(a.out, "disagreements.csv"), index=False)
    acc.to_csv(os.path.join(a.out, "accuracy.csv"), index=False)
    if len(site):
        figure(c6, c3, acc, site, a.out)
    else:
        print("  no site breakdown, so the figure is not drawn")

    print()
    print("=" * 92)
    print("WHERE THE MODEL AND THE REFERENCE PART COMPANY")
    print("=" * 92)
    print("  %d of %d scored patients (%.1f%%)" % (len(dis), len(scored),
                                                   100 * len(dis) / len(scored)))
    for _, r in dis.iterrows():
        print("    %-18s %-11s reference %-14s model %-14s%s"
              % (r["omop_person_id"], r["institution"] or "-", r["ref6"], r["model6"],
                 "  [provisional]" if r["gold_provisional"] else ""))

    print()
    print("wrote confusion_six.csv, confusion_three.csv, accuracy.csv, "
          "accuracy_by_site.csv, disagreements.csv, accuracy.pdf, accuracy.png")


if __name__ == "__main__":
    main()
