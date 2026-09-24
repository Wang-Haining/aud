#!/usr/bin/env python3
"""Panels d and e of Figure 1: the values themselves, and who is right where they differ.

Panel d is a forest plot of what each reader achieved, not of differences. Sensitivity and
specificity for the instrument, the hepatology fellow and the diagnosis code, each shown as
observed in the reviewed sample and again weighted to the registry. Everything sits on one
absolute scale, so the reader sees our numbers directly and can also see the single place we
do not come out ahead: the code's weighted specificity. Nothing is hidden by showing values
rather than contrasts, and the paired differences behind the same comparisons are written to
a csv beside the figure.

Panel e asks the paired question twice, and draws each answer as a two-by-two matrix in the
same visual language as panel b, so a reader who has just read that panel does not have to
learn a new convention. Against the diagnosis code the off-diagonal is lopsided. Against the
fellow the two readers disagree on sixteen patients out of 399 and neither is reliably right,
which is the honest form of the claim that the instrument reads a chart as a fellow does.

Intervals come from a paired stratified bootstrap: the same resampled patients are scored by
every reader, resampled within the sampling stratum, so the design and the pairing both
survive into the interval.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "downstream")
sys.path.insert(0, str(HERE))

sys.path.insert(0, "downstream")
import stats_core as SCORE  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

import panel_a as PA                                                # noqa: E402
import ref_standard as R                                            # noqa: E402
from nature_style import apply_nature_style, save_nature_figure      # noqa: E402

NB = 2000
SEED = 20260811
MODEL, FELLOW, CODE = "M", "F", "I"
READERS = [(MODEL, "LLM (Qwen3.6-27B)", PA.C_POS, "o"),
           (FELLOW, "Hepatologist", PA.INK, "^"),
           (CODE, "ICD F10.x", PA.C_NEG, "s")]
NAME = dict((k, n) for k, n, _, _ in READERS)
COLOUR = dict((k, c) for k, _, c, _ in READERS)
MARK = dict((k, m) for k, _, _, m in READERS)


# ------------------------------------------------------------------ the data
def load():
    g, _ = R.build()
    r = pd.read_csv("output/raters/rater_long.csv", dtype=str)
    w3 = r.pivot_table(index="omop_person_id", columns="rater", values="label_3",
                       aggfunc="first")
    ros = pd.read_csv("resources/patient_roster.csv", dtype={"omop_person_id": str})
    d = g[["omop_person_id", "gold_three"]].merge(
        ros[["omop_person_id", "has_aud_icd"]], on="omop_person_id")
    d[MODEL] = d["omop_person_id"].map(w3["aud_qwen36_27b"])
    d[FELLOW] = d["omop_person_id"].map(w3["maaz"])
    d[CODE] = d["has_aud_icd"].map({1: "AUD+", 0: "AUD-"})
    sc = d[d["gold_three"].isin(["AUD+", "AUD-"])].copy()

    import fig6_bayes_strata as B
    reg = B.build()
    N = reg["aud"].notna().astype(int).value_counts()
    n = sc.groupby("has_aud_icd").size()
    w = {u: N.get(u, 0) / n.get(u, 1) for u in (0, 1)}
    sc["w"] = sc["has_aud_icd"].map(w)
    return sc


def est(s, who, met, weighted):
    if met == "sensitivity":
        den, num = s["gold_three"] == "AUD+", s[who] == "AUD+"
    else:
        den, num = s["gold_three"] == "AUD-", s[who] == "AUD-"
    k = s[den]
    ww = k["w"] if weighted else pd.Series(1.0, index=k.index)
    return float((ww * num[den]).sum() / ww.sum())


def _boot_indices(sc, rng):
    u = sc["has_aud_icd"].to_numpy()
    strata = [np.flatnonzero(u == v) for v in np.unique(u)]
    for _ in range(NB):
        yield np.concatenate([rng.choice(v, len(v), True) for v in strata])


def estimates(sc):
    """Each reader's value, with an interval, and the paired difference against ours."""
    rng = np.random.default_rng(SEED)
    reps = [sc.iloc[i] for i in _boot_indices(sc, rng)]
    rows = []
    for who in (MODEL, FELLOW, CODE):
        for met in ("sensitivity", "specificity"):
            for weighted, basis in ((False, "in the reviewed sample"),
                                    (True, "weighted to the registry")):
                point = est(sc, who, met, weighted)
                draws = np.array([est(t, who, met, weighted) for t in reps])
                row = {"reader": who, "metric": met, "weighted": weighted,
                       "basis": basis, "value": point,
                       "lo": float(np.nanpercentile(draws, 2.5)),
                       "hi": float(np.nanpercentile(draws, 97.5))}
                if who != MODEL:
                    dd = np.array([est(t, MODEL, met, weighted)
                                   - est(t, who, met, weighted) for t in reps])
                    row["diff_vs_model"] = est(sc, MODEL, met, weighted) - point
                    row["diff_lo"] = float(np.nanpercentile(dd, 2.5))
                    row["diff_hi"] = float(np.nanpercentile(dd, 97.5))
                rows.append(row)
    return pd.DataFrame(rows)


def test_label(t):
    """The paired test as the panel prints it: chi-square where the rule uses it, else the
    exact P."""
    if t["method"] == "chi2_cc":
        return "$\\chi^2$ = %.1f" % t["statistic"]
    return "exact $P$ = %.2f" % t["p_value"]


def discordance(sc):
    """The full two-by-two of who was right, for each comparator."""
    out = {}
    for other in (CODE, FELLOW):
        both = sc[sc[MODEL].isin(["AUD+", "AUD-"]) & sc[other].isin(["AUD+", "AUD-"])]
        mok = both[MODEL] == both["gold_three"]
        ook = both[other] == both["gold_three"]
        mat = np.array([[int((mok & ook).sum()), int((mok & ~ook).sum())],
                        [int((~mok & ook).sum()), int((~mok & ~ook).sum())]])
        disagree = int(mat[0, 1] + mat[1, 0])
        test = SCORE.mcnemar_counts(int(mat[0, 1]), int(mat[1, 0]))
        out[other] = {"mat": mat, "n": len(both), "disagree": disagree, "test": test}
    return out


# ------------------------------------------------------------------ the panels
def draw_panel_d(ax, est_df):
    """What the move from the reviewed sample to the registry does to each reader.

    Two readers, not three. The hepatologist was drawn here until 13 August and is now
    left out: the reference standard equals his own label on 367 of the 425 determinate
    charts, so his value there is 1.000 by definition and the panel would be reporting the
    escalation rather than his reading. The comparison with a second human belongs to the
    raw first-pass kappa, which no construction touches.

    The panel draws the movement rather than two free-standing markers. The claim is about
    the distance, not the positions: the code loses a third of its sensitivity when the
    stratum mix becomes the registry, and the LLM does not move.

    A caution the caption carries. The sampling stratum is whether an AUD code ever
    appears, and the code comparator is that same variable, so within the coded stratum the
    code calls every patient positive and within the uncoded stratum every patient
    negative. It has no within-stratum discrimination at all, and its sensitivity and
    specificity are a restatement of the stratum mix. Its weighted specificity is high for
    that reason and not because it discriminates: it calls 29 per cent of the registry
    positive where the LLM calls 42 per cent. Youden J is printed so that neither rate is
    read on its own.
    """
    y, yticks, ylabels, group_marks = 0.0, [], [], []
    jvals = {}
    for met in ("specificity", "sensitivity"):          # bottom-up
        for who in (CODE, MODEL):
            sub = est_df[(est_df["reader"] == who) & (est_df["metric"] == met)]
            s = sub[~sub["weighted"].astype(bool)].iloc[0]
            w = sub[sub["weighted"].astype(bool)].iloc[0]
            jvals.setdefault(who, {})[met] = (s["value"], w["value"])
            col = COLOUR[who]
            if abs(w["value"] - s["value"]) > 0.004:
                ax.annotate("", xy=(w["value"], y), xytext=(s["value"], y),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9,
                                            shrinkA=1.5, shrinkB=1.0, mutation_scale=6))
            ax.plot(s["value"], y, MARK[who], ms=3.0, mfc="white", mec=col, mew=0.8,
                    zorder=3)
            ax.errorbar(w["value"], y,
                        xerr=[[w["value"] - w["lo"]], [w["hi"] - w["value"]]],
                        fmt=MARK[who], ms=3.8, lw=0.8, capsize=1.6, color=col,
                        mfc=col, mec=col, mew=0.8, zorder=4)
            # The LLM does not move, so its two markers overlap and no arrow is drawn.
            # That is the finding, but a reader cannot measure an arrow that is not there,
            # so every row carries its movement as text.
            d_ = w["value"] - s["value"]
            lab_ = ("%.3f to %.3f  (unchanged)" % (s["value"], w["value"])
                    if abs(d_) < 0.0005 else
                    "%.3f to %.3f  (%+.3f)" % (s["value"], w["value"], d_))
            ax.text(1.02, y, lab_,
                    transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=4.6, color=col, clip_on=False)
            yticks.append(y)
            # The code has no call of its own: in the sensitivity block what is scored is
            # the presence of a code among reference-positive patients, and in the
            # specificity block the absence of one among reference-negative patients. The
            # row says which, so the reader does not have to infer it.
            if who == CODE:
                ylabels.append("AUD code present" if met == "sensitivity"
                               else "No AUD code")
            else:
                ylabels.append(NAME[who])
            y += 1.0
        group_marks.append((y - 2.0, met))
        y += 0.7

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=5)
    ax.set_ylim(-0.75, y - 0.9)
    ax.set_xlim(0.55, 1.015)
    ax.set_xlabel("Value against the reference standard")
    ax.tick_params(axis="y", length=0)
    for ypos, met in group_marks:
        ax.text(0.556, ypos + 1.30, met.capitalize(), fontsize=5.6, fontweight="bold",
                color=PA.INK, va="center")
    ax.text(0.02, 0.02,
            "open, in the reviewed sample\narrow, moving to the registry",
            transform=ax.transAxes, fontsize=4.5, color=PA.MUTED, va="bottom", ha="left")
    js = []
    for who in (MODEL, CODE):
        se = jvals[who]["sensitivity"][1]
        sp = jvals[who]["specificity"][1]
        js.append("%s  J %.2f" % (NAME[who].split(" ")[0], se + sp - 1))
    ax.text(0.98, 0.98, "weighted to the registry\n" + "\n".join(js),
            transform=ax.transAxes, fontsize=4.5, color=PA.MUTED, va="top", ha="right")
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    return ax


def draw_panel_e(axes, disc):
    """Each paired comparison as a two-by-two, in panel b's visual language."""
    short = ["Correct", "Wrong"]
    for ax, other in zip(axes, (CODE, FELLOW)):
        d = disc[other]
        mat = d["mat"]
        rows = mat.sum(axis=1, keepdims=True)
        pct = np.where(rows > 0, mat / rows * 100, 0)
        ax.imshow(mat, cmap="Blues", aspect="equal", origin="upper")
        for i in range(2):
            for j in range(2):
                v = mat[i, j]
                c = "white" if v > mat.max() * 0.6 else "black"
                # A percentage is printed only where its denominator can carry one. The
                # second row holds the handful of patients the instrument got wrong, and
                # "86%" of seven says less than "6" does.
                lab = ("%d\n(%.0f%%)" % (v, pct[i, j])) if rows[i, 0] >= 20 else str(v)
                ax.text(j, i, lab, ha="center", va="center", fontsize=5.4, color=c)
        ax.set_xticks(range(2))
        ax.set_xticklabels(short, fontsize=5)
        ax.set_yticks(range(2))
        ax.set_yticklabels(short, fontsize=5)
        ax.set_xlabel(NAME[other], fontsize=5.6, color=COLOUR[other])
        ax.set_ylabel("LLM", fontsize=5.6, color=PA.C_POS)
        # Two short lines. The earlier single long line ran past its own axes and met
        # the neighbouring panel's title in the gutter; the verdict moves to the caption.
        ax.set_title("%d of %d disagree\n%s" % (d["disagree"], d["n"], test_label(d["test"])),
                     fontsize=5.0, color=PA.MUTED, loc="center", pad=3.0)
    return axes


# ------------------------------------------------------- panel f, the inputs
# NIAAA, 2024 National Survey on Drug Use and Health: past-year AUD in US adults 18+.
# https://www.niaaa.nih.gov/alcohols-effects-health/alcohol-topics/alcohol-facts-and-statistics/
#   alcohol-use-disorder-aud-united-states-age-groups-and-demographic-characteristics
NSDUH_2024_ADULTS = 0.103


def predictive_inputs(sc):
    """The two rates the Bayes conversion needs, and the counts behind them.

    A binary test has two free rates, the true-positive rate and the false-positive rate,
    with the other two fixed as their complements. A test with three outcomes has four,
    because an indeterminate output is neither a positive call nor a negative one:

        tpr = P(says AUD+ | AUD+)      fnr = P(says AUD- | AUD+)      tpr + fnr <= 1
        fpr = P(says AUD+ | AUD-)      tnr = P(says AUD- | AUD-)      fpr + tnr <= 1

    Predictive value needs the pair that matches the call being made. PPV conditions on a
    positive call and needs tpr and fpr. NPV conditions on a negative call and needs tnr and
    fnr. Substituting 1 - fpr for tnr, which is correct for a binary test, would count the
    indeterminate outputs on reference-negative patients as though they had been negative
    calls, and would report an NPV the reader never earned.

    Panel d reports tpr as its sensitivity and tnr as its specificity, both over the full
    reference group, so the two panels rest on one set of counts.
    """
    rows = []
    P = sc["gold_three"] == "AUD+"
    N = sc["gold_three"] == "AUD-"
    npos, nneg = int(P.sum()), int(N.sum())
    for who in (MODEL, FELLOW, CODE):
        tp = int((P & (sc[who] == "AUD+")).sum())
        fn = int((P & (sc[who] == "AUD-")).sum())
        fp = int((N & (sc[who] == "AUD+")).sum())
        tn = int((N & (sc[who] == "AUD-")).sum())
        rows.append({"reader": who, "name": NAME[who], "n_pos": npos, "n_neg": nneg,
                     "tp": tp, "fn": fn, "indeterminate_on_pos": npos - tp - fn,
                     "tn": tn, "fp": fp, "indeterminate_on_neg": nneg - tn - fp,
                     "tpr": tp / npos, "fnr": fn / npos,
                     "fpr": fp / nneg, "tnr": tn / nneg})
    return pd.DataFrame(rows)


def registry_prevalence(sc, seed=SEED):
    """Horvitz-Thompson prevalence of a reference AUD+ in the registry the sample came from.

    The reviewed charts are a stratified probability sample of the registry, so the sampling
    weight already in the frame carries the sample prevalence, which is enriched by design,
    back to the population the index test would be applied to.
    """
    w = sc["w"].to_numpy()
    pos = (sc["gold_three"] == "AUD+").to_numpy()
    point = float((w * pos).sum() / w.sum())
    rng = np.random.default_rng(seed)
    draws = np.array([float((w[i] * pos[i]).sum() / w[i].sum())
                      for i in _boot_indices(sc, rng)])
    return point, float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def draw_panel_f(ax, sc, rates=None, prev=None):
    """Predictive value against prevalence, with the two populations that bound the range.

    Sensitivity and specificity do not move with prevalence and predictive value does, so a
    predictive value measured on this sample would describe no real population: the sample is
    enriched for coded patients by design and its own prevalence is 0.585. The panel therefore
    reports predictive value as a function of prevalence and marks where two real populations
    sit, rather than asserting one number.
    """
    if rates is None:
        rates = predictive_inputs(sc)
    if prev is None:
        prev = registry_prevalence(sc)
    q, lo, hi = prev

    ax.axvspan(lo, hi, color=PA.TINT[PA.C_POS], alpha=0.55, lw=0, zorder=0)
    for x in (NSDUH_2024_ADULTS, q):
        ax.axvline(x, color=PA.MUTED, lw=0.6, ls=(0, (2.2, 2.2)), zorder=1)

    p = np.linspace(0.01, 0.50, 300)
    # Two readers, not three. The hepatologist was drawn here until 13 August, on the
    # reasoning that his higher PPV was an unfavourable comparison and should not be
    # hidden. That was wrong: the comparison is not valid, so drawing it reported an
    # artefact rather than a result. His four rates come from a standard that equals his
    # own label on 367 of the 425 determinate charts, where he records 221 of 221 true
    # positives and 0 of 146 false positives by construction. His overall false-positive
    # rate of 1 in 174 is the reason his PPV curve sat above ours, and it is not a
    # property of his reading. The valid comparison with a second human reader is panel e,
    # where all 16 disagreements fall on adjudicated charts and a third reader set the
    # label, and the raw first-pass kappa reported in the text.
    for who in (CODE, MODEL):
        r = rates[rates["reader"] == who].iloc[0]
        tpr, fpr = float(r["tpr"]), float(r["fpr"])
        tnr, fnr = float(r["tnr"]), float(r["fnr"])
        ppv = tpr * p / (tpr * p + fpr * (1 - p))
        npv = tnr * (1 - p) / (tnr * (1 - p) + fnr * p)
        col = COLOUR[who]
        ax.plot(p, ppv, "-", color=col, lw=1.0, zorder=3, label=NAME[who])
        ax.plot(p, npv, "--", color=col, lw=1.0, zorder=3)

    ax.set_xlim(0.0, 0.52)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Prevalence in the population read")
    ax.set_ylabel("Predictive value")
    # The labels sit at the top. The lower half of the panel holds the two PPV curves at low
    # prevalence, which is the part a reader is most likely to want to read a value off.
    bb = dict(fc="white", ec="none", pad=0.6, alpha=0.85)
    ax.text(NSDUH_2024_ADULTS - 0.010, 0.985, "US adults, %.3f" % NSDUH_2024_ADULTS,
            fontsize=4.2, color=PA.MUTED, ha="right", va="top", rotation=90, bbox=bb,
            zorder=5)
    ax.text(q - 0.010, 0.985, "this registry, %.2f" % q,
            fontsize=4.2, color=PA.MUTED, ha="right", va="top", rotation=90, bbox=bb,
            zorder=5)
    leg = ax.legend(fontsize=4.3, loc="lower right", frameon=False, borderaxespad=0.15,
                    handlelength=1.5, labelspacing=0.22,
                    title="solid, PPV\ndashed, NPV")
    plt.setp(leg.get_title(), fontsize=4.3, color=PA.MUTED, multialignment="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return ax


def main():
    out = "downstream/npj/out/f01_accuracy"
    os.makedirs(out, exist_ok=True)
    sc = load()
    est_df = estimates(sc)
    disc = discordance(sc)
    rates = predictive_inputs(sc)
    prev = registry_prevalence(sc)
    est_df.to_csv(os.path.join(out, "reader_estimates.csv"), index=False)
    rates.to_csv(os.path.join(out, "predictive_value_inputs.csv"), index=False)
    pd.DataFrame([{"quantity": "registry prevalence of a reference AUD+",
                   "estimator": "Horvitz-Thompson from the sampling weights",
                   "value": prev[0], "lo": prev[1], "hi": prev[2],
                   "sample_prevalence": float((sc["gold_three"] == "AUD+").mean())}]
                 ).to_csv(os.path.join(out, "registry_prevalence.csv"), index=False)

    print("=" * 96)
    print("WHAT EACH READER ACHIEVED, AND THE PAIRED DIFFERENCE AGAINST OURS")
    print("=" * 96)
    print("  %-18s %-12s %-24s %8s %-18s %s"
          % ("reader", "metric", "basis", "value", "95% CI", "diff vs instrument"))
    for _, r in est_df.iterrows():
        ci = "%.3f to %.3f" % (r["lo"], r["hi"])
        dd = ("" if r["reader"] == MODEL else
              "%+.3f (%+.3f to %+.3f)%s" % (r["diff_vs_model"], r["diff_lo"], r["diff_hi"],
                                            "  <- ours lower"
                                            if r["diff_vs_model"] < 0 else ""))
        print("  %-18s %-12s %-24s %8.3f %-18s %s"
              % (NAME[r["reader"]], r["metric"], r["basis"], r["value"], ci, dd))
    print()
    print("=" * 96)
    print("WHAT PANEL f CONVERTS: THE TRUE-POSITIVE AND FALSE-POSITIVE RATES")
    print("=" * 96)
    print("  %-18s %-24s %-24s %7s %7s %7s %7s"
          % ("reader", "on reference AUD+ (%d)" % int(rates["n_pos"].iloc[0]),
             "on reference AUD- (%d)" % int(rates["n_neg"].iloc[0]),
             "tpr", "fnr", "fpr", "tnr"))
    for _, r in rates.iterrows():
        print("  %-18s %-24s %-24s %7.4f %7.4f %7.4f %7.4f"
              % (r["name"],
                 "%d+ %d- %d indet" % (r["tp"], r["fn"], r["indeterminate_on_pos"]),
                 "%d+ %d- %d indet" % (r["fp"], r["tn"], r["indeterminate_on_neg"]),
                 r["tpr"], r["fnr"], r["fpr"], r["tnr"]))
    print("  PPV uses tpr and fpr, NPV uses tnr and fnr. tnr is panel d specificity, and "
          "it is not 1 - fpr")
    print("  prevalence in the reviewed sample %.4f, weighted to the registry %.4f "
          "(%.4f to %.4f)"
          % (float((sc["gold_three"] == "AUD+").mean()), prev[0], prev[1], prev[2]))
    print()
    for other in (CODE, FELLOW):
        d = disc[other]
        print("  instrument vs %-18s both right %3d, ours only %3d, theirs only %3d, "
              "both wrong %3d   %s"
              % (NAME[other], d["mat"][0, 0], d["mat"][0, 1], d["mat"][1, 0],
                 d["mat"][1, 1], test_label(d["test"])))

    apply_nature_style()
    fig = plt.figure(figsize=(7.205, 2.55))
    fig.set_layout_engine("none")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 0.78, 0.78], wspace=0.52,
                          left=0.135, right=0.99, top=0.86, bottom=0.19)
    axd = fig.add_subplot(gs[0])
    draw_panel_d(axd, est_df)
    axe = [fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
    draw_panel_e(axe, disc)
    axd.text(-0.34, 1.12, "d", transform=axd.transAxes, fontsize=8, fontweight="bold",
             va="top", ha="left")
    axe[0].text(-0.42, 1.30, "e", transform=axe[0].transAxes, fontsize=8,
                fontweight="bold", va="top", ha="left")
    save_nature_figure(fig, os.path.join(out, "panels_de"))
    plt.close(fig)
    print()
    print("wrote %s/panels_de.pdf and .png" % out)


if __name__ == "__main__":
    main()
