#!/usr/bin/env python3
"""Assemble the equity figure: is the burden even across patients?

  a  conditional sensitivity of four candidate models, stratum against its complement,
     ordered by the width of the interval the design delivers
  b  how often no classification is returned, the model beside the chart reviewers,
     on axes the structured record supplies
  c  sensitivity of the AUD diagnosis code by subgroup at registry scale

Every panel is cut by one variable, the patient subgroup, and the figure reads along it:
does the index test work equally, does it answer equally often, and who does the
comparator miss.

Panel a is ordered by interval width rather than by effect. On this cohort most
comparisons are uninformative, and an ordering by effect lets an absence of marked
dots read as evidence of equality.

Panel b reports the indeterminate rate as coverage and puts the chart reviewers on the
same axis. Three results place the cause in the record rather than in the model: the
reviewers returned an indeterminate label for 62 of 487 charts against the model's 55,
53 of the model's 55 are charts the reviewers also could not resolve, and the
rural-urban gradient appears in both readers at the same size.

Panel b uses only axes the structured record supplies. Housing, tobacco, insurance and
labour-force status were read out of the notes by the same model, so a split on "this
field is not documented" and an outcome of "no classification was returned" measure
chart sparsity on both sides. On tobacco that pairing gives 59.5 per cent against 0.6
per cent, a ratio of about a hundred.

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
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402

import f06_subgroup_frame as SF                                    # noqa: E402

OUT = "downstream/npj/out/figure6"
PARITY = "downstream/npj_goal2_outputs/parity_table.csv"
FAIR = "output/fairness"

INK, MUTED, RULE = "#1f1e1d", "#55534e", "#B9B7B2"
BLUE, VERM, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#B4B4B4"
PRIMARY = "Qwen3.6-27B"
# Both indeterminate grades, which is the abstain set of downstream/data.yaml and the
# set run_parity.py and build_canonical_panel.py use. It has to match the reference side:
# gold_binary == "abstain" collapses Uncertain and Not Enough Data together, so counting
# only Not Enough Data on the model side compared 62 reviewer abstentions against 55 model
# abstentions when the like-for-like comparison is 62 against 64. Corrected 14 August.
IND = ("Uncertain", "Not Enough Data")

# Panel b axes. Each comes from the structured record or from the ZIP, so the split is
# formed without reading the note whose content the outcome depends on.
COVER_AXES = [
    ("Site", "site", ["Academic", "Safety net", "Other"]),
    ("Gender", "gender", ["Man", "Woman"]),
    ("Race", "race_2", ["White", "Non-White"]),
    ("Age", "age_bracket_50", ["Under 50", "50 to 64", "65 and over"]),
    ("Decompensation", "decompensation", ["Compensated", "Decompensated"]),
    ("Index period", "covid_era", ["Index before Mar 2020", "Index Mar 2020 onward"]),
    ("Rural or urban", "ruca_category", ["urban", "suburban", "rural"]),
]
# Panel a axes: the stratifying variables the record carries for at least 30 per cent of
# the cohort, which is the prespecified reporting rule. care_site_id is excluded because
# it has one level per unit rather than per hospital system, and adi_quintile because the
# ZIP5 the cohort supplies does not support the linkage.
PARITY_AXES = ["race", "gender", "ethnicity", "age_bin", "cci_bin", "ruca_category",
               "housing", "labor_force_status"]
# Two of the eight were read out of the notes by the same model whose output is being
# scored. They are kept for conditional sensitivity, where the outcome is whether the
# label is right, and dropped from coverage, where the outcome and the split both
# measure whether the chart carried information.
NOTE_READ = {"housing", "labor_force_status"}
CLEAN = {"under 40": "Under 40", "40 to 64": "40 to 64", "65 and over": "65 and over",
         "urban": "Urban", "suburban": "Suburban", "rural": "Rural",
         "Index before Mar 2020": "Before Mar 2020",
         "Index Mar 2020 onward": "Mar 2020 onward"}
AXIS_C = {"sex": "Sex", "race": "Race", "agegrp": "Age",
          "charlson_noliver": "Charlson, liver excluded"}


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    z, p = 1.959963985, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h))


def coverage(a):
    """Indeterminate rate per level, for the model and for the chart reviewers."""
    class _A:
        panel = a.panel
    s, _, _ = SF.build_axes(_A())
    s["m_ind"] = s["label_canonical"].isin(IND)
    s["r_ind"] = s["gold_binary"].eq("abstain")
    rows = []
    for label, col, order in COVER_AXES:
        for lev in order:
            g = s[s[col].astype("string").eq(lev)]
            if not len(g):
                continue
            m = wilson(int(g["m_ind"].sum()), len(g))
            r = wilson(int(g["r_ind"].sum()), len(g))
            rows.append({"axis": label, "level": CLEAN.get(lev, lev), "n": len(g),
                         "model": m[0], "m_lo": m[1], "m_hi": m[2],
                         "review": r[0], "r_lo": r[1], "r_hi": r[2]})
    both = int((s["m_ind"] & s["r_ind"]).sum())
    head = {"n": len(s), "model": int(s["m_ind"].sum()),
            "review": int(s["r_ind"].sum()), "both": both,
            "model_only": int((s["m_ind"] & ~s["r_ind"]).sum())}
    return pd.DataFrame(rows), head


# The offset that separates the methods inside a stratum row. It was
# (hash(method) % 4 - 1.5) * 0.16 until 14 August. Python randomises the hash of a string
# per process, so the offsets changed between runs, and %% 4 gave four slots to five
# methods, so two or three of them were drawn on top of each other and which ones collided
# changed as well. The order below is the leaderboard order of Figure 4, with the
# diagnosis code last.
# The four candidate models, in the leaderboard order of Figure 4. The parity analysis is
# defined on these four: 26 strata x 4 models x 2 metrics is the 208 stratum-level
# comparisons the Results report. The diagnosis code is in the parity table as well, and
# was drawn in this panel until 14 August, which put a fifth series in a panel the caption
# and the count describe as four. Its own quantity is panel c.
METHOD_SLOTS = ["Qwen3.6-27B", "GPT-OSS-120B", "Nemotron-3-120B-FP8", "GPT-OSS-20B"]


def jitter(method, step=0.16):
    i = METHOD_SLOTS.index(method) if method in METHOD_SLOTS else len(METHOD_SLOTS)
    return (i - (len(METHOD_SLOTS) - 1) / 2) * step


def draw_parity(ax, p):
    """Every stratum, the four candidate models, ordered by the stratum's interval width."""
    d = p[p["metric"].eq("conditional_sensitivity")
          & p["axis"].isin(PARITY_AXES)
          & p["method"].isin(METHOD_SLOTS)].copy()
    d["half"] = (d["delta_primary_ci95_high"] - d["delta_primary_ci95_low"]) / 2
    order = (d.groupby(["axis", "stratum"])["half"].median()
             .sort_values().reset_index())
    key = {(r.axis, r.stratum): i for i, r in enumerate(order.itertuples())}
    d["row"] = list(zip(d["axis"], d["stratum"]))
    d = d[d["row"].isin(key)]
    y = d["row"].map(key)
    n = len(order)

    ax.axvline(0, color=RULE, lw=0.7, zorder=0)
    ax.axvspan(-5, 5, color=GREEN, alpha=0.07, lw=0, zorder=0)
    for (_, r), yy in zip(d.iterrows(), y):
        c = {"equivalent": GREEN, "disparity": VERM}.get(
            str(r.get("conclusion", "")), GREY)
        j = jitter(str(r["method"]))
        ax.plot([100 * r["delta_primary_ci95_low"], 100 * r["delta_primary_ci95_high"]],
                [yy + j, yy + j], color=c, lw=0.6, alpha=0.8, zorder=2)
        ax.plot(100 * r["delta_stratum_minus_complement"], yy + j, "o", color=c,
                markersize=2.4, markeredgecolor="none", zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels(["%s: %s%s" % (r.axis.replace("_", " "), r.stratum,
                                      "  *" if r.axis in NOTE_READ else "")
                        for r in order.itertuples()], fontsize=4.4)
    ax.set_ylim(-0.8, n - 0.2)
    ax.invert_yaxis()
    ax.set_xlabel("Stratum minus complement, percentage points")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[
        Line2D([0], [0], color=GREEN, marker="o", lw=1, markersize=2.6,
               label="equivalence"),
        Line2D([0], [0], color=VERM, marker="o", lw=1, markersize=2.6,
               label="disparity"),
        Line2D([0], [0], color=GREY, marker="o", lw=1, markersize=2.6,
               label="no conclusion")],
        loc="lower left", bbox_to_anchor=(0.0, -0.10), fontsize=4.8, ncol=3,
        frameon=False, handlelength=1.0, handletextpad=0.35, columnspacing=0.9)
    ax.set_title("Conditional sensitivity, four models\n"
                 "rows ordered by interval width, narrowest first;\n"
                 "shading is the $\\pm$5 point equivalence band. * read from notes",
                 fontsize=6.3, color="#333333", pad=4)


def draw_coverage(ax, cov, head):
    rows, y = [], 0
    for label in [a[0] for a in COVER_AXES]:
        sub = cov[cov["axis"].eq(label)]
        if not len(sub):
            continue
        rows.append(("head", label, None, y))
        y += 1
        for _, r in sub.iterrows():
            rows.append(("row", r["level"], r, y))
            y += 1
        y += 0.4
    for kind, lab, r, yy in rows:
        if kind == "head":
            ax.text(-0.5, yy, lab, ha="right", va="center", fontsize=5.2,
                    color=INK, fontweight="bold")
            continue
        ax.plot([r["review"], r["model"]], [yy, yy], color=RULE, lw=3.4,
                alpha=0.55, solid_capstyle="round", zorder=1)
        ax.plot([r["m_lo"], r["m_hi"]], [yy, yy], color=BLUE, lw=0.8, alpha=0.5,
                zorder=2)
        ax.plot(r["review"], yy, "o", color="white", markeredgecolor=INK,
                markeredgewidth=0.9, markersize=4.0, zorder=3)
        ax.plot(r["model"], yy, "o", color=BLUE, markeredgecolor="white",
                markeredgewidth=0.7, markersize=4.4, zorder=4)
        ax.text(-0.5, yy, "%s   %d" % (lab, r["n"]), ha="right", va="center",
                fontsize=5.0, color=MUTED)
    ax.set_yticks([])
    ax.set_ylim(max(y for *_, y in rows) + 0.8, -1.0)
    ax.set_xlim(0, 34)
    ax.set_xlabel("No classification returned (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", lw=0, color=BLUE, markersize=4.0,
               label="LLM (Qwen3.6-27B)"),
        Line2D([0], [0], marker="o", lw=0, color="white", markeredgecolor=INK,
               markeredgewidth=0.9, markersize=4.0, label="chart review")],
        loc="upper left", bbox_to_anchor=(0.0, -0.115), fontsize=5.0, frameon=False,
        handlelength=0.8, handletextpad=0.35, ncol=2, columnspacing=1.2)
    ax.set_title("How often no classification is returned\n"
                 "%d of the model's %d are charts\nthe reviewers could not resolve either"
                 % (head["both"], head["model"]), fontsize=6.3, color="#333333", pad=4)


def draw_code(ax, f):
    rows, y = [], 0
    for axis in ("agegrp", "sex", "race", "charlson_noliver"):
        sub = f[f["axis"].eq(axis) & ~f["level"].str.contains(" minus ")]
        if not len(sub):
            continue
        sub = sub.sort_values("sensitivity_pct", ascending=False)
        rows.append(("head", AXIS_C.get(axis, axis), None, y))
        y += 1
        for _, r in sub.iterrows():
            rows.append(("row", r["level"], r, y))
            y += 1
        y += 0.4
    for kind, lab, r, yy in rows:
        if kind == "head":
            ax.text(-1.0, yy, lab, ha="right", va="center", fontsize=5.2,
                    color=INK, fontweight="bold")
            continue
        ax.plot([r["ci_low"], r["ci_high"]], [yy, yy], color=VERM, lw=0.9, zorder=2)
        for b in (r["ci_low"], r["ci_high"]):
            ax.plot([b, b], [yy - 0.16, yy + 0.16], color=VERM, lw=0.9, zorder=2)
        ax.plot(r["sensitivity_pct"], yy, "o", color=VERM, markeredgecolor="white",
                markeredgewidth=0.7, markersize=4.4, zorder=3)
        ax.text(-1.0, yy, "%s   %s" % (lab, "{:,d}".format(int(r["registry_n"]))),
                ha="right", va="center", fontsize=5.0, color=MUTED)
    ax.set_yticks([])
    ax.set_ylim(max(y for *_, y in rows) + 0.8, -1.0)
    ax.set_xlim(0, 62)
    ax.set_xlabel("Code sensitivity at the cirrhosis diagnosis (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Who the AUD diagnosis code misses,\nat registry scale, design-weighted "
                 "to the\nstratum sizes; counts are registry patients",
                 fontsize=6.3, color="#333333", pad=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default="output/canonical_panel.csv")
    ap.add_argument("--parity", default=PARITY)
    ap.add_argument("--fair", default=FAIR)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    p = pd.read_csv(a.parity)
    cov, head = coverage(a)
    f = pd.concat([pd.read_csv(os.path.join(a.fair, "icd_sensitivity_by_group.csv")),
                   pd.read_csv(os.path.join(a.fair, "cci_sensitivity.csv"))],
                  ignore_index=True)

    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 3, "ytick.major.size": 3, "legend.frameon": False,
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 300})

    fig = plt.figure(figsize=(7.09, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.98], height_ratios=[1.0, 1.02],
                          hspace=0.40, wspace=0.86,
                          left=0.185, right=0.955, top=0.885, bottom=0.075)
    axa = fig.add_subplot(gs[:, 0])
    axb, axc = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    draw_parity(axa, p)
    draw_coverage(axb, cov, head)
    draw_code(axc, f)
    for ax, lab, dx, dy in ((axa, "a", -0.42, 1.005), (axb, "b", -0.40, 1.02),
                            (axc, "c", -0.40, 1.02)):
        ax.text(dx, dy, lab, transform=ax.transAxes, fontsize=8, fontweight="bold",
                va="bottom", ha="left")
    # The coverage numbers are reported in the manuscript and tabulated in the
    # supplement, so they are written out rather than living only inside the panel.
    cov.to_csv(os.path.join(a.out, "coverage.csv"), index=False)
    print("  coverage.csv  %d strata; model %d of %d charts, reviewers %d, both %d"
          % (len(cov), head["model"], head["n"], head["review"], head["both"]))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "figure6.%s" % ext))
    plt.close(fig)
    print("wrote %s/figure6.pdf and .png" % a.out)

    q = p[p["axis"].isin(PARITY_AXES)]
    d = q[q["metric"].eq("conditional_sensitivity")]
    allm = q[q["metric"].isin(["conditional_sensitivity", "abstention_rate"])]
    print("\npanel a, on the %d reportable axes: %d strata, %d conditional-sensitivity "
          "evaluations, %d over both metrics"
          % (len(PARITY_AXES), d.groupby(["axis", "stratum"]).ngroups, len(d),
             len(allm)))
    print("  conclusions, conditional sensitivity: %s"
          % d["conclusion"].value_counts().to_dict())
    print("  conclusions, both metrics: %s" % allm["conclusion"].value_counts().to_dict())
    for m in ("conditional_sensitivity", "abstention_rate"):
        g = q[q["metric"].eq(m)]
        for c in ("equivalent", "disparity"):
            hit = g[g["conclusion"].eq(c)]
            if len(hit):
                print("    %-24s %-11s %s" % (m, c, hit["axis"].value_counts().to_dict()))
    print("\npanel b, coverage")
    print("  cohort %d; model returned no classification %d (%.1f%%); chart review "
          "indeterminate %d (%.1f%%)"
          % (head["n"], head["model"], 100 * head["model"] / head["n"],
             head["review"], 100 * head["review"] / head["n"]))
    print("  both %d, which is %.0f%% of the model's; model only %d (%.1f%% of the "
          "cohort)" % (head["both"], 100 * head["both"] / head["model"],
                       head["model_only"], 100 * head["model_only"] / head["n"]))
    print(cov.to_string(index=False, float_format=lambda x: "%.1f" % x))
    print("\npanel c, code sensitivity at registry scale")
    print(f.to_string(index=False, float_format=lambda x: "%.1f" % x))


if __name__ == "__main__":
    main()
