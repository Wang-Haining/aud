#!/usr/bin/env python3
"""Supplementary Figure 3: the indeterminate rate in the undocumented stratum.

Why the figure exists. The equity analysis reports only stratifying variables the
structured record or the patient's ZIP supplies. Four candidate variables were read out
of the notes by a model, and a stratum defined by "this field is not documented" cannot
be crossed with an outcome defined by "no classification was returned", because both
measure whether the chart carried information. That exclusion removes four variables
from the analysis, so the evidence for it is given rather than asserted: seven
note-derived fields, four models, 28 comparisons.

Why it is a forest plot. The version carried until 25 August drew the same 28 estimates
as a scatter against documentation availability. Availability is not a quantity the
paragraph makes a claim about, and the transportation field sits at the lowest
difference in the middle of that axis, so the layout suggested a dose-response the data
do not show. Availability is a column here instead. The panel also had no legend, four
series that only marker shape separated, and intervals clipped by the top spine.

Estimand. For field g the stratum is the charts where the extraction returned no value,
the complement is every other chart, and the quantity is the difference in the rate of
indeterminate output between them, in percentage points. Intervals are the parity run's
own, from downstream/npj_goal2_outputs/parity_strata.csv.

Writes out/figure4/supp_documentation.{pdf,png,csv}.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, "downstream")
sys.path.insert(0, "downstream/npj")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

import npj_style as S                                                    # noqa: E402

SRC = "downstream/npj_goal2_outputs/parity_strata.csv"
OUT = "downstream/npj/out/figure4"
N_COHORT = 487
PRIMARY = "Qwen3.6-27B"

# The field, the level the extraction writes when it found nothing, and the name the
# supplement uses for it.
FIELDS = [("housing", "not_documented", "Housing"),
          ("labor_force_status", "not_documented", "Labour force status"),
          ("transportation_barrier", "not_documented", "Transportation barrier"),
          ("insurance", "not_documented", "Insurance"),
          ("food_security", "not_documented", "Food security"),
          ("disability_status", "unknown_not_separately_assessed", "Disability"),
          ("incarceration_history", "not_documented", "Incarceration history")]

# The four models the parity run covers, the primary first. Marker shape separates them
# and the primary carries the index colour, as it does in every other panel. No entry is
# bold: the caption names the primary model.
MODELS = [(PRIMARY, "Qwen3.6-27B", "o"),
          ("Nemotron-3-120B-FP8", "Nemotron-3-120B", "s"),
          ("GPT-OSS-120B", "GPT-OSS-120B", "D"),
          ("GPT-OSS-20B", "GPT-OSS-20B", "^")]

# Millimetres, on npj's 183 mm double column, which is what every other figure in the
# paper is built at. The manuscript includes them at \textwidth and they are all scaled
# by the same factor, so type set here at S.FS_BODY reaches the page at the size the
# other panels' does. Built at 166 this was the one figure at scale 1.0 and its type was
# a point larger than its siblings'.
W_MM, H_MM = 183.0, 88.0
X_LAB = 4.5
AX_X0, AX_W = 44.0, 101.0
NUM1_X, NUM2_X = 156.0, 180.0
Y_AX0, AX_H = 14.0, 62.0
Y_HEAD = Y_AX0 + AX_H + 1.5
XLIM = (-11.0, 30.0)
XTICKS = [-10, 0, 10, 20, 30]
XLAB = "Difference in the indeterminate rate (percentage points)"


def build(src):
    p = pd.read_csv(src)
    p = p[p["metric"].eq("abstention_rate")]
    rows = []
    for field, level, name in FIELDS:
        for tag, short, marker in MODELS:
            m = p[p["method"].eq(tag) & p["axis"].eq(field) & p["stratum"].eq(level)]
            if m.empty:
                raise RuntimeError("no row for %s / %s" % (field, tag))
            r = m.iloc[0]
            n_und = int(r["n_cohort"])
            rows.append({"field": name, "model": short, "marker": marker,
                         "n_undocumented": n_und, "n_documented": N_COHORT - n_und,
                         "documented_pct": 100.0 * (N_COHORT - n_und) / N_COHORT,
                         "rate_undocumented": 100.0 * float(r["stratum_metric"]),
                         "rate_documented": 100.0 * float(r["complement_metric"]),
                         "delta": 100.0 * float(r["delta_stratum_minus_complement"]),
                         "ci_low": 100.0 * float(r["delta_primary_ci95_low"]),
                         "ci_high": 100.0 * float(r["delta_primary_ci95_high"])})
    return pd.DataFrame(rows)


def draw(d, out):
    S.apply_npj_style()
    matplotlib.rcParams.update({"font.sans-serif": S.SANS})
    fig = plt.figure(figsize=S.mm(W_MM, H_MM))
    ax = fig.add_axes([AX_X0 / W_MM, Y_AX0 / H_MM, AX_W / W_MM, AX_H / H_MM])

    names = [n for _, _, n in FIELDS]
    pos = {n: len(names) - 1 - k for k, n in enumerate(names)}
    step = 0.22

    ax.axvline(0.0, color=S.INK, lw=S.LW_MIN, zorder=1)
    for k, (tag, short, marker) in enumerate(MODELS):
        first = short == "Qwen3.6-27B"
        col = S.C_INDEX if first else S.C_FORE
        ms = 3.2 if first else 2.6
        off = (1.5 - k) * step
        g = d[d["model"].eq(short)]
        for _, r in g.iterrows():
            yy = pos[r["field"]] + off
            ax.plot([r["ci_low"], r["ci_high"]], [yy, yy], color=col,
                    lw=S.LW_MIN, solid_capstyle="butt", zorder=2)
            ax.plot([r["delta"]], [yy], marker=marker, ms=ms, mfc=col, mec="none",
                    linestyle="none", zorder=3)

    ax.set_yticks([])
    ax.set_ylim(-0.62, len(names) - 1 + 0.62)
    ax.set_xlim(*XLIM)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([("\u2212%d" % -t) if t < 0 else str(t) for t in XTICKS])
    ax.set_xlabel(XLAB, fontsize=S.FS_BODY, color=S.INK, labelpad=4)
    ax.tick_params(axis="x", pad=2, labelsize=S.FS_BODY)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    def y_fig(v):
        span = (len(names) - 1) + 1.24
        return (Y_AX0 + AX_H * (v + 0.62) / span) / H_MM

    for name in names:
        fig.text(X_LAB / W_MM, y_fig(pos[name]), name, fontsize=S.FS_SMALL,
                 color=S.INK, ha="left", va="center")
        r = d[d["field"].eq(name)].iloc[0]
        fig.text(NUM1_X / W_MM, y_fig(pos[name]), "%.1f%%" % r["documented_pct"],
                 fontsize=S.FS_SMALL, color=S.MUTED, ha="right", va="center")
        fig.text(NUM2_X / W_MM, y_fig(pos[name]),
                 "%d / %d" % (r["n_undocumented"], r["n_documented"]),
                 fontsize=S.FS_SMALL, color=S.MUTED, ha="right", va="center")

    fig.text(NUM1_X / W_MM, Y_HEAD / H_MM, "Documented", fontsize=S.FS_SMALL,
             color=S.MUTED, ha="right", va="bottom")
    fig.text(NUM2_X / W_MM, Y_HEAD / H_MM, "Charts", fontsize=S.FS_SMALL,
             color=S.MUTED, ha="right", va="bottom")

    # The legend, drawn rather than placed, so the primary model can be bold as it is in
    # every other panel that names models.
    x = X_LAB
    y = (H_MM - 3.6) / H_MM
    for tag, short, marker in MODELS:
        first = short == "Qwen3.6-27B"
        col = S.C_INDEX if first else S.C_FORE
        fig.add_artist(plt.Line2D([x / W_MM], [y], marker=marker,
                                  ms=3.2 if first else 2.6, mfc=col, mec="none",
                                  linestyle="none", transform=fig.transFigure))
        fig.text((x + 2.2) / W_MM, y, short, fontsize=S.FS_SMALL, color=S.INK,
                 ha="left", va="center")
        x += 7.0 + 1.30 * len(short)

    S.check_text_fits(fig, W_MM, H_MM)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "supp_documentation.%s" % ext),
                    facecolor="white", dpi=600)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = build(a.src)
    d.to_csv(os.path.join(a.out, "supp_documentation.csv"), index=False)
    draw(d, a.out)
    n_pos = int((d["delta"] > 0).sum())
    n_excl = int(((d["ci_low"] > 0) | (d["ci_high"] < 0)).sum())
    print("  %d comparisons, %d positive, %d intervals excluding zero"
          % (len(d), n_pos, n_excl))
    print("  smallest %.1f points (%s, %s), largest %.1f points (%s, %s)"
          % (d["delta"].min(), d.loc[d["delta"].idxmin(), "field"],
             d.loc[d["delta"].idxmin(), "model"],
             d["delta"].max(), d.loc[d["delta"].idxmax(), "field"],
             d.loc[d["delta"].idxmax(), "model"]))
    p = d[d["model"].eq("Qwen3.6-27B")]
    print("  primary model: %.1f to %.1f points, %d of 7 intervals excluding zero"
          % (p["delta"].min(), p["delta"].max(),
             int(((p["ci_low"] > 0) | (p["ci_high"] < 0)).sum())))
    print("  wrote supp_documentation.pdf, .png, .csv in %s" % a.out)


if __name__ == "__main__":
    main()
