#!/usr/bin/env python3
"""Supplementary figure for the subgroup section: the specificity half.

Panel d of Figure 4 draws sensitivity. This draws specificity, on the same sixteen columns
and in the same grammar, and nothing else: repeating the sensitivity half here would print
the main panel twice.

Specificity across strata is not estimable at this cohort size for any reader, and this is
the figure that shows it. Every reader straddles zero with a median within about one point
of it, the counts outside the band run from 4 to 11 in no order, and the widest spreads
belong to the readers that make the most errors. Two strata hold 17 and 18 reference-negative
charts, where one chart moves the rate by 5.9 and 5.6 points, so a band of 5 points cannot be
met there by arithmetic.

Each point is one stratum's specificity minus the specificity of the rest of that stratum's
axis, in points. The box is the interquartile range, the line across it is the median, zero
is no difference and the band is 5 points. The row above the panel counts the strata outside
it.

Columns, order and grammar come from panel_reader_swarm, so the two figures cannot diverge.

Writes out/figure4/panel_supp_reader_swarm.pdf and .png.

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
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.patches import Rectangle                                 # noqa: E402

import npj_style as S                                                    # noqa: E402
import panel_reader_swarm as PD                                          # noqa: E402

OUT = "downstream/npj/out/figure4"
METRIC = "specificity"

W_MM, H_MM = PD.W_MM, PD.H_MM
X_AX0, AX_W = PD.X_AX0, PD.AX_W
Y_AX0, AX_H = PD.Y_AX0, PD.AX_H
Y_COUNT = PD.Y_COUNT
YLIM = (-35.0, 30.0)
YTICKS = [-30, -15, 0, 15, 30]
BAND = PD.BAND
# The swarm spacing is set in data units, so it is rescaled to the wider axis to keep the
# gap between points the same on the page as it is in panel d.
SEP_Y = PD.SEP_Y * (YLIM[1] - YLIM[0]) / (PD.YLIM[1] - PD.YLIM[0])


def columns():
    """The same sixteen columns panel d uses, on the specificity differences."""
    def load(src):
        d = pd.read_csv(src)
        d = d[d["metric"].eq(METRIC)]
        return {n: g["difference"].to_numpy(dtype=float) for n, g in d.groupby("reader")}

    dm, dp = load(PD.SRC_M), load(PD.SRC_P)
    cols = list(PD.COLUMNS)
    seen = {name for src, name, _ in cols if src == "m"}
    cols += [("m", n, n) for n in pd.read_csv(PD.LEADER)["name"] if n not in seen]
    return [(label, (dm if src == "m" else dp)[key], key == PD.CODE)
            for src, key, label in cols]


def draw(ax, cols):
    ax.axhspan(-BAND, BAND, color=S.RULE, alpha=0.28, lw=0, zorder=0)
    ax.axhline(0.0, color=S.INK, lw=S.LW_MIN, zorder=1)
    PD.SEP_Y = SEP_Y
    for x, (label, v, is_code) in enumerate(cols):
        col = S.C_CODE if is_code else S.C_INDEX
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.add_patch(Rectangle((x - 0.28, q1), 0.56, q3 - q1, facecolor="white",
                               edgecolor=col, lw=S.LW_MIN * 0.8, alpha=0.9, zorder=2))
        ax.plot([x - 0.32, x + 0.32], [med, med], color=col, lw=S.LW_MIN * 1.4,
                solid_capstyle="butt", zorder=4)
        ax.plot(x + PD.swarm(v), v, marker="o", ms=PD.MS, mfc=col, mec="white",
                mew=S.LW_MIN * 0.35, linestyle="none", zorder=3)
    for k in PD.RULES_AFTER:
        ax.axvline(k + 0.5, color=S.RULE, lw=S.LW_MIN, zorder=1)

    ax.set_xlim(-0.62, len(cols) - 0.38)
    ax.set_ylim(*YLIM)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols], rotation=PD.ROT, ha="right",
                       rotation_mode="anchor", fontsize=PD.FS_TICK)
    for t, (label, _, _) in zip(ax.get_xticklabels(), cols):
        t.set_color(S.INK)
        if label in PD.BOLD:
            t.set_fontweight("bold")
    ax.tick_params(axis="x", pad=1.5, length=2)
    ax.set_yticks(YTICKS)
    ax.set_yticklabels([str(t) for t in YTICKS])
    ax.set_ylabel("Specificity difference", fontsize=S.FS_BODY, color=S.INK, labelpad=3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cols = columns()
    print("  %d columns, %s" % (len(cols), METRIC))
    for label, v, _ in cols:
        print("    %-22s median %+5.2f  range %+6.2f to %+6.2f  outside the band %d"
              % (label, np.median(v), v.min(), v.max(), int(np.sum(np.abs(v) > BAND))))

    S.apply_npj_style()
    fig = plt.figure(figsize=S.mm(W_MM, H_MM))
    fig.set_layout_engine("none")
    ax = fig.add_axes([X_AX0 / W_MM, Y_AX0 / H_MM, AX_W / W_MM, AX_H / H_MM])
    fig.canvas.draw()
    draw(ax, cols)

    for x, (label, v, _) in enumerate(cols):
        ax.annotate("%d" % int(np.sum(np.abs(v) > BAND)), xy=(x, 1.0), xytext=(0, 2.4),
                    textcoords="offset points", xycoords=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=S.FS_SMALL, color=S.MUTED,
                    annotation_clip=False)
    for k, text in enumerate(reversed(PD.NUMHEAD)):
        fig.text((X_AX0 - 1.2) / W_MM, (Y_COUNT + k * 2.6) / H_MM, text,
                 fontsize=S.FS_SMALL, color=S.MUTED, ha="right", va="bottom")

    S.check_text_fits(fig, W_MM, H_MM)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "panel_supp_reader_swarm." + ext), dpi=600,
                    bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    print("wrote %s/panel_supp_reader_swarm.pdf and .png at %.1f by %.1f mm"
          % (a.out, W_MM, H_MM))


if __name__ == "__main__":
    main()
