#!/usr/bin/env python3
"""Panel d of Figure 4. Every stratum is a point, on one axis.

The earlier version of this panel drew each reader's 18 stratum differences as a segment
from the smallest to the largest with a marker at the median. A range is biased away from
zero by an amount that grows with the noise behind each estimate, so a reader that makes
more errors gets a wider segment for that reason alone. Eighteen values per reader is few
enough to draw all of them, which removes the summary and the bias with it.

One column per reader, in three groups on one axis, separated by rules. The diagnosis code
comes first, because the section opens with it. The primary model follows under the frozen
protocol and then under the four prespecified variations of Fig. 3c, which are the same
model asked in five ways rather than five models. The other ten models close the axis in
the order of Fig. 3a. The frozen protocol and the primary model were two columns in the
earlier version and are one here: they place all 487 charts identically, so drawing them
twice said nothing.

Each point is one stratum's sensitivity minus the sensitivity of the rest of that stratum's
axis, in points. The box is the interquartile range, the line across it is the median, zero
is no difference and the band is 5 points. The number above each column counts the strata
outside the band.

No kernel density. A violin at 18 points draws a shape the 18 points do not support, so the
distribution is the points themselves with the median and the quartiles over them.

Sensitivity only. The 174 reference-negative charts leave two strata holding 17 and 18
charts, where one chart moves specificity by 5.9 and 5.6 points, so the band is not
reachable there by arithmetic; those values are in the supplement.

Values come from out/figure4/model_disparity.csv and out/figure4/protocol_disparity.csv.
This file draws; it does not compute.

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

SRC_M = "downstream/npj/out/figure4/model_disparity.csv"
SRC_P = "downstream/npj/out/figure4/protocol_disparity.csv"
LEADER = "downstream/npj/out/figure4/panel_a_leaderboard.csv"
OUT = "downstream/npj/out/figure4"

W_MM, H_MM = 183.0, 49.5
X_AX0, AX_W = 21.0, 158.0
Y_AX0, AX_H = 14.0, 28.5
Y_COUNT = Y_AX0 + AX_H + 1.2
YLIM = (-20.0, 15.0)
YTICKS = [-20, -10, 0, 10]
BAND = 5.0
CODE = "AUD diagnosis code"
# The column header states what is counted, the threshold and the denominator, so the
# reader does not have to infer any of the three from the band. Two lines, right-aligned,
# the lower one on the baseline of the numbers themselves. The word disparity is not used:
# the band is a reading convention and not a test, and the section says so.
NUMHEAD = ("Of 18 strata,", "beyond 5 points")
PRIMARY = "Qwen3.6-27B"
MS = 1.8
FS_TICK = 6.0        # the rotated reader names, inside Nature's 5 to 7 pt
# A shallower rotation than 45 degrees costs horizontal room and buys vertical room. At
# 30 degrees the longest name occupies about 10 mm below the axis instead of 13, and
# adjacent labels stay 5.0 mm apart perpendicular to their own direction against a cap
# height of 2.1 mm, so nothing collides.
ROT = 30.0
SEP_Y = 1.25          # data units; a marker is about 0.8 of one at this height
SEP_X = 0.095         # column units, one column being 1.0
HALF = 0.34

# (source, key in that source, label). The primary model carries its own name and its
# decoding; the four variations carry only what varies, because the rule that follows the
# code column and the rule that closes the group say whose variations they are.
COLUMNS = [
    ("m", CODE, CODE),
    ("m", PRIMARY, "Qwen3.6-27B, T = 0"),
    ("p", "T = 0.3", "T = 0.3"),
    ("p", "T = 0.7", "T = 0.7"),
    ("p", "Paraphrase A", "Paraphrase A"),
    ("p", "Paraphrase B", "Paraphrase B"),
]
RULES_AFTER = (0, 5)          # column indices a rule follows
BOLD = ("Qwen3.6-27B, T = 0", )


def load(src):
    d = pd.read_csv(src)
    d = d[d["metric"].eq("sensitivity")]
    return {n: g["difference"].to_numpy(dtype=float) for n, g in d.groupby("reader")}


def columns(dm, dp):
    """The three groups, with the ten remaining models in Fig. 3a's order."""
    cols = list(COLUMNS)
    seen = {name for src, name, _ in cols if src == "m"}
    order = [n for n in pd.read_csv(LEADER)["name"] if n not in seen]
    cols += [("m", n, n) for n in order]
    out = []
    for src, key, label in cols:
        d = dm if src == "m" else dp
        if key not in d:
            raise SystemExit("%s is not in the %s file" % (key, src))
        out.append((label, d[key], key == CODE))
    return out


def swarm(vals):
    """Offsets within the column: the slot nearest the centre that does not collide.

    Deterministic, so the panel redraws identically. Points are taken from the lowest value
    upward; a candidate offset is rejected when a placed point sits within SEP_Y in value
    and SEP_X in offset.
    """
    off = np.zeros(len(vals))
    placed = []
    for i in np.argsort(vals, kind="stable"):
        v = vals[i]
        chosen, k = 0.0, 0
        while True:
            hit = None
            for c in ((0.0, ) if k == 0 else (k * SEP_X, -k * SEP_X)):
                if all(abs(v - pv) >= SEP_Y or abs(c - po) >= SEP_X
                       for pv, po in placed):
                    hit = c
                    break
            if hit is not None:
                chosen = hit
                break
            k += 1
            if k * SEP_X > HALF:
                break
        placed.append((v, chosen))
        off[i] = chosen
    return off


def draw(ax, cols):
    ax.axhspan(-BAND, BAND, color=S.RULE, alpha=0.28, lw=0, zorder=0)
    ax.axhline(0.0, color=S.INK, lw=S.LW_MIN, zorder=1)
    for x, (label, v, is_code) in enumerate(cols):
        col = S.C_CODE if is_code else S.C_INDEX
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.add_patch(Rectangle((x - 0.28, q1), 0.56, q3 - q1, facecolor="white",
                               edgecolor=col, lw=S.LW_MIN * 0.8, alpha=0.9, zorder=2))
        ax.plot([x - 0.32, x + 0.32], [med, med], color=col, lw=S.LW_MIN * 1.4,
                solid_capstyle="butt", zorder=4)
        ax.plot(x + swarm(v), v, marker="o", ms=MS, mfc=col, mec="white",
                mew=S.LW_MIN * 0.35, linestyle="none", zorder=3)
    for k in RULES_AFTER:
        ax.axvline(k + 0.5, color=S.RULE, lw=S.LW_MIN, zorder=1)

    ax.set_xlim(-0.62, len(cols) - 0.38)
    ax.set_ylim(*YLIM)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols], rotation=ROT, ha="right",
                       rotation_mode="anchor", fontsize=FS_TICK)
    for t, (label, _, _) in zip(ax.get_xticklabels(), cols):
        t.set_color(S.INK)
        if label in BOLD:
            t.set_fontweight("bold")
    ax.tick_params(axis="x", pad=1.5, length=2)
    ax.set_yticks(YTICKS)
    ax.set_yticklabels([str(t) for t in YTICKS])
    ax.set_ylabel("Sensitivity difference", fontsize=S.FS_BODY, color=S.INK, labelpad=3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--letter", default="d")
    ap.add_argument("--letter-visible", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for src, who in ((SRC_M, "f04_model_disparity.py"),
                     (SRC_P, "f04_protocol_disparity.py")):
        if not Path(src).exists():
            raise SystemExit("%s is missing; run %s first" % (src, who))
    cols = columns(load(SRC_M), load(SRC_P))
    print("  %d columns" % len(cols))
    for label, v, _ in cols:
        print("    %-22s median %+5.2f  range %+6.2f to %+5.2f  outside the band %d"
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
    for k, line in enumerate(reversed(NUMHEAD)):
        fig.text((X_AX0 - 1.2) / W_MM, (Y_COUNT + k * 2.6) / H_MM, line,
                 fontsize=S.FS_SMALL, color=S.MUTED, ha="right", va="bottom")

    S.panel_letter(fig, a.letter, a.letter_visible)
    S.check_text_fits(fig, W_MM, H_MM)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "panel_reader_swarm." + ext), dpi=600,
                    bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    print("wrote %s/panel_reader_swarm.pdf and .png at %.1f by %.1f mm"
          % (a.out, W_MM, H_MM))


if __name__ == "__main__":
    main()
