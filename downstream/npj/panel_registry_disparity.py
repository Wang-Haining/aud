#!/usr/bin/env python3
"""Registry-scale coverage of the AUD diagnosis code, each stratum against the rest.

Panel of Figure 4. The earlier version of this panel drew the share coded by t0 in each
stratum and the number of patients behind it. A panel of levels has no null, so it could
show who is under-coded but not whether the under-coding is unequal. This one draws the
ratio of a stratum's coverage to the coverage of the rest of its axis, where one is no
difference, and keeps the share and the count as columns so nothing is lost.

A ratio here, a difference in the chart-scale panel. The registry-scale shares run from 24
to 45 per cent, so nothing sits near a boundary and a ratio is stable and reads directly.
At chart scale the model reaches 100 per cent in five strata, where a ratio of error rates
is undefined, which is why that panel is on the difference scale.

The band is 0.8 to 1.25, the four-fifths convention, which is symmetric on a log axis
because 0.8 times 1.25 is one. It is drawn as a reference and not as a test. The convention
comes from employment selection and has been criticised for being carried into algorithmic
fairness as though it were a criterion, so the panel uses it the way the chart-scale panel
uses its own band: to say which strata are worth reading, not to decide anything.

Only the diagnosis code appears. Its coverage is observed on all 14,406 registry patients
once chart review calibrates it; the model was run on the sampled charts only, so it has no
registry-scale estimate. The comparison between readers is the chart-scale panel's job.

Values come from out/figure4/registry_disparity.csv and registry_burden_subgroup.csv. This
file draws; it does not compute.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "downstream")
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator               # noqa: E402

import npj_style as S                                                    # noqa: E402

SRC = "downstream/npj/out/figure4/registry_disparity.csv"
BURDEN = "downstream/npj/out/figure4/registry_burden_subgroup.csv"
OUT = "downstream/npj/out/figure4"

W_MM, H_MM = 88.0, 80.0
X_LAB, X_INDENT = 1.5, 3.0
X_AX0, AX_W = 25.0, 41.0
NUM1_X = 67.5
NUM2_X = 75.0
Y_AX0, AX_H = 10.5, 62.0
Y_LEGEND = 74.0
XLIM = (0.5, 2.0)
XTICKS = [0.5, 0.8, 1.0, 1.25, 2.0]
XLABELS = ["0.5", "0.8", "1", "1.25", "2"]
BAND = (0.8, 1.25)
ORDER = ["Age", "Sex", "Race", "Charlson index", "Neighbourhood deprivation"]
SHOW = {"Male": "Men", "Female": "Women"}


def load(src=SRC, burden=BURDEN):
    d = pd.read_csv(src)
    b = pd.read_csv(burden).set_index(["axis", "level"])
    rows = []
    for axis in ORDER:
        rows.append({"kind": "head", "label": axis})
        for _, r in d[d["axis"].eq(axis)].iterrows():
            unc = float(b.loc[(axis, r["level"]), "uncoded_at_t0"])
            rows.append({"kind": "data", "label": SHOW.get(r["level"], r["level"]),
                         "ratio": float(r["ratio"]), "lo": float(r["ci_low"]),
                         "hi": float(r["ci_high"]), "cov": float(r["coverage_pct"]),
                         "unc": unc})
    return rows, d


def draw(ax, rows):
    n = len(rows)
    ys = {k: n - 1 - k for k in range(n)}
    ax.axvspan(BAND[0], BAND[1], color=S.RULE, alpha=0.28, lw=0, zorder=0)
    ax.axvline(1.0, color=S.INK, lw=S.LW_MIN, zorder=1)
    for k, r in enumerate(rows):
        if r["kind"] != "data":
            continue
        yy = ys[k]
        ax.errorbar(r["ratio"], yy,
                    xerr=[[r["ratio"] - r["lo"]], [r["hi"] - r["ratio"]]],
                    fmt="s", ms=2.6, lw=S.LW_MIN, capsize=1.0, color=S.C_CODE,
                    mfc=S.C_CODE, mec=S.C_CODE, mew=S.LW_MIN, zorder=3)
        for x_mm, txt in ((NUM1_X, "%.1f" % r["cov"]),
                          (NUM2_X, format(int(round(r["unc"])), ",d"))):
            ax.annotate(txt, xy=(1.0, yy),
                        xytext=((x_mm - X_AX0 - AX_W) * 72.0 / 25.4, 0),
                        textcoords="offset points", xycoords=ax.get_yaxis_transform(),
                        ha="left", va="center", fontsize=S.FS_SMALL, color=S.MUTED,
                        annotation_clip=False)

    ax.set_xscale("log")
    ax.set_yticks([])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(*XLIM)
    ax.xaxis.set_major_locator(FixedLocator(XTICKS))
    ax.xaxis.set_major_formatter(FixedFormatter(XLABELS))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("Code sensitivity ratio versus complement",
                  fontsize=S.FS_BODY, color=S.INK, labelpad=4)
    ax.tick_params(axis="x", pad=2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    return ys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--letter", default="c")
    ap.add_argument("--letter-visible", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in (SRC, BURDEN):
        if not Path(f).exists():
            raise SystemExit("%s is missing; run f04_registry_disparity.py first" % f)
    rows, d = load()

    out = (d["ci_low"] > 1.0) | (d["ci_high"] < 1.0)
    band = (d["ratio"] < BAND[0]) | (d["ratio"] > BAND[1])
    print("  strata %d; ratio outside the band %d; intervals excluding one %d"
          % (len(d), int(band.sum()), int(out.sum())))
    for _, r in d[out].iterrows():
        print("    %-26s %-17s %5.3f (%5.3f to %5.3f)"
              % (r["axis"], r["level"], r["ratio"], r["ci_low"], r["ci_high"]))

    S.apply_npj_style()
    fig = plt.figure(figsize=S.mm(W_MM, H_MM))
    fig.set_layout_engine("none")
    ax = fig.add_axes([X_AX0 / W_MM, Y_AX0 / H_MM, AX_W / W_MM, AX_H / H_MM])
    fig.canvas.draw()
    ys = draw(ax, rows)

    n = len(rows)
    for k, r in enumerate(rows):
        head = r["kind"] == "head"
        y_fig = (Y_AX0 + AX_H * (ys[k] + 0.6) / n) / H_MM
        x_mm = X_LAB if head else X_LAB + X_INDENT
        tx = fig.text(x_mm / W_MM, y_fig, r["label"], fontsize=S.FS_SMALL, color=S.INK,
                      ha="left", va="center", fontweight="bold" if head else "normal")
        bb = tx.get_window_extent(renderer=fig.canvas.get_renderer())
        if not head and bb.x1 / fig.dpi * 25.4 > X_AX0 - 0.8:
            print("  WARNING label reaches the plot: %.1f mm  %r"
                  % (bb.x1 / fig.dpi * 25.4, r["label"]))

    for x_mm, txt in ((NUM1_X, "%"), (NUM2_X, "Uncoded")):
        fig.text(x_mm / W_MM, (Y_AX0 + AX_H + 1.4) / H_MM, txt, fontsize=S.FS_SMALL,
                 color=S.MUTED, ha="left", va="bottom")

    S.panel_letter(fig, a.letter, a.letter_visible)
    S.check_text_fits(fig, W_MM, H_MM)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "panel_registry_disparity." + ext), dpi=600,
                    bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    print("wrote %s/panel_registry_disparity.pdf and .png at %.1f by %.1f mm"
          % (a.out, W_MM, H_MM))


if __name__ == "__main__":
    main()
