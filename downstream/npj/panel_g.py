#!/usr/bin/env python3
"""Panel g of Figure 1: every comparator on one axis, ordered by Youden's J.

The last paragraph of the accuracy section builds a ladder of comparators and ranks them.
This panel is that ladder. Rows are ordered with the highest J at the top, and the ordering is
not arranged to flatter: the keyword rule over the same notes comes second and its specificity
equals the LLM's, so a reader sees that much of the gap over the diagnosis code is reading the
notes at all and that the remaining gap is sensitivity. The supervised text model reads the
notes too and comes last, which is why reading notes is not the whole account.

Drawing decisions.

  Colour records what each comparator was allowed to read: orange the clinical notes, blue the
  structured record alone. The distinction was carried by the colour of the row label until
  npj's rule was applied, which asks for coloured marks with black text; it is now a legend
  with two colour swatches and every label is black.

  Marker shape records the metric, a filled circle for sensitivity and an open square for
  specificity, and the legend carries that too.

  A broken rule sits under the top row. The claim of the panel is that one comparator sits
  above the rest, and not that the ordering below it is resolved.

  Two row labels are shortened here and nowhere else: the two gradient-boosted models are
  written with + code and - code, because the longer forms are 36.1 and 34.2 mm wide at 8 pt
  and the label column has 33.0 mm. Supplementary Table 6 carries the full specifications.

Values are as observed on the charts with a determinate reference, with Wilson intervals, from
the frame build_supp_table6.py writes, which is the frame Supplementary Table 6 renders. This
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
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.lines import Line2D                                     # noqa: E402
from matplotlib.patches import Patch                                    # noqa: E402

import npj_style as S                                                   # noqa: E402

TABLE = "downstream/npj/out/supp_table6.csv"
OUT = "downstream/npj/out/figure1"
PRIMARY = "LLM (Qwen3.6-27B)"

W_MM, H_MM = 78.0, 64.0
X_AX0, AX_W = 34.5, 38.0
Y_AX0, AX_H = 10.5, 41.5
Y_LEGEND = 62.0

READS = {"notes": S.C_INDEX, "structured record": S.C_CODE}
RENAME = {"Gradient boosting, with code": "Gradient boosting + code",
          "Gradient boosting, no code": "Gradient boosting − code",
          "Fine-tuned Bio_ClinicalBERT": "Bio_ClinicalBERT",
          "TF-IDF logistic regression": "TF-IDF logistic"}


def load(table=TABLE):
    if not Path(table).exists():
        raise SystemExit("run build_supp_table6.py first; %s is missing" % table)
    d = pd.read_csv(table)
    d["label"] = d["short"].str.replace(r"\s+", " ", regex=True).str.strip()
    d["label"] = d["label"].map(lambda v: RENAME.get(v, v))
    return d


def draw_panel_g(ax, d):
    n = len(d)
    for k, r in d.iterrows():
        y = n - 1 - k                                    # highest J at the top
        col = READS[r["reads"]]
        for met, off, mark, face in (("sens", 0.22, "o", None),
                                     ("spec", -0.22, "s", "white")):
            v = r[met] / 100.0
            lo, hi = r["%s_lo" % met] / 100.0, r["%s_hi" % met] / 100.0
            ax.errorbar(v, y + off, xerr=[[v - lo], [hi - v]], fmt=mark, ms=3.2,
                        lw=S.LW_MIN, capsize=1.4, color=col,
                        mfc=face if face else col, mec=col, mew=S.LW_MIN, zorder=3)

    labels = list(d["label"])[::-1]
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=S.FS_BODY, color=S.INK)
    for t, name in zip(ax.get_yticklabels(), labels):
        if name == PRIMARY:
            t.set_fontweight("bold")
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(0.65, 1.01)
    ax.set_xticks([0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("Value against the reference standard", fontsize=S.FS_BODY,
                  color=S.INK, labelpad=4)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", pad=2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.axhline(n - 1.5, color=S.RULE, lw=S.LW_MIN, ls=(0, (2.4, 2.0)), zorder=1)
    return ax


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--letter", default="g")
    ap.add_argument("--letter-visible", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load(a.table)

    print("%-26s %6s %-20s %-20s %s"
          % ("comparator", "J", "sensitivity", "specificity", "reads"))
    for _, r in d.iterrows():
        print("%-26s %6.3f %5.3f (%.3f-%.3f) %5.3f (%.3f-%.3f)  %s"
              % (r["label"], r["youden"], r["sens"] / 100, r["sens_lo"] / 100,
                 r["sens_hi"] / 100, r["spec"] / 100, r["spec_lo"] / 100,
                 r["spec_hi"] / 100, r["reads"]))
    d[["label", "reads", "youden", "sens", "sens_lo", "sens_hi", "spec", "spec_lo",
       "spec_hi", "n"]].to_csv(os.path.join(a.out, "panel_g_ladder.csv"), index=False)

    S.apply_npj_style()
    fig = plt.figure(figsize=S.mm(W_MM, H_MM))
    fig.set_layout_engine("none")
    ax = fig.add_axes([X_AX0 / W_MM, Y_AX0 / H_MM, AX_W / W_MM, AX_H / H_MM])
    fig.canvas.draw()
    draw_panel_g(ax, d)
    S.panel_letter(fig, a.letter, a.letter_visible)

    # Colour swatches for what a comparator reads, and the two markers for the metric. The
    # swatch is a patch and not a marker, so that a filled circle means the metric and
    # nothing else.
    handles = [Patch(facecolor=READS["notes"], edgecolor="none"),
               Patch(facecolor=READS["structured record"], edgecolor="none"),
               Line2D([], [], ls="none", marker="o", ms=3.4, color=S.C_FORE),
               Line2D([], [], ls="none", marker="s", ms=3.4, mfc="white",
                      mec=S.C_FORE, mew=S.LW_MIN)]
    labels = ["Reads clinical notes", "Reads structured record",
              "Sensitivity", "Specificity"]
    leg = fig.legend(handles, labels, loc="upper left",
                     bbox_to_anchor=(S.LETTER_BOX_W / W_MM, Y_LEGEND / H_MM),
                     ncol=2, frameon=False, fontsize=S.FS_BODY, handlelength=1.4,
                     handleheight=0.9, handletextpad=0.6, columnspacing=2.0,
                     labelspacing=0.4, borderpad=0.0, borderaxespad=0.0)
    for t_ in leg.get_texts():
        t_.set_color(S.INK)

    fig.canvas.draw()
    for name, art in (("legend", leg), ("x label", ax.xaxis.label)):
        bb = art.get_window_extent()
        x0, x1 = bb.x0 / fig.dpi * 25.4, bb.x1 / fig.dpi * 25.4
        if x0 < -0.2 or x1 > W_MM + 0.2:
            print("  WARNING %s runs outside the panel: %.1f to %.1f mm"
                  % (name, x0, x1))
    for t_ in ax.get_yticklabels():
        bb = t_.get_window_extent()
        if bb.x0 / fig.dpi * 25.4 < -0.2:
            print("  WARNING row label runs outside the panel: %.1f mm  %r"
                  % (bb.x0 / fig.dpi * 25.4, t_.get_text()))

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "panel_g." + ext), dpi=600,
                    bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    print("wrote %s/panel_g.pdf and .png at %.1f by %.1f mm" % (a.out, W_MM, H_MM))


if __name__ == "__main__":
    main()
