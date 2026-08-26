#!/usr/bin/env python3
"""The six-category matrix, on its own, for the supplement.

The main text carries the three-category collapse, because that is the label every analysis
uses. This one supports a narrower claim: that classifications which miss do so by one grade
far more often than they invert, which is the reason the kappa is weighted linearly.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, "downstream")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import render_fig2_headline as H   # noqa: E402
import matplotlib.pyplot as plt    # noqa: E402
from nature_style import apply_nature_style, save_nature_figure   # noqa: E402

OUT = "downstream/npj/out/figure1"


def main():
    os.makedirs(OUT, exist_ok=True)
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.24, right=0.98, top=0.97, bottom=0.26)
    H.draw_confusion(ax)
    save_nature_figure(fig, os.path.join(OUT, "supp_confusion_six"))
    plt.close(fig)
    print("wrote %s/supp_confusion_six.pdf and .png" % OUT)
    tables()


# Supplementary Tables 1 and 2 are this figure written out. Until 15 August both were
# typed from an earlier run: Table 1 gave 210 where the matrix gives 211 and a diagonal of
# 86.2 per cent against 86.4, and Table 2's six model-grade counts summed to 496, the
# cohort before the standard closed. Both are emitted here so the figure and the tables
# cannot disagree again.
DISPLAY = ["Definitely Yes", "Probably Yes", "Uncertain", "Probably No", "Definitely No",
           "Not Enough Data"]
SHORT = {"Definitely Yes": "Def.\\ Yes", "Probably Yes": "Prob.\\ Yes",
         "Uncertain": "Uncertain", "Probably No": "Prob.\\ No",
         "Definitely No": "Def.\\ No", "Not Enough Data": "Not enough data"}
POS = ("Definitely Yes", "Probably Yes")
NEG = ("Definitely No", "Probably No")
# The ordinal scale, on which "one grade away" is defined. Not Enough Data has no place on
# it and is excluded from that count rather than given a position.
SCALE = ["Definitely No", "Probably No", "Uncertain", "Probably Yes", "Definitely Yes"]


def tables():
    import numpy as np
    import pandas as pd

    m = pd.read_csv("downstream/npj/out/f01_accuracy/confusion_six.csv", index_col=0)
    m = m.reindex(index=DISPLAY, columns=DISPLAY).fillna(0).astype(int)
    n = int(m.values.sum())

    diag = int(np.trace(m.values))
    pos = {c: i for i, c in enumerate(SCALE)}
    one = sum(int(m.loc[r, c]) for r in SCALE for c in SCALE
              if abs(pos[r] - pos[c]) == 1)
    print("\nSupplementary Table 1, n = %d" % n)
    print("  diagonal %d, %.1f%%" % (diag, 100 * diag / n))
    print("  one grade away on the ordinal scale %d, %.1f%%" % (one, 100 * one / n))
    print("  Not Enough Data has no position on that scale and is not counted in it")
    for r in DISPLAY:
        cells = " & ".join("--" if m.loc[r, c] == 0 else str(int(m.loc[r, c]))
                           for c in DISPLAY)
        print("%-16s & %s \\\\" % (SHORT[r], cells))

    def three(x):
        return "AUD+" if x in POS else "AUD-" if x in NEG else "Indeterminate"

    print("\nSupplementary Table 2, agreement within each model grade")
    rows = []
    for c in DISPLAY:
        col = m[c]
        d = int(col.sum())
        if d == 0:
            continue
        agree = int(sum(v for r, v in col.items() if three(r) == three(c)))
        z, p = 1.959963985, agree / d
        den = 1 + z * z / d
        cen = (p + z * z / (2 * d)) / den
        half = z * np.sqrt(p * (1 - p) / d + z * z / (4 * d * d)) / den
        lo, hi = 100 * max(0.0, cen - half), 100 * min(1.0, cen + half)
        rows.append({"model_grade": c, "n": d, "agree": agree,
                     "pct": 100 * p, "lo": lo, "hi": hi})
        print("%-16s & %3d & %.1f (%.1f--%.1f) \\\\" % (SHORT[c], d, 100 * p, lo, hi))
    t = pd.DataFrame(rows)
    print("  grades sum to %d" % int(t["n"].sum()))
    t.to_csv(os.path.join(OUT, "supp_table2_by_grade.csv"), index=False)
    m.to_csv(os.path.join(OUT, "supp_table1_confusion_six.csv"))

    # The two shares the main text and the supplement both state in prose. They were
    # printed and never written, so both were typed into the .tex by hand; the supplement
    # was corrected from this matrix on 15 August and the main text was not, and it stood
    # at 10.5 per cent where the matrix gives 10.9. The anchor is the phrase the sentence
    # uses. check_manuscript.py reads the percentage that stands nearest before it.
    pd.DataFrame([
        {"quantity": "diagonal", "anchor": "diagonal",
         "count": diag, "n": n, "pct": round(100.0 * diag / n, 1)},
        {"quantity": "one grade away", "anchor": "one grade",
         "count": one, "n": n, "pct": round(100.0 * one / n, 1)},
    ]).to_csv(os.path.join(OUT, "supp_table1_shares.csv"), index=False)


if __name__ == "__main__":
    main()
