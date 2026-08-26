#!/usr/bin/env python3
"""Assemble Figure 1: the LLM, the reference standard, and every comparator.

  a  what each reader was given, and the scale they all answered on
  b  three-category agreement, reference standard against the LLM
  c  where the diagnosis code's misses go, and which of them the LLM recovers
  d  what each reader achieved, in the sample and weighted to the registry
  e  where the LLM and a comparator disagree, which of the two was right,
     asked twice: against the code, and against the fellow
  f  whether the advantage survives at any prevalence, with the two populations
     that bound the range marked
  g  every comparator on one axis, ordered by Youden's J

Panels b and c come from the existing renderer in downstream/render_fig2_headline.py,
unchanged. Panel f was drawn from there until 13 August and is now drawn by panels_de.py,
so that it and panel d share one frame and one set of counts. Panel a's scale is drawn
by panel_a.py and its left half is supplied as a file.
Panels d and e are drawn by panels_de.py and panel g by panel_g.py.

The reader comparison moved up into the second row to make room for panel g. The last
paragraph of the section ranks eight comparators and the figure did not carry that ranking, so
the one claim a reader could not check was the one that decides how much of the result is the
language model and how much is reading the notes at all.

Sensitivity by hospital system stays out: Figure 2c asks it. The eleven-model leaderboard also
stays out; that is a robustness question and Figure 4b asks it.

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
# Imported before the house style is applied: this module sets its own rcParams as it loads,
# and applying ours afterwards keeps one type size across the whole figure.
import render_fig2_headline as H                                    # noqa: E402
import matplotlib.image as mpimg                                   # noqa: E402
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import Rectangle                           # noqa: E402

import panel_a as PA                                                # noqa: E402
import panels_de as DE                                              # noqa: E402
import panel_g as PG                                                # noqa: E402
from nature_style import apply_nature_style, save_nature_figure      # noqa: E402

OUT = "downstream/npj/out/figure1"
ILLUSTRATION = "downstream/npj/assets/panel_a_left.png"
ESTIMATES = "downstream/npj/out/f01_accuracy/reader_estimates.csv"


def panel_a(fig, cell, illustration):
    # The illustration is very nearly square, so its cell is narrowed until the image is
    # limited by the cell's width rather than its height. Left wider, the image shrank to fit
    # the row and sat in a pool of white on both sides.
    inner = cell.subgridspec(1, 2, width_ratios=[0.95, 1.15], wspace=0.04)
    ax = fig.add_subplot(inner[0])
    ax.axis("off")
    if Path(illustration).exists():
        ax.imshow(mpimg.imread(illustration))
        ax.set_anchor("N")
    else:
        ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, fc="#FAFAF8", ec=PA.RULE,
                               lw=0.6, ls=(0, (3, 2))))
        ax.text(0.5, 0.52, "panel a, left half\n\nwhat each reader was given",
                ha="center", va="center", fontsize=6, color=PA.MUTED)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    PA.draw_scale(fig.add_subplot(inner[1]))
    return ax


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--illustration", default=ILLUSTRATION)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if not Path(ESTIMATES).exists():
        raise SystemExit("run panels_de.py first; %s is missing" % ESTIMATES)
    est_df = pd.read_csv(ESTIMATES)
    sc = DE.load()
    disc = DE.discordance(sc)
    comp_df = PG.load()

    apply_nature_style()
    fig = plt.figure(figsize=(7.205, 8.0))
    fig.set_layout_engine("none")
    # Row one is sized to the illustration, which is anchored to the top of its cell.
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 0.96, 0.88], hspace=0.32,
                          left=0.068, right=0.988, top=0.984, bottom=0.056)
    # Three panels in the middle row. The alluvial diagram was the widest thing in the
    # figure and did not need to be; the room it gives back holds the reader comparison.
    mid = gs[1].subgridspec(1, 3, width_ratios=[0.80, 1.06, 1.12], wspace=0.60)
    # The last row is nested rather than a single four-column grid. A uniform wspace wide
    # enough to clear panel g's comparator names would also have prised apart the two
    # two-by-twos, which have to stay adjacent to read as one question asked twice.
    bot = gs[2].subgridspec(1, 2, width_ratios=[2.08, 1.30], wspace=0.34)
    botl = bot[0].subgridspec(1, 3, width_ratios=[0.56, 0.56, 0.96], wspace=0.62)

    axa = panel_a(fig, gs[0], a.illustration)

    axb = fig.add_subplot(mid[0])
    H.draw_confusion_3class(axb)
    axb.set_xlabel("")

    axc = fig.add_subplot(mid[1])
    H.draw_alluvial(axc)

    axd = fig.add_subplot(mid[2])
    DE.draw_panel_d(axd, est_df)

    axe = [fig.add_subplot(botl[0]), fig.add_subplot(botl[1])]
    DE.draw_panel_e(axe, disc)

    axf = fig.add_subplot(botl[2])
    # Drawn here rather than by render_fig2_headline.draw_prevalence, which read a stale
    # csv for the index test and carried the comparator as two hard-coded constants from a
    # 433-patient cohort. Both arms now come from the same frame panel d uses.
    DE.draw_panel_f(axf, sc)

    axg = fig.add_subplot(bot[1])
    PG.draw_panel_g(axg, comp_df)

    for ax, lab, dx, dy in ((axa, "a", -0.050, 1.02), (axb, "b", -0.30, 1.07),
                            (axc, "c", -0.07, 1.07), (axd, "d", -0.40, 1.10),
                            (axe[0], "e", -0.48, 1.34), (axf, "f", -0.33, 1.10),
                            (axg, "g", -0.52, 1.10)):
        ax.text(dx, dy, lab, transform=ax.transAxes, fontsize=8, fontweight="bold",
                va="top", ha="left")

    # Figure-wide alignment groups x labels by gridspec column, and this figure's root
    # gridspec has one column, so it would drag the reader comparison's x label out of the
    # middle row and print it over panel g's. Only the two value axes of the last row are
    # aligned. The two-by-twos are square and much shorter, and aligning them too left their
    # comparator names floating an inch below the matrix they name.
    fig.align_xlabels([axf, axg])
    save_nature_figure(fig, os.path.join(a.out, "figure1"), align=False)
    plt.close(fig)
    print("wrote %s/figure1.pdf and .png" % a.out)
    if not Path(a.illustration).exists():
        print()
        print("!" * 74)
        print("Panel a's left half is a placeholder. Put the illustration at")
        print("  %s" % a.illustration)
        print("!" * 74)


if __name__ == "__main__":
    main()
