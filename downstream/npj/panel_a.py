#!/usr/bin/env python3
"""Panel a of Figure 1: one worked patient, beside the scale every reader used.

The two halves answer different questions and belong together. The left half shows what the
instrument does to a single record, from the sentence it read to the label it returned and
the reason it gave. The right half shows the scale that label sits on, and which of the three
analysis classes each grade belongs to. A connector runs from the label the patient received
to its rung, so the reader learns the vocabulary and sees it used in the same glance.

The pipeline boxes are neutral. They are stages of a process, not categories, and colouring
them would spend the colour vocabulary on something that carries no meaning. Only the label
box is coloured, in the class its grade collapses to.

Written as a standalone renderer so the panel can be checked before the figure is assembled.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch     # noqa: E402

from nature_style import apply_nature_style                        # noqa: E402

# The values of npj_style: the AUD present class carries the index colour and the AUD
# absent class the code colour. Until 24 August the two were the other way round here,
# which is the palette this figure used before the orange convention. Geometry and the
# nature_style call are unchanged, because the panel is drawn at its own type scale.
C_POS, C_NEG, C_INDET = "#D55E00", "#0072B2", "#767570"
TINT = {C_POS: "#F7DCCC", C_NEG: "#CCE3F0", C_INDET: "#E8E6E1"}
INK, MUTED, RULE = "#1f1e1d", "#55534e", "#B9B7B2"

GRADES = [("Definitely yes", C_POS), ("Probably yes", C_POS),
          ("Uncertain", C_INDET), ("Probably no", C_NEG),
          ("Definitely no", C_NEG), ("Not enough data", C_INDET)]
CLASSES = [("AUD+", C_POS), ("Indeterminate", C_INDET), ("AUD−", C_NEG)]


def box(ax, x, y, w, h, text, fs=5.2, fc="white", ec=RULE, lw=0.6, weight="normal",
        align="center"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x if align == "center" else x - w / 2 + 0.012, y, text, ha=align,
            va="center", fontsize=fs, color=INK, fontweight=weight, zorder=3,
            linespacing=1.45)


def arrow(ax, x, y0, y1, colour=RULE):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=6, color=colour, lw=0.6, zorder=1))


def draw_panel_a(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------------- left: what the study did --------------------------------
    # The one thing this half has to make obvious is that the two readers whose
    # agreement we measure both read the notes, while the comparator sees only the
    # coded fields. Every comparison in the paper follows from that asymmetry, and it
    # is the reason a diagnosis code can miss a patient whose record says so plainly.
    # Coordinates are laid out on an explicit grid, with every box's span written down,
    # because three reader boxes in a narrow half will silently collide otherwise.
    lx, lw = 0.25, 0.455
    X_CODE, W_CODE = 0.115, 0.160        # spans 0.035 to 0.195
    X_NOTE, W_NOTE = 0.355, 0.240        # spans 0.235 to 0.475
    X_MODEL, W_MODEL = 0.285, 0.130      # spans 0.220 to 0.350
    X_REF, W_REF = 0.418, 0.122          # spans 0.357 to 0.479

    ax.text(lx, 0.985, "What each reader was given", ha="center", va="top",
            fontsize=5.4, color=MUTED)

    box(ax, lx, 0.912, lw, 0.058, "Cirrhosis registry   14,406 patients", fs=5.6)
    arrow(ax, lx, 0.883, 0.858)
    box(ax, lx, 0.818, lw, 0.066,
        "487 charts, sampled on whether an AUD code\nappears anywhere in the record",
        fs=5.2)

    # the record splits into what a code can see and what it cannot
    for x in (X_CODE, X_NOTE):
        ax.add_patch(FancyArrowPatch((lx, 0.785), (x, 0.742), arrowstyle="-|>",
                                     mutation_scale=6, color=RULE, lw=0.6,
                                     connectionstyle="arc3,rad=0.0"))
    box(ax, X_CODE, 0.712, W_CODE, 0.056, "Coded fields", fs=5.4)
    box(ax, X_NOTE, 0.712, W_NOTE, 0.056, "Clinical notes", fs=5.4)

    arrow(ax, X_CODE, 0.684, 0.636, C_NEG)
    for xt, col in ((X_MODEL, C_POS), (X_REF, INK)):
        ax.add_patch(FancyArrowPatch((X_NOTE, 0.684), (xt, 0.636), arrowstyle="-|>",
                                     mutation_scale=6, color=col, lw=0.6))

    box(ax, X_CODE, 0.590, W_CODE, 0.084, "ICD F10.x code\nthe comparator", fs=5.2,
        fc=TINT[C_NEG], ec=C_NEG, lw=0.9)
    box(ax, X_MODEL, 0.590, W_MODEL, 0.084, "Qwen3.6-27B\nthe index test", fs=5.2,
        fc=TINT[C_POS], ec=C_POS, lw=0.9)
    box(ax, X_REF, 0.590, W_REF, 0.084, "Two fellows and\ntwo attendings", fs=5.2,
        ec=INK, lw=0.9)

    for x, lab in ((X_CODE, "scored"), (X_MODEL, "scored"),
                   (X_REF, "the reference standard")):
        ax.text(x, 0.532, lab, ha="center", va="center", fontsize=4.9, color=MUTED,
                style="italic" if x == X_REF else "normal")

    ax.text(lx, 0.462,
            "The code sees only the coded fields. The instrument and the clinicians read\n"
            "the same notes, so a code and a reader can differ on a patient whose record\n"
            "states the diagnosis in a sentence.",
            ha="center", va="top", fontsize=5.0, color=MUTED, linespacing=1.5)


def draw_scale(ax):
    """The six grades, their order, and the three classes they collapse to."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # ---------------- right: the scale every reader used ----------------------
    rx = 0.235
    ax.text(0.5, 0.985, "The scale every reader used, human or model",
            ha="center", va="top", fontsize=5.4, color=MUTED)

    # Sized to fill the row: the illustration on the left sets the height, and a
    # short ladder beside a tall drawing reads as an unfinished panel.
    top, step, gw, gh = 0.892, 0.136, 0.205, 0.086
    ys = []
    for k, (name, col) in enumerate(GRADES):
        y = top - k * step
        ys.append(y)
        box(ax, rx, y, gw, gh, name, fs=5.2, fc=TINT[col], ec=col, lw=0.7)

    # The four directional grades are one ordered scale; not enough data is not on it.
    ax.plot([rx - gw / 2 - 0.030] * 2, [ys[0] + gh / 2, ys[4] - gh / 2], lw=0.6,
            color=MUTED)
    for y in (ys[0] + gh / 2, ys[4] - gh / 2):
        ax.plot([rx - gw / 2 - 0.036, rx - gw / 2 - 0.024], [y, y], lw=0.6, color=MUTED)
    ax.text(rx - gw / 2 - 0.048, (ys[0] + ys[4]) / 2, "ordered", rotation=90,
            ha="center", va="center", fontsize=5.0, color=MUTED)
    # Below the box, not beside it: the key sits to the right and the two would collide.
    ax.text(rx, ys[5] - gh / 2 - 0.022, "off the scale", ha="center", va="top",
            fontsize=5.0, color=MUTED)

    # No connector across to the left half. The header above this scale already says it
    # applies to every reader, and a line drawn across three reader boxes and a rung
    # ladder has nowhere to run without crossing one of them.

    # ---------------- the collapse, beside the scale rather than under it ------
    # Indeterminate is not contiguous on the scale, since uncertain sits in the middle
    # and not enough data is off the end, so the grouping is carried by fill colour and
    # named in a key. Brackets would imply an adjacency that does not exist.
    kx = rx + gw / 2 + 0.055
    ky = ys[1] + 0.015
    ax.text(kx, ky + 0.052, "collapsed for analysis", ha="left", va="center",
            fontsize=4.9, color=MUTED)
    for k, (name, col) in enumerate(CLASSES):
        y = ky - k * 0.098
        ax.add_patch(FancyBboxPatch((kx, y - 0.014), 0.026, 0.028,
                                    boxstyle="round,pad=0.002,rounding_size=0.006",
                                    fc=TINT[col], ec=col, lw=0.7))
        ax.text(kx + 0.036, y, name, ha="left", va="center", fontsize=5.2, color=INK)
    ax.text(kx, ky - 0.300,
            "The two indeterminate grades\ndiffer in kind and are counted\napart. A patient "
            "the reference\nstandard cannot place is\nexcluded from accuracy.",
            ha="left", va="top", fontsize=4.8, color=MUTED, linespacing=1.55)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "downstream/npj/out/f01_accuracy"
    os.makedirs(out, exist_ok=True)
    apply_nature_style()
    fig = plt.figure(figsize=(7.205, 3.45))
    fig.set_layout_engine("none")
    ax = fig.add_axes([0.005, 0.0, 0.99, 0.985])
    draw_panel_a(ax)
    ax.text(-0.002, 1.0, "a", transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="left")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "panel_a." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s/panel_a.pdf and .png" % out)


if __name__ == "__main__":
    main()
