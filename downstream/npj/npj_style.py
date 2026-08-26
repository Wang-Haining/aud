#!/usr/bin/env python3
"""npj_style.py -- the project's semantic palette, and npj Digital Medicine's own numbers.

Three things live here and nothing else.

  The semantic palette. Colour encodes category and never magnitude, and a colour keeps one
  meaning across every figure in the paper. Tier 1 is the category vocabulary. Tier 2 is the
  value channel, which separates foreground from background and carries no category. Tier 3
  holds the ramps and accents that appear in exactly one figure each.

  The typeface. npj asks for Arial or Helvetica and for one typeface across every figure.
  Neither is installed on Quartz, so nature_style.apply_nature_style() fell through its
  stack to DejaVu Sans, which is a different typeface with wider advance widths. Liberation
  Sans is metric-compatible with Arial and Nimbus Sans with Helvetica, and both are present,
  so they are inserted ahead of the fallback. Every figure built before 22 August 2026 is
  set in DejaVu Sans.

  apply_npj_style(). nature_style.apply_nature_style() sets Nature Portfolio's general
  figure defaults, which give 5 to 7 pt text and 0.5 pt rules. npj Digital Medicine states
  8 pt as the optimum at print size, a thinnest line of 1 pt at final size, and panel labels
  in the same type size as the figure text. This function applies the general defaults and
  then those overrides, so the two documents are not conflated.

nature_style.py is left as the unmodified helper from the nature-visualizer skill. Producers
that have not been migrated keep importing from it and are unaffected by this file.

Verified 22 August 2026. CIE76 dE between the two Tier 1 readers, under Machado, Oliveira
and Fernandes (2009) simulation at severity 1.0: 114.6 normal, 91.9 protanopia, 108.3
deuteranopia, 102.0 tritanopia. The two tints alone give 23.4 and 18.1, so the border
carries the category and the tint supports it. Perceptual grey of the four Tier 1 colours is
50.4, 42.5, 13.9 and 45.6, and the first, second and fourth lie within 8 of one another,
which is why the indeterminate class also carries a dashed border.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.font_manager            # noqa: F401

from nature_style import apply_nature_style, nature_figsize            # noqa: F401

# --- Typeface -------------------------------------------------------------
SANS = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]

# --- Tier 1: category. One meaning per colour, across every figure. ---------
C_INDEX = "#D55E00"      # the LLM (index test), and the AUD present class
C_CODE = "#0072B2"       # the AUD diagnosis code, and the AUD absent class
C_REF = "#1F1E1D"        # the clinicians, and the reference standard
C_INDET = "#767570"      # indeterminate, and no conclusion

# The clinicians carry a light fill as well as the ink, so that where they are a plotted
# series beside the other two readers they have the same visual weight. Its lightness is
# matched to the other two: perceptual grey 88.8 against 87.9 and 87.2, and contrast against
# white 1.28 against 1.31 and 1.33. Adding a hue for the clinicians was tested and rejected:
# the third reader has to separate from orange, blue and two greys at once, and every
# candidate in the Okabe-Ito set fell to dE 25 or below against one of them under simulated
# colour vision deficiency, while ink stays above 38 against every colour it co-occurs with
# and is the darkest of the three in grey print.
TINT = {C_INDEX: "#F7DCCC", C_CODE: "#CCE3F0", C_REF: "#E6E3DC",
        C_INDET: "#EDEDEA"}

# The indeterminate class carries a dashed edge wherever it is drawn as a region. It is the
# only channel that separates it from the other two under grey printing.
DASH_INDET = (0, (2.6, 1.9))

# --- Tier 2: value. Foreground against background. No category. ------------
C_FORE = "#404040"       # other models, foreground
C_BACK = "#B9B7B2"       # other models in the background; rules; reference lines
INK = "#1F1E1D"
MUTED = "#55534E"
RULE = "#B9B7B2"
HAIR = "#DCDAD5"         # cell separators inside a matrix

# --- Tier 3: one figure each. ----------------------------------------------
# Figure 5, the four ordered states of quotation grounding. Monotone in perceptual
# grey (33.3, 50.4, 67.6, 87.9) so it survives grey printing and every simulated
# colour vision deficiency; adjacent steps are dE 25.6, 22.5 and 38.1 in deuteranopia.
RAMP_INDEX = ["#8C3D00", "#D55E00", "#ED9B5E", "#F7DCCC"]
# The same four stops in the diagnosis code's hue, for the one panel that has to show three
# ordered states of the code at once. Lightness runs 30, 45, 71, 89 in CIE L*, so the stops
# stay apart in grey print.
RAMP_CODE = ["#00507D", "#0072B2", "#7FB6D9", "#CCE3F0"]
# The clinicians' hue has no chroma to run a ramp through, so its four stops are warm
# greys at the same lightness spacing as the two above: 8, 30, 57, 90 in CIE L*.
RAMP_REF = ["#0F0E0D", "#4A4844", "#8C8981", "#E6E3DC"]

# --- Which stop fills a bar -------------------------------------------------
# A bar that carries an interval takes the third stop as its fill and the first stop for the
# interval. The palest stop, which TINT gives, is for a cell that carries text, where the
# fill is a background and the ink on top has to win. A saturated fill needs a white
# interval, and a white interval disappears in the part of it that extends past the end of
# the bar, which is the part a reader looks at. Settled 24 August by rendering Figure 2c and
# 2d under all three.
FILL_BAR = {C_INDEX: RAMP_INDEX[2], C_CODE: RAMP_CODE[2], C_REF: RAMP_REF[2]}
FILL_DARK = {C_INDEX: RAMP_INDEX[0], C_CODE: RAMP_CODE[0], C_REF: RAMP_REF[0]}

# --- npj Digital Medicine's own numbers ------------------------------------
FS_BODY, FS_SMALL, FS_PANEL = 8.0, 7.0, 8.0
LW_MIN = 1.0
MM = 1 / 25.4

# --- Figure 1, the width budget -------------------------------------------
# One double column is 183 mm. Panel a is supplied as a vector file, so the panels beside
# it have to be built at a declared width rather than at whatever a tight bounding box
# leaves. GUTTER is the space between two panels in a row.
W_DOUBLE, W_HALF, W_SINGLE = 183.0, 120.0, 89.0
H_MAX = 247.0
GUTTER = 5.0
# Every panel reserves this box at its top left for its letter, so that switching a
# letter on, off, or to another letter moves nothing else. Content in a panel's top
# band starts at LETTER_BOX_W.
LETTER_X, LETTER_Y, LETTER_BOX_W = 1.0, 1.0, 6.0


def mm(*values):
    """Millimetres to inches. mm(183, 64) returns a figsize tuple."""
    out = tuple(v * MM for v in values)
    return out[0] if len(out) == 1 else out


def apply_npj_style():
    """Nature Portfolio defaults, then npj Digital Medicine's own numbers."""
    apply_nature_style()
    mpl.rcParams.update({
        "font.sans-serif": SANS,
        "font.size": FS_BODY,
        "axes.labelsize": FS_BODY,
        "axes.titlesize": FS_BODY,
        "xtick.labelsize": FS_BODY,
        "ytick.labelsize": FS_BODY,
        "legend.fontsize": FS_SMALL,
        "legend.title_fontsize": FS_SMALL,
        "axes.linewidth": LW_MIN,
        "xtick.major.width": LW_MIN,
        "ytick.major.width": LW_MIN,
        "xtick.minor.width": LW_MIN,
        "ytick.minor.width": LW_MIN,
        "lines.linewidth": LW_MIN,
        "patch.linewidth": LW_MIN,
        "hatch.linewidth": LW_MIN,
        # Panels are assembled side by side at a known width, so a figure must be saved at
        # exactly the size it declares. A tight bounding box crops to the drawn content and
        # the widths stop matching. savefig(bbox_inches=None) falls back to this rcParam,
        # so it has to be None and not the string.
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
    })
    # Mathtext keeps its own font set and it defaults to DejaVu, so a panel carrying a
    # $t_0$ in a label came out in two typefaces at once and the PDF embedded both. The
    # font this machine actually resolves from SANS is given to mathtext as well.
    name = mpl.font_manager.FontProperties(
        fname=mpl.font_manager.findfont(
            mpl.font_manager.FontProperties(family=SANS))).get_name()
    mpl.rcParams.update({
        "mathtext.fontset": "custom",
        "mathtext.rm": name,
        "mathtext.it": "%s:italic" % name,
        "mathtext.bf": "%s:bold" % name,
    })


def check_text_fits(fig, w_mm, h_mm=None, tol=0.0):
    """Warn for any drawn text whose box leaves the panel.

    Tick labels, axis labels and legend text are not in `ax.texts`. A check that walks only
    that list reports nothing while an axis label runs off the page, which is what the first
    version of this check did.
    """
    fig.canvas.draw()
    items = list(fig.texts)
    for ax in fig.axes:
        items += list(ax.texts)
        items += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        items += [ax.xaxis.label, ax.yaxis.label]
        leg = ax.get_legend()
        if leg is not None:
            items += list(leg.get_texts())
    for leg in fig.legends:
        items += list(leg.get_texts())

    bad = 0
    seen = 0
    margin = w_mm
    for t in items:
        # A hidden axis keeps its tick labels as objects. They are not drawn and their
        # positions are meaningless, so they are not checked.
        if not t.get_text() or not t.get_visible():
            continue
        seen += 1
        bb = t.get_window_extent()
        margin = min(margin, w_mm - bb.x1 / fig.dpi * 25.4, bb.x0 / fig.dpi * 25.4)
        x0, x1 = bb.x0 / fig.dpi * 25.4, bb.x1 / fig.dpi * 25.4
        if x1 > w_mm + tol or x0 < -tol:
            print("  WARNING %r spans %.1f to %.1f mm across a %.1f mm panel"
                  % (t.get_text(), x0, x1, w_mm))
            bad += 1
        if h_mm is not None:
            y0, y1 = bb.y0 / fig.dpi * 25.4, bb.y1 / fig.dpi * 25.4
            if y1 > h_mm + tol or y0 < -tol:
                print("  WARNING %r spans %.1f to %.1f mm up a %.1f mm panel"
                      % (t.get_text(), y0, y1, h_mm))
                bad += 1
    if not bad:
        print("  %d text items inside the panel, the tightest %.1f mm from an edge"
              % (seen, margin))
    return bad


def panel_letter(fig, letter, visible=False, x_mm=LETTER_X, y_mm=LETTER_Y):
    """Reserve the panel letter's place and, by default, draw it invisibly.

    Panels are exported on their own and lettered where they are assembled, so no producer
    prints a visible letter. The placeholder is still drawn, at alpha 0, so the space is
    reserved in every panel and the letter can be made visible without moving anything else.
    The text object is in the PDF, so it can also be selected and given full opacity in a
    vector editor.
    """
    w_mm = fig.get_size_inches()[0] * 25.4
    h_mm = fig.get_size_inches()[1] * 25.4
    return fig.text(x_mm / w_mm, 1.0 - y_mm / h_mm, letter, fontsize=FS_PANEL,
                    fontweight="bold", color=INK, ha="left", va="top",
                    alpha=1.0 if visible else 0.0)
