#!/usr/bin/env python3
"""External validation across three hospital systems.

  a  weighted kappa for the primary model, pooled and at each system
  b  weighted kappa for all 11 models at each system, one line per model
  c  sensitivity of the LLM and of the AUD diagnosis code at each system, with the
     paired gain printed beside it

Every panel is cut by one variable, the hospital system, and the figure reads along it:
whether agreement holds away from the academic centre, whether that is one model's
property or the leaderboard's, and whether the separation between the two readers holds.

The stratum panel left this figure on 14 August. It is cut by the structured record at
$t_0$, not by system, and it now sits in the registry figure beside the other panels that
share that axis.

Data comes from downstream/site_figure_data.py, which reads the closed reference standard.
Colours and sizes follow downstream/npj/nature_style.py so that this figure and the others
are one system: the Wong colourblind-safe palette, blue for what the notes give and
vermillion for what the structured record gives.

The file prefix is the figure number in the manuscript. It was swapped with the
other one on 14 August: the prefixes had followed a proposed reorganisation that was
never applied, so f02 built Figure 3 and f03 built Figure 2.

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
import matplotlib.pyplot as plt                                     # noqa: E402
import nature_style as NS                                           # noqa: E402

SRC = "downstream/npj_goal7_outputs"
OUT = "downstream/npj/out/figure2"

SITES = ["ALL", "IUH", "Eskenazi", "Other"]
# The three systems are named as the manuscript names them: an academic health system, a
# safety-net hospital, and a group of other contributing hospitals. The third is "other",
# not "community"; it is a residual group, not a hospital type.
LABEL = {"ALL": "All systems", "IUH": "Academic", "Eskenazi": "Safety net",
         "Other": "Other"}
FOOTNOTE = ("Academic, the academic health system.  Safety net, the safety-net hospital.  "
            "Other, the group of other contributing hospitals.")
PRIMARY = "Qwen3.6-27B"
C_NOTES, C_CODE = "#0072B2", "#D55E00"
INK, MUTED, RULE = "#1f1e1d", "#55534e", "#B9B7B2"
GREY = "#9d9b97"


def _sites(df, drop_all=False):
    s = [x for x in SITES if x in set(df["site"])]
    return [x for x in s if x != "ALL"] if drop_all else s


def draw_kappa(ax, k):
    """Weighted kappa for the primary model, pooled and at each system."""
    k = k[k["is_primary"]].set_index("site")
    order = _sites(k.reset_index())
    xs = np.arange(len(order))
    ax.axhline(float(k.loc["ALL", "kappa_w"]), color=RULE, lw=0.6, ls=(0, (2.0, 2.0)),
               zorder=0)
    for i, s in enumerate(order):
        r = k.loc[s]
        y, lo, hi = float(r["kappa_w"]), float(r["ci_low"]), float(r["ci_high"])
        ax.errorbar(i, y, yerr=[[y - lo], [hi - y]], fmt="o", color=C_NOTES,
                    markerfacecolor=C_NOTES if s != "ALL" else "white",
                    markeredgecolor=C_NOTES, markeredgewidth=0.7,
                    markersize=4.2, elinewidth=0.8, capsize=1.8, capthick=0.6, zorder=3)
        ax.text(i, hi + 0.012, "%.3f" % y, ha="center", va="bottom", fontsize=5.2,
                color=INK)
        ax.text(i, 0.632, "n=%d" % int(r["n"]), ha="center", va="bottom", fontsize=5.0,
                color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[s] for s in order], fontsize=5.8, rotation=20, ha="right")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0.63, 1.01)
    ax.set_ylabel("Weighted $\\kappa$")
    ax.set_title("Agreement holds at every system", fontsize=6.5, color="#333333", pad=3)
    return ax


def draw_models(ax, k):
    """Every model's kappa at each system, one line per model."""
    order = _sites(k, drop_all=True)
    xs = np.arange(len(order))
    w = k.pivot_table(index="display_name", columns="site", values="kappa_w",
                      aggfunc="first").reindex(columns=order)
    for name, row in w.iterrows():
        if row.isna().any():
            continue
        prim = name == PRIMARY
        ax.plot(xs, row.to_numpy(float), "-o",
                color=C_NOTES if prim else GREY,
                lw=1.1 if prim else 0.6, alpha=1.0 if prim else 0.55,
                markersize=3.4 if prim else 1.9,
                markeredgecolor="black" if prim else "none",
                markeredgewidth=0.3 if prim else 0,
                zorder=4 if prim else 2, label=PRIMARY if prim else None)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[s] for s in order], fontsize=5.8, rotation=20, ha="right")
    ax.set_xlim(-0.35, len(order) - 0.65)
    ax.set_ylabel("Weighted $\\kappa$")
    ax.legend(loc="lower left", fontsize=5.2, handlelength=1.0, handletextpad=0.35,
              frameon=False, bbox_to_anchor=(0.0, 0.10))
    ax.text(0.98, 0.115, "grey, the other 10 models", transform=ax.transAxes,
            fontsize=5.0, color=MUTED, va="bottom", ha="right")
    ax.set_title("The leaderboard is stable across systems",
                 fontsize=6.5, color="#333333", pad=3)
    return ax


def draw_primary(ax, p, pr):
    """Sensitivity of the two readers at each system, with the paired gain."""
    p = p.copy()
    p["who"] = np.where(p["instrument"] == "instrument", "LLM", "AUD code")
    order = _sites(p)
    ys = np.arange(len(order))[::-1]
    gain = pr.set_index("site")
    for y, s in zip(ys, order):
        for who, off, col in (("LLM", 0.15, C_NOTES), ("AUD code", -0.15, C_CODE)):
            r = p[(p["site"] == s) & (p["who"] == who)].iloc[0]
            v, lo, hi = float(r["sens"]), float(r["sens_lo"]), float(r["sens_hi"])
            ax.plot([lo, hi], [y + off, y + off], color=col, lw=1.0, zorder=2)
            ax.plot(v, y + off, "o", color=col, markersize=4.0, markeredgecolor="black",
                    markeredgewidth=0.3, zorder=3)
            ax.text(v, y + off + 0.11, "%.1f" % v, ha="center", va="bottom", fontsize=5.0,
                    color=col)
        if s in gain.index:
            g = gain.loc[s]
            ax.text(1.03, y, "%+.1f  (%+.1f to %+.1f)"
                    % (g["sens_gain"], g["gain_lo"], g["gain_hi"]),
                    transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=5.2, color=INK, clip_on=False)
    # The gain column is a text column of the same forest, so it carries its own header
    # rather than relying on the caption to say what the numbers on the right are.
    ax.text(1.03, len(order) - 0.52, "Paired gain (95% CI)",
            transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.2,
            color=MUTED, clip_on=False)
    ax.set_yticks(ys)
    # The denominator of a sensitivity is the reference-positive charts, n_pos, not every
    # chart with a determinate label, n. The two differ by roughly a factor of two here.
    ax.set_yticklabels(["%s\n%d with AUD" % (LABEL[s], int(p[p["site"] == s]["n_pos"]
                                                           .iloc[0]))
                        for s in order], fontsize=5.8)
    ax.set_ylim(-0.6, len(order) - 0.32)
    ax.set_xlim(48, 104)
    ax.set_xlabel("Sensitivity (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.plot([], [], "o", color=C_NOTES, markersize=3.6, label="LLM (Qwen3.6-27B)")
    ax.plot([], [], "o", color=C_CODE, markersize=3.6, label="AUD diagnosis code")
    ax.legend(loc="upper left", fontsize=5.2, handlelength=0.8, handletextpad=0.35,
              frameon=False, ncol=2, bbox_to_anchor=(-0.01, -0.14), columnspacing=1.2)
    ax.set_title("The two readers separate at every system, and by the most "
                 "where the code is weakest", fontsize=6.5, color="#333333", pad=3)
    return ax


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    P = Path(a.src)
    os.makedirs(a.out, exist_ok=True)
    NS.apply_nature_style()
    # apply_nature_style turns on constrained layout and a tight bounding box. This figure
    # places the paired gain outside the axes of panel c, and the layout engine shrinks the
    # axes to make room for it until the plotting area collapses to a line. The gridspec
    # below sets the margins explicitly instead.
    matplotlib.rcParams["figure.constrained_layout.use"] = False
    matplotlib.rcParams["savefig.bbox"] = None

    k = pd.read_csv(P / "site_fig_kappa.csv")
    p = pd.read_csv(P / "site_fig_primary.csv")
    pr = pd.read_csv(P / "site_fig_paired.csv")

    # Panels a and b are small and share a y axis of the same quantity, so they sit side by
    # side. Panel c is a forest with a text column and needs the full width.
    fig = plt.figure(figsize=(7.09, 4.5))
    gs = fig.add_gridspec(2, 2, wspace=0.42, hspace=0.72, left=0.085, right=0.825,
                          top=0.895, bottom=0.135, height_ratios=[1.0, 1.05])
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, :])
    draw_kappa(axa, k)
    draw_models(axb, k)
    draw_primary(axc, p, pr)
    # Panel labels are placed in figure coordinates. In axes coordinates they land on the
    # y tick labels of panel c and on the titles, because the panels have different left
    # margins.
    for x, y, lab in ((0.020, 0.980, "a"), (0.470, 0.980, "b"), (0.020, 0.487, "c")):
        fig.text(x, y, lab, fontsize=8, fontweight="bold", va="top", ha="left")
    fig.text(0.085, 0.004, FOOTNOTE, fontsize=5.0, color=MUTED, va="bottom", ha="left")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "figure2.%s" % ext), dpi=300)
    plt.close(fig)
    print("wrote %s/figure2.pdf and .png" % a.out)
    print("\npanel a, weighted kappa, primary model")
    print(k[k["is_primary"]][["site", "n", "kappa_w", "ci_low", "ci_high"]]
          .to_string(index=False, float_format=lambda x: "%.3f" % x))
    print("\npanel b, spread across the 11 models")
    w = k.pivot_table(index="display_name", columns="site", values="kappa_w")
    print(w[[c for c in ["IUH", "Eskenazi", "Other"] if c in w.columns]]
          .to_string(float_format=lambda x: "%.3f" % x))
    print("\npanel c, sensitivity and the paired gain")
    print(p[["site", "instrument", "n", "n_pos", "sens", "sens_lo", "sens_hi"]]
          .to_string(index=False, float_format=lambda x: "%.1f" % x))
    print(pr[["site", "n", "sens_gain", "gain_lo", "gain_hi", "p_exact"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
