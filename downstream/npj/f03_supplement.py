#!/usr/bin/env python3
"""The two panels the registry figure gave up, each in a figure cut by its own variable.

Writes two files.

  calendar.pdf    chart-documented AUD and registry composition by the period of the
                  cirrhosis diagnosis
    a, b, c  one panel per code stratum: period on the x axis, the coding indicator as
             the marker fill
    d        how the three strata compose the registry in each period

  specifications.pdf  the treatment contrast under every prespecified and added
                      specification
    a  risk ratio against the record-silent stratum
    b  risk ratio against the alcohol-coded-only stratum

The registry figure carried the calendar dimension inside its first panel, which put
stratum, period and coding indicator on one set of axes. Splitting by stratum leaves one
organising axis per figure, the period, and the coding indicator stays visible as the
marker fill.

The specification forest was the fourth panel of the registry figure. It showed the
record-silent contrast alone and titled itself "four specifications" over more rows than
that. Both contrasts are shown here and every variant is named.

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
import render_fig6_strata as S                                      # noqa: E402
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.lines import Line2D                                 # noqa: E402

PANELS = "output/fig6_registry/panels"
OUT = "downstream/npj/out/figure3"

INK, MUTED, RULE = "#1f1e1d", "#55534e", "#B9B7B2"
PN = ["2015-16", "2017-18", "2019+"]
ORDER = ["AUD code", "Alcohol coded only", "Record silent"]
SHORT = {"AUD code": "AUD code at $t_0$",
         "Alcohol coded only": "Alcohol coded only at $t_0$",
         "Record silent": "Record silent at $t_0$"}

# Every variant that the fit writes, named as the Methods names it. The registry figure
# mapped four of these and printed the raw tag for the rest.
SENS = {
    "primary": "Primary analysis",
    "acamp_disulf": "Acamprosate or disulfiram only",
    "narrow_signal": "Alcohol-associated liver disease alone",
    "theta_pooled": "$\\theta$ pooled across periods",
    "everrx": "Treatment counted at any time",
    "uroster": "Sampling indicator from the roster",
    "s400_100_50": "Inner chain 100 steps, 50 discarded",
    "joint": "Full joint posterior",
    # t0 is left-censored at the extract floor of 1 January 2015, so the 2015
    # entrants carry a mean of 132 days before t0 against 2,010 for the 2020
    # entrants, and stratum membership is defined at t0. This fit drops them.
    "t0from2016": "$t_0$ in 2016 or later",
}
# The order the Methods lists them: prespecified first, then the two added after the
# reference standard closed, then the joint posterior.
SORDER = ["primary", "acamp_disulf", "narrow_signal", "theta_pooled", "everrx",
          "uroster", "s400_100_50", "t0from2016", "joint"]
CONTRAST = {"silent": "against the record-silent stratum",
            "alcohol": "against the alcohol-coded-only stratum"}


# ----------------------------------------------------------------- calendar

def draw_stratum_period(ax, th, stratum, n_charts):
    """One stratum: chart-documented AUD in each period, split on the coding indicator."""
    col = S.COL[stratum]
    d = th[(th["stratum"] == stratum) & (th["ever_coded"].isin([0, 1]))]
    x = np.arange(len(PN))
    for u, off, filled in ((0, -0.10, True), (1, 0.10, False)):
        g = d[d["ever_coded"] == u].set_index("period").reindex(PN)
        if g["mean"].isna().all():
            continue
        y, lo, hi = (100 * g[c].to_numpy(float) for c in ("mean", "ci_low", "ci_high"))
        ok = ~np.isnan(y)
        ax.errorbar(x[ok] + off, y[ok],
                    yerr=[np.maximum(0.0, (y - lo)[ok]), np.maximum(0.0, (hi - y)[ok])],
                    fmt=S.MK[stratum], color=col,
                    markerfacecolor=col if filled else "white",
                    markeredgecolor=col if not filled else "black",
                    markeredgewidth=0.8 if not filled else 0.35,
                    markersize=4.0, elinewidth=0.8, capsize=1.8, capthick=0.55,
                    zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(PN, fontsize=5.8)
    ax.set_xlim(-0.5, len(PN) - 0.5)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Period of the cirrhosis diagnosis")
    ax.set_ylabel("Chart-documented AUD (%)")
    ax.set_title("%s\n%d validated charts in this stratum" % (SHORT[stratum], n_charts),
                 fontsize=6.5, color="#333333", pad=3)
    return ax


def draw_compose(ax, co):
    """How the three strata compose the registry in each period."""
    x = np.arange(len(PN))
    bottom = np.zeros(len(PN))
    for s in ORDER:
        g = co[co["stratum"] == s].set_index("period").reindex(PN)
        v = g["pct"].fillna(0).to_numpy(float)
        n = g["n"].fillna(0).to_numpy(float)
        ax.bar(x, v, bottom=bottom, width=0.62, color=S.COL[s], edgecolor="black",
               linewidth=0.4, label=s)
        for xi, vi, bi, ni in zip(x, v, bottom, n):
            if vi >= 8:
                ax.text(xi, bi + vi / 2, "%.0f%%\n%s" % (vi, "{:,d}".format(int(ni))),
                        ha="center", va="center", fontsize=5.5,
                        color="white" if s != "Alcohol coded only" else "#3A2A00")
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(PN, fontsize=5.8)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Period of the cirrhosis diagnosis")
    ax.set_ylabel("Registry patients (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=5.2,
              handlelength=0.9, handletextpad=0.35, frameon=False, columnspacing=1.0)
    ax.set_title("All 14,406 registry patients\ncounts inside the bars",
                 fontsize=6.5, color="#333333", pad=3)
    return ax


def figure_calendar(th, co, counts, out):
    fig = plt.figure(figsize=(7.09, 5.0))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.42, left=0.085, right=0.975,
                          top=0.895, bottom=0.145)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, s in zip(axes[:3], ORDER):
        draw_stratum_period(ax, th, s, counts.get(s, 0))
    # One legend for the three stratum panels; the fills mean the same thing in each. It
    # is built from explicit proxy marks, because the AUD-code panel holds only the open
    # series and its own handles would explain half the figure.
    handles = [Line2D([], [], marker="o", linestyle="none", color=INK, markersize=3.0,
                      markeredgecolor="black", markeredgewidth=0.35),
               Line2D([], [], marker="o", linestyle="none", color="white",
                      markeredgecolor=INK, markeredgewidth=0.8, markersize=3.0)]
    axes[0].legend(handles, ["no AUD code at any time", "an AUD code at some time"],
                   loc="lower left", bbox_to_anchor=(0.0, 0.02), fontsize=4.8,
                   handlelength=0.8, handletextpad=0.35, frameon=False,
                   labelspacing=0.35)
    draw_compose(axes[3], co)
    for ax, lab in zip(axes, "abcd"):
        S.lab(ax, lab, -0.22)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "figure3_calendar.%s" % ext))
    plt.close(fig)


# ------------------------------------------------------------ specifications

def draw_forest(ax, items, title, xlim, ticks):
    y = np.arange(len(items))[::-1]
    ax.axvline(1.0, color="#999999", linewidth=0.6, linestyle=":")
    for yy, (nm, e, lo, hi) in zip(y, items):
        prim = nm == SENS["primary"]
        c = S.COL["AUD code"] if prim else "#404040"
        lw = 1.2 if prim else 0.9
        ax.plot([lo, hi], [yy, yy], color=c, linewidth=lw, zorder=2)
        for b in (lo, hi):
            ax.plot([b, b], [yy - 0.11, yy + 0.11], color=c, linewidth=lw, zorder=2)
        ax.plot(e, yy, "o", color=c, markersize=4.0, markeredgecolor="black",
                markeredgewidth=0.35, zorder=3)
        ax.text(1.03, yy, "%.2f (%.2f–%.2f)" % (e, lo, hi),
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=5.4, clip_on=False, fontweight="bold" if prim else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items], fontsize=5.7)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    # A log axis labels its minor ticks when the decade is short, which puts "4 x 10^0"
    # next to the named ticks.
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Risk ratio (log scale)")
    ax.set_ylim(-0.6, len(items) - 0.35)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=6.5, color="#333333", pad=3)
    return ax


def figure_specifications(ra, se, out):
    d = ra.set_index("stratum")
    prim = {"silent": d.loc["risk ratio vs record silent"],
            "alcohol": d.loc["risk ratio vs alcohol coded only"]}
    panels = {}
    for contrast in ("silent", "alcohol"):
        items = []
        for tag in SORDER:
            if tag == "primary":
                r = prim[contrast]
                items.append((SENS[tag], float(r["pct"]), float(r["ci_low"]),
                              float(r["ci_high"])))
                continue
            g = se[(se["variant"] == tag) & (se["contrast"] == contrast)]
            if not len(g):
                continue
            r = g.iloc[0]
            items.append((SENS[tag], float(r["mean"]), float(r["ci_low"]),
                          float(r["ci_high"])))
        panels[contrast] = items

    # The two contrasts stack rather than sit side by side: each forest carries a value
    # column on its right, and side by side that column lands on the other panel's row
    # labels.
    fig = plt.figure(figsize=(7.09, 4.3))
    gs = fig.add_gridspec(2, 1, hspace=0.52, left=0.235, right=0.845,
                          top=0.915, bottom=0.135)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    draw_forest(axa, panels["silent"], "Risk ratio %s" % CONTRAST["silent"],
                (0.85, 130), [1, 2, 4, 8, 16, 32, 64])
    draw_forest(axb, panels["alcohol"], "Risk ratio %s" % CONTRAST["alcohol"],
                (0.85, 3.2), [1, 1.5, 2, 2.5, 3])
    S.lab(axa, "a", -0.36)
    S.lab(axb, "b", -0.36)
    # The collapsed check estimates a different contrast, one uncoded stratum against the
    # AUD-code stratum, so it cannot sit on either axis above.
    cc = se[se["variant"] == "collapse_check"]
    if len(cc):
        r = cc.iloc[0]
        fig.text(0.235, 0.012,
                 "Internal check, a different contrast: the two uncoded strata collapsed "
                 "into one gives a risk ratio of %.2f (%.2f–%.2f) for that single "
                 "uncoded stratum against the AUD-code stratum."
                 % (r["mean"], r["ci_low"], r["ci_high"]),
                 fontsize=5.0, color=MUTED, va="bottom", ha="left")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "figure3_specifications.%s" % ext))
    plt.close(fig)
    return panels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panels", default=PANELS)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    P = Path(a.panels)
    os.makedirs(a.out, exist_ok=True)

    th = pd.read_csv(P / "f6s_theta.csv")
    co = pd.read_csv(P / "f6s_compose.csv")
    ra = pd.read_csv(P / "f6s_rates.csv")
    se = pd.read_csv(P / "f6s_sens.csv")

    # The validated-chart count per stratum comes from the model frame, not from a
    # hand-typed constant; the registry figure carried a stale one.
    import f03_registry_burden as RB
    m = RB.B.build()
    v = m[m["val"].astype(bool)]           # val marks the validation subsample
    counts = {ORDER[s]: int((v["S"] == s).sum()) for s in range(3)}

    figure_calendar(th, co, counts, a.out)
    panels = figure_specifications(ra, se, a.out)
    print("wrote %s/figure3_calendar.pdf and figure3_specifications.pdf" % a.out)
    print("\nvalidated charts per stratum: %s (total %d)"
          % (counts, sum(counts.values())))
    print("\ncalendar, chart-documented AUD by stratum, period and coding indicator")
    t = th[th["ever_coded"].isin([0, 1])].copy()
    t[["mean", "ci_low", "ci_high"]] *= 100
    print(t[["stratum", "period", "ever_coded", "mean", "ci_low", "ci_high"]]
          .to_string(index=False, float_format=lambda x: "%.1f" % x))
    for contrast, items in panels.items():
        print("\nspecifications, %s" % CONTRAST[contrast])
        for nm, e, lo, hi in items:
            print("  %-42s %6.2f (%.2f-%.2f)" % (nm, e, lo, hi))


if __name__ == "__main__":
    main()
