#!/usr/bin/env python3
"""Assemble the registry figure: code status at the cirrhosis diagnosis, who finds the
patients, what they receive, and how many the registry holds.

  a  chart-documented AUD in each code stratum, split by whether an AUD code ever appears
  b  which reader finds those patients, in each stratum
  c  receipt of the three approved AUD medications among patients with true AUD
  d  how many of the 14,406 registry patients each route identifies as carrying AUD

Every panel is cut by one variable, the code stratum at $t_0$, and the figure reads along it:
how many patients in this stratum carry AUD, who can find them, what they receive, and how
many that is at registry scale.

Two dimensions were removed on 14 August. Calendar period left panel a for a supplementary
figure, because the section reports that trend without attributing it and the panel carried
three dimensions at once. The sensitivity-analysis forest left the figure for the same
supplementary figure. Panel b arrived from the external-validation figure, which is cut by
hospital system and had no business carrying a stratum panel.

Denominators differ by panel and each panel states its own. Panels a, c and d are registry
scale; panel b is the reviewed charts.

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
import render_fig6_strata as S                                      # noqa: E402
import matplotlib.pyplot as plt                                     # noqa: E402

import f03_registry_burden as RB                                    # noqa: E402

PANELS = "output/fig6_registry/panels"
DRAWS = "output/fig6_registry/draws_strata3.npy"
SITE = "downstream/npj_goal7_outputs/site_fig_stratum.csv"
OUT = "downstream/npj/out/figure3"

INK, MUTED, RULE = "#1f1e1d", "#55534e", "#B9B7B2"
C_NOTES, C_CODE = "#0072B2", "#D55E00"
SN = ["AUD code", "Alcohol coded only", "Record silent"]
# The stratum names the site producer writes, mapped onto the display names used here.
SITE_SN = {"AUD code": "AUD code", "alcohol coded only": "Alcohol coded only",
           "record silent": "Record silent"}
STYLE = {
    "AUD diagnosis code at $t_0$": (C_CODE, "s"),
    "AUD diagnosis code at any time": (C_CODE, "D"),
    "Hepatologist": (MUTED, "^"),
    "Chart review": (INK, "o"),
    "LLM": (C_NOTES, "o"),
}
ORDER = ["AUD diagnosis code at $t_0$", "AUD diagnosis code at any time",
         "Hepatologist", "Chart review", "LLM"]


def theta_by_u(m, draws_path):
    """Registry-weighted chart-documented AUD per stratum, split on the coding indicator.

    The panel csv carries theta per (stratum, period, coding indicator) cell and a pooled
    value per stratum, but not the value per stratum and indicator pooled over periods, which
    is what the panel needs once the calendar dimension moves out. It is formed here inside
    each posterior draw, so its interval is a posterior interval and not a sum of marginals.
    """
    D = np.load(draws_path)
    key = list(zip(m["S"].to_numpy(int), m["period"].to_numpy(int), m["U"].to_numpy(int)))
    cells = sorted(set(key))
    cellmap = {c: i for i, c in enumerate(cells)}
    thidx = np.array([cellmap[k] for k in key])
    nth = len(cells)
    rows = []
    for s in range(int(m["S"].max()) + 1):
        for u, lab in ((None, "pooled"), (0, "no code ever"), (1, "a code at some time")):
            w = np.zeros(nth)
            for c, i in cellmap.items():
                if c[0] != s or (u is not None and c[2] != u):
                    continue
                w[i] = (thidx == i).sum()
            if w.sum() == 0:          # an AUD code at $t_0$ forces U = 1, so that stratum
                continue              # has no never-coded cell
            n_grp = int(w.sum())
            w = w / w.sum()
            v = np.concatenate([D[c][:, :nth] @ w for c in range(D.shape[0])])
            rows.append({"stratum": SN[s], "group": lab, "n": n_grp,
                         "mean": 100 * v.mean(),
                         "ci_low": 100 * np.percentile(v, 2.5),
                         "ci_high": 100 * np.percentile(v, 97.5)})
    return pd.DataFrame(rows)


def draw_theta(ax, tu):
    """Chart-documented AUD in each stratum, and the split the sampling design forces.

    Charts were drawn stratified on whether an AUD diagnosis code appears at any time, while
    the strata are defined at the cirrhosis diagnosis, so within a stratum the two groups are
    shown separately and never pooled into one marker. The number above each pair is the
    registry-weighted value for the stratum, which is what the text reports.
    """
    xs = np.arange(len(SN))
    for i, stratum in enumerate(SN):
        col = S.COL[stratum]
        d = tu[tu["stratum"] == stratum].set_index("group")
        for grp, off, filled in (("no code ever", -0.17, True),
                                 ("a code at some time", 0.17, False)):
            if grp not in d.index:
                continue
            r = d.loc[grp]
            v, lo, hi = float(r["mean"]), float(r["ci_low"]), float(r["ci_high"])
            ax.errorbar(i + off, v, yerr=[[max(0.0, v - lo)], [max(0.0, hi - v)]],
                        fmt=S.MK[stratum], color=col,
                        markerfacecolor=col if filled else "white",
                        markeredgecolor=col if not filled else "black",
                        markeredgewidth=0.8 if not filled else 0.35,
                        markersize=4.0, elinewidth=0.8, capsize=1.6, capthick=0.5,
                        zorder=3)
            ax.text(i + off, 3.0, "%s" % "{:,d}".format(int(r["n"])), ha="center",
                    va="bottom", fontsize=4.6, color=col)
        if "pooled" in d.index:
            p = d.loc["pooled"]
            ax.text(i, 104, "%.1f%%" % float(p["mean"]), ha="center", va="bottom",
                    fontsize=6.2, color=col, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(["AUD code", "Alcohol\ncoded only", "Record\nsilent"], fontsize=5.8)
    ax.set_xlim(-0.55, len(SN) - 0.45)
    ax.set_ylim(0, 114)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Chart-documented AUD (%)")
    ax.set_xlabel("Structured record at $t_0$")
    ax.plot([], [], "o", color=INK, markersize=3.0, markeredgecolor="black",
            markeredgewidth=0.35, label="no AUD code at any time")
    ax.plot([], [], "o", color="white", markeredgecolor=INK, markeredgewidth=0.8,
            markersize=3.0, label="an AUD code at some time")
    ax.legend(loc="upper left", bbox_to_anchor=(0.015, 0.60), fontsize=4.8,
              handlelength=0.8, handletextpad=0.35, frameon=False, labelspacing=0.35)
    ax.text(0.030, 0.34, "counts are registry patients", transform=ax.transAxes,
            fontsize=4.8, color=MUTED, va="top", ha="left")
    ax.set_title("Chart-documented AUD in each stratum\nbold, the registry-weighted value",
                 fontsize=6.5, color="#333333", pad=3)
    return ax


def draw_detection(ax, st):
    """Which reader finds the reference-positive patients, inside each stratum.

    Pooled over the three hospital systems; the open marks on each bar are the systems. The
    record-silent stratum is defined by the structured record holding no alcohol code at
    $t_0$, so the code can only reach those patients by acquiring one after the index date.
    """
    st = st.copy()
    st["stratum"] = st["stratum"].map(SITE_SN).fillna(st["stratum"])
    sites = [s for s in sorted(set(st["site"])) if s != "ALL"]
    xs = np.arange(len(SN))
    W = 0.34
    for who, off, col in (("LLM", -W / 2, C_NOTES), ("AUD diagnosis code", W / 2, C_CODE)):
        for i, stratum in enumerate(SN):
            g = st[(st["stratum"] == stratum) & (st["instrument"] == who)]
            p = g[g["site"] == "ALL"]
            if not len(p):
                continue
            p = p.iloc[0]
            v, lo, hi = float(p["sens"]), float(p["ci_low"]), float(p["ci_high"])
            x = i + off
            ax.bar(x, v, width=W * 0.92, color=col, alpha=0.85, edgecolor="black",
                   linewidth=0.4, zorder=2)
            ax.errorbar(x, v, yerr=[[max(0.0, v - lo)], [max(0.0, hi - v)]], fmt="none",
                        ecolor="black", elinewidth=0.6, capsize=1.6, capthick=0.5, zorder=4)
            ax.text(x, hi + 1.2, "%.0f%%" % v, ha="center", va="bottom", fontsize=5.6,
                    color=col, fontweight="bold")
            for q, site in enumerate(sites):
                r = g[g["site"] == site]
                if len(r):
                    ax.plot(x + (q - 1) * 0.075, float(r["sens"].iloc[0]), "o",
                            color="white", markeredgecolor=INK, markeredgewidth=0.5,
                            markersize=2.4, zorder=5)
            if who == "LLM":
                ax.text(i, 2.5, "n=%d" % int(p["n"]), ha="center", va="bottom",
                        fontsize=5.0, color="white", zorder=6)
    ax.set_xticks(xs)
    ax.set_xticklabels(["AUD code", "Alcohol\ncoded only", "Record\nsilent"], fontsize=5.8)
    ax.set_xlim(-0.55, len(SN) - 0.45)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Sensitivity (%)")
    ax.set_xlabel("Structured record at $t_0$")
    ax.plot([], [], "s", color=C_NOTES, markersize=3.6, label="LLM (Qwen3.6-27B)")
    ax.plot([], [], "s", color=C_CODE, markersize=3.6, label="AUD diagnosis code")
    ax.plot([], [], "o", color="white", markeredgecolor=INK, markeredgewidth=0.5,
            markersize=2.4, label="each system")
    ax.legend(loc="lower left", fontsize=5.0, handlelength=0.8, handletextpad=0.35,
              frameon=False, ncol=3, bbox_to_anchor=(-0.02, -0.36), columnspacing=0.9)
    ax.set_title("Reference-positive charts each route finds\n"
                 "pooled over the three hospital systems",
                 fontsize=6.5, color="#333333", pad=3)
    return ax


def draw_burden(ax, r):
    """How many registry patients each route identifies, on one count axis."""
    tab = pd.concat([r["enumerated"], r["primary"]], ignore_index=True).set_index("arm")
    N = r["n_registry"]
    y = np.arange(len(ORDER))[::-1]

    ax.axvline(r["model_total"], color=RULE, lw=0.7, ls=(0, (2.5, 2.0)), zorder=0)
    ax.text(r["model_total"] + 110, len(ORDER) - 1.55,
            "model in a\n%s" % "{:,.0f}".format(r["model_total"]),
            fontsize=5.0, color=MUTED, ha="left", va="center")

    for yy, name in zip(y, ORDER):
        row = tab.loc[name]
        col, mk = STYLE[name]
        est = row["count"]
        if not np.isnan(row["lo"]):
            ax.plot([row["lo"], row["hi"]], [yy, yy], color=col, lw=1.0, zorder=2)
            for b in (row["lo"], row["hi"]):
                ax.plot([b, b], [yy - 0.12, yy + 0.12], color=col, lw=1.0, zorder=2)
        else:
            ax.plot([0, est], [yy, yy], color=col, lw=0.7, alpha=0.35, zorder=1)
        ax.plot(est, yy, mk, color=col, markersize=4.0, markeredgecolor="black",
                markeredgewidth=0.35, zorder=3)
        lab = "{:,.0f}  ({:.1f}%)".format(est, 100 * row["fraction"])
        ax.text(1.02, yy, lab, transform=ax.get_yaxis_transform(), ha="left",
                va="center", fontsize=5.6, clip_on=False)

    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("AUD diagnosis code", "AUD code") for n in ORDER],
                       fontsize=5.9)
    ax.set_xlim(0, 7200)
    ax.set_xticks([0, 2000, 4000, 6000])
    ax.set_xticklabels(["0", "2,000", "4,000", "6,000"])
    ax.set_ylim(-0.6, len(ORDER) - 0.30)
    ax.set_xlabel("Registry patients identified as having AUD")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("All %s registry patients\nintervals are carried from the sample"
                 % "{:,d}".format(N), fontsize=6.5, color="#333333", pad=3)
    return ax


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panels", default=PANELS)
    ap.add_argument("--draws", default=DRAWS)
    ap.add_argument("--site", default=SITE)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    P = Path(a.panels)
    os.makedirs(a.out, exist_ok=True)

    ra = pd.read_csv(P / "f6s_rates.csv")
    st = pd.read_csv(a.site)
    burden = RB.build()
    m = RB.B.build()
    tu = theta_by_u(m, a.draws)

    fig = plt.figure(figsize=(7.09, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.66,
                          left=0.09, right=0.88, top=0.905, bottom=0.105)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc, axd = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    draw_theta(axa, tu)
    draw_detection(axb, st)
    S.draw_b(axc, ra)
    draw_burden(axd, burden)
    for ax, lab, dx in ((axa, "a", -0.24), (axb, "b", -0.24),
                        (axc, "c", -0.24), (axd, "d", -0.60)):
        S.lab(ax, lab, dx)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "figure3.%s" % ext))
    plt.close(fig)
    print("wrote %s/figure3.pdf and .png" % a.out)
    print("\npanel a, chart-documented AUD by stratum and coding indicator")
    print(tu.to_string(index=False, float_format=lambda x: "%.1f" % x))
    print("\npanel b, sensitivity by stratum, pooled")
    print(st[st["site"] == "ALL"][["stratum", "instrument", "n", "sens", "ci_low",
                                   "ci_high"]]
          .to_string(index=False, float_format=lambda x: "%.1f" % x))
    print("\npanel c, standardised treatment")
    print(ra.to_string(index=False, float_format=lambda x: "%.2f" % x))
    p = burden["paired"]
    print("\npanel d, LLM minus chart review: %+.0f (%+.0f to %+.0f)"
          % (p["difference"], p["lo"], p["hi"]))
    print("panel d, model total: %.0f (%.0f to %.0f)"
          % ((burden["model_total"],) + (burden["model_ci"] or (np.nan, np.nan))))


if __name__ == "__main__":
    main()
