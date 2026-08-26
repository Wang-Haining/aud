#!/usr/bin/env python3
"""Figure 6: what the diagnosis code misses, where it misses, and what it costs.

Three strata at the cirrhosis diagnosis, by what the structured record contains:
  1  an AUD diagnosis code
  2  an alcohol-attributed code but no AUD diagnosis code
  3  no alcohol code of any kind

Chart review supplies P(true AUD | stratum). The 14,406-patient registry supplies the
treatment rate. Stratum 2 is a documentation gap: chart-confirmed AUD is as high as in
stratum 1, so an AUD code adds no detection information there. Stratum 3 is the detection
gap, and it holds 70% of the registry.

Contrasts are prognostic associations. No model output reached a clinician.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.direction": "out", "ytick.direction": "out",
    "lines.linewidth": 0.8, "lines.markersize": 3, "legend.frameon": False,
    "legend.borderpad": 0.3, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
S1, S2, S3 = "#0072B2", "#E8B25A", "#D55E00"
COL = {"AUD code": S1, "Alcohol coded only": S2, "Record silent": S3}
MK = {"AUD code": "o", "Alcohol coded only": "^", "Record silent": "s"}
PN = ["2015-16", "2017-18", "2019+"]
ORDER = ["AUD code", "Alcohol coded only", "Record silent"]
SHORT = {"AUD code": "AUD code", "Alcohol coded only": "Alcohol only", "Record silent": "Silent"}
SENS = {"acamp_disulf": "Acamprosate or disulfiram",
        "narrow_signal": "Narrow alcohol-signal definition",
        "theta_pooled": "$\\theta$ pooled across periods",
        "collapse_check": "Collapsed, sampler check"}


def lab(ax, s, dx=-0.18):
    ax.text(dx, 1.02, s, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left")


def draw_a(ax, th):
    x = np.arange(len(PN))
    for i, s in enumerate(ORDER):
        d = th[th["stratum"] == s].set_index("period").reindex(PN)
        y, lo, hi = (100 * d[c].to_numpy() for c in ("mean", "ci_low", "ci_high"))
        off = (i - 1) * 0.16
        ax.errorbar(x + off, y, yerr=[y - lo, hi - y], fmt=MK[s], color=COL[s],
                    markeredgecolor="black", markeredgewidth=0.35, markersize=4.2,
                    elinewidth=0.8, capsize=2.0, capthick=0.6, label=SHORT[s])
    ax.set_xticks(x)
    ax.set_xticklabels(PN)
    ax.set_xlim(-0.5, len(PN) - 0.5)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Period of the cirrhosis diagnosis")
    ax.set_ylabel("Chart-confirmed AUD (%)")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.42), handlelength=1.0,
              handletextpad=0.35)
    ax.set_title("Chart review supplies this\nn = 487 reviewed charts",
                 fontsize=6.5, color="#333333", pad=3)


def draw_b(ax, ra):
    d = ra.set_index("stratum")
    for i, s in enumerate(ORDER):
        r = d.loc[s]
        ax.bar(i, r["pct"], width=0.55, color=COL[s], edgecolor="black", linewidth=0.4)
        ax.errorbar(i, r["pct"], yerr=[[r["pct"] - r["ci_low"]], [r["ci_high"] - r["pct"]]],
                    fmt="none", ecolor="black", elinewidth=0.7, capsize=2.4, capthick=0.6)
        ax.text(i, r["ci_high"] + 0.25, "%.1f%%" % r["pct"], ha="center", va="bottom",
                fontsize=6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["AUD\ncode", "Alcohol\ncoded only", "Record\nsilent"], fontsize=6)
    ax.set_ylabel("Received AUD pharmacotherapy (%)")
    ax.set_ylim(0, 15.5)
    r1 = d.loc["risk ratio vs alcohol coded only"]
    r2 = d.loc["risk ratio vs record silent"]
    for y, xa, xb, r in ((12.4, 0, 1, r1), (14.0, 0, 2, r2)):
        ax.plot([xa, xa, xb, xb], [y, y + 0.25, y + 0.25, y], color="black", linewidth=0.5)
        ax.text((xa + xb) / 2, y + 0.32, "%.2f (%.2f–%.2f)"
                % (r["pct"], r["ci_low"], r["ci_high"]), ha="center", va="bottom", fontsize=5.7)
    ax.set_title("Among patients with true AUD\nrisk ratios against the AUD-code stratum",
                 fontsize=6.5, color="#333333", pad=3)


def draw_c(ax, ra, se):
    rows = [("Primary, vs record silent",
             ra.set_index("stratum").loc["risk ratio vs record silent"]),
            ("Primary, vs alcohol coded only",
             ra.set_index("stratum").loc["risk ratio vs alcohol coded only"])]
    items = [(n, r["pct"], r["ci_low"], r["ci_high"]) for n, r in rows]
    if se is not None:
        # show the record-silent contrast for each variant; that is the detection claim
        sub = se[se["contrast"].isin(["silent", "collapsed"])] if "contrast" in se.columns else se
        for _, r in sub.iterrows():
            items.append((SENS.get(r["variant"], r["variant"]), r["mean"],
                          r["ci_low"], r["ci_high"]))
    y = np.arange(len(items))[::-1]
    ax.axvline(1.0, color="#999999", linewidth=0.6, linestyle=":")
    for yy, (nm, e, lo, hi) in zip(y, items):
        c = S1 if nm.startswith("Primary") else "#404040"
        lw = 1.1 if nm.startswith("Primary") else 0.9
        ax.plot([lo, hi], [yy, yy], color=c, linewidth=lw)
        for b in (lo, hi):
            ax.plot([b, b], [yy - 0.11, yy + 0.11], color=c, linewidth=lw)
        ax.plot(e, yy, "o", color=c, markersize=4.0, markeredgecolor="black",
                markeredgewidth=0.35, zorder=3)
        ax.text(1.02, yy, "%.2f (%.2f–%.2f)" % (e, lo, hi),
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=5.6, clip_on=False)
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items], fontsize=5.9)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 70)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("Risk ratio (log scale)")
    ax.set_ylim(-0.6, len(items) - 0.35)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Record-silent contrast under four specifications",
                 fontsize=6.5, color="#333333", pad=3)


def draw_d(ax, co):
    x = np.arange(len(PN))
    bottom = np.zeros(len(PN))
    for s in ORDER:
        v = co[co["stratum"] == s].set_index("period").reindex(PN)["pct"].fillna(0).to_numpy()
        n = co[co["stratum"] == s].set_index("period").reindex(PN)["n"].fillna(0).to_numpy()
        ax.bar(x, v, bottom=bottom, width=0.62, color=COL[s], edgecolor="black",
               linewidth=0.4, label=SHORT[s])
        for xi, vi, bi, ni in zip(x, v, bottom, n):
            if vi >= 8:
                ax.text(xi, bi + vi / 2, "%.0f%%\nn=%d" % (vi, ni), ha="center",
                        va="center", fontsize=5.5,
                        color="white" if s != "Alcohol coded only" else "#3A2A00")
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(PN)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Period of the cirrhosis diagnosis")
    ax.set_ylabel("Registry patients (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3,
              handlelength=1.0, handletextpad=0.35, columnspacing=1.0)
    ax.set_title("All 14,406 registry patients", fontsize=6.5, color="#333333", pad=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="output/fig6_registry/panels")
    ap.add_argument("--out", default="output/fig6_registry/figures")
    a = ap.parse_args()
    P, outdir = Path(a.panels), Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    th = pd.read_csv(P / "f6s_theta.csv")
    ra = pd.read_csv(P / "f6s_rates.csv")
    co = pd.read_csv(P / "f6s_compose.csv")
    sp = P / "f6s_sens.csv"
    se = pd.read_csv(sp) if sp.exists() else None

    fig = plt.figure(figsize=(7.09, 5.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.62, wspace=0.66,
                  left=0.09, right=0.90, top=0.92, bottom=0.115)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc, axd = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    draw_a(axa, th)
    lab(axa, "a", -0.24)
    draw_b(axb, ra)
    lab(axb, "b", -0.24)
    draw_c(axc, ra, se)
    lab(axc, "c", -0.72)
    draw_d(axd, co)
    lab(axd, "d", -0.24)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / ("Fig6_strata.%s" % ext))
    print("wrote %s/Fig6_strata.pdf and .png" % outdir)
    print("\nlegend numbers")
    for _, r in ra.iterrows():
        print("  %-34s %6.2f (%.2f-%.2f)" % (r["stratum"], r["pct"], r["ci_low"], r["ci_high"]))
    for _, r in th.iterrows():
        print("  theta %-20s %-8s %5.1f%% (%.1f-%.1f)" % (r["stratum"], r["period"],
              100 * r["mean"], 100 * r["ci_low"], 100 * r["ci_high"]))


if __name__ == "__main__":
    main()
