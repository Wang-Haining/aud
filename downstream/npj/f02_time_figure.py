#!/usr/bin/env python3
"""The registry treatment contrast on a time axis, drawn from fig2_time_to_rx.py.

  a  what the record shows: cumulative incidence of first AUD pharmacotherapy after t0 in
     every new user of each stratum, Aalen-Johansen with death competing, no model. The
     record-silent curve is diluted, because most of that stratum does not have AUD.
  b  what the measurement model recovers: the same curve among patients with
     chart-documented AUD, standardised over the registry, with 95% credible bands.
  c  the contrast against the AUD-code stratum as a risk ratio at each horizon, formed
     within each posterior draw.

Panel a is the plot a reader expects under the name Kaplan-Meier, and it is the one that
misleads: its gap mixes treatment with prevalence. Panel b removes the prevalence, and
panel c is what the paper reports. Numbers at risk sit under panel a.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud after
`fig2_time_to_rx.py --summarise 1`.
"""
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
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

import npj_style as N  # noqa: E402
import render_fig6_strata as RS  # noqa: E402

SN = ["AUD code", "alcohol coded only", "record silent"]
DISPLAY = {"AUD code": "AUD code", "alcohol coded only": "Alcohol coded only",
           "record silent": "Record silent"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--censor", default="admin")
    ap.add_argument("--out", default="downstream/npj/out/figure2_time")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    c = pd.read_csv("output/fig2_time/cif_time_%s.csv" % a.censor)
    N.apply_npj_style()

    fig = plt.figure(figsize=N.mm(180, 76))
    ax_a = fig.add_axes([0.07, 0.40, 0.26, 0.50])
    ax_r = fig.add_axes([0.07, 0.03, 0.26, 0.13])
    ax_b = fig.add_axes([0.42, 0.40, 0.26, 0.50])
    ax_c = fig.add_axes([0.79, 0.40, 0.19, 0.50])

    def q(name, s):
        x = c[(c["quantity"] == name) & (c["stratum"] == s)].sort_values("t")
        return (np.r_[0, x["t"]], np.r_[0, x["mean"]] * 100,
                np.r_[0, x["ci_low"]] * 100, np.r_[0, x["ci_high"]] * 100)

    ymax = 0
    for s in SN:
        col = RS.COL[DISPLAY[s]]
        t, y, _, _ = q("crude_aj_rx", s)
        ax_a.step(t, y, where="post", color=col, lw=1.2, label=DISPLAY[s])
        ax_a.text(t[-1] + 0.08, y[-1], "%.1f%%" % y[-1], color=col, va="center",
                  fontsize=N.FS_SMALL)
        t, y, lo, hi = q("cif_rx", s)
        ax_b.fill_between(t, lo, hi, color=col, alpha=0.18, lw=0)
        ax_b.plot(t, y, color=col, lw=1.2)
        ax_b.text(t[-1] + 0.08, y[-1], "%.1f%%" % y[-1], color=col, va="center",
                  fontsize=N.FS_SMALL)
        ymax = max(ymax, np.nanmax(hi))
    for ax, title in ((ax_a, "Every new user, as recorded"),
                      (ax_b, "Chart-documented AUD, standardised")):
        ax.set_xlim(0, 6)
        ax.set_xticks(range(0, 7))
        ax.set_ylim(0, np.ceil(ymax))
        ax.set_xlabel("Years since cirrhosis diagnosis ($t_0$)")
        ax.set_ylabel("AUD pharmacotherapy (%)")
        ax.set_title(title, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    ax_a.legend(frameon=False, loc="upper left", fontsize=N.FS_SMALL)

    # Numbers at risk.
    ax_r.set_xlim(0, 6)
    ax_r.set_ylim(-0.5, 2.5)
    ax_r.axis("off")
    for k, s in enumerate(SN):
        x = c[(c["quantity"] == "at_risk") & (c["stratum"] == s)].sort_values("t")
        for t, n in zip(x["t"], x["mean"]):
            ax_r.text(t, 2 - k, "{:,d}".format(int(n)), ha="center", va="center",
                      fontsize=N.FS_SMALL, color=RS.COL[DISPLAY[s]])
    ax_r.text(-0.2, 2.9, "Number at risk", fontsize=N.FS_SMALL, color=N.MUTED, ha="left")

    for s, ls in (("record silent", "-"), ("alcohol coded only", (0, (3, 1.5)))):
        col = RS.COL[DISPLAY[s]]
        x = c[(c["quantity"] == "rr") & (c["stratum"] == s)].sort_values("t")
        ax_c.fill_between(x["t"], x["ci_low"], x["ci_high"], color=col, alpha=0.18, lw=0)
        ax_c.plot(x["t"], x["mean"], color=col, lw=1.2, ls=ls,
                  label=DISPLAY[s])
    ax_c.axhline(1, color=N.RULE, lw=N.LW_MIN)
    ax_c.set_yscale("log")
    ax_c.set_ylim(0.8, 30)
    ax_c.set_yticks([1, 2, 5, 10, 20])
    ax_c.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax_c.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_c.set_xlim(0, 6)
    ax_c.set_xticks(range(0, 7))
    ax_c.set_xlabel("Years since $t_0$")
    ax_c.set_ylabel("Risk ratio")
    ax_c.set_title("Risk ratio vs AUD code", loc="left")
    ax_c.spines[["top", "right"]].set_visible(False)
    ax_c.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.62), ncol=1,
                fontsize=N.FS_SMALL)

    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):
        ax.text(-0.22, 1.08, letter, transform=ax.transAxes, fontweight="bold",
                fontsize=N.FS_BODY + 1, va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "figure2_time_%s.%s" % (a.censor, ext)), dpi=450)
    print("wrote %s/figure2_time_%s.{pdf,png}" % (a.out, a.censor))


if __name__ == "__main__":
    main()
