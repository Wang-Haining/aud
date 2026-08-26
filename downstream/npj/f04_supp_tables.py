#!/usr/bin/env python3
"""LaTeX bodies for the supplementary tables of the subgroup section.

The section reports summaries; the stratum-level values live in the supplement. Every
number here comes from a file this figure's producers already wrote, so the tables cannot
drift from the panels.

  supp_registry_ratio.tex    registry-scale coverage and its ratio, 15 strata
  supp_chart_strata.tex      chart-scale differences, both readers, both metrics, 18 strata
  supp_reader_spans.tex      spans and counts outside the band, 12 readers and 5 readings
  supp_deprivation_site.tex  the leave-out check against hospital system
  supp_indeterminate.tex     the indeterminate rate by stratum, both readers, 7 axes

Writes to out/figure4/tex/.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

SRC = "downstream/npj/out/figure4"
COV = "downstream/npj/out/figure6"
OUT = "downstream/npj/out/figure4/tex"
CODE = "AUD diagnosis code"
SEX = {"Male": "Men", "Female": "Women"}
NOTE = {"all sites": "All sites", "safety net dropped": "Safety net out",
        "all quintiles": "All quintiles",
        "most deprived dropped": "Most deprived out"}


def f(v, nd=1):
    """A number with LaTeX minus signs, blank when absent."""
    if v is None or not np.isfinite(v):
        return "---"
    return ("%.*f" % (nd, v)).replace("-", "$-$")


def head(cols, axis, first=False):
    """A group heading row. No leading space on the first group: a tabular cannot
    open with a \noalign, and \\input has already started the cell."""
    lead = [] if first else ["\\addlinespace"]
    return lead + ["\\multicolumn{%d}{l}{\\textit{%s}} \\\\" % (cols, axis)]


def registry_ratio(out):
    d = pd.read_csv(os.path.join(SRC, "registry_disparity.csv"))
    lines, last = [], None
    for _, r in d.iterrows():
        if r["axis"] != last:
            lines += head(5, r["axis"], not lines)
            last = r["axis"]
        lines.append("\\quad %s & %s & %s & %s & %s to %s \\\\" % (
            SEX.get(r["level"], r["level"]), "{:,}".format(int(r["registry_n"])),
            f(r["coverage_pct"]),
            f(r["ratio"], 3), f(r["ci_low"], 3), f(r["ci_high"], 3)))
    open(os.path.join(out, "supp_registry_ratio.tex"), "w").write("\n".join(lines) + "\n")
    print("  supp_registry_ratio.tex   %d strata" % len(d))


def chart_strata(out):
    d = pd.read_csv(os.path.join(SRC, "subgroup_disparity.csv"))
    lines, last = [], None
    for _, r in d.iterrows():
        if r["axis"] != last:
            lines += head(6, r["axis"], not lines)
            last = r["axis"]
        cells = ["\\quad %s" % r["level"],
                 "%d / %d" % (int(r["n_sens"]), int(r["n_spec"]))]
        for who in ("code", "llm"):
            for m in ("sens", "spec"):
                v, lo, hi = (r["%s_%s" % (m, who)], r["%s_%s_lo" % (m, who)],
                             r["%s_%s_hi" % (m, who)])
                cells.append("%s (%s to %s)" % (f(v), f(lo), f(hi))
                             if np.isfinite(v) else "---")
        lines.append(" & ".join(cells) + " \\\\")
    open(os.path.join(out, "supp_chart_strata.tex"), "w").write("\n".join(lines) + "\n")
    print("  supp_chart_strata.tex     %d strata" % len(d))


def reader_spans(out):
    m = pd.read_csv(os.path.join(SRC, "model_disparity_summary.csv"))
    p = pd.read_csv(os.path.join(SRC, "protocol_disparity_summary.csv"))
    lines = []
    for title, t, drop_code in (("Model", m, False),
                                ("Prompt and decoding", p, True)):
        lines += head(7, title, not lines)
        for _, r in t.iterrows():
            if drop_code and r["reader"] == CODE:
                continue
            lines.append("\\quad %s & %s & %s & %d & %s & %s & %d \\\\" % (
                r["reader"], f(r["sens_med"]), f(r["sens_span"]), int(r["sens_out"]),
                f(r["spec_med"]), f(r["spec_span"]), int(r["spec_out"])))
    open(os.path.join(out, "supp_reader_spans.tex"), "w").write("\n".join(lines) + "\n")
    print("  supp_reader_spans.tex     %d readers, %d readings" % (len(m), len(p) - 1))


def deprivation_site(out):
    d = pd.read_csv(os.path.join(SRC, "deprivation_site.csv"))
    show = {"AUD diagnosis code": "Code"}
    lines = []
    for _, r in d.iterrows():
        who = "Code" if r["comparison"].startswith(CODE) else "LLM"
        metric = "Specificity" if " spec," in r["comparison"] else "Sensitivity"
        stratum = ("Most deprived" if " spec," in r["comparison"]
                   else "Safety net")
        lines.append("%s & %s & %s & %s & %d / %d & %s & %s & %s (%s to %s) \\\\" % (
            who, metric, stratum, NOTE.get(r["note"], r["note"]), int(r["n_in"]), int(r["n_out"]),
            f(r["rate_in"]), f(r["rate_out"]), f(r["difference"]),
            f(r["ci_low"]), f(r["ci_high"])))
    open(os.path.join(out, "supp_deprivation_site.tex"), "w").write(
        "\n".join(lines) + "\n")
    print("  supp_deprivation_site.tex %d rows" % len(d))
    del show


# The coverage producer names its axes and levels for a panel. The supplement uses the
# vocabulary of the other tables in this section, and orders race as they do.
COV_AXIS = {"Site": "Hospital system", "Gender": "Sex", "Rural or urban": "Residence"}
COV_LEVEL = {"Other": "Other hospitals", "Man": "Men", "Woman": "Women",
             "Non-White": "Other than White"}
COV_ORDER = {"Race": ["Non-White", "White"]}


def indeterminate(out):
    """The indeterminate rate per stratum, the model beside the chart reviewers."""
    d = pd.read_csv(os.path.join(COV, "coverage.csv"))
    parts = []
    for axis in d["axis"].drop_duplicates():
        g = d[d["axis"].eq(axis)]
        if axis in COV_ORDER:
            g = g.set_index("level").loc[COV_ORDER[axis]].reset_index()
        parts.append(g)
    d = pd.concat(parts, ignore_index=True)
    lines, last = [], None
    for _, r in d.iterrows():
        if r["axis"] != last:
            lines += head(4, COV_AXIS.get(r["axis"], r["axis"]), not lines)
            last = r["axis"]
        lines.append("\\quad %s & %d & %s (%s to %s) & %s (%s to %s) \\\\" % (
            COV_LEVEL.get(r["level"], r["level"]), int(r["n"]),
            f(r["model"]), f(r["m_lo"]), f(r["m_hi"]),
            f(r["review"]), f(r["r_lo"]), f(r["r_hi"])))
    open(os.path.join(out, "supp_indeterminate.tex"), "w").write("\n".join(lines) + "\n")
    print("  supp_indeterminate.tex    %d strata" % len(d))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    registry_ratio(a.out)
    chart_strata(a.out)
    reader_spans(a.out)
    deprivation_site(a.out)
    indeterminate(a.out)


if __name__ == "__main__":
    main()
