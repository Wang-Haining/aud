#!/usr/bin/env python3
"""Subgroup differences for all eleven models and for the diagnosis code.

Producer for the fourth panel of Figure 4. The chart-scale panel shows that the primary
model's differences across strata are small where the code's are not. That leaves the
question of whether the flatness belongs to the primary model or to the approach, which is
the same question the model leaderboard answers for accuracy.

For each reader and each of the 18 strata, the difference in percentage points between that
stratum and the rest of its axis, computed on the reference-positive charts for sensitivity
and on the reference-negative charts of the same stratum for specificity. Zero is no
difference.

One convention, the one Fig. 1 uses: on a chart the reference standard placed, a reader is
correct only if it returns that direction, so an indeterminate output counts against it.

The panel summarises each reader by the smallest, the median and the largest of its 18
differences, and by how many of them fall outside a band of plus or minus 5 points. These
are descriptive. The smallest and the largest of 18 estimates are biased away from zero even
when every stratum is identical, by an amount that depends on the counts rather than on the
reader, so the comparison the panel supports is between readers on the same 18 strata and
not between one reader's two halves or between strata.

Writes out/figure4/model_disparity.csv, one row per reader and stratum, and
model_disparity_summary.csv.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "downstream")
sys.path.insert(0, "downstream/npj")
import f04_subgroup_rates as SR                                           # noqa: E402
from f04_figure4 import DISPLAY                                           # noqa: E402

RATERS = "output/raters/rater_long.csv"
ROSTER = "resources/patient_roster.csv"
LEADER = "downstream/npj/out/figure4/panel_a_leaderboard.csv"
OUT = "downstream/npj/out/figure4"
CODE = "AUD diagnosis code"
BAND = 5.0


def frame():
    """One row per chart: the reference label, the code, and all eleven model labels."""
    r = pd.read_csv(RATERS, dtype=str)
    r = r[r["in_analysis_cohort"].eq("True")]
    w = r.pivot_table(index="omop_person_id", columns="rater", values="label_3",
                      aggfunc="first")
    missing = [t for t in DISPLAY if t not in w.columns]
    if missing:
        raise SystemExit("rater_long lacks %s" % ", ".join(missing))
    d = pd.DataFrame({"pid": w.index.astype(str), "gold": w["gold"].values})
    for tag in DISPLAY:
        d[tag] = w[tag].values
    ros = pd.read_csv(ROSTER, dtype={"omop_person_id": str})
    d = d.merge(ros[["omop_person_id", "has_aud_icd"]].rename(
        columns={"omop_person_id": "pid"}), on="pid", how="left")
    d["code"] = d["has_aud_icd"].astype(str).str.lower().isin(["true", "1", "1.0"])
    d["person_id"] = d["pid"].astype("int64")
    d["llm"] = d["aud_qwen36_27b"]
    return d


def vs_complement(codes, ok, k):
    n = np.bincount(codes, minlength=k).astype(float)
    s = np.bincount(codes, weights=ok, minlength=k)
    tot_n, tot_s = n.sum(), s.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        inside = np.where(n > 0, s / n, np.nan)
        outside = np.where(tot_n - n > 0, (tot_s - s) / (tot_n - n), np.nan)
    return 100.0 * (inside - outside)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = SR.attach(frame())
    pos = d[d["gold"].eq("AUD+")].copy()
    neg = d[d["gold"].eq("AUD-")].copy()
    print("  reference-positive %d, reference-negative %d" % (len(pos), len(neg)))

    readers = [(CODE, None)] + [(DISPLAY[t], t) for t in DISPLAY]
    rows = []
    for name, tag in readers:
        for col, axis, levels in SR.AXES:
            idx = {lev: i for i, lev in enumerate(levels)}
            for tag2, frame_, what in (("sens", pos, "sensitivity"),
                                       ("spec", neg, "specificity")):
                f = frame_[frame_[col].isin(levels)]
                codes = f[col].map(idx).to_numpy().astype(int)
                if tag is None:
                    ok = (f["code"].astype(bool) if tag2 == "sens"
                          else ~f["code"].astype(bool)).to_numpy(dtype=float)
                else:
                    ok = (f[tag].eq("AUD+") if tag2 == "sens"
                          else f[tag].eq("AUD-")).to_numpy(dtype=float)
                diff = vs_complement(codes, ok, len(levels))
                for lvl, v in zip(levels, diff):
                    rows.append({"reader": name, "tag": tag or "code", "axis": axis,
                                 "level": lvl, "metric": what, "difference": v,
                                 "n": int((f[col] == lvl).sum())})

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.out, "model_disparity.csv"), index=False)

    order = [CODE] + list(pd.read_csv(LEADER)["name"])
    srows = []
    for name in order:
        g = t[t["reader"].eq(name)]
        if g.empty:
            print("  no rows for %s" % name)
            continue
        rec = {"reader": name}
        for tag2, what in (("sens", "sensitivity"), ("spec", "specificity")):
            v = g[g["metric"].eq(what)]["difference"].to_numpy()
            rec["%s_min" % tag2] = np.nanmin(v)
            rec["%s_med" % tag2] = np.nanmedian(v)
            rec["%s_max" % tag2] = np.nanmax(v)
            rec["%s_out" % tag2] = int(np.sum(np.abs(v) > BAND))
            rec["%s_span" % tag2] = np.nanmax(v) - np.nanmin(v)
        srows.append(rec)
        print("  %-19s sensitivity %6.1f to %6.1f (span %5.1f, %2d of 18 outside)  "
              "specificity %6.1f to %6.1f (span %5.1f, %2d outside)"
              % (name, rec["sens_min"], rec["sens_max"], rec["sens_span"],
                 rec["sens_out"], rec["spec_min"], rec["spec_max"],
                 rec["spec_span"], rec["spec_out"]))

    s = pd.DataFrame(srows)
    s.to_csv(os.path.join(a.out, "model_disparity_summary.csv"), index=False)
    m = s[s["reader"].ne(CODE)]
    print("\n  the code spans %.1f points on sensitivity and %.1f on specificity"
          % (s.loc[s["reader"].eq(CODE), "sens_span"].iloc[0],
             s.loc[s["reader"].eq(CODE), "spec_span"].iloc[0]))
    print("  the eleven models span %.1f to %.1f on sensitivity and %.1f to %.1f on "
          "specificity" % (m["sens_span"].min(), m["sens_span"].max(),
                           m["spec_span"].min(), m["spec_span"].max()))
    print("  models whose sensitivity span is under the code's: %d of %d"
          % (int((m["sens_span"] < s.loc[s["reader"].eq(CODE), "sens_span"].iloc[0]).sum()),
             len(m)))
    print("  overall difference range %.1f to %.1f points"
          % (t["difference"].min(), t["difference"].max()))
    print("  wrote %s/model_disparity.csv and model_disparity_summary.csv" % a.out)


if __name__ == "__main__":
    main()
