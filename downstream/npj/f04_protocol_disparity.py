#!/usr/bin/env python3
"""Subgroup flatness under four prespecified variations of the protocol.

The right half of panel d of Figure 4. The left half asks whether the flatness measured on
the primary model belongs to models in general, and answers it across eleven. This asks the
other question a reader has: whether it belongs to the way the model was asked. The five
conditions are the ones Figure 3c already reports for overall agreement, read by the primary
model on the same 487 charts: the frozen protocol at greedy decoding, the same prompt at
temperature 0.3 and 0.7, and two prespecified paraphrases of it.

Same estimator as f04_model_disparity: for each of the 18 strata, the condition's
sensitivity in the stratum minus its sensitivity in the rest of that axis, in points. The
summary carries the smallest, the median, the largest, the span and the number of strata
outside the band.

Sensitivity only. The 174 reference-negative charts leave two strata with 17 and 18 charts,
where one chart moves specificity by 5.9 and 5.6 points, so a band of 5 points is not
reachable there by arithmetic and a span of specificity differences measures the count
behind each stratum rather than the reader. The specificity differences are written to the
csv for the supplement and are not drawn.

The frozen condition is the primary model, so its row reproduces the primary row of the left
half. That is the intended check, not a duplication: it ties the two halves to one number.

Writes out/figure4/protocol_disparity.csv and protocol_disparity_summary.csv.

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
import f04_model_disparity as MD                                          # noqa: E402

RATERS = "output/raters/rater_long.csv"
OUT = "downstream/npj/out/figure4"
CODE = "AUD diagnosis code"
BAND = 5.0

# The five conditions of Figure 3c, in that order, and the names the manuscript uses.
CONDITIONS = [
    ("goal5_qwen_v1_t0p0", "T = 0"),
    ("goal5_qwen_v1_t0p3", "T = 0.3"),
    ("goal5_qwen_v1_t0p7", "T = 0.7"),
    ("goal5_qwen_paraphrase_a_t0p0", "Paraphrase A"),
    ("goal5_qwen_paraphrase_b_t0p0", "Paraphrase B"),
]


def frame():
    """One row per chart: the reference label, the code, and the five condition labels."""
    r = pd.read_csv(RATERS, dtype=str)
    r = r[r["in_analysis_cohort"].eq("True")]
    w = r.pivot_table(index="omop_person_id", columns="rater", values="label_3",
                      aggfunc="first")
    missing = [t for t, _ in CONDITIONS if t not in w.columns]
    if missing:
        raise SystemExit("rater_long lacks %s" % ", ".join(missing))
    d = pd.DataFrame({"pid": w.index.astype(str), "gold": w["gold"].values})
    for tag, _ in CONDITIONS:
        d[tag] = w[tag].values
    d["primary"] = w["aud_qwen36_27b"].values
    ros = pd.read_csv(MD.ROSTER, dtype={"omop_person_id": str})
    d = d.merge(ros[["omop_person_id", "has_aud_icd"]].rename(
        columns={"omop_person_id": "pid"}), on="pid", how="left")
    d["code"] = d["has_aud_icd"].astype(str).str.lower().isin(["true", "1", "1.0"])
    d["person_id"] = d["pid"].astype("int64")
    d["llm"] = d["primary"]
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = SR.attach(frame())
    same = int((d["goal5_qwen_v1_t0p0"] == d["primary"]).sum())
    print("  frozen condition agrees with the primary model on %d of %d charts"
          % (same, len(d)))
    pos = d[d["gold"].eq("AUD+")].copy()
    neg = d[d["gold"].eq("AUD-")].copy()
    print("  reference-positive %d, reference-negative %d" % (len(pos), len(neg)))

    readers = [(CODE, None)] + [(name, tag) for tag, name in CONDITIONS]
    rows = []
    for name, tag in readers:
        for col, axis, levels in SR.AXES:
            idx = {lev: i for i, lev in enumerate(levels)}
            for key, fr, what in (("sens", pos, "sensitivity"),
                                  ("spec", neg, "specificity")):
                f = fr[fr[col].isin(levels)]
                codes = f[col].map(idx).to_numpy().astype(int)
                if tag is None:
                    ok = (f["code"].astype(bool) if key == "sens"
                          else ~f["code"].astype(bool)).to_numpy(dtype=float)
                else:
                    ok = (f[tag].eq("AUD+") if key == "sens"
                          else f[tag].eq("AUD-")).to_numpy(dtype=float)
                diff = MD.vs_complement(codes, ok, len(levels))
                for lvl, v in zip(levels, diff):
                    rows.append({"reader": name, "tag": tag or "code", "axis": axis,
                                 "level": lvl, "metric": what, "difference": v,
                                 "n": int((f[col] == lvl).sum())})

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.out, "protocol_disparity.csv"), index=False)

    srows = []
    for name, _ in readers:
        g = t[t["reader"].eq(name)]
        row = {"reader": name}
        for key, what in (("sens", "sensitivity"), ("spec", "specificity")):
            v = g[g["metric"].eq(what)]["difference"].to_numpy()
            v = v[np.isfinite(v)]
            row[key + "_min"] = v.min()
            row[key + "_med"] = float(np.median(v))
            row[key + "_max"] = v.max()
            row[key + "_out"] = int((np.abs(v) > BAND).sum())
            row[key + "_span"] = v.max() - v.min()
            row[key + "_n"] = len(v)
        srows.append(row)
    s = pd.DataFrame(srows)
    s.to_csv(os.path.join(a.out, "protocol_disparity_summary.csv"), index=False)
    print()
    print(s.to_string(index=False, float_format=lambda v: "%.2f" % v))
    v = s[s["reader"].ne(CODE)]
    print("\n  sensitivity: code spans %.1f, conditions span %.1f to %.1f; "
          "outside the band, code %d, conditions %d to %d"
          % (float(s[s["reader"].eq(CODE)]["sens_span"].iloc[0]),
             v["sens_span"].min(), v["sens_span"].max(),
             int(s[s["reader"].eq(CODE)]["sens_out"].iloc[0]),
             int(v["sens_out"].min()), int(v["sens_out"].max())))


if __name__ == "__main__":
    main()
