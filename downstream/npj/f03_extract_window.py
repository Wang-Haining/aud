#!/usr/bin/env python3
"""What the 2015 floor on $t_0$ is, and what depends on it.

`fig6_bayes_strata.MIN_DATE` is 2005-01-01 and the OMOP extract holds no condition
row before 2015-01-01, so that filter never binds. The Methods sentence "the first
cirrhosis code on or after 1 January 2015" therefore states the extract boundary and
not a cohort-entry rule, and $t_0$ is left-censored at that boundary: a patient whose
first cirrhosis code fell in 2013 enters with $t_0$ in early 2015.

Two consequences are measured here.

Composition. The count of first cirrhosis codes in 2015 is larger than in any later
year. Cases prevalent at the extract boundary are assigned a $t_0$ there, so the 2015
stratum mixes them with incident cases and the later strata do not.

Lookback. The registry strata are defined at $t_0$: stratum 0 requires an AUD code
dated at or before the first cirrhosis code. A patient with $t_0$ in January 2015 has
no record before $t_0$ in which such a code could appear, and a patient with $t_0$ in
2020 has five years of it. The share of patients in stratum 0 therefore rises with the
year of $t_0$ whether or not coding practice changed. This file reports that share
beside the mean lookback, so the two can be told apart.

The reviewed charts are reported on the same axis, because Supplementary Figure 8
bands them 2015--2016, 2017--2018 and 2019--2020 and reports the sensitivity of the
diagnosis code rising across the three. The first band is where the boundary sits.

The code indicator on the reviewed charts comes from the roster's `has_aud_icd`,
which is an AUD code anywhere in the record rather than at or before $t_0$, so the
lookback argument above does not apply to it in the same direction. It is reported
so the two definitions are not confused.

This file writes and draws nothing else. It is a diagnostic, in the sense of
`f03_frame_probe.py`, except that it writes its numbers to disk.

    python3 downstream/npj/f03_extract_window.py

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "downstream")
sys.path.insert(0, str(HERE))

import fig6_bayes_strata as B                                      # noqa: E402
import ref_standard as RS                                          # noqa: E402

OUT = "downstream/npj/out/f03_registry"
ROSTER = "resources/patient_roster.csv"
FLOOR = pd.Timestamp("2015-01-01")


def extract_window():
    """The observed span of the extract, from the table the strata are built on."""
    co = pd.read_parquet(B.D + "/condition_occurrence.parquet",
                         columns=["condition_start_date"])
    s = pd.to_datetime(co["condition_start_date"], errors="coerce")
    return {"min_date_constant": str(B.MIN_DATE)[:10],
            "first_condition_row": str(s.min())[:10],
            "last_condition_row": str(s.max())[:10],
            "rows": int(len(s)),
            "rows_before_2015": int((s < FLOOR).sum())}


def registry_by_year(reg):
    """Stratum share and mean lookback, by the calendar year of t0."""
    reg = reg.copy()
    reg["year"] = reg["t0"].dt.year
    reg["lookback_days"] = (reg["t0"] - FLOOR).dt.days
    rows = []
    for y, g in reg.groupby("year"):
        n = len(g)
        rows.append({
            "t0_year": int(y),
            "n": n,
            "mean_lookback_days": round(float(g["lookback_days"].mean()), 1),
            "coded_at_t0": int((g["S"] == 0).sum()),
            "coded_at_t0_pct": round(100.0 * float((g["S"] == 0).mean()), 2),
            "alcohol_only_pct": round(100.0 * float((g["S"] == 1).mean()), 2),
            "silent_pct": round(100.0 * float((g["S"] == 2).mean()), 2),
            "period": int(g["period"].iloc[0]),
        })
    return pd.DataFrame(rows).sort_values("t0_year")


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    z, p = 1.959963985, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def reviewed_by_year(reg):
    """The reviewed charts on the same axis as Supplementary Figure 8.

    Sensitivity of the index test and of the diagnosis code among the reference-positive
    charts of each t0 year, and of each band with 2015 removed. Definitions are the ones
    f02_supp_era.py uses, so the two are comparable.
    """
    ref, man = RS.build()
    assert len(ref) == 487, "reference standard is not the 487"
    d = ref[["omop_person_id", "gold_binary"]].copy()
    d["pid"] = pd.to_numeric(d["omop_person_id"], errors="coerce").astype("int64")

    ros = pd.read_csv(ROSTER)
    ros["pid"] = pd.to_numeric(ros["omop_person_id"], errors="coerce").astype("Int64")
    d = (d.merge(reg[["pid", "t0", "period", "S"]], on="pid", how="left")
          .merge(ros[["pid", "has_aud_icd"]], on="pid", how="left"))
    d["year"] = d["t0"].dt.year
    d["ref_pos"] = d["gold_binary"].eq("AUD+")
    d["code_pos"] = d["has_aud_icd"].astype("Int64").fillna(0).astype(int).eq(1)

    rows = []

    def add(label, g):
        pos = g[g["ref_pos"]]
        cs = wilson(int(pos["code_pos"].sum()), len(pos))
        rows.append({"group": label, "n_charts": len(g), "n_ref_pos": len(pos),
                     "code_sens": round(cs[0], 4) if len(pos) else np.nan,
                     "code_lo": round(cs[1], 4) if len(pos) else np.nan,
                     "code_hi": round(cs[2], 4) if len(pos) else np.nan,
                     "coded_at_t0_pct": round(100.0 * float((g["S"] == 0).mean()), 2)
                     if len(g) else np.nan})

    for y in sorted(d["year"].dropna().unique()):
        add("t0 %d" % int(y), d[d["year"].eq(y)])
    for k, name in {0: "2015-2016", 1: "2017-2018", 2: "2019-2020"}.items():
        add("band %s" % name, d[d["period"].eq(k)])
    add("band 2015-2016 without 2015", d[d["period"].eq(0) & d["year"].ne(2015)])
    add("2016-2020, 2015 removed", d[d["year"].between(2016, 2020)])
    add("2015 only", d[d["year"].eq(2015)])
    add("no t0", d[d["t0"].isna()])
    return pd.DataFrame(rows)


def coding_trend(reg):
    """The trend the Results reports, on the whole registry and on a matched lookback.

    manuscript.tex:538 gives the share of registry patients with an AUD code at the
    cirrhosis diagnosis rising from 10.3 to 21.8 per cent across the three periods. That
    share is the stratum-0 share, and stratum 0 requires an AUD code dated at or before
    $t_0$. The window in which such a code can appear is the distance from the extract
    floor to $t_0$, so it is 132 days on average in 2015 and 2,010 days in 2020.

    Three versions are reported. As published, which is the whole registry. With the 2015
    boundary year removed, which leaves every remaining patient at least a year of
    lookback. And on a matched window, where the qualifying code must fall inside the 365
    days before $t_0$ and every patient has those 365 days, so the window is the same width
    for everyone.
    """
    reg = reg.copy()
    reg["year"] = reg["t0"].dt.year
    reg["lookback_days"] = (reg["t0"] - FLOOR).dt.days
    has_aud_date = "aud" in reg.columns
    if has_aud_date:
        reg["coded_365"] = (reg["aud"].notna()
                            & (reg["aud"] <= reg["t0"])
                            & (reg["aud"] >= reg["t0"] - pd.Timedelta(days=365)))

    rows = []

    def add(version, frame, col):
        for p in (0, 1, 2):
            g = frame[frame["period"].eq(p)]
            rows.append({"version": version, "period": p, "n": len(g),
                         "coded_pct": round(100.0 * float(g[col].mean()), 2)
                         if len(g) else np.nan})

    reg["coded_at_t0"] = reg["S"].eq(0)
    add("as published, whole registry", reg, "coded_at_t0")
    add("2015 boundary year removed", reg[reg["year"].ne(2015)], "coded_at_t0")
    if has_aud_date:
        add("matched 365-day window, t0 from 2016",
            reg[reg["lookback_days"] >= 365], "coded_365")
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    win = extract_window()
    print("extract window")
    for k, v in win.items():
        print("  %-22s %s" % (k, v))
    pd.DataFrame([win]).to_csv(os.path.join(OUT, "extract_window.csv"), index=False)

    reg = B.build()
    reg["pid"] = reg["person_id"].astype("int64")
    print("\nregistry, %d patients with a cirrhosis code" % len(reg))
    ry = registry_by_year(reg)
    print(ry.to_string(index=False))
    ry.to_csv(os.path.join(OUT, "extract_window_registry_by_year.csv"), index=False)

    # 2021 holds 25 patients and 2022 holds 2, so a correlation over all eight years is
    # decided by 27 of 14,406. It is reported over the six years that carry the registry.
    six = ry[ry["t0_year"].between(2015, 2020)]
    print("\n2015 to 2020 carry %d of %d patients. Over those six years the rank "
          "correlation between mean lookback and the coded share is %.3f"
          % (int(six["n"].sum()), int(ry["n"].sum()),
             six["mean_lookback_days"].corr(six["coded_at_t0_pct"], method="spearman")))

    print("\nthe trend the Results reports")
    ct = coding_trend(reg)
    print(ct.to_string(index=False))
    ct.to_csv(os.path.join(OUT, "extract_window_coding_trend.csv"), index=False)

    print("\nreviewed charts")
    rv = reviewed_by_year(reg)
    print(rv.to_string(index=False))
    rv.to_csv(os.path.join(OUT, "extract_window_reviewed.csv"), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
