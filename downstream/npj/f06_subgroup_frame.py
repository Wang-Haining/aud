#!/usr/bin/env python3
"""What each proposed subgroup can and cannot answer, before any analysis is run.

The team proposed twelve subgroup axes. This file counts them on the two denominators
that matter and reports, for every level, the width of the interval the design can
deliver. A level whose interval is 30 points wide cannot answer a question about a
10-point difference, and that is worth knowing before the analysis rather than after.

Two denominators, and they differ by a factor of fifty.

  reviewed charts   487 with a reference label, of which 251 are reference-positive.
                    Sensitivity of a reader is estimated on the 251. This is the only
                    denominator on which the LLM can be measured, because notes were
                    obtained for the sampled patients only.
  registry          14,406 patients with a cirrhosis code. The AUD diagnosis code is
                    observed on all of them, so the code's sensitivity by subgroup is
                    estimable at this scale once chart review calibrates it.

Axes that come from the structured record carry to the registry. Axes the LLM read out
of the notes do not: they exist for the sampled patients only. The team's list contains
both kinds and the table names which is which.

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

import ref_standard as RS                                          # noqa: E402
import fig6_bayes_strata as B                                      # noqa: E402
import fig6_bayes_me_v2 as M                                       # noqa: E402

OUT = "downstream/npj/out/subgroups"
PANEL = "output/canonical_panel.csv"
PRIMARY_TAG = "aud_qwen36_27b"

# Ascites, bleeding varices and hepatic encephalopathy. This is the narrow definition of
# the three the denominator probe compared on 13 August; the wider two add spontaneous
# bacterial peritonitis, hepatorenal syndrome and the cirrhosis codes that name a
# complication, and they move the split by a few points.
DECOMP = r"^(R18|I85\.01|I85\.11|K72|789\.5|456\.0|456\.20|572\.2)"
COVID = pd.Timestamp("2020-03-01")


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    z, p = 1.959963985, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h))


def build_axes(a):
    """One row per sampled patient, carrying every axis the team asked for."""
    ref, man = RS.build()
    ref = ref[["omop_person_id", "gold_binary"]].copy()
    ref["pid"] = pd.to_numeric(ref["omop_person_id"], errors="coerce").astype("int64")

    reg = B.build()
    reg["pid"] = reg["person_id"].astype("int64")

    # Decompensation at or before the cirrhosis diagnosis, for the whole registry.
    co = pd.read_parquet(M.D + "/condition_occurrence.parquet",
                         columns=["person_id", "condition_source_value",
                                  "condition_start_date"])
    co["pid"] = co["person_id"].astype("int64")
    co = co[co["pid"].isin(set(reg["pid"]))]
    co["d"] = pd.to_datetime(co["condition_start_date"], errors="coerce")
    co = co[co["d"].notna()]
    icd = co["condition_source_value"].astype(str).str.extract(r"\^\^([^\^]+)\^")[0]
    co["icd"] = icd.fillna(co["condition_source_value"].astype(str))
    co = co.merge(reg[["pid", "t0"]], on="pid", how="inner")
    hit = co[co["d"].le(co["t0"]) & co["icd"].str.match(DECOMP, na=False)]
    decomp = set(hit["pid"])

    person = pd.read_parquet(M.D + "/person.parquet",
                             columns=["person_id", "race_concept_id"])
    person["pid"] = person["person_id"].astype("int64")
    RACE = {8527: "White", 8516: "Black"}

    r = reg[["pid", "age", "male", "period", "t0", "val"]].copy()
    r["decompensation"] = np.where(r["pid"].isin(decomp), "Decompensated", "Compensated")
    r = r.merge(person[["pid", "race_concept_id"]], on="pid", how="left")
    r["race_3"] = r["race_concept_id"].map(RACE).fillna("Other or unknown")
    r["race_2"] = np.where(r["race_3"].eq("White"), "White", "Non-White")
    r["gender"] = np.where(r["male"].astype(float) > 0, "Man", "Woman")
    r["age_bracket"] = pd.cut(r["age"], [-np.inf, 39, 64, np.inf],
                              labels=["under 40", "40 to 64", "65 and over"]
                              ).astype("string")
    # The cut every reported age stratification uses. f04_subgroup_rates cuts at 50 and
    # the registry-scale panel cuts at 50, so the coverage axes of f06_figure6 cut at 50
    # as well: one age grouping across the paper. age_bracket is kept for the axis
    # inventory, which records what the record carries rather than what is reported.
    r["age_bracket_50"] = pd.cut(r["age"], [-np.inf, 49, 64, np.inf],
                                 labels=["Under 50", "50 to 64", "65 and over"]
                                 ).astype("string")
    r["covid_era"] = np.where(r["t0"] < COVID, "Index before Mar 2020",
                              "Index Mar 2020 onward")
    r["covid_era_2018"] = np.where(
        r["t0"] < pd.Timestamp("2018-01-01"), "Index before 2018",
        r["covid_era"])

    # Note-derived axes exist for the sampled patients only.
    sd = pd.read_parquet("output/sdoh/aggregate_recoded.parquet")
    sd["pid"] = pd.to_numeric(sd["person_id"], errors="coerce").astype("Int64")
    geo = pd.read_parquet("output/geo/adi_ruca.parquet")
    geo["pid"] = pd.to_numeric(geo["omop_person_id"], errors="coerce").astype("Int64")
    inst = pd.read_csv("output/fairness/patient_institution.csv")
    inst["pid"] = pd.to_numeric(inst["person_id"], errors="coerce").astype("Int64")
    SITE = {"IUH": "Academic", "Eskenazi": "Safety net"}
    inst["site"] = inst["institution"].map(SITE).fillna("Other")

    p = pd.read_csv(a.panel, dtype={"omop_person_id": str})
    p = p[p["model"].eq(PRIMARY_TAG)][["omop_person_id", "label_canonical"]].copy()
    p["pid"] = pd.to_numeric(p["omop_person_id"], errors="coerce").astype("int64")

    ros = pd.read_csv("resources/patient_roster.csv")
    ros["pid"] = pd.to_numeric(ros["omop_person_id"], errors="coerce").astype("Int64")

    s = (ref.merge(r, on="pid", how="left")
            .merge(sd[["pid", "insurance", "tobacco", "housing"]], on="pid", how="left")
            .merge(geo[["pid", "ruca_category", "adi_quintile"]], on="pid", how="left")
            .merge(inst[["pid", "site"]], on="pid", how="left")
            .merge(ros[["pid", "has_aud_icd"]], on="pid", how="left")
            .merge(p[["pid", "label_canonical"]], on="pid", how="left"))
    POS = {"Definitely Yes", "Probably Yes"}
    s["llm_pos"] = s["label_canonical"].isin(POS)
    s["code_pos"] = s["has_aud_icd"].astype("Int64").fillna(0).astype(int).eq(1)
    s["ref_pos"] = s["gold_binary"].eq("AUD+")
    return s, r, man


AXES = [
    ("Site", "site", "structured"),
    ("Gender", "gender", "structured"),
    ("Race, two levels", "race_2", "structured"),
    ("Race, three levels", "race_3", "structured"),
    ("Age bracket", "age_bracket", "structured"),
    ("Decompensation at $t_0$", "decompensation", "structured"),
    ("Index relative to COVID", "covid_era", "structured"),
    ("Index relative to COVID, from 2018", "covid_era_2018", "structured"),
    ("Insurance", "insurance", "read from notes"),
    ("Rural, urban or suburban", "ruca_category", "from ZIP"),
    # ADI is carried here so its absence is on the record. The cohort supplies ZIP5 and
    # the Neighborhood Atlas does not validate an ADI linkage at ZIP5 or ZCTA, so the
    # axis has no levels for any patient.
    ("Area deprivation quintile", "adi_quintile", "not linkable"),
    ("Tobacco", "tobacco", "read from notes"),
    ("Housing", "housing", "read from notes"),
]


def table(s, r):
    rows = []
    for label, col, source in AXES:
        if col not in s.columns:
            continue
        # A structured axis is missing only for the two sampled charts that carry no
        # cirrhosis code and so are outside the registry frame. Saying "not documented"
        # for those two would read as a data-quality problem rather than a frame one.
        fill = ("outside the registry frame" if source == "structured"
                else "not documented")
        v = s[col].astype("string").fillna(fill)
        for lev in sorted(v.dropna().unique()):
            g = s[v.eq(lev)]
            pos = g[g["ref_pos"]]
            n_pos = len(pos)
            code = wilson(int(pos["code_pos"].sum()), n_pos)
            llm = wilson(int(pos["llm_pos"].sum()), n_pos)
            reg_n = (int(r[col].astype("string").eq(lev).sum())
                     if source == "structured" and col in r.columns else np.nan)
            rows.append({
                "axis": label, "level": lev, "source": source,
                "charts": len(g), "reference_positive": n_pos,
                "code_sens": code[0], "code_lo": code[1], "code_hi": code[2],
                "code_halfwidth": (code[2] - code[1]) / 2 if n_pos else np.nan,
                "llm_sens": llm[0], "llm_lo": llm[1], "llm_hi": llm[2],
                "gain": llm[0] - code[0] if n_pos else np.nan,
                "registry": reg_n,
            })
    d = pd.DataFrame(rows)
    # A level answers a 10-point question only if its interval is narrower than that.
    d["verdict"] = np.where(
        d["reference_positive"] < 10, "too few positives to estimate",
        np.where(d["code_halfwidth"] > 15, "estimable, interval too wide to compare",
                 "can support a 10-point comparison"))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    s, r, man = build_axes(a)
    d = table(s, r)
    d.to_csv(os.path.join(a.out, "subgroup_feasibility.csv"), index=False)

    print("reviewed charts with a reference label: %d" % len(s))
    print("  by reference label: %s"
          % s["gold_binary"].value_counts().to_dict())
    print("registry patients: %d" % len(r))
    print("\nwrote %s/subgroup_feasibility.csv\n" % a.out)

    show = d.copy()
    for c in ("code_sens", "llm_sens", "gain", "code_halfwidth"):
        show[c] = show[c].round(1)
    show["registry"] = show["registry"].map(
        lambda x: "" if pd.isna(x) else "{:,d}".format(int(x)))
    print(show[["axis", "level", "source", "charts", "reference_positive",
                "code_sens", "code_halfwidth", "llm_sens", "gain", "registry",
                "verdict"]].to_string(index=False))

    print("\nsummary")
    print("  levels that can support a 10-point comparison: %d of %d"
          % ((d["verdict"] == "can support a 10-point comparison").sum(), len(d)))
    print("  axes that carry to the registry: %s"
          % ", ".join(sorted(d.loc[d["source"] == "structured", "axis"].unique())))
    print("  axes that exist for the sampled patients only: %s"
          % ", ".join(sorted(d.loc[d["source"] != "structured", "axis"].unique())))


if __name__ == "__main__":
    main()
