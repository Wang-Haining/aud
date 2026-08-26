#!/usr/bin/env python3
"""Table 1, in the layout of the R table1 package.

Continuous variables are summarised as mean (SD) and median [min, max]; categorical
variables as n (%). A Missing row appears wherever a variable is incomplete, so the
denominator of every percentage is visible rather than implied. Percentages are within
column.

Columns are the AUD diagnosis code strata, because that is what the sample was drawn
on: the reviewed charts are a stratified sample of the registry with roughly equal
numbers coded and uncoded, so a single undifferentiated column would describe a
population that does not exist. An overall column is given for completeness and should
not be read as a registry estimate.

No significance tests. The two columns are sampling strata of known and unequal
fractions, so a p-value would test the sampling plan rather than a hypothesis about
patients.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import re
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "downstream")
import fig6_bayes_strata as B  # noqa: E402
import stats_core as sc  # noqa: E402

OUT = "downstream/npj_table1"
L6 = ["Definitely Yes", "Probably Yes", "Uncertain", "Probably No", "Definitely No",
      "Not Enough Data"]
YES = {"Definitely Yes", "Probably Yes"}
NO = {"Definitely No", "Probably No"}
# Human-readable levels for fields that arrive as machine labels.
RELABEL = {
    "not_documented": "Not documented", "dual_medicaid_medicare": "Dual eligible",
    "medicaid": "Medicaid", "medicare": "Medicare", "private": "Private",
    "uninsured": "Uninsured", "other": "Other",
    "stable": "Stable", "unstable_or_homeless": "Unstable or homeless",
    "Others": "Other",
    "institutional": "Institutional",
    "employed": "Employed", "unemployed": "Unemployed",
    "not_in_labor_force": "Not in the labour force",
}
MIN_DOCUMENTED = 0.30
# Unstable housing and homelessness are reported as one level. Separately they held two
# patients each in one stratum, and a percentage on a count of two is a precision the
# count cannot carry. Both are housing instability, so the merge loses no distinction
# the table was drawing.
HOUSING = ["stable", "unstable_or_homeless", "institutional", "not_documented"]
HOUSING_MERGE = {"unstable": "unstable_or_homeless", "homeless": "unstable_or_homeless"}
LABOR = ["employed", "unemployed", "not_in_labor_force", "not_documented"]
SDOH = ["housing", "labor_force_status", "insurance", "transportation_barrier",
        "food_security", "incarceration_history"]
SDOH_NAME = {"housing": "Housing", "labor_force_status": "Labour force status",
             "insurance": "Insurance", "food_security": "Food security",
             "transportation_barrier": "Transportation barrier",
             "incarceration_history": "Incarceration history"}
CHARLSON_SHOW = [
    ("Liver_Disease_Moderate_Severe", "Decompensated liver disease"),
    ("Diabetes_with_Chronic_Complications", "Diabetes with complications"),
    ("Renal_Disease_Mild_Moderate", "Renal disease"),
    ("Congestive_Heart_Failure", "Congestive heart failure"),
    ("Chronic_Pulmonary_Disease", "Chronic pulmonary disease"),
    ("Malignancy", "Malignancy"),
]


def t3(x):
    return "AUD present" if x in YES else "AUD absent" if x in NO else "Indeterminate"


def fmt_cont(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if not len(s):
        return "--", "--"
    return ("%.1f (%.1f)" % (s.mean(), s.std()),
            "%.0f [%.0f, %.0f]" % (s.median(), s.min(), s.max()))


def fmt_cat(s, level):
    s = s.astype(str)
    n = int((s == level).sum())
    d = int((s != "nan").sum())
    return "%d (%.1f%%)" % (n, 100 * n / d) if d else "0"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--min-cell", type=int, default=20, dest="min_cell",
                    help="report categorical cells below this many patients")
    # The reference standard moved out of this script when the findings pipeline was
    # split up. It is passed in so that Table 1 and the flow diagram cannot silently
    # describe two different cohorts.
    ap.add_argument("--gold", default="resources/gold_reference_v2.csv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ref = pd.read_csv(a.gold, dtype={"omop_person_id": str})
    ros = pd.read_csv("resources/patient_roster.csv", dtype={"omop_person_id": str})
    feat = pd.read_parquet("output/features_structured.parquet")
    feat["omop_person_id"] = feat["person_id"].astype("int64").astype(str)
    sdoh = pd.read_parquet("output/sdoh/aggregate_recoded.parquet")
    sdoh["omop_person_id"] = sdoh["person_id"].astype("int64").astype(str)
    inst = pd.read_csv("output/fairness/patient_institution.csv")
    inst["omop_person_id"] = inst["person_id"].astype(str)

    d = (ref.merge(ros[["omop_person_id", "has_aud_icd", "n_notes"]], on="omop_person_id")
            .merge(feat, on="omop_person_id")
            .merge(sdoh[["omop_person_id"] + [c for c in SDOH if c in sdoh.columns]],
                   on="omop_person_id", how="left")
            .merge(inst[["omop_person_id", "institution"]], on="omop_person_id",
                   how="left"))
    d = d[d["gold_label"].notna()].copy()

    # features_structured carries age as of extraction, not as of the index date; the
    # implied reference year is 2026 for every patient. Index dates span 2015 to 2023,
    # so that column runs seven to eight years high and does not describe the cohort at
    # the moment it entered the study. Recompute it.
    idx = pd.read_csv("resources/index_dates.csv")
    idx["omop_person_id"] = idx["omop_person_id"].astype(str).str.replace(
        r"\.0$", "", regex=True)
    idx["index_year"] = pd.to_datetime(idx["index_date"], errors="coerce").dt.year
    pe = pd.read_parquet(B.D + "/person.parquet",
                         columns=["person_id", "year_of_birth"])
    pe["omop_person_id"] = pe["person_id"].astype("int64").astype(str)
    d = d.merge(idx[["omop_person_id", "index_year"]].drop_duplicates("omop_person_id"),
                on="omop_person_id", how="left")
    d = d.merge(pe[["omop_person_id", "year_of_birth"]], on="omop_person_id", how="left")
    d["age_at_index"] = d["index_year"] - d["year_of_birth"]
    if d["age_at_index"].isna().mean() > 0.02:
        raise SystemExit("age at index missing for %.1f%% of patients"
                         % (100 * d["age_at_index"].isna().mean()))
    if "housing" in d.columns:
        d["housing"] = d["housing"].replace(HOUSING_MERGE)
    d["reference"] = sc.normalize_labels(d["gold_label"], L6, strict=False).map(t3)
    d["stratum"] = np.where(d["has_aud_icd"] == 1, "AUD code present", "No AUD code")
    d["site"] = d["institution"].map({"IUH": "Academic health system",
                                      "Eskenazi": "Safety-net hospital",
                                      "Other": "Other hospitals"})
    for k, _ in CHARLSON_SHOW:
        d[k] = pd.to_numeric(d[k], errors="coerce").fillna(0)

    cols = [("No AUD code", d[d["stratum"] == "No AUD code"]),
            ("AUD code present", d[d["stratum"] == "AUD code present"]),
            ("Overall", d)]
    print("Table 1 on %d patients: %s" % (len(d), ", ".join(
        "%s %d" % (n, len(g)) for n, g in cols)))

    rows = []

    def cont(label, col):
        rows.append(("head", label, [fmt_cont(g[col])[0] for _, g in cols]))
        rows.append(("sub", "Median [min, max]", [fmt_cont(g[col])[1] for _, g in cols]))

    def cat(label, col, levels=None, missing=True):
        rows.append(("section", label, ["", "", ""]))
        lv = levels or [x for x in sorted(d[col].dropna().astype(str).unique())]
        for lev in lv:
            rows.append(("sub", RELABEL.get(lev, lev),
                         [fmt_cat(g[col], lev) for _, g in cols]))
        if missing and d[col].isna().any():
            rows.append(("sub", "Missing", [
                "%d (%.1f%%)" % (int(g[col].isna().sum()),
                                 100 * g[col].isna().mean()) for _, g in cols]))

    def binary(label, col):
        rows.append(("head", label, [
            "%d (%.1f%%)" % (int((g[col] == 1).sum()), 100 * (g[col] == 1).mean())
            for _, g in cols]))

    cont("Age at the index date, mean (SD)", "age_at_index")
    cat("Sex", "gender", ["Man", "Woman"])
    cat("Race", "race", ["White", "Black", "Others"])
    cat("Ethnicity", "ethnicity", ["Hispanic", "Others"])
    cat("Hospital system", "site", ["Academic health system", "Safety-net hospital",
                                    "Other hospitals"])
    cont("Charlson comorbidity index, mean (SD)", "CCI")
    rows.append(("section", "Selected comorbidity", ["", "", ""]))
    for k, lab in CHARLSON_SHOW:
        rows.append(("sub", lab, [
            "%d (%.1f%%)" % (int((g[k] == 1).sum()), 100 * (g[k] == 1).mean())
            for _, g in cols]))
    cont("Notes reviewed per patient, mean (SD)", "n_notes")
    # Social-determinant fields are shown only where the structured record documents
    # them for at least 30 per cent of the cohort. Insurance (11.5 per cent), food
    # security (8.3 per cent) and incarceration history (5.0 per cent) fall below that
    # and describe too few patients to characterise the cohort. Housing (44.0 per cent)
    # and labour force status (32.3 per cent) clear it. Whatever is shown keeps its
    # undocumented level, so no percentage is computed on a hidden denominator.
    cat("Housing", "housing", HOUSING)
    cat("Labour force status", "labor_force_status", LABOR)
    cat("Reference standard", "reference",
        ["AUD present", "AUD absent", "Indeterminate"], missing=False)

    hdr = [("%s (N=%d)" % (n, len(g))) for n, g in cols]

    def tex(s):
        """A per-cent sign is a comment character in LaTeX and would swallow the row
        terminator, silently destroying every row after it."""
        return str(s).replace("%", "\\%")

    body = []
    for kind, label, vals in rows:
        vals = [tex(v) for v in vals]
        label = tex(label)
        if kind == "section":
            body.append("\\addlinespace\n\\textit{%s} & & & \\\\" % label)
        elif kind == "head":
            body.append("%s & %s \\\\" % (label, " & ".join(vals)))
        else:
            body.append("\\quad %s & %s \\\\" % (label, " & ".join(vals)))

    tex = """\\begin{table}[htbp]
\\centering
\\caption*{\\textbf{Table 1.} Characteristics of the reviewed charts.
Continuous variables are mean (SD) with median [minimum, maximum] beneath;
categorical variables are n (\\%%) within column. Columns are the sampling strata: the
reviewed charts were drawn stratified on whether the structured record carried an AUD
diagnosis code, in roughly equal numbers, so the overall column describes the reviewed
sample and not the registry. Registry-standardised estimates are given in the text. No
significance tests are shown, because the two columns are sampling strata of known and
unequal fractions and a test between them would describe the sampling plan rather than
the patients. Unstable housing and homelessness are reported as one level.}
\\footnotesize
\\renewcommand{\\arraystretch}{0.96}
\\begin{tabular}{l ccc}
\\toprule
 & %s \\\\
\\midrule
%s
\\bottomrule
\\end{tabular}
\\end{table}
""" % (" & ".join(hdr), "\n".join(body))

    # ------------------------------------------------------------------ SDOH audit
    # How complete each structured social-determinant field is, and whether that
    # completeness depends on the outcome. A field documented more often in patients who
    # turn out to have AUD does not give a stratum that can be compared with another
    # stratum on equal terms, whatever the overall completeness is.
    sd = []
    for f in SDOH:
        if f not in d.columns:
            continue
        doc = (d[f].astype(str) != "not_documented") & d[f].notna()
        neg, pos = d["reference"] == "AUD absent", d["reference"] == "AUD present"
        sd.append({"field": SDOH_NAME[f], "documented_pct": 100 * doc.mean(),
                   "documented_aud_absent_pct": 100 * doc[neg].mean(),
                   "documented_aud_present_pct": 100 * doc[pos].mean(),
                   "difference_pct": 100 * (doc[pos].mean() - doc[neg].mean()),
                   "reported": doc.mean() >= MIN_DOCUMENTED})
    sd = pd.DataFrame(sd).sort_values("documented_pct", ascending=False)
    sd.round(1).to_csv(os.path.join(a.out, "supp_sdoh.csv"), index=False)
    print("\nStructured social-determinant fields, completeness and its dependence "
          "on the outcome:")
    print(sd.round(1).to_string(index=False))
    if (sd["difference_pct"] <= 0).any():
        print("\nNOTE: a field is documented no more often in AUD-positive patients. "
              "The text says every field is; rewrite it.")

    # ------------------------------------------------------------------ small cells
    # Every categorical cell below the threshold, so a minimum-cell-size requirement can
    # be checked against a list. Nothing here alters the table.
    small = []
    for kind, label, vals in rows:
        if kind == "section":
            continue
        for (name, _), v in zip(cols, vals):
            # A count, not a mean: continuous rows are written "63.0 (11.6)" and
            # would otherwise be read as cells of sixty-three patients.
            m = re.match(r"^(\d+) \(", str(v))
            if not m:
                continue
            n = int(m.group(1))
            if n < a.min_cell:
                small.append((label, name, n))
    print("\ncategorical cells below %d patients: %d" % (a.min_cell, len(small)))
    for label, name, n in sorted(small, key=lambda r: r[2]):
        print("  %-34s %-18s %3d" % (label, name, n))
    if not small:
        print("  none")

    # A context manager, not a bare open().write(). Unclosed, the text sits in a buffer
    # and a signal that kills the process before the destructor runs leaves the previous
    # table on disk with nothing to say it is stale.
    with open(os.path.join(a.out, "table1.tex"), "w") as fh:
        fh.write(tex)
    plain = pd.DataFrame([(("  " if k == "sub" else "") + lab, *v)
                          for k, lab, v in rows],
                         columns=["", *[n for n, _ in cols]])
    plain.to_csv(os.path.join(a.out, "table1.csv"), index=False)
    print(plain.to_string(index=False))
    print("\nwrote %s/table1.tex and table1.csv" % a.out)


if __name__ == "__main__":
    main()
