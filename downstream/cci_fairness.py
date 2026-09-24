#!/usr/bin/env python3
"""Charlson comorbidity as a fairness axis for the AUD diagnosis code.

Why this is worth asking. Comorbidity burden is a standard, interpretable proxy for
health-care contact. If code sensitivity rises with it, the code is tracking how often
a patient is seen rather than whether the patient drinks. That is the paper's thesis,
stated in a measure a clinician already reads.

Code sets: NCI SEER-Medicare 2021 macro with the Glasheen 2019 hierarchy, taken verbatim
from the DualR ETL reference. Dot-stripped, prefix-matched, ICD-9 and ICD-10 together
because the cohort spans the 2015 transition.

Two corrections that matter and are easy to get wrong:
  AIDS is HIV AND an opportunistic infection, not the OI codes alone. OI-only flagging
  inflates AIDS from about 0.1% to 5-15%.
  Six hierarchy pairs: the severe condition zeroes the mild one.

Cohort caveat. Every patient here has cirrhosis, so the liver components are close to
universal and would dominate the score. The index is reported with and without them;
the liver-free version is the honest measure of other comorbidity burden.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "downstream")
import fig6_bayes_strata as B  # noqa: E402
from icd_fairness import sensitivity  # noqa: E402

D = B.D
OUT = "output/fairness"
SEED = 42
NBOOT = 4000

CODESETS = {
    "Myocardial_Infarction": {"9": ["410", "412"], "10": ["I21", "I22", "I252"]},
    "Congestive_Heart_Failure": {
        "9": ["39891", "40201", "40211", "40291", "40401", "40403", "40411", "40413",
              "40491", "40493", "4254", "4255", "4256", "4257", "4258", "4259", "428"],
        "10": ["I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426", "I427",
               "I428", "I429", "I43", "I50", "P290"]},
    "Peripheral_Vascular_Disease": {
        "9": ["0930", "440", "441", "4431", "4432", "4433", "4434", "4435", "4436",
              "4437", "4438", "4439", "4471", "5571", "5579", "V434"],
        "10": ["I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792",
               "K551", "K558", "K559", "Z958", "Z959"]},
    "Cerebrovascular_Disease": {
        "9": ["36234", "430", "431", "432", "433", "434", "435", "436", "437", "438"],
        "10": ["G45", "G46", "H340", "I60", "I61", "I62", "I63", "I64", "I65", "I66",
               "I67", "I68", "I69"]},
    "Dementia": {"9": ["290", "2941", "3312"],
                 "10": ["F00", "F01", "F02", "F03", "F051", "G30", "G311"]},
    "Chronic_Pulmonary_Disease": {
        "9": ["4168", "4169", "490", "491", "492", "493", "494", "495", "496", "497",
              "498", "499", "500", "501", "502", "503", "504", "505", "5064", "5081", "5088"],
        "10": ["I278", "I279", "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47",
               "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67", "J684", "J701", "J703"]},
    "Rheumatic_Disease": {
        "9": ["4465", "7100", "7101", "7102", "7103", "7104", "7140", "7141", "7142",
              "7148", "725"],
        "10": ["M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"]},
    "Peptic_Ulcer_Disease": {"9": ["531", "532", "533", "534"],
                             "10": ["K25", "K26", "K27", "K28"]},
    "Liver_Disease_Mild": {
        "9": ["07022", "07023", "07032", "07033", "07044", "07054", "0706", "0709",
              "570", "571", "5733", "5734", "5738", "5739", "V427"],
        "10": ["B18", "K700", "K701", "K702", "K703", "K709", "K713", "K714", "K715",
               "K717", "K73", "K74", "K760", "K762", "K763", "K764", "K768", "K769", "Z944"]},
    "Liver_Disease_Moderate_Severe": {
        "9": ["4560", "4561", "4562", "5722", "5723", "5724", "5725", "5726", "5727", "5728"],
        "10": ["I850", "I859", "I864", "I982", "K704", "K711", "K721", "K729",
               "K765", "K766", "K767"]},
    "Diabetes_without_Chronic_Complications": {
        "9": ["2500", "2501", "2502", "2503", "2508", "2509"],
        "10": ["E100", "E101", "E106", "E108", "E109", "E110", "E111", "E116", "E118",
               "E119", "E130", "E131", "E136", "E138", "E139"]},
    "Diabetes_with_Chronic_Complications": {
        "9": ["2504", "2505", "2506", "2507"],
        "10": ["E102", "E103", "E104", "E105", "E107", "E112", "E113", "E114", "E115",
               "E117", "E132", "E133", "E134", "E135", "E137"]},
    "Hemiplegia_Paraplegia": {
        "9": ["3341", "342", "343", "3440", "3441", "3442", "3443", "3444", "3445",
              "3446", "3449"],
        "10": ["G041", "G114", "G801", "G802", "G81", "G82", "G830", "G831", "G832",
               "G833", "G834", "G839"]},
    "Renal_Disease_Mild_Moderate": {
        "9": ["40300", "40310", "40390", "40400", "40401", "40410", "40411", "40490",
              "40491", "582", "583", "5851", "5852", "5853", "5854", "5859", "V420"],
        "10": ["I129", "I130", "I1310", "N032", "N033", "N034", "N035", "N036", "N037",
               "N052", "N053", "N054", "N055", "N056", "N057",
               "N181", "N182", "N183", "N184", "N189", "Z940"]},
    "Renal_Disease_Severe": {
        "9": ["40301", "40311", "40391", "40402", "40403", "40412", "40413", "40492",
              "40493", "5855", "5856", "586", "5880", "V451", "V56"],
        "10": ["I120", "I1311", "I132", "N185", "N186", "N19", "N250", "Z49", "Z992"]},
    "HIV": {"9": ["042", "043", "044"], "10": ["B20", "B21", "B22", "B24"]},
    "Metastatic_Solid_Tumor": {"9": ["196", "197", "198", "1990"],
                               "10": ["C77", "C78", "C79", "C800", "C802"]},
    "Malignancy": {
        "9": ["14", "15", "16", "170", "171", "172", "174", "175", "176", "179", "18",
              "190", "191", "192", "193", "194", "195", "1991", "200", "201", "202",
              "203", "204", "205", "206", "207", "208", "2386"],
        "10": ["C0", "C1", "C2", "C30", "C31", "C32", "C33", "C34", "C37", "C38", "C39",
               "C40", "C41", "C43", "C45", "C46", "C47", "C48", "C49", "C50", "C51",
               "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C60", "C61", "C62",
               "C63", "C76", "C801", "C81", "C82", "C83", "C84", "C85", "C88", "C9"]},
}
AIDS_OI = {
    "9": ["112", "180", "114", "1175", "0074", "0785", "3483", "054", "115", "0072",
          "176", "200", "201", "202", "203", "204", "205", "206", "207", "208", "209",
          "031", "010", "011", "012", "013", "014", "015", "016", "017", "018",
          "1363", "V1261", "0463", "0031", "130", "7994"],
    "10": ["B37", "C53", "B38", "B45", "A072", "B25", "G934", "B00", "B39", "A073",
           "C46", "C81", "C82", "C83", "C84", "C85", "C86", "C87", "C88", "C9",
           "C90", "C91", "C92", "C93", "C94", "C95", "C96", "A31", "A15", "A16",
           "A17", "A18", "A19", "B59", "Z8701", "A812", "A021", "B58", "R64"],
}
HIERARCHY_PAIRS = [
    ("Hemiplegia_Paraplegia", "Cerebrovascular_Disease"),
    ("Liver_Disease_Moderate_Severe", "Liver_Disease_Mild"),
    ("Diabetes_with_Chronic_Complications", "Diabetes_without_Chronic_Complications"),
    ("Renal_Disease_Severe", "Renal_Disease_Mild_Moderate"),
    ("Metastatic_Solid_Tumor", "Malignancy"),
    ("AIDS", "HIV"),
]
WEIGHTS = {
    "Myocardial_Infarction": 1, "Congestive_Heart_Failure": 1,
    "Peripheral_Vascular_Disease": 1, "Cerebrovascular_Disease": 1, "Dementia": 1,
    "Chronic_Pulmonary_Disease": 1, "Rheumatic_Disease": 1, "Peptic_Ulcer_Disease": 1,
    "Liver_Disease_Mild": 1, "Diabetes_without_Chronic_Complications": 1,
    "Hemiplegia_Paraplegia": 2, "Renal_Disease_Mild_Moderate": 1,
    "Diabetes_with_Chronic_Complications": 2, "Malignancy": 2,
    "Liver_Disease_Moderate_Severe": 3, "Renal_Disease_Severe": 3, "HIV": 3,
    "Metastatic_Solid_Tumor": 6, "AIDS": 6,
}
LIVER = ["Liver_Disease_Mild", "Liver_Disease_Moderate_Severe"]


def prefixes(cs):
    return sorted({str(c).replace(".", "").upper() for v in cs.values() for c in v})


def main():
    os.makedirs(OUT, exist_ok=True)
    m = B.build(0)
    ids = set(m["person_id"])
    co = pd.read_parquet(D + "/condition_occurrence.parquet",
                         columns=["person_id", "condition_source_value"])
    co["person_id"] = co["person_id"].astype("int64")
    co = co[co["person_id"].isin(ids)].copy()
    icd = co["condition_source_value"].astype(str).str.extract(r"\^\^([^\^]+)\^")[0]
    icd = icd.fillna(co["condition_source_value"].astype(str))
    co["p"] = icd.str.replace(".", "", regex=False).str.upper().str.strip()
    del icd

    flags = pd.DataFrame({"person_id": sorted(ids)}).set_index("person_id")
    for name, cs in CODESETS.items():
        pats = tuple(prefixes(cs))
        hit = co.loc[co["p"].str.startswith(pats, na=False), "person_id"].unique()
        flags[name] = 0
        flags.loc[flags.index.isin(hit), name] = 1
    oi = co.loc[co["p"].str.startswith(tuple(prefixes(AIDS_OI)), na=False),
                "person_id"].unique()
    flags["has_oi"] = 0
    flags.loc[flags.index.isin(oi), "has_oi"] = 1
    del co

    # AIDS requires HIV AND an opportunistic infection
    flags["AIDS"] = ((flags["HIV"] == 1) & (flags["has_oi"] == 1)).astype(int)
    flags["HIV"] = ((flags["HIV"] == 1) & (flags["AIDS"] == 0)).astype(int)
    flags = flags.drop(columns=["has_oi"])
    for severe, mild in HIERARCHY_PAIRS:
        flags.loc[flags[severe] == 1, mild] = 0

    assert flags["AIDS"].mean() < 0.02, "AIDS prevalence above 2 percent, check the OI logic"
    print("AIDS %.3f%%   HIV %.3f%%   (Glasheen reference: AIDS 0.01-0.10%%)"
          % (100 * flags["AIDS"].mean(), 100 * flags["HIV"].mean()))
    print()
    print("prevalence after the hierarchy rules")
    for c in sorted(WEIGHTS, key=lambda k: -flags[k].mean()):
        print("  %-42s %6.2f%%  (%5d)" % (c, 100 * flags[c].mean(), int(flags[c].sum())))

    flags["cci"] = sum(flags[c] * w for c, w in WEIGHTS.items())
    flags["cci_noliver"] = sum(flags[c] * w for c, w in WEIGHTS.items() if c not in LIVER)
    m = m.merge(flags.reset_index()[["person_id", "cci", "cci_noliver"]],
                on="person_id", how="left")
    print()
    print("CCI including liver     median %d  IQR %d-%d" %
          (m["cci"].median(), m["cci"].quantile(.25), m["cci"].quantile(.75)))
    print("CCI excluding liver     median %d  IQR %d-%d" %
          (m["cci_noliver"].median(), m["cci_noliver"].quantile(.25),
           m["cci_noliver"].quantile(.75)))

    m["ccigrp"] = np.where(m["cci_noliver"] == 0, "0",
                           np.where(m["cci_noliver"] <= 2, "1 to 2", "3 or more"))
    rng = np.random.default_rng(SEED)
    rows = []
    print()
    print("=" * 74)
    print("CODE SENSITIVITY BY COMORBIDITY BURDEN, liver excluded, design-weighted")
    print("=" * 74)
    ct = pd.crosstab(m.loc[m["val"], "ccigrp"], m.loc[m["val"], "S"])
    ct.columns = [B.SNAMES[c] for c in ct.columns]
    print("  validated charts per cell:")
    print("    " + ct.to_string().replace("\n", "\n    "))
    point = sensitivity(m, "ccigrp")
    boots = {k: [] for k in point}
    for _ in range(NBOOT):
        b = sensitivity(m, "ccigrp", rng, resample=True)
        for k, val in b.items():
            boots[k].append(val)
    print()
    for lvl in sorted(point):
        arr = np.array([x for x in boots[lvl] if np.isfinite(x)])
        if not np.isfinite(point[lvl]) or len(arr) < 100:
            print("  %-12s not estimable" % lvl)
            continue
        lo, hi = np.percentile(arr, [2.5, 97.5])
        print("  CCI %-10s %5.1f%%  (95%% CI %.1f-%.1f)   registry n = %d"
              % (lvl, 100 * point[lvl], 100 * lo, 100 * hi, int((m["ccigrp"] == lvl).sum())))
        rows.append({"axis": "charlson_noliver", "level": lvl,
                     "sensitivity_pct": 100 * point[lvl], "ci_low": 100 * lo,
                     "ci_high": 100 * hi, "registry_n": int((m["ccigrp"] == lvl).sum())})
    hi_lo = [b for b in zip(boots.get("3 or more", []), boots.get("0", []))
             if np.isfinite(b[0]) and np.isfinite(b[1])]
    if hi_lo:
        d = np.array([a - b for a, b in hi_lo])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print("  difference, 3 or more minus 0: %+.1f points (95%% CI %+.1f to %+.1f)"
              % (100 * (point["3 or more"] - point["0"]), 100 * lo, 100 * hi))
        rows.append({"axis": "charlson_noliver", "level": "3 or more minus 0",
                     "sensitivity_pct": 100 * (point["3 or more"] - point["0"]),
                     "ci_low": 100 * lo, "ci_high": 100 * hi, "registry_n": np.nan})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cci_sensitivity.csv"), index=False)
    flags.reset_index().to_csv(os.path.join(OUT, "charlson_flags.csv"), index=False)
    print("\n  wrote %s" % OUT)


if __name__ == "__main__":
    main()
