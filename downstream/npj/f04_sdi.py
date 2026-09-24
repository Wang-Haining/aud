#!/usr/bin/env python3
"""Neighbourhood deprivation as an axis for the coding gap and for the index test.

Producer for panel c of Figure 4. The other panels stratify on attributes of the patient
or of the encounter. This one stratifies on where the patient lives, which is the axis a
reader asks about when the word disparity appears and the only axis in the figure that is
ordered.

Why the Social Deprivation Index and not the Area Deprivation Index. The cohort's OMOP
location table carries a 5-digit ZIP and nothing finer. The Neighborhood Atlas states that
mapping its Area Deprivation Index to ZIP5 or ZCTA is not validated, which is why
output/geo/adi_ruca.parquet records the ADI axis as unavailable for every patient. The
Social Deprivation Index is published at ZCTA, so the only approximation left is matching a
5-digit ZIP to the ZCTA of the same five digits. The match rate is reported below and no ZIP
value is printed by this script.

Quintile definition. SDI_score in the source file is a population-weighted national
percentile rank: each 20-point band of it holds one fifth of the United States population,
64.9 million per band, verified against the ZCTA5_population column of the same file.
Quintiles are therefore the fixed bands 1-20, 21-40, 41-60, 61-80 and 81-100. No cut point
is estimated from this cohort, and quintile 5 means the most deprived fifth of the country
rather than the most deprived fifth of the cohort.

Two estimands are computed, on the two scales the figure already uses.

  Registry scale. Among the registry patients estimated to have AUD, the share carrying an
  AUD diagnosis code by t0. This uses icd_fairness.sensitivity with sdi_q as the axis, so
  it is the same estimator as panels c and d of the earlier draft and cannot disagree with
  them. Intervals are a nonparametric bootstrap over validated charts resampled within code
  stratum, seed and count as in icd_fairness.py.

  Chart scale. Among the reference-positive charts of a quintile, the sensitivity of the
  AUD diagnosis code at any time and of the primary model, with Wilson intervals and
  McNemar's paired test. This is panel a's estimand on a new axis. It is computed at five
  levels and at a collapsed three-level version, because the figure states a feasibility
  rule and the two versions do not both clear it.

Writes downstream/npj/out/figure4/sdi_quintile.csv, sdi_chart_level.csv and sdi_coverage.csv.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "downstream")
import stats_core as sc                                                   # noqa: E402
import fig6_bayes_strata as B                                             # noqa: E402
from icd_fairness import build, sensitivity, NBOOT, SEED                  # noqa: E402

SDI = "resources/sdi_zcta.csv"
LOCATION = "/N/project/cirrhosis_regv20/data_tables_v1/csv/location.csv"
RATERS = "output/raters/rater_long.csv"
ROSTER = "resources/patient_roster.csv"
OUT = "downstream/npj/out/figure4"
PRIMARY = "aud_qwen36_27b"

# Fixed bands of the population-weighted national percentile. Each holds a fifth of the
# United States population by construction.
BANDS = [(1, 20), (21, 40), (41, 60), (61, 80), (81, 100)]
Q5 = ["Least deprived", "Second", "Middle", "Fourth", "Most deprived"]
Q3 = ["Least deprived", "Middle", "Most deprived"]
# Which quintiles the three-level version pools.
TO3 = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}


def sdi_reference(path=SDI):
    d = pd.read_csv(path, usecols=["ZCTA5_FIPS", "ZCTA5_population", "SDI_score"],
                    dtype={"ZCTA5_FIPS": "string"})
    d["ZCTA5_FIPS"] = d["ZCTA5_FIPS"].str.zfill(5)
    if d["ZCTA5_FIPS"].duplicated().any():
        raise RuntimeError("the SDI reference is not unique by ZCTA")
    # Confirm the band definition against the file rather than trusting the docstring.
    v = d.dropna(subset=["SDI_score"])
    pop = v.groupby(pd.cut(v["SDI_score"], [0, 20, 40, 60, 80, 100], labels=range(5)),
                    observed=True)["ZCTA5_population"].sum()
    share = (pop / pop.sum()).to_numpy()
    if not np.all(np.abs(share - 0.2) < 0.005):
        raise RuntimeError("SDI_score bands are not population quintiles: %s" % share)
    print("  SDI reference: %d ZCTAs, %d without a score; the five 20-point bands hold "
          "%s of the United States population"
          % (len(d), int(d["SDI_score"].isna().sum()),
             ", ".join("%.1f%%" % (100 * s) for s in share)))
    return d[["ZCTA5_FIPS", "SDI_score"]]


def zip5_by_person(person_parquet=None, location=LOCATION):
    """person_id to the ZCTA of the same five digits. No ZIP value is printed."""
    d = B.D if person_parquet is None else person_parquet
    pe = pd.read_parquet(d + "/person.parquet", columns=["person_id", "location_id"])
    pe["person_id"] = pe["person_id"].astype("int64")
    pe["location_id"] = pe["location_id"].astype("Int64").astype("string")
    loc = pd.read_csv(location, dtype="string")
    if loc["location_id"].duplicated().any():
        raise RuntimeError("location table is not unique by location_id")
    g = pe.merge(loc[["location_id", "zip", "address_1"]], on="location_id", how="left",
                 validate="many_to_one")
    if len(g) != len(pe):
        raise RuntimeError("person/location join changed the row count")
    z = g["zip"].astype("string").str.strip()
    if g["address_1"].notna().any() or z.str.fullmatch(r"\d{5}-?\d{4}", na=False).any():
        raise RuntimeError("a street address or ZIP+4 appeared; this builder is ZIP5 only")
    g["zcta"] = z.where(z.str.fullmatch(r"\d{5}", na=False))
    return g[["person_id", "zcta"]]


def attach(m, ref, zp):
    n0 = len(m)
    m = m.merge(zp, on="person_id", how="left", validate="one_to_one")
    m = m.merge(ref.rename(columns={"ZCTA5_FIPS": "zcta"}), on="zcta", how="left")
    if len(m) != n0:
        raise RuntimeError("the SDI join changed the row count")
    q = pd.Series(np.nan, index=m.index)
    for k, (lo, hi) in enumerate(BANDS):
        q = q.mask(m["SDI_score"].between(lo, hi), k)
    m["q5"] = q
    m["sdi_q"] = q.map(lambda k: Q5[int(k)] if np.isfinite(k) else "not linked")
    m["sdi_q3"] = q.map(lambda k: Q3[TO3[int(k)]] if np.isfinite(k) else "not linked")
    return m


def coverage(m, label):
    rows = []
    n = len(m)
    have_zip = int(m["zcta"].notna().sum())
    have_sdi = int(m["SDI_score"].notna().sum())
    print("  %s: %d rows; ZIP5 present %d (%.1f%%); matched to a scored ZCTA %d (%.1f%%)"
          % (label, n, have_zip, 100 * have_zip / n, have_sdi, 100 * have_sdi / n))
    rows.append({"frame": label, "n": n, "zip5_present": have_zip,
                 "matched_to_scored_zcta": have_sdi,
                 "match_rate_pct": 100.0 * have_sdi / n})
    return rows


def registry_scale(m, axis, order):
    """Design-weighted share coded by t0, by quintile, with a bootstrap interval."""
    print("\n  validated charts per cell, %s:" % axis)
    ct = pd.crosstab(m.loc[m["val"], axis], m.loc[m["val"], "S"])
    print("    " + ct.to_string().replace("\n", "\n    "))

    rng = np.random.default_rng(SEED)
    point = sensitivity(m, axis)
    boots = {k: [] for k in point}
    for _ in range(NBOOT):
        b = sensitivity(m, axis, rng, resample=True)
        for k, val in b.items():
            boots[k].append(val)

    rows = []
    for lvl in order:
        if lvl not in point:
            continue
        arr = np.array([x for x in boots[lvl] if np.isfinite(x)])
        n_reg = int((m[axis] == lvl).sum())
        if not np.isfinite(point[lvl]) or len(arr) < 100:
            print("    %-16s not estimable" % lvl)
            continue
        lo, hi = np.percentile(arr, [2.5, 97.5])
        print("    %-16s %5.1f%%  (95%% CI %.1f-%.1f)   registry n = %5d"
              % (lvl, 100 * point[lvl], 100 * lo, 100 * hi, n_reg))
        rows.append({"axis": axis, "level": lvl, "coded_by_t0_pct": 100 * point[lvl],
                     "ci_low": 100 * lo, "ci_high": 100 * hi, "registry_n": n_reg,
                     "halfwidth": (100 * hi - 100 * lo) / 2.0})
    a, b_ = order[-1], order[0]
    if len(order) >= 2 and a in point and b_ in point:
        d = np.array([boots[a][i] - boots[b_][i] for i in range(NBOOT)
                      if np.isfinite(boots[a][i]) and np.isfinite(boots[b_][i])])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print("    %s minus %s: %+.1f points (95%% CI %+.1f to %+.1f)"
              % (a, b_, 100 * (point[a] - point[b_]), 100 * lo, 100 * hi))
        rows.append({"axis": axis, "level": "%s minus %s" % (a, b_),
                     "coded_by_t0_pct": 100 * (point[a] - point[b_]),
                     "ci_low": 100 * lo, "ci_high": 100 * hi,
                     "registry_n": np.nan, "halfwidth": (100 * hi - 100 * lo) / 2.0})
    return rows


def charts(ref, zp):
    r = pd.read_csv(RATERS, dtype=str)
    r = r[r["in_analysis_cohort"].eq("True")]
    w = r.pivot_table(index="omop_person_id", columns="rater", values="label_3",
                      aggfunc="first")
    d = pd.DataFrame({"pid": w.index.astype(str), "gold": w["gold"].values,
                      "llm": w[PRIMARY].values})
    ros = pd.read_csv(ROSTER, dtype={"omop_person_id": str})
    d = d.merge(ros[["omop_person_id", "has_aud_icd"]].rename(
        columns={"omop_person_id": "pid"}), on="pid", how="left")
    d["code"] = d["has_aud_icd"].astype(str).str.lower().isin(["true", "1", "1.0"])
    d["person_id"] = d["pid"].astype("int64")
    d = attach(d, ref, zp)
    print("  charts %d, reference-positive %d" % (len(d), int(d["gold"].eq("AUD+").sum())))
    return d


def chart_scale(d, axis, order):
    """Sensitivity of the code and of the primary model on reference-positive charts."""
    rows = []
    for lvl in order:
        sub = d[d[axis].eq(lvl)]
        pos = sub[sub["gold"].eq("AUD+")]
        n = len(pos)
        if n == 0:
            continue
        kc = int(pos["code"].sum())
        kl = int(pos["llm"].eq("AUD+").sum())
        c_lo, c_hi = sc.wilson_ci(kc, n)
        l_lo, l_hi = sc.wilson_ci(kl, n)
        half = 100.0 * (c_hi - c_lo) / 2.0
        mc = sc.mcnemar(np.ones(n, dtype=bool), pos["llm"].eq("AUD+").to_numpy(),
                        pos["code"].to_numpy())
        verdict = ("too few positives to estimate" if n < 20 else
                   "can support a 10-point comparison" if half <= 10.0 else
                   "estimable, interval too wide to compare")
        print("    %-16s charts %4d  positives %4d  code %5.1f (%.1f-%.1f, half %4.1f)  "
              "model %5.1f  gain %+5.1f  LLM alone %2d  code alone %2d  p %.3f  %s"
              % (lvl, len(sub), n, 100.0 * kc / n, 100 * c_lo, 100 * c_hi, half,
                 100.0 * kl / n, 100.0 * (kl - kc) / n, mc["b"], mc["c"], mc["p_value"],
                 verdict))
        rows.append({"axis": axis, "level": lvl, "charts": len(sub),
                     "reference_positive": n,
                     "code_sens": 100.0 * kc / n, "code_lo": 100.0 * c_lo,
                     "code_hi": 100.0 * c_hi, "code_halfwidth": half,
                     "llm_sens": 100.0 * kl / n, "llm_lo": 100.0 * l_lo,
                     "llm_hi": 100.0 * l_hi, "gain": 100.0 * (kl - kc) / n,
                     "llm_only": mc["b"], "code_only": mc["c"],
                     "p_paired": mc["p_value"], "paired_method": mc["method"],
                     "verdict": verdict})
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    print("SDI reference")
    ref = sdi_reference()
    zp = zip5_by_person()

    print("\nregistry frame")
    m = attach(build(), ref, zp)
    cov = coverage(m, "registry")
    print("\n  registry composition against fixed national quintiles "
          "(each quintile is a fifth of the United States population)")
    vc = m["sdi_q"].value_counts()
    for lvl in Q5 + ["not linked"]:
        n = int(vc.get(lvl, 0))
        print("    %-16s %6d  %5.1f%% of the registry" % (lvl, n, 100 * n / len(m)))

    # The pooled value first, so the panel has a reference line computed the same way.
    m["registry"] = "Registry"
    reg = registry_scale(m, "registry", ["Registry"])
    reg += registry_scale(m, "sdi_q", Q5)
    reg += registry_scale(m, "sdi_q3", Q3)

    print("\nchart frame")
    d = charts(ref, zp)
    cov += coverage(d, "reviewed charts")
    print("  five levels:")
    ch = chart_scale(d, "sdi_q", Q5)
    print("  three levels:")
    ch += chart_scale(d, "sdi_q3", Q3)

    pd.DataFrame(reg).to_csv(os.path.join(OUT, "sdi_quintile.csv"), index=False)
    pd.DataFrame(ch).to_csv(os.path.join(OUT, "sdi_chart_level.csv"), index=False)
    pd.DataFrame(cov).to_csv(os.path.join(OUT, "sdi_coverage.csv"), index=False)
    print("\nwrote %s/sdi_quintile.csv, sdi_chart_level.csv and sdi_coverage.csv" % OUT)


if __name__ == "__main__":
    main()
