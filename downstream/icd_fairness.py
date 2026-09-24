#!/usr/bin/env python3
"""Is the AUD diagnosis code equally sensitive across patient groups?

The question the manuscript does not yet answer. The equity analysis on the
reviewed charts alone could not estimate code sensitivity by subgroup at that
sample size. The registry design can, because the reviewed charts supply
P(true AUD | stratum) and the registry supplies the stratum sizes.

Why a raw proportion will not do. The validation subsample was drawn stratified
on code status, so the coded stratum is oversampled roughly twofold relative to
the silent stratum. Any subgroup comparison must reweight to the registry
stratum sizes before it means anything.

Estimator. For subgroup g and stratum s,

    N_AUD(g, s)  =  registry count of g in s  x  theta(g, s)
    sensitivity(g) = N_AUD(g, stratum 1) / sum_s N_AUD(g, s)

where stratum 1 carries an AUD diagnosis code at the cirrhosis diagnosis and
theta comes from the validated charts in that subgroup and stratum. Confidence
intervals are a nonparametric bootstrap over validated charts, resampled within
stratum so the sampling design is preserved.

A difference in theta alone is not evidence of unequal coding. AUD prevalence
differs by sex, so the uncoded will differ by sex even under a code that
performs identically. Sensitivity is the quantity that separates the two.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "downstream")
import fig6_bayes_strata as B  # noqa: E402

D = B.D
OUT = "output/fairness"
SEED = 42
NBOOT = 4000
SNAMES = B.SNAMES



# ---------------------------------------------------------------------------------------
# theta(g, s) and its bootstrap, shared by every registry-scale subgroup producer.
#
# The charts were drawn stratified on whether an AUD code appears at any time (U), and the
# patients coded only after t0 were oversampled relative to the registry: in the
# record-silent stratum they are 9 per cent of the registry and far more of the sample, and
# about four in five of them have chart-documented AUD. Averaging the validated charts of a
# cell without regard to U therefore overstated theta in strata 2 and 3, and with it the
# number of uncoded patients with AUD: until 24 September 2026 the subgroup analysis implied
# 4,919 uncoded patients and a code sensitivity of 32.0 per cent over the whole registry,
# where the primary registry fit gives 3,261 and 41.1. The primary fit also estimates the
# coded-later cells of strata 2 and 3 from the never-coded charts of the same stratum,
# because the notes of the coded-later charts were drawn around the code that placed them
# (fig6_bayes_strata.ANCHOR_DEFAULT). THETA_RULE mirrors that: in strata 2 and 3 theta comes
# from the validated charts with U = 0. "raw" reproduces the earlier estimator.
THETA_RULE = "asuncoded"


def theta_frame(m, rule=None):
    """The validated charts that estimate theta(g, s) under the rule."""
    rule = rule or THETA_RULE
    v = m[m["val"]]
    if rule == "asuncoded":
        v = v[(v["S"] == 0) | (v["U"] == 0)]
    elif rule != "raw":
        raise SystemExit("unknown theta rule: %s" % rule)
    return v


def resample_design(v, rng):
    """Resample validated charts within code stratum and sampling status, as drawn."""
    parts = [g.iloc[rng.integers(0, len(g), len(g))] for _, g in v.groupby(["S", "U"])]
    return pd.concat(parts)

def build():
    m = B.build(0)
    pe = pd.read_parquet(D + "/person.parquet",
                         columns=["person_id", "race_concept_id", "ethnicity_concept_id"])
    pe["person_id"] = pe["person_id"].astype("int64")
    m = m.merge(pe, on="person_id", how="left")
    con = pd.read_parquet(D + "/concept.parquet", columns=["concept_id", "concept_name"])
    cmap = dict(zip(con["concept_id"].astype("int64"), con["concept_name"]))
    m["race"] = m["race_concept_id"].map(cmap).fillna("Unknown")
    # Two levels, not three. Splitting White, Black or African American, and Other or
    # unknown left the third level with too few validated charts to estimate at all: it
    # was dropped as not estimable and 408 registry patients disappeared from the axis,
    # so the reported contrast rested on 54 validated charts and covered 13,998 of the
    # 14,406 patients. Collapsing everything other than White keeps all 14,406 and rests
    # the contrast on 60 of the 423 charts that carry a determinate label and a cirrhosis
    # date. Table 1 reports the three levels separately.
    print("  race, raw composition of the registry before collapsing:")
    for name, n in m["race"].value_counts().items():
        print("    %-32s %6d" % (name, n))
    m["race"] = np.where(m["race"].eq("White"), "White", "Other than White")
    m["sex"] = np.where(m["male"] == 1, "Male", "Female")
    m["agegrp"] = np.where(m["age"] < 50, "Under 50",
                           np.where(m["age"] < 65, "50 to 64", "65 and over"))
    return m


def sensitivity(m, axis, rng=None, resample=False):
    """Design-weighted code sensitivity by subgroup. Returns {level: sensitivity}."""
    v = theta_frame(m)
    if resample:
        v = resample_design(v, rng)
    out = {}
    for lvl in sorted(m[axis].unique()):
        num = den = 0.0
        ok = True
        for s in range(3):
            n_reg = int(((m[axis] == lvl) & (m["S"] == s)).sum())
            sub = v[(v[axis] == lvl) & (v["S"] == s)]
            if len(sub) == 0:
                ok = False
                break
            th = float(sub["Aobs"].mean())
            n_aud = n_reg * th
            den += n_aud
            if s == 0:
                num = n_aud
        out[lvl] = (num / den) if (ok and den > 0) else np.nan
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    m = build()
    print("registry %d ; validated %d" % (len(m), m["val"].sum()))
    rng = np.random.default_rng(SEED)
    rows = []
    for axis in ("sex", "race", "agegrp"):
        print("\n" + "=" * 74)
        print("CODE SENSITIVITY BY %s, design-weighted" % axis.upper())
        print("=" * 74)
        # validated cell counts first, so thin cells are visible
        print("  validated charts per cell:")
        ct = pd.crosstab(m.loc[m["val"], axis], m.loc[m["val"], "S"])
        ct.columns = [SNAMES[c] for c in ct.columns]
        print(ct.to_string().replace("\n", "\n    "))
        point = sensitivity(m, axis)
        boots = {k: [] for k in point}
        for _ in range(NBOOT):
            b = sensitivity(m, axis, rng, resample=True)
            for k, val in b.items():
                boots[k].append(val)
        print()
        for lvl in point:
            arr = np.array([x for x in boots[lvl] if np.isfinite(x)])
            if not np.isfinite(point[lvl]) or len(arr) < 100:
                print("  %-26s not estimable" % lvl)
                continue
            lo, hi = np.percentile(arr, [2.5, 97.5])
            n_reg = int((m[axis] == lvl).sum())
            print("  %-26s %5.1f%%  (95%% CI %.1f-%.1f)   registry n = %d"
                  % (lvl, 100 * point[lvl], 100 * lo, 100 * hi, n_reg))
            rows.append({"axis": axis, "level": lvl, "sensitivity_pct": 100 * point[lvl],
                         "ci_low": 100 * lo, "ci_high": 100 * hi, "registry_n": n_reg})
        # difference between the two largest groups
        lv = [lev for lev in point if np.isfinite(point[lev])]
        if len(lv) >= 2:
            lv = sorted(lv, key=lambda lev: -int((m[axis] == lev).sum()))[:2]
            d = np.array([boots[lv[0]][i] - boots[lv[1]][i] for i in range(NBOOT)
                          if np.isfinite(boots[lv[0]][i]) and np.isfinite(boots[lv[1]][i])])
            lo, hi = np.percentile(d, [2.5, 97.5])
            print("  difference, %s minus %s: %+.1f points (95%% CI %+.1f to %+.1f)"
                  % (lv[0], lv[1], 100 * (point[lv[0]] - point[lv[1]]), 100 * lo, 100 * hi))
            rows.append({"axis": axis, "level": "%s minus %s" % (lv[0], lv[1]),
                         "sensitivity_pct": 100 * (point[lv[0]] - point[lv[1]]),
                         "ci_low": 100 * lo, "ci_high": 100 * hi, "registry_n": np.nan})

    print("\n" + "=" * 74)
    print("WHY theta ALONE IS NOT THE ANSWER")
    print("=" * 74)
    v = m[m["val"]]
    for axis in ("sex",):
        for lvl in sorted(m[axis].unique()):
            g = v[(v[axis] == lvl) & (v["S"] == 2)]
            print("  theta in the silent stratum, %-8s %3d/%3d = %.1f%%"
                  % (lvl, int(g["Aobs"].sum()), len(g), 100 * g["Aobs"].mean()))
    print("  Those differ, but they reflect how common AUD is in each group as much")
    print("  as how well the code finds it. The sensitivities above separate the two.")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "icd_sensitivity_by_group.csv"), index=False)
    print("\n  wrote %s/icd_sensitivity_by_group.csv" % OUT)


if __name__ == "__main__":
    main()
