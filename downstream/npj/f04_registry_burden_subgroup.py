#!/usr/bin/env python3
"""How many registry patients with AUD carried no AUD code at t0, by subgroup.

Panel c of Figure 4 reports the share coded by t0. A share does not say how many people it
is, and the subgroups differ in size by a factor of five, so the group with the worst share
is not necessarily the group with the most uncoded patients. This computes the count.

Estimator, the same one icd_fairness.py uses for the share, so the two panels cannot
disagree. For subgroup g and code stratum s,

    N_AUD(g, s) = registry count of g in s  x  theta(g, s)

with theta from icd_fairness.theta_frame: the validated charts of the cell, the never-
coded ones only in strata 2 and 3, as the primary registry fit uses them. Then

    with AUD      = sum over s of N_AUD(g, s)
    coded by t0   = N_AUD(g, s = 0)
    uncoded at t0 = with AUD - coded by t0

Intervals are a nonparametric bootstrap over validated charts resampled within code stratum
and sampling status,
seed and count as in icd_fairness.py, so the design is preserved.

Charlson comes from output/fairness/charlson_flags.csv on person_id, bucketed as the
companion panel buckets it. Any patient the merge does not match is reported and dropped
from that axis alone.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "downstream")
sys.path.insert(0, "downstream/npj")
from icd_fairness import build, NBOOT, SEED, theta_frame, resample_design                             # noqa: E402
import f04_sdi as F4                                                    # noqa: E402

OUT = "downstream/npj/out/figure4"
FAIR = "output/fairness"
AXES = [("agegrp", "Age"), ("sex", "Sex"), ("race", "Race"),
        ("charlson_noliver", "Charlson index"),
        ("sdi_q", "Neighbourhood deprivation")]
# Levels that carry no comparison and are reported as counts elsewhere.
SKIP = {"unmatched", "not linked"}


def attach_charlson(m, fair=FAIR):
    c = pd.read_csv(os.path.join(fair, "charlson_flags.csv"))
    c["person_id"] = c["person_id"].astype("int64")
    m = m.merge(c[["person_id", "cci_noliver"]], on="person_id", how="left")
    miss = int(m["cci_noliver"].isna().sum())
    if miss:
        print("  charlson unmatched for %d of %d registry patients" % (miss, len(m)))
    m["charlson_noliver"] = np.where(m["cci_noliver"].isna(), "unmatched",
                            np.where(m["cci_noliver"] <= 0, "0",
                            np.where(m["cci_noliver"] <= 2, "1 to 2", "3 or more")))
    return m


def burden(m, axis, rng=None, resample=False):
    """{level: (with AUD, coded by t0, uncoded at t0)} on the registry scale."""
    v = theta_frame(m)
    if resample:
        v = resample_design(v, rng)
    out = {}
    for lvl in sorted(m[axis].unique()):
        if lvl in SKIP:
            continue
        tot = coded = 0.0
        ok = True
        for s in range(3):
            n_reg = int(((m[axis] == lvl) & (m["S"] == s)).sum())
            sub = v[(v[axis] == lvl) & (v["S"] == s)]
            if len(sub) == 0:
                ok = False
                break
            n_aud = n_reg * float(sub["Aobs"].mean())
            tot += n_aud
            if s == 0:
                coded = n_aud
        out[lvl] = (tot, coded, tot - coded) if ok else (np.nan, np.nan, np.nan)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    m = attach_charlson(build())
    # The deprivation axis joins here rather than in a separate producer, so the share and
    # the count in the same panel come from one estimator on one frame.
    m = F4.attach(m, F4.sdi_reference(), F4.zip5_by_person())
    print("registry %d ; validated %d ; linked to a scored ZCTA %d"
          % (len(m), int(m["val"].sum()), int(m["SDI_score"].notna().sum())))
    rng = np.random.default_rng(SEED)

    rows = []
    for axis, display in AXES:
        point = burden(m, axis)
        boots = {k: [] for k in point}
        for _ in range(NBOOT):
            b = burden(m, axis, rng, resample=True)
            for k, val in b.items():
                boots[k].append(val[2])                  # the uncoded count
        for lvl, (tot, coded, unc) in point.items():
            d = np.array([x for x in boots[lvl] if np.isfinite(x)])
            lo, hi = (np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan))
            n_reg = int((m[axis] == lvl).sum())
            rows.append({"axis": display, "level": lvl, "registry_n": n_reg,
                         "with_aud": tot, "coded_by_t0": coded, "uncoded_at_t0": unc,
                         "ci_low": lo, "ci_high": hi,
                         "pct_coded": 100.0 * coded / tot if tot else np.nan})
        print("  %s done" % display)

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "registry_burden_subgroup.csv"), index=False)
    print()
    print(t.round(0).to_string(index=False))
    print()
    print("total uncoded at t0 over the whole registry, by axis:")
    for axis, _ in AXES:
        g = t[t["axis"].eq(dict(AXES)[axis])]
        print("  %-15s %8.0f uncoded of %8.0f with AUD"
              % (dict(AXES)[axis], g["uncoded_at_t0"].sum(), g["with_aud"].sum()))
    print()
    worst_pct = t.loc[t["pct_coded"].idxmin()]
    worst_n = t.loc[t["uncoded_at_t0"].idxmax()]
    print("lowest share coded: %s %s at %.1f%%, %.0f uncoded"
          % (worst_pct["axis"], worst_pct["level"], worst_pct["pct_coded"],
             worst_pct["uncoded_at_t0"]))
    print("most uncoded patients: %s %s, %.0f uncoded at %.1f%% coded"
          % (worst_n["axis"], worst_n["level"], worst_n["uncoded_at_t0"],
             worst_n["pct_coded"]))
    print("wrote %s/registry_burden_subgroup.csv" % OUT)


if __name__ == "__main__":
    main()
