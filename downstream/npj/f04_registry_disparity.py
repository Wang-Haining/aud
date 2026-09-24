#!/usr/bin/env python3
"""Registry-scale coverage of the AUD diagnosis code, each stratum against the rest.

Producer for the registry-scale panel of Figure 4. The earlier version of that panel drew
the share coded by t0 in each stratum and the number of patients behind it. Both are worth
reporting and neither answers the question the section asks, because a panel of levels has
no null. This computes the quantity that does: the ratio of one stratum's coverage to the
coverage of the rest of its axis, where one is no difference.

A ratio rather than a difference, and a ratio of shares rather than an odds ratio. The
registry-scale shares run from 24 to 45 per cent, so nothing sits near a boundary and the
ratio is stable; and a ratio of shares reads directly, coverage 1.48 times as high, where an
odds ratio does not. The chart-scale panel of this figure uses a difference instead, because
there the model reaches 100 per cent in five strata and a ratio of error rates is undefined.

The estimand is the one icd_fairness defines. For stratum g and code stratum s,

    N_AUD(g, s) = registry count of g in s  x  theta(g, s)

coverage(g) = N_AUD(g, s = 0) / sum_s N_AUD(g, s), and the complement's coverage is formed
from the same cells summed over the other strata of that axis, so the two are disjoint.

Intervals are the bootstrap the rest of the figure uses: validated charts resampled within
code stratum and sampling status, theta from icd_fairness.theta_frame, seed and count from
icd_fairness.

Writes out/figure4/registry_disparity.csv.

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
from icd_fairness import build, NBOOT, SEED, theta_frame, resample_design                               # noqa: E402
import f04_sdi as F4                                                      # noqa: E402
import f04_registry_burden_subgroup as RB                                 # noqa: E402

OUT = "downstream/npj/out/figure4"
AXES = [("agegrp", "Age", ["Under 50", "50 to 64", "65 and over"]),
        ("sex", "Sex", ["Male", "Female"]),
        ("race", "Race", ["Other than White", "White"]),
        ("charlson_noliver", "Charlson index", ["0", "1 to 2", "3 or more"]),
        ("sdi_q", "Neighbourhood deprivation", F4.Q5)]


def cells(m, axis, levels, rng=None, resample=False):
    """{level: (N_AUD coded by t0, N_AUD in total)} on the registry scale."""
    v = theta_frame(m)
    if resample:
        v = resample_design(v, rng)
    out = {}
    for lvl in levels:
        num = den = 0.0
        ok = True
        for s in range(3):
            n_reg = int(((m[axis] == lvl) & (m["S"] == s)).sum())
            sub = v[(v[axis] == lvl) & (v["S"] == s)]
            if len(sub) == 0:
                ok = False
                break
            n_aud = n_reg * float(sub["Aobs"].mean())
            den += n_aud
            if s == 0:
                num = n_aud
        out[lvl] = (num, den) if ok else (np.nan, np.nan)
    return out


def ratios(c, levels):
    """Coverage of each level divided by the coverage of the rest of the axis."""
    tot_n = sum(c[lev][0] for lev in levels)
    tot_d = sum(c[lev][1] for lev in levels)
    out = {}
    for lvl in levels:
        num, den = c[lvl]
        rn, rd = tot_n - num, tot_d - den
        if not np.isfinite(num) or den <= 0 or rd <= 0 or rn <= 0:
            out[lvl] = np.nan
            continue
        out[lvl] = (num / den) / (rn / rd)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    m = RB.attach_charlson(build())
    m = F4.attach(m, F4.sdi_reference(), F4.zip5_by_person())
    print("registry %d ; validated %d" % (len(m), int(m["val"].sum())))

    rows = []
    for axis, display, levels in AXES:
        rng = np.random.default_rng(SEED)
        obs_c = cells(m, axis, levels)
        obs = ratios(obs_c, levels)
        boots = {lev: [] for lev in levels}
        for _ in range(NBOOT):
            b = ratios(cells(m, axis, levels, rng, resample=True), levels)
            for lev in levels:
                boots[lev].append(b[lev])
        print("\n  %s, each level against the rest of the axis" % display)
        for lvl in levels:
            arr = np.array([x for x in boots[lvl] if np.isfinite(x)])
            n_reg = int((m[axis] == lvl).sum())
            cov = 100.0 * obs_c[lvl][0] / obs_c[lvl][1]
            if not np.isfinite(obs[lvl]) or len(arr) < 100:
                print("    %-17s not estimable" % lvl)
                continue
            lo, hi = np.percentile(arr, [2.5, 97.5])
            print("    %-17s coverage %5.1f%%  ratio %5.3f (95%% CI %5.3f to %5.3f)  "
                  "registry n = %5d" % (lvl, cov, obs[lvl], lo, hi, n_reg))
            rows.append({"axis": display, "level": lvl, "coverage_pct": cov,
                         "ratio": obs[lvl], "ci_low": lo, "ci_high": hi,
                         "registry_n": n_reg})

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.out, "registry_disparity.csv"), index=False)
    out = ((t["ci_low"] > 1.0) | (t["ci_high"] < 1.0))
    band = (t["ratio"] < 0.8) | (t["ratio"] > 1.25)
    print("\n  strata %d; ratio outside 0.8 to 1.25 %d; intervals excluding one %d"
          % (len(t), int(band.sum()), int(out.sum())))
    for _, r in t[out].iterrows():
        print("    %-26s %-17s %5.3f (%5.3f to %5.3f)"
              % (r["axis"], r["level"], r["ratio"], r["ci_low"], r["ci_high"]))
    print("  wrote %s/registry_disparity.csv" % a.out)


if __name__ == "__main__":
    main()
