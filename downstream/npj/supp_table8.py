#!/usr/bin/env python3
"""Supplementary Table 8: the hepatology fellow's independent review, on the closed standard.

S.N. read a random subsample blind to M.A.'s ratings and to all model output. The table
reports how much of each reference category he covered, and how far the two first-pass
readings agree.

Two values in the version carried until 14 August cannot stand.

The unseen stratum was given as 438. On the closed standard it is 422.

Agreement was reported on the unseen stratum alone, at 95.2 per cent. That quantity cannot
be measured there. S.N.'s disagreement with M.A. is the second referral trigger, so a chart
on which the two differ is referred to attending review by construction, and the unseen
stratum holds no disagreement by definition: the count is 156 of 156. The informative
figures are the agreement over every chart S.N. read and the two kappas, and those are what
this script emits.

The kappas use the primary definition: three categories, AUD present, AUD absent and
uncertain, unweighted, with a chart dropped when either reader returned not enough data.
The denominators are therefore smaller than the 186 charts S.N. read, and differ between
the two comparisons.

The coverage table above uses a different three-category collapse, and the difference is
deliberate: it counts two readers who both returned an indeterminate grade as agreeing,
which is what a coverage table should report.

Values under the earlier six-category linear-weighted metric, for reference: 0.872 for
M.A. against S.N. and 0.841 for the primary model, both on 186 charts. The 0.843 the
manuscript carried until 14 August was the 189-chart value, computed on the roster rather
than the cohort.

Intervals are percentile bootstrap, 2000 resamples, seed 20260814. The count matches the
figure producers, so an interval here and an interval in a figure are drawn the same way.

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
sys.path.insert(0, "downstream/npj")

import ref_standard as RS                                          # noqa: E402
import stats_core as sc                                            # noqa: E402

OUT = "downstream/npj/out/supp_table8"
RATERS = "output/raters/rater_long.csv"
PRIMARY = "aud_qwen36_27b"
ORD6 = ["Definitely No", "Probably No", "Uncertain", "Not Enough Data",
        "Probably Yes", "Definitely Yes"]
CAT = [("AUD+", "AUD present"), ("AUD-", "AUD absent"), ("abstain", "Indeterminate")]
N_BOOT, SEED = 2000, 20260814

POS = ("Definitely Yes", "Probably Yes")
NEG = ("Definitely No", "Probably No")
NED = "Not Enough Data"
ORD3_PRIMARY = ["AUD-", "Uncertain", "AUD+"]


def collapse3(s):
    """The primary definition. Not Enough Data becomes missing and drops the chart."""
    out = pd.Series(np.where(s.isin(POS), "AUD+",
                    np.where(s.isin(NEG), "AUD-",
                    np.where(s.eq("Uncertain"), "Uncertain", None))), index=s.index)
    return out.where(s.notna() & s.ne(NED))


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    z, p = 1.959963985, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h))


def boot_kappa(a, b):
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(a), len(a))
        try:
            out.append(sc.cohens_kappa(a[i], b[i], ORD3_PRIMARY, weights=None))
        except Exception:
            continue
    return np.nanpercentile(out, [2.5, 97.5])


def load(a):
    ref, man = RS.build()
    assert len(ref) == man["cohort"]["analysed"] == 487, "reference standard is not the 487"
    ref = ref[["omop_person_id", "gold_binary", "gold_source"]].copy()
    ref["omop_person_id"] = ref["omop_person_id"].astype(str)

    r = pd.read_csv(a.raters, dtype={"omop_person_id": str})
    r = r[r["in_analysis_cohort"].astype(str).str.lower().eq("true")]

    # The rater table must agree with the reference standard on the escalation, or the
    # coverage column describes a different partition from the one the caption names.
    n_gaw = int((r["rater"] == "gawrieh").sum())
    n_ref = int(ref["gold_source"].eq("gawrieh_adjudicated").sum())
    if n_gaw != n_ref:
        raise SystemExit(
            "rater_long.csv is stale: %d gawrieh rows against the standard's %d. "
            "Run downstream/build_rater_table.py first." % (n_gaw, n_ref))

    w3 = r.pivot_table(index="omop_person_id", columns="rater", values="label_3",
                       aggfunc="first").add_suffix("_3")
    w6 = r.pivot_table(index="omop_person_id", columns="rater", values="label_6",
                       aggfunc="first").add_suffix("_6")
    w = w3.join(w6).reset_index()
    return ref.merge(w, on="omop_person_id", how="left"), man


def coverage(d):
    rows = []
    for key, label in CAT:
        g = d[d["gold_binary"].eq(key)]
        rows.append({"category": label, "charts": len(g),
                     "rated": int(g["sai_3"].notna().sum()),
                     "coverage": 100 * g["sai_3"].notna().mean()})
    rows.append({"category": "All", "charts": len(d),
                 "rated": int(d["sai_3"].notna().sum()),
                 "coverage": 100 * d["sai_3"].notna().mean()})
    return pd.DataFrame(rows)


def agreement(d):
    g = d[d["sai_3"].notna() & d["maaz_3"].notna()]
    dis = g[g["maaz_3"].ne(g["sai_3"])]
    referred = g["gold_source"].ne("maaz_unreviewed")
    pc = wilson(len(g) - len(dis), len(g))
    return {
        "n_read": len(g),
        "n_agree": len(g) - len(dis),
        "pct": pc[0], "lo": pc[1], "hi": pc[2],
        "n_disagree": len(dis),
        "disagree_referred": int(dis["gold_source"].ne("maaz_unreviewed").sum()),
        "unseen_read": int((~referred).sum()),
        "unseen_agree": int(((~referred) & g["maaz_3"].eq(g["sai_3"])).sum()),
        "referred_read": int(referred.sum()),
        "referred_agree": int((referred & g["maaz_3"].eq(g["sai_3"])).sum()),
    }


def kappas(d):
    out = []
    for name, col in (("M.A. against S.N.", "maaz_6"),
                      ("Primary model against S.N.", PRIMARY + "_6")):
        sai, other = collapse3(d["sai_6"]), collapse3(d[col])
        m = sai.notna() & other.notna()
        a, b = other[m].to_numpy(), sai[m].to_numpy()
        k = sc.cohens_kappa(a, b, ORD3_PRIMARY, weights=None)
        lo, hi = boot_kappa(a, b)
        out.append({"comparison": name, "n": int(m.sum()), "kappa": k, "lo": lo, "hi": hi})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raters", default=RATERS)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d, man = load(a)
    cov, ag, k = coverage(d), agreement(d), kappas(d)
    cov.to_csv(os.path.join(a.out, "supp_table8_coverage.csv"), index=False)
    k.to_csv(os.path.join(a.out, "supp_table8_kappa.csv"), index=False)
    pd.DataFrame([ag]).to_csv(os.path.join(a.out, "supp_table8_agreement.csv"), index=False)

    print("analysis cohort   %d" % len(d))
    print("unseen stratum    %d" % int(d["gold_source"].eq("maaz_unreviewed").sum()))
    print()
    print(cov.to_string(index=False))
    print()
    print("agreement, raw first-pass labels, three categories")
    print("  read by both readers        %d" % ag["n_read"])
    print("  agree                       %d  (%.1f%%, %.1f--%.1f)"
          % (ag["n_agree"], ag["pct"], ag["lo"], ag["hi"]))
    print("  disagree                    %d" % ag["n_disagree"])
    print("  of those, referred to A.D.  %d   <- the second trigger, so this is always all"
          % ag["disagree_referred"])
    print("  unseen stratum              %d of %d, which is 100%% by construction"
          % (ag["unseen_agree"], ag["unseen_read"]))
    print("  referred stratum            %d of %d"
          % (ag["referred_agree"], ag["referred_read"]))
    print()
    print(k.to_string(index=False))
    print()
    print("LaTeX")
    for _, r in cov.iterrows():
        print("%-19s & %3d & %3d & %.0f\\%% \\\\"
              % (r["category"], r["charts"], r["rated"], r["coverage"]))
    print()
    for _, r in k.iterrows():
        print("%-30s & %3d & %.3f (%.3f--%.3f) \\\\"
              % (r["comparison"], r["n"], r["kappa"], r["lo"], r["hi"]))


if __name__ == "__main__":
    main()
