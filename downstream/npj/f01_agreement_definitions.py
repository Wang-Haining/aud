#!/usr/bin/env python3
"""Every agreement definition the study can support, on one page.

Agreement between two readers on this scale can be defined several ways, and each answers a
slightly different question. The paper reports one of them in the main text. This file
computes all of them so the supplement can carry the set, and so a reader who wants a
different definition can find its value rather than recompute it.

The definitions, and what each is for.

  three categories, unweighted      The decision the paper is about: AUD present, AUD
                                    absent, indeterminate. Indeterminate is an output
                                    state, not a missing value, and it is not between the
                                    other two, so no weights are applied. This is the
                                    proposed primary.
  three categories, linear weights  The same collapse, treating indeterminate as lying
                                    between the two directions.
  three categories, Gwet's AC1      The same collapse under a chance-agreement model that
                                    does not fall away when one category dominates.
  two categories                    Restricted to charts where both readers returned a
                                    direction. Answers how well they agree when both
                                    commit, and says nothing about who abstains.
  five categories, linear weights   The four directional grades plus uncertain, dropping
                                    not enough data. Every remaining category lies on one
                                    ordinal scale, so linear weights have a meaning here
                                    that they do not have on six.
  six categories, linear weights    The full scale. Reported twice, under the two category
                                    orderings this pipeline uses, because the value depends
                                    on where the non-ordinal not-enough-data category is
                                    placed and the difference is not negligible.

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

import stats_core as sc                                            # noqa: E402
import ref_standard as RS                                          # noqa: E402

OUT = "downstream/npj/out/agreement"
PANEL = "output/canonical_panel.csv"
RATERS = "output/raters/rater_long.csv"
PRIMARY_TAG = "aud_qwen36_27b"

POS = {"Definitely Yes", "Probably Yes"}
NEG = {"Definitely No", "Probably No"}
NED = "Not Enough Data"

# The two orderings in use. NED_LAST is f04_figure4.LEVELS, which the leaderboard and the
# abstract carry. NED_MID reproduces the reported 0.872 for the two first-pass readers.
NED_LAST = ["Definitely Yes", "Probably Yes", "Uncertain", "Probably No",
            "Definitely No", "Not Enough Data"]
NED_MID = ["Definitely No", "Probably No", "Uncertain", "Not Enough Data",
           "Probably Yes", "Definitely Yes"]
ORD5 = ["Definitely No", "Probably No", "Uncertain", "Probably Yes", "Definitely Yes"]
ORD3 = ["AUD-", "Indeterminate", "AUD+"]
ORD2 = ["AUD-", "AUD+"]
N_BOOT, SEED = 4000, 20260814


def to3(s):
    """Collapse to three, preserving absence. A chart the reader never saw is not
    indeterminate; it has no label, and np.where would silently make it one."""
    out = pd.Series(np.where(s.isin(POS), "AUD+",
                    np.where(s.isin(NEG), "AUD-", "Indeterminate")), index=s.index)
    return out.where(s.notna())


def gwet_ac1(a, b, cats):
    """AC1: chance agreement from the propensity to pick a category, not from its rate."""
    n = len(a)
    po = float((np.asarray(a) == np.asarray(b)).mean())
    pi = [((np.sum(np.asarray(a) == c) + np.sum(np.asarray(b) == c)) / (2 * n))
          for c in cats]
    k = len(cats)
    pe = sum(p * (1 - p) for p in pi) / (k - 1)
    return (po - pe) / (1 - pe)


def boot(fn, a, b):
    rng = np.random.default_rng(SEED)
    a, b = np.asarray(a), np.asarray(b)
    out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(a), len(a))
        try:
            out.append(fn(a[i], b[i]))
        except Exception:
            continue
    return np.nanpercentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def one(name, a, b, fn, note=""):
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    k = fn(a, b)
    lo, hi = boot(fn, a, b)
    return {"definition": name, "n": len(a), "kappa": k, "lo": lo, "hi": hi,
            "percent_agreement": 100 * float((a.to_numpy() == b.to_numpy()).mean()),
            "note": note}


def block(a6, b6, who):
    """Every definition, for one pair of readers given their six-grade labels."""
    a3, b3 = to3(a6), to3(b6)
    rows = []

    rows.append(one("Three categories, unweighted", a3, b3,
                    lambda x, y: sc.cohens_kappa(x, y, ORD3, None),
                    "the decision the paper reports"))
    rows.append(one("Three categories, linear weights", a3, b3,
                    lambda x, y: sc.cohens_kappa(x, y, ORD3, "linear"),
                    "treats indeterminate as lying between the two directions"))
    rows.append(one("Three categories, Gwet's AC1", a3, b3,
                    lambda x, y: gwet_ac1(x, y, ORD3),
                    "chance agreement that does not collapse when one category dominates"))

    # Two categories: both readers committed to a direction.
    m2 = a3.isin(["AUD+", "AUD-"]) & b3.isin(["AUD+", "AUD-"])
    rows.append(one("Two categories, both readers committed", a3[m2], b3[m2],
                    lambda x, y: sc.cohens_kappa(x, y, ORD2, None),
                    "excludes every chart either reader left indeterminate"))

    # Five categories: the one ordinal scale, dropping not enough data.
    m5 = a6.ne(NED) & b6.ne(NED)
    rows.append(one("Five categories, linear weights", a6[m5], b6[m5],
                    lambda x, y: sc.cohens_kappa(x, y, ORD5, "linear"),
                    "drops not enough data, so the remaining categories are ordinal"))

    # Six categories, every position not enough data could occupy. Reporting two
    # orderings was worse than reporting none: the two this pipeline happens to contain
    # are 0.004 apart, which understates the choice by a factor of six.
    m6 = a6.notna() & b6.notna()
    vals = []
    for i in range(6):
        order = ORD5[:i] + [NED] + ORD5[i:]
        vals.append(sc.cohens_kappa(a6[m6], b6[m6], order, "linear"))
    rows.append({"definition": "Six categories, linear weights", "n": int(m6.sum()),
                 "kappa": float(np.median(vals)), "lo": float(min(vals)),
                 "hi": float(max(vals)),
                 "percent_agreement": 100 * float(
                     (a6[m6].to_numpy() == b6[m6].to_numpy()).mean()),
                 "note": "lo and hi are the range over the six positions not enough "
                          "data can occupy, not a confidence interval"})

    d = pd.DataFrame(rows)
    d.insert(0, "comparison", who)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--raters", default=RATERS)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ref, man = RS.build()
    assert len(ref) == man["cohort"]["analysed"] == 487
    ref["omop_person_id"] = ref["omop_person_id"].astype(str)

    p = pd.read_csv(a.panel, dtype={"omop_person_id": str}, low_memory=False)
    p = p[p["model"].eq(PRIMARY_TAG)][["omop_person_id", "label_canonical"]]
    d = ref.merge(p, on="omop_person_id", how="left")

    r = pd.read_csv(a.raters, dtype={"omop_person_id": str})
    r = r[r["in_analysis_cohort"].astype(str).str.lower().eq("true")]
    r6 = r.pivot_table(index="omop_person_id", columns="rater", values="label_6",
                       aggfunc="first")
    d = d.merge(r6[["maaz", "sai"]], left_on="omop_person_id", right_index=True,
                how="left")

    blocks = [
        block(d["gold_label"], d["label_canonical"],
              "LLM against the reference standard"),
        block(d["maaz"], d["sai"],
              "The two first-pass readers, before any adjudication"),
        block(d["label_canonical"], d["sai"],
              "LLM against the second reader, first-pass labels"),
    ]
    t = pd.concat(blocks, ignore_index=True)
    t.to_csv(os.path.join(a.out, "agreement_definitions.csv"), index=False)

    for who, g in t.groupby("comparison", sort=False):
        print("=== %s" % who)
        s = g[["definition", "n", "kappa", "lo", "hi", "percent_agreement"]].copy()
        for c in ("kappa", "lo", "hi"):
            s[c] = s[c].round(3)
        s["percent_agreement"] = s["percent_agreement"].round(1)
        print(s.to_string(index=False))
        print()

    print("LaTeX rows, LLM against the reference standard")
    g = t[t["comparison"].str.startswith("LLM against the reference")]
    for _, r_ in g.iterrows():
        print("%-58s & %3d & %.3f (%.3f--%.3f) & %.1f\\%% \\\\"
              % (r_["definition"], r_["n"], r_["kappa"], r_["lo"], r_["hi"],
                 r_["percent_agreement"]))
    print("\nwrote %s/agreement_definitions.csv" % a.out)


if __name__ == "__main__":
    main()
