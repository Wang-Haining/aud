#!/usr/bin/env python3
"""Every agreement value the manuscript reports, under both definitions.

The paper reports linearly weighted kappa on the six-grade scale. That metric depends on
where the non-ordinal `Not Enough Data` category is placed, and this pipeline places it in
two positions: f04_figure4.LEVELS puts it last, which is where the abstract's 0.92 comes
from, and the ordering that reproduces the reported 0.872 for the two first-pass readers
puts it fourth. Same data, two answers.

The proposed replacement is unweighted kappa on the three categories the paper already
collapses to, which carries no ordering assumption and is the collapse Figure 1b draws.

This file changes nothing. It recomputes every value under both definitions and prints the
pair, so the scope of the migration and its cost are visible before anything is edited.

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
import fig6_bayes_strata as B                                      # noqa: E402

OUT = "downstream/npj/out/kappa_inventory"
PANEL = "output/canonical_panel.csv"
RATERS = "output/raters/rater_long.csv"
INST = "output/fairness/patient_institution.csv"

# The ordering the leaderboard and the abstract use.
LEVELS = ["Definitely Yes", "Probably Yes", "Uncertain", "Probably No",
          "Definitely No", "Not Enough Data"]
ORD3 = ["AUD-", "Indeterminate", "AUD+"]
POS = {"Definitely Yes", "Probably Yes"}
NEG = {"Definitely No", "Probably No"}

DISPLAY = {
    "aud_qwen36_27b": "Qwen3.6-27B", "aud_nemotron_fp8": "Nemotron-3-120B",
    "aud": "GPT-OSS-120B", "aud_glm52": "GLM-5.2",
    "aud_nemotron_llama": "Nemotron-Llama-49B", "aud_gptoss_20b": "GPT-OSS-20B",
    "aud_deepseek_v4_flash_0731": "DeepSeek-V4-Flash", "aud_gemma4": "Gemma4-31B",
    "aud_llama4_scout": "Llama-4-Scout", "aud_mistral32_24b": "Mistral-3.2-24B",
    "aud_llama33_70b": "Llama-3.3-70B",
}
PRIMARY = "Qwen3.6-27B"


def to3(s):
    return pd.Series(np.where(s.isin(POS), "AUD+",
                     np.where(s.isin(NEG), "AUD-", "Indeterminate")), index=s.index)


def both(a, b):
    """(six-class linear, three-category unweighted) on the same two label vectors."""
    k6 = sc.cohens_kappa(a, b, LEVELS, "linear")
    k3 = sc.cohens_kappa(to3(a), to3(b), ORD3, None)
    return k6, k3


def load(a):
    ref, man = RS.build()
    assert len(ref) == man["cohort"]["analysed"] == 487
    ref["omop_person_id"] = ref["omop_person_id"].astype(str)
    ref["pid"] = pd.to_numeric(ref["omop_person_id"], errors="coerce").astype("int64")

    p = pd.read_csv(a.panel, dtype={"omop_person_id": str}, low_memory=False)
    p = p[p["model"].isin(DISPLAY)].copy()
    p["name"] = p["model"].map(DISPLAY)
    w = p.pivot(index="omop_person_id", columns="name", values="label_canonical")

    d = ref.set_index("omop_person_id").join(w)
    inst = pd.read_csv(a.inst)
    inst["omop_person_id"] = inst["person_id"].astype(str)
    d = d.join(inst.set_index("omop_person_id")["institution"])

    reg = B.build()
    reg["pid"] = reg["person_id"].astype("int64")
    # merge on a column drops the index, so carry omop_person_id as a column first
    d = d.reset_index().merge(reg[["pid", "t0"]], on="pid", how="left")
    d["period"] = pd.cut(pd.to_datetime(d["t0"]).dt.year,
                         [2014, 2016, 2018, 2021],
                         labels=["2015-2016", "2017-2018", "2019-2020"])

    r = pd.read_csv(a.raters, dtype={"omop_person_id": str})
    r = r[r["in_analysis_cohort"].astype(str).str.lower().eq("true")]
    r6 = r.pivot_table(index="omop_person_id", columns="rater", values="label_6",
                       aggfunc="first")
    d = d.merge(r6[["maaz", "sai"]], left_on="omop_person_id", right_index=True,
                how="left")
    return d, man


def rows(d):
    out = []

    def add(where, what, a, b, n=None):
        m = a.notna() & b.notna()
        k6, k3 = both(a[m], b[m])
        out.append({"where": where, "quantity": what, "n": int(m.sum()),
                    "six_class_linear": k6, "three_cat_unweighted": k3,
                    "delta": k3 - k6})

    g = d["gold_label"]

    # Abstract and the leaderboard, all 11 models against the reference standard.
    for name in sorted(DISPLAY.values()):
        add("Fig 4a, Supp Table 3", "%s against the reference" % name, g, d[name])

    # Findings 1 headline: the two kappas the text calls equal at 0.92.
    add("Findings 1", "LLM against the reference, full standard", g, d[PRIMARY])

    # The reliability pair, on the charts the second reader read.
    add("Findings 1, Supp Table 8", "M.A. against S.N., first-pass labels",
        d["maaz"], d["sai"])
    add("Findings 1, Supp Table 8", "LLM against S.N., first-pass labels",
        d[PRIMARY], d["sai"])

    # External validation, the primary model per hospital system.
    SITE = {"IUH": "academic", "Eskenazi": "safety net"}
    d = d.copy()
    d["site"] = d["institution"].map(SITE).fillna("other hospitals")
    for s in ("academic", "safety net", "other hospitals"):
        m = d["site"].eq(s)
        add("Fig 2a, Supp Table 9", "LLM at the %s" % s, g[m], d.loc[m, PRIMARY])

    # The referred stratum against the rest.
    ref_set = d["gold_source"].ne("maaz_unreviewed")
    add("Fig 4c", "LLM on the 422 never referred", g[~ref_set], d.loc[~ref_set, PRIMARY])
    add("Fig 4c", "LLM on the 65 referred", g[ref_set], d.loc[ref_set, PRIMARY])

    # Calendar era.
    for band in ("2015-2016", "2017-2018", "2019-2020"):
        m = d["period"].eq(band)
        add("Supp Fig 8", "LLM in %s" % band, g[m], d.loc[m, PRIMARY])

    # The pairwise matrix, summarised rather than listed.
    names = sorted(DISPLAY.values())
    p6, p3 = [], []
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            m = d[x].notna() & d[y].notna()
            a, b = both(d.loc[m, x], d.loc[m, y])
            p6.append(a)
            p3.append(b)
    for lab, f in (("minimum", np.min), ("median", np.median), ("maximum", np.max)):
        out.append({"where": "Fig 4b", "quantity": "pairwise, %s of 55 pairs" % lab,
                    "n": 487, "six_class_linear": f(p6),
                    "three_cat_unweighted": f(p3), "delta": f(p3) - f(p6)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--raters", default=RATERS)
    ap.add_argument("--inst", default=INST)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d, man = load(a)
    t = rows(d)
    t.to_csv(os.path.join(a.out, "kappa_inventory.csv"), index=False)

    show = t.copy()
    for c in ("six_class_linear", "three_cat_unweighted", "delta"):
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    print()
    print("values that move by more than 0.02: %d of %d"
          % (int(t["delta"].abs().gt(0.02).sum()), len(t)))
    print("largest move: %+.3f, %s"
          % (t.loc[t["delta"].abs().idxmax(), "delta"],
             t.loc[t["delta"].abs().idxmax(), "quantity"]))
    print("mean move: %+.3f" % t["delta"].mean())
    print()
    lb = t[t["where"].eq("Fig 4a, Supp Table 3")].copy()
    lb["r6"] = lb["six_class_linear"].rank(ascending=False)
    lb["r3"] = lb["three_cat_unweighted"].rank(ascending=False)
    moved = lb[lb["r6"].ne(lb["r3"])]
    print("leaderboard: %d of 11 models change rank" % len(moved))
    if len(moved):
        print(moved[["quantity", "six_class_linear", "r6",
                     "three_cat_unweighted", "r3"]].to_string(index=False))
    print("\nnot covered here, needs its own rerun: the five prompt and temperature "
          "conditions of Fig 4d, whose labels live in downstream/npj_goal5_outputs")
    print("\nwrote %s/kappa_inventory.csv" % a.out)


if __name__ == "__main__":
    main()
