#!/usr/bin/env python3
"""Refuse to finish when two products disagree, or when one is older than its input.

Written 14 August, after the manuscript was found to carry two versions of the reference
standard at once. Table 1 and the STARD flow gave 248, 176 and 63; Figure 1 and
Supplementary Table 8 gave 251, 174 and 62. Both were internally consistent. Nothing
compared them, so nothing failed.

Two kinds of check, and the first matters more.

Staleness. A product older than the rating files it derives from is wrong whatever it
says, and no count comparison can catch it if the product is simply not read by anything
else. Every product below must be newer than every input below. This is the check that
would have caught Table 1 on 13 August, the day the standard closed.

Identity. Where two products carry the same quantity, they must agree, and both must agree
with ref_standard.build(). The reference standard is the single source; a product is only
allowed to restate it.

The script also writes resources/canonical_counts.json, which is the registry the
manuscript-side check reads. Nothing in the paper may state a cohort count that is not in
that file.

Exit status is 1 on any failure, so rebuild_all.sh stops.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import json
import pathlib
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "downstream")
sys.path.insert(0, "downstream/npj")

import ref_standard as RS                                          # noqa: E402

OUT = Path("downstream/npj/out")
PANELS = Path("downstream/npj_goal6_outputs/panels")
REGISTRY = "resources/canonical_counts.json"

# If any of these is newer than a product, that product is stale.
INPUTS = [
    "chart/ratings/desai.json",
    "chart/ratings/gawrieh.json",
    "resources/patient_roster.csv",
    "resources/3-3 Maaz and Sai inter-rater reliability.xlsx",
]

# Every product the manuscript rests on, with the figure or table it feeds.
PRODUCTS = [
    (OUT / "f01_flow_table1/flow_counts.csv", "Supplementary Fig. 1, the STARD flow"),
    (OUT / "f01_flow_table1/table1.csv", "Table 1"),
    (PANELS / "f1_confusion_6class.csv", "Fig. 1b and Supplementary Fig. 3"),
    (PANELS / "f2_alluvial.csv", "Fig. 1c"),
    (OUT / "f01_accuracy/predictive_value_inputs.csv", "Fig. 1d and 1f"),
    (OUT / "f01_accuracy/reader_estimates.csv", "Fig. 1d"),
    (OUT / "f01_accuracy/confusion_three.csv", "no live consumer; kept current so it "
                                              "cannot be read as if it were"),
    (OUT / "figure2/figure2.pdf", "Fig. 2"),
    (OUT / "figure2/figure2_era.csv", "Supplementary Fig. 8"),
    (OUT / "figure3/figure3.pdf", "Fig. 3"),
    (OUT / "f03_registry/registry_burden.csv", "Fig. 3d"),
    (OUT / "figure4/figure4.pdf", "Fig. 4"),
    (OUT / "figure5/figure5.pdf", "Fig. 5"),
    (OUT / "figure6/figure6.pdf", "Fig. 6"),
    (OUT / "supp_table8/supp_table8_coverage.csv", "Supplementary Table 8"),
    (OUT / "purity/purity_by_site.csv", "Supplementary Table 9, the note-purity rows"),
    (OUT / "supp_table6.tex", "Supplementary Table 6"),
]

fails: list[str] = []
notes: list[str] = []


def fail(msg):
    fails.append(msg)


def ok(msg):
    notes.append(msg)


def canonical():
    """The numbers the manuscript is allowed to state, from the single source."""
    ref, man = RS.build()
    ref["omop_person_id"] = ref["omop_person_id"].astype(str)
    ros = pd.read_csv("resources/patient_roster.csv", dtype={"omop_person_id": str})
    d = ref.merge(ros[["omop_person_id", "has_aud_icd"]], on="omop_person_id", how="left")
    coded = d["has_aud_icd"].fillna(0).astype(int).eq(1)
    pos = d["gold_binary"].eq("AUD+")
    c = {
        "analysed": int(man["cohort"]["analysed"]),
        "roster_rows": int(man["cohort"]["roster_rows"]),
        "ref_pos": int(pos.sum()),
        "ref_neg": int(d["gold_binary"].eq("AUD-").sum()),
        "ref_indet": int(d["gold_binary"].eq("abstain").sum()),
        "sampled_coded": int(coded.sum()),
        "sampled_uncoded": int((~coded).sum()),
        "uncoded_ref_pos": int((~coded & pos).sum()),
        "referred": int(d["gold_source"].ne("maaz_unreviewed").sum()),
        "not_referred": int(d["gold_source"].eq("maaz_unreviewed").sum()),
        "adjudicated": int(d["gold_source"].eq("gawrieh_adjudicated").sum()),
    }
    c["determinate"] = c["ref_pos"] + c["ref_neg"]
    return c


def check_internal(c):
    if c["ref_pos"] + c["ref_neg"] + c["ref_indet"] != c["analysed"]:
        fail("the three reference categories do not sum to the analysis cohort")
    if c["sampled_coded"] + c["sampled_uncoded"] != c["analysed"]:
        fail("coded plus uncoded does not sum to the analysis cohort")
    if c["referred"] + c["not_referred"] != c["analysed"]:
        fail("referred plus not referred does not sum to the analysis cohort")
    ok("reference standard internally consistent at %d" % c["analysed"])


def check_staleness():
    newest, newest_name = 0.0, ""
    for p in INPUTS:
        if not os.path.exists(p):
            fail("input missing: %s" % p)
            continue
        t = os.path.getmtime(p)
        if t > newest:
            newest, newest_name = t, p
    stale = 0
    for path, who in PRODUCTS:
        if not path.exists():
            fail("product missing: %s  (%s)" % (path, who))
            continue
        if os.path.getmtime(path) < newest:
            fail("STALE: %s is older than %s  (%s)" % (path, newest_name, who))
            stale += 1
    if not stale:
        ok("every product is newer than the newest rating input, %s" % newest_name)


def check_flow(c):
    f = pd.read_csv(OUT / "f01_flow_table1/flow_counts.csv").set_index("key")["n"]
    for key, want in (("ref_pos", c["ref_pos"]), ("ref_neg", c["ref_neg"]),
                      ("ref_indet", c["ref_indet"]), ("sampled", c["analysed"]),
                      ("sampled_coded", c["sampled_coded"]),
                      ("sampled_uncoded", c["sampled_uncoded"]),
                      ("analysed_accuracy", c["determinate"]),
                      ("ref_read_twice", c["referred"]),
                      ("ref_single", c["not_referred"])):
        if key not in f.index:
            fail("flow_counts.csv has no row %s" % key)
        elif int(f[key]) != want:
            fail("flow_counts.csv %s = %d, reference standard says %d"
                 % (key, int(f[key]), want))
    ok("the STARD flow agrees with the reference standard")


def check_table1(c):
    t = pd.read_csv(OUT / "f01_flow_table1/table1.csv", index_col=0)
    col = t.columns[-1]
    want = {"AUD present": c["ref_pos"], "AUD absent": c["ref_neg"],
            "Indeterminate": c["ref_indet"]}
    for label, n in want.items():
        rows = [i for i in t.index if str(i).strip() == label]
        if not rows:
            fail("Table 1 has no reference-standard row %r" % label)
            continue
        cell = str(t.loc[rows[0], col])
        if not cell.strip().startswith(str(n)):
            fail("Table 1 %s reads %r, reference standard says %d" % (label, cell, n))
    ok("Table 1 agrees with the reference standard")


def check_confusion(c):
    d = pd.read_csv(PANELS / "f1_confusion_6class.csv")
    q = d[d["method"] == "Qwen3.6-27B"]
    if int(q["n"].sum()) != c["analysed"]:
        fail("the confusion matrix totals %d, not %d" % (int(q["n"].sum()), c["analysed"]))
    C = {"Definitely Yes": "AUD+", "Probably Yes": "AUD+", "Definitely No": "AUD-",
         "Probably No": "AUD-", "Uncertain": "Ind", "Not Enough Data": "Ind"}
    tot = q.assign(r=q["reference_label"].map(C)).groupby("r")["n"].sum()
    for key, want in (("AUD+", c["ref_pos"]), ("AUD-", c["ref_neg"]),
                      ("Ind", c["ref_indet"])):
        if int(tot.get(key, -1)) != want:
            fail("confusion matrix reference row %s = %s, standard says %d"
                 % (key, tot.get(key), want))
    ok("Fig. 1b agrees with the reference standard")


def check_alluvial(c):
    a = pd.read_csv(PANELS / "f2_alluvial.csv")
    q = a[a["method"] == "Qwen3.6-27B"]
    if int(q["n"].sum()) != c["analysed"]:
        fail("the alluvial totals %d, not %d" % (int(q["n"].sum()), c["analysed"]))
    # The recovery denominator: no AUD code, reference positive.
    m = q[q["icd"].astype(str).str.lower().str.contains("no|absent|negative")
          & q["reference"].astype(str).str.contains(r"AUD\+|present", case=False)]
    if len(m) and int(m["n"].sum()) != c["uncoded_ref_pos"]:
        fail("the alluvial's uncoded reference-positive band is %d, the standard says %d"
             % (int(m["n"].sum()), c["uncoded_ref_pos"]))
    ok("Fig. 1c totals the analysis cohort")


def check_predictive(c):
    p = pd.read_csv(OUT / "f01_accuracy/predictive_value_inputs.csv")
    for _, r in p.iterrows():
        if int(r["n_pos"]) != c["ref_pos"] or int(r["n_neg"]) != c["ref_neg"]:
            fail("predictive_value_inputs %s has n_pos %d and n_neg %d, standard says "
                 "%d and %d" % (r["name"], int(r["n_pos"]), int(r["n_neg"]),
                                c["ref_pos"], c["ref_neg"]))
        if int(r["tp"] + r["fn"] + r["indeterminate_on_pos"]) != int(r["n_pos"]):
            fail("predictive_value_inputs %s: tp + fn + indeterminate != n_pos" % r["name"])
        if int(r["tn"] + r["fp"] + r["indeterminate_on_neg"]) != int(r["n_neg"]):
            fail("predictive_value_inputs %s: tn + fp + indeterminate != n_neg" % r["name"])
    icd = p[p["name"].str.contains("ICD")]
    if len(icd) and int(icd["fn"].iloc[0]) != c["uncoded_ref_pos"]:
        fail("the code's false negatives are %d, the uncoded reference-positive count is %d"
             % (int(icd["fn"].iloc[0]), c["uncoded_ref_pos"]))
    ok("Fig. 1d and 1f rest on the reference standard's denominators")


def check_supp8(c):
    s = pd.read_csv(OUT / "supp_table8/supp_table8_coverage.csv")
    want = {"AUD present": c["ref_pos"], "AUD absent": c["ref_neg"],
            "Indeterminate": c["ref_indet"], "All": c["analysed"]}
    for _, r in s.iterrows():
        if r["category"] in want and int(r["charts"]) != want[r["category"]]:
            fail("Supplementary Table 8 %s = %d, standard says %d"
                 % (r["category"], int(r["charts"]), want[r["category"]]))
    ok("Supplementary Table 8 agrees with the reference standard")


def check_era(c):
    e = pd.read_csv(OUT / "figure2/figure2_era.csv")
    if int(e["n"].sum()) > c["analysed"]:
        fail("the calendar bands hold %d charts, more than the cohort" % int(e["n"].sum()))
    if int(e["n_pos"].sum()) > c["ref_pos"]:
        fail("the calendar bands hold %d reference-positive charts, more than the standard's"
             % int(e["n_pos"].sum()))
    ok("Supplementary Fig. 8 sits inside the cohort, %d of %d charts in a band"
       % (int(e["n"].sum()), c["analysed"]))


# The manuscript repository carries its own copy of every figure, under the name the .tex
# includes. Nothing compared the two until 15 August, and a stale six-class confusion
# matrix sat in the manuscript for several days: its Uncertain row held 8 charts where the
# closed standard has 7. Both versions totalled 487, so no count check saw it. The hashes
# below travel with the counts, and check_manuscript.py compares them against the copies
# beside the .tex.
# The four main figures are assembled by hand from the panel pdfs, so their bytes are
# not a pipeline product and cannot be hashed against one. What is listed here is every
# figure the manuscript includes verbatim from this pipeline. A figure that leaves here
# and reaches the repository unchanged belongs in this list; one that is composed by
# hand does not, and is covered instead by the panel that feeds it.
FIGURES = [
    ("cohort_flow.pdf", OUT / "f01_flow_table1/flow.pdf"),
    ("supp_reader_swarm.pdf", OUT / "figure4/panel_supp_reader_swarm.pdf"),
    ("methods_model_schematic.pdf", OUT / "methods/methods_model_schematic.pdf"),
    ("transportability_era.pdf", OUT / "figure2/figure2_era.pdf"),
    ("supp_documentation.pdf", OUT / "figure4/supp_documentation.pdf"),
]


def figure_md5():
    out = {}
    for name, path in FIGURES:
        path = pathlib.Path(path)
        if not path.exists():
            fail("no product for %s at %s" % (name, path))
            continue
        out[name] = hashlib.md5(path.read_bytes()).hexdigest()
    return out


# Every kappa the manuscript quotes as "value (low--high)", with the producer that draws
# the panel it belongs to. Eighteen of these had drifted from their producers by 15 August,
# because the text was populated from an independent check that ran a different number of
# bootstrap resamples. The point estimates agreed and only the endpoints moved, which is
# why nothing caught it. check_manuscript.py now looks for each triple in the .tex.
KAPPA_SOURCES = [
    (OUT / "figure4/panel_a_leaderboard.csv", "name", "kappa_w", "ci_low", "ci_high",
     None, "Fig. 4a and Supplementary Table 3"),
    (OUT / "figure4/diagnosis_code.csv", "name", "kappa_w", "ci_low", "ci_high",
     None, "the diagnosis code, Findings 2"),
    (OUT / "figure4/panel_c_by_route.csv", "name", "kappa_w", "ci_low", "ci_high",
     ("set", "referred", "name", "Qwen3.6-27B"), "Fig. 4c, the referred stratum"),
    (OUT / "supp_table8/supp_table8_kappa.csv", "comparison", "kappa", "lo", "hi",
     None, "Supplementary Table 8"),
    (pathlib.Path("downstream/npj_goal7_outputs/site_fig_kappa.csv"),
     "site", "kappa_w", "ci_low", "ci_high",
     ("is_primary", "True", "site", ("IUH", "Eskenazi", "Other")), "Fig. 2a"),
]


def published_kappa():
    out = []
    for path, label, k, lo, hi, keep, who in KAPPA_SOURCES:
        path = pathlib.Path(path)
        if not path.exists():
            fail("no product for %s at %s" % (who, path))
            continue
        d = pd.read_csv(path)
        if keep:
            col, want, col2, want2 = keep
            d = d[d[col].astype(str).eq(want)]
            d = d[d[col2].astype(str).isin(want2 if isinstance(want2, tuple) else (want2,))]
        for _, r in d.iterrows():
            out.append({"who": who, "what": str(r[label]),
                        "kappa": round(float(r[k]), 3),
                        "lo": round(float(r[lo]), 3), "hi": round(float(r[hi]), 3)})
    return out


# Percentages the manuscript states in prose, with the producer that computes them. A
# percentage in a sentence is covered by no other check here: the count checks read
# products, the figure check reads bytes and the kappa check reads triples. The anchor is
# the phrase the sentence uses, and check_manuscript.py requires the percentage nearest
# before that phrase to be this value.
PCT_SOURCES = [
    (OUT / "figure1/supp_table1_shares.csv", "the six-class confusion matrix"),
]


def published_pct():
    out = []
    for path, who in PCT_SOURCES:
        path = pathlib.Path(path)
        if not path.exists():
            fail("no product for %s at %s" % (who, path))
            continue
        for _, r in pd.read_csv(path).iterrows():
            out.append({"who": who, "what": str(r["quantity"]),
                        "anchor": str(r["anchor"]), "pct": round(float(r["pct"]), 1)})
    return out


def main():
    c = canonical()
    check_internal(c)
    check_staleness()
    for fn in (check_flow, check_table1, check_confusion, check_alluvial,
               check_predictive, check_supp8, check_era):
        try:
            fn(c)
        except FileNotFoundError as e:
            fail("%s could not run: %s" % (fn.__name__, e))
        except Exception as e:                                     # noqa: BLE001
            fail("%s raised %s: %s" % (fn.__name__, type(e).__name__, e))

    record = dict(c)
    record["figure_md5"] = figure_md5()
    record["published_kappa"] = published_kappa()
    record["published_pct"] = published_pct()

    with open(REGISTRY, "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)

    print("canonical counts, written to %s" % REGISTRY)
    for k in sorted(c):
        print("  %-18s %d" % (k, c[k]))
    print()
    for n in notes:
        print("  ok    %s" % n)
    if fails:
        print()
        for f in fails:
            print("  FAIL  %s" % f)
        print("\n%d check%s failed. The products do not describe one cohort."
              % (len(fails), "" if len(fails) == 1 else "s"))
        return 1
    print("\nall checks passed; every product describes the same %d charts" % c["analysed"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
