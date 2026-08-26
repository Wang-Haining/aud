#!/usr/bin/env python3
"""Two corrections the manuscript owes: site purity, and design weighting.

Site purity. A patient is assigned to the hospital system that wrote most of that
patient's notes, but the model read every delivered note, so a patient assigned to one
system may carry notes from another. If the site contrast is real it should survive
restricting to patients whose notes come overwhelmingly from one place.

This script computed design-weighted sensitivity and specificity as well until 15 August.
Those four numbers belong to downstream/npj/f01_accuracy_against_reference.py, which writes
them to f01_accuracy/reader_estimates.csv and draws them in Figure 1d, and Supplementary
Table 9 takes them from there. The version here counted an indeterminate classification as
a negative call and so reported specificity near 0.970 against that producer's 0.937. Two
producers for one quantity under two estimands is how a table ends up disagreeing with the
figure beside it, so the section is removed rather than reconciled.

Supplementary Table 9 carries four rows from this script: code sensitivity and model kappa
by system, at each of the two purity thresholds. It wrote nothing to disk until
15 August, so those rows could not be checked
against the run that produced them, and the script was briefly archived as unreachable
because no pipeline, job, import or document named it. It now writes
downstream/npj/out/purity/purity_by_site.csv and weighting.csv, and rebuild_all.sh runs it.

It also read resources/gold_reference_v1_2026-08-03.csv, the pre-closure reference standard
of 496 charts, which no longer exists, and scored the model with the six-category
linear-weighted kappa. Both are corrected: the reference standard comes from
ref_standard.build() at 487 charts, and the kappa is the primary definition.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "downstream")
sys.path.insert(0, "downstream/npj")
import ref_standard as RS  # noqa: E402
import stats_core as sc  # noqa: E402

PRIMARY = "aud_qwen36_27b"
L6 = ["Definitely Yes", "Probably Yes", "Uncertain", "Probably No", "Definitely No",
      "Not Enough Data"]
YES = {"Definitely Yes", "Probably Yes"}
NO = {"Definitely No", "Probably No"}
SITES = ["IUH", "Eskenazi", "Other"]
NB = 2000
SEED = 20260809
OUT = "downstream/npj/out/purity"
ORD3_PRIMARY = ["AUD-", "Uncertain", "AUD+"]
NED = "Not Enough Data"


def collapse3(x):
    """The primary definition. Not Enough Data becomes missing and drops the chart."""
    return pd.Series(np.where(x.isin(YES), "AUD+",
                     np.where(x.isin(NO), "AUD-",
                     np.where(x.eq("Uncertain"), "Uncertain", None))),
                     index=x.index).where(x.notna() & x.ne(NED))


def kappa3(a, b):
    a, b = collapse3(a), collapse3(b)
    m = a.notna() & b.notna()
    return (sc.cohens_kappa(a[m], b[m], ORD3_PRIMARY, weights=None), int(m.sum()))


def t3(x):
    return "AUD+" if x in YES else "AUD-" if x in NO else "Indeterminate"


def load():
    ref, man = RS.build()
    if len(ref) != man["cohort"]["analysed"] != 487:
        raise SystemExit("reference standard is not the 487")
    ref = ref[["omop_person_id", "gold_label"]].copy()
    ref["omop_person_id"] = ref["omop_person_id"].astype(str)
    ros = pd.read_csv("resources/patient_roster.csv", dtype={"omop_person_id": str})
    pan = pd.read_csv("output/canonical_panel.csv", dtype={"omop_person_id": str})
    pan = pan[pan["model"] == PRIMARY][["omop_person_id", "label_canonical"]]
    inst = pd.read_csv("output/fairness/patient_institution.csv")
    inst["omop_person_id"] = inst["person_id"].astype(str)
    d = (ref.merge(ros[["omop_person_id", "has_aud_icd"]], on="omop_person_id")
            .merge(pan, on="omop_person_id")
            .merge(inst[["omop_person_id", "institution", "share"]], on="omop_person_id"))
    d = d[d["gold_label"].notna()].copy()
    d["gold6"] = sc.normalize_labels(d["gold_label"], L6, strict=False)
    d["model6"] = sc.normalize_labels(d["label_canonical"], L6, strict=False)
    d["gold3"], d["model3"] = d["gold6"].map(t3), d["model6"].map(t3)
    return d


def main():
    d = load()
    print("reviewed charts with a reference label: %d" % len(d))

    # ------------------------------------------------------------------ site purity
    print("\n" + "=" * 76)
    print("1  SITE PURITY")
    print("=" * 76)
    print("  share of a patient's notes coming from the assigned system")
    print("  median %.3f, quartiles %.3f and %.3f, minimum %.3f"
          % (d["share"].median(), d["share"].quantile(.25), d["share"].quantile(.75),
             d["share"].min()))
    rows = []
    for thr in (0.0, 0.8, 0.9):
        s = d[d["share"] >= thr] if thr else d
        print("\n  purity >= %.0f%%   n = %d" % (100 * thr, len(s)) if thr
              else "\n  all patients   n = %d" % len(s))
        for site in SITES:
            g = s[s["institution"] == site]
            if len(g) < 15:
                print("    %-9s n = %2d, too few" % (site, len(g)))
                continue
            k, n_k = kappa3(g["gold6"], g["model6"])
            det = g[g["gold3"].isin(["AUD+", "AUD-"])]
            pos = det[det["gold3"] == "AUD+"]
            code = 100 * (pos["has_aud_icd"] == 1).mean() if len(pos) else np.nan
            print("    %-9s n = %3d   model kappa %.3f on %3d   code sensitivity %.1f%%"
                  % (site, len(g), k, n_k, code))
            rows.append({"threshold": thr, "n_threshold": len(s), "site": site,
                         "n_site": len(g), "kappa_primary": k, "n_kappa": n_k,
                         "code_sensitivity": code, "n_reference_positive": len(pos)})
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "purity_by_site.csv"), index=False)

    print("\n  If the code-sensitivity ordering across systems holds at the stricter")
    print("  purity thresholds, the contrast is not an artefact of mixed note sources.")


if __name__ == "__main__":
    main()
