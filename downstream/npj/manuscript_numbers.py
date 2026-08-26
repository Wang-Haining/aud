#!/usr/bin/env python3
"""Recompute every number the manuscript claims about the cohort and the index test.

The manuscript was written against a reference standard built before the eight charts
from the second fellow's random sample were adjudicated, and against the 499-patient
roster rather than the 487-patient analysis cohort. Every count and every proportion that
depends on either has moved. Editing them by hand invites the failure this file exists to
prevent, so each quantity is recomputed here from the current reference standard and
printed beside the value the manuscript currently states.

Only quantities owned by f00 and f01 appear. Grounding rates, the pairwise model matrix
and the prompt-robustness results are produced elsewhere and are listed at the end as
outstanding, not guessed at.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "downstream")
sys.path.insert(0, str(HERE))
import ref_standard as R  # noqa: E402
import stats_core as sc  # noqa: E402

L6 = ["Definitely Yes", "Probably Yes", "Uncertain", "Probably No", "Definitely No",
      "Not Enough Data"]
# "A definite grade" in the manuscript means the unhedged ends of the scale only. Reading
# it as any directional grade, which would include the two "probably" categories, moves
# both figures into the high eighties and reverses which reader is higher, so the two
# readings are not interchangeable.
DEFINITE = {"Definitely Yes", "Definitely No"}
MODEL = "aud_qwen36_27b"
NB = 2000
SEED = 20260811


def kappa_ci(a, b, labels, weights, rng, nb=NB):
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    k = sc.cohens_kappa(d["a"], d["b"], labels, weights=weights)
    idx = np.arange(len(d))
    reps = []
    for _ in range(nb):
        t = d.iloc[rng.choice(idx, len(idx), replace=True)]
        try:
            reps.append(sc.cohens_kappa(t["a"], t["b"], labels, weights=weights))
        except Exception:
            reps.append(np.nan)
    return k, float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5)), \
        len(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", default=None,
                    help="manuscript.tex, to report which stated values are stale")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    g, man = R.build()
    r = pd.read_csv("output/raters/rater_long.csv", dtype=str)
    w6 = r.pivot_table(index="omop_person_id", columns="rater", values="label_6",
                       aggfunc="first")
    w3 = r.pivot_table(index="omop_person_id", columns="rater", values="label_3",
                       aggfunc="first")
    ros = pd.read_csv("resources/patient_roster.csv", dtype={"omop_person_id": str})

    d = g[["omop_person_id", "gold_label", "gold_three", "maaz_label",
           "desai_label", "gawrieh_label"]].copy()
    d["ref6"] = sc.normalize_labels(d["gold_label"], L6, strict=False)
    d["model6"] = sc.normalize_labels(d["omop_person_id"].map(w6[MODEL]), L6,
                                      strict=False)
    d["maaz6"] = sc.normalize_labels(d["maaz_label"], L6, strict=False)
    d["ref3"] = d["gold_three"]
    d["model3"] = d["omop_person_id"].map(w3[MODEL])
    d = d.merge(ros[["omop_person_id", "has_aud_icd"]], on="omop_person_id")
    d["code3"] = d["has_aud_icd"].map({1: "AUD+", 0: "AUD-"})
    scored = d[d["ref3"].isin(["AUD+", "AUD-"])]

    rows = []

    def rec(key, was, now, note=""):
        rows.append({"quantity": key, "manuscript_says": was, "recomputed": now,
                     "note": note})

    # ---- cohort ---------------------------------------------------------------
    rec("analysis cohort", "496", "%d" % len(d),
        "the 487 carrying an era-A identifier")
    rec("reference AUD present", "256", "%d" % int((d["ref3"] == "AUD+").sum()))
    rec("reference AUD absent", "177", "%d" % int((d["ref3"] == "AUD-").sum()))
    rec("reference indeterminate", "63", "%d" % int((d["ref3"] == "Indeterminate").sum()))
    rec("scored for accuracy", "433", "%d" % len(scored))
    n_unseen = int(d["desai_label"].isna().sum())
    rec("charts no attending saw", "438", "%d" % n_unseen)
    rec("charts an attending read", "58", "%d" % int(d["desai_label"].notna().sum()))
    rec("charts a third reader adjudicated", "34",
        "%d" % int(d["gawrieh_label"].notna().sum()))

    # ---- agreement ------------------------------------------------------------
    k, lo, hi, n = kappa_ci(d["ref6"], d["model6"], L6, "linear", rng)
    rec("model linear-weighted kappa", "0.92 (0.90-0.94)",
        "%.2f (%.2f-%.2f)" % (k, lo, hi), "n = %d" % n)
    kf, lof, hif, nf = kappa_ci(d["ref6"], d["maaz6"], L6, "linear", rng)
    rec("fellow linear-weighted kappa", "0.91 (0.89-0.94)",
        "%.2f (%.2f-%.2f)" % (kf, lof, hif), "n = %d, not independent" % nf)
    # The two kappas are computed on the same patients, so the difference has to be
    # resampled patient-wise with both raters carried together. Bootstrapping them
    # separately would treat the comparison as unpaired and widen the interval.
    pair = d[["ref6", "model6", "maaz6"]].dropna()
    idx, reps = np.arange(len(pair)), []
    for _ in range(NB):
        t = pair.iloc[rng.choice(idx, len(idx), replace=True)]
        try:
            reps.append(sc.cohens_kappa(t["ref6"], t["model6"], L6, weights="linear")
                        - sc.cohens_kappa(t["ref6"], t["maaz6"], L6, weights="linear"))
        except Exception:
            reps.append(np.nan)
    rec("paired difference in kappa", "+0.008 (-0.020 to +0.037)",
        "%+.3f (%+.3f to %+.3f)" % (k - kf, np.nanpercentile(reps, 2.5),
                                    np.nanpercentile(reps, 97.5)),
        "paired bootstrap over %d patients" % len(pair))

    diag = float((d["ref6"] == d["model6"]).sum()) / int(d["ref6"].notna().sum())
    rec("six-category exact agreement", "87.3%", "%.1f%%" % (100 * diag))

    def definite_rate(col):
        s = d[col].dropna()
        return 100 * s.isin(DEFINITE).mean()
    rec("model reached a definite grade", "75.8%", "%.1f%%" % definite_rate("model6"))
    rec("reference reached a definite grade", "69.4%", "%.1f%%" % definite_rate("ref6"))

    kq, _, _, _ = kappa_ci(d["ref6"], d["model6"], L6, "quadratic", rng, nb=1)
    kqf, _, _, _ = kappa_ci(d["ref6"], d["maaz6"], L6, "quadratic", rng, nb=1)
    rec("quadratic-weighted kappa, model", "0.96", "%.2f" % kq)
    rec("quadratic-weighted kappa, fellow", "0.94", "%.2f" % kqf)

    # ---- three-category detail --------------------------------------------------
    d["maaz3"] = d["maaz_label"].map(
        lambda x: "AUD+" if x in ("Definitely Yes", "Probably Yes")
        else "AUD-" if x in ("Definitely No", "Probably No")
        else ("Indeterminate" if pd.notna(x) else np.nan))
    agree3 = float((d["model3"] == d["ref3"]).sum()) / len(d)
    rec("three-category exact agreement", "96.0%", "%.1f%%" % (100 * agree3))
    fn = int(((d["ref3"] == "AUD+") & (d["model3"] == "AUD-")).sum())
    fp = int(((d["ref3"] == "AUD-") & (d["model3"] == "AUD+")).sum())
    ind = int((((d["ref3"] != "Indeterminate") & (d["model3"] == "Indeterminate"))
               | ((d["ref3"] == "Indeterminate") & (d["model3"] != "Indeterminate"))).sum())
    rec("reference-positive called negative", "two", "%d" % fn)
    rec("reference-negative called positive", "five", "%d" % fp)
    rec("charts in an indeterminate cell", "13", "%d" % ind)

    def sscore(col):
        # Rebuilt from d rather than reusing the earlier `scored`, which was taken
        # before the fellow's collapsed column existed.
        s = d[d["ref3"].isin(["AUD+", "AUD-"])]
        pos, neg = s[s["ref3"] == "AUD+"], s[s["ref3"] == "AUD-"]
        se = (pos[col] == "AUD+").mean()
        sp = (neg[col] == "AUD-").mean()
        sp_ = s[s[col] == "AUD+"]
        sn_ = s[s[col] == "AUD-"]
        ppv = (sp_["ref3"] == "AUD+").mean() if len(sp_) else np.nan
        npv = (sn_["ref3"] == "AUD-").mean() if len(sn_) else np.nan
        return se, sp, ppv, npv
    for col, who, was in (("model3", "model", ("0.99", "0.99", "0.98", "0.99")),
                          ("maaz3", "fellow", ("0.96", "0.99", "1.00", "0.95")),
                          ("code3", "code", ("0.84", "-", "-", "0.80"))):
        se, sp, ppv, npv = sscore(col)
        for nm, v, w in zip(("sensitivity", "specificity", "PPV", "NPV"),
                            (se, sp, ppv, npv), was):
            rec("%s %s, as sampled" % (who, nm), w, "%.2f" % v)

    # ---- the code comparison --------------------------------------------------
    miss = d[(d["has_aud_icd"] == 0) & (d["ref3"] == "AUD+")]
    rec("code-negative, reference-positive", "41", "%d" % len(miss))
    rec("of those, the model recovered", "39",
        "%d" % int((miss["model3"] == "AUD+").sum()))
    nocode = d[(d["has_aud_icd"] == 0) & d["ref3"].isin(["AUD+", "AUD-"])]
    rec("patients with no AUD code and a determinate reference", "201",
        "%d" % len(nocode))
    if len(miss):
        rec("instrument identified, among code-negative positives", "95.1%",
            "%.1f%%" % (100 * (miss["model3"] == "AUD+").mean()))

    both = scored[scored["model3"].isin(["AUD+", "AUD-"])]
    disc = both[both["model3"] != both["code3"]]
    mr = int((disc["model3"] == disc["ref3"]).sum())
    cr = int((disc["code3"] == disc["ref3"]).sum())
    rec("discordant model-versus-code classifications", "60", "%d" % len(disc))
    rec("of those, the model was right", "55", "%d" % mr)
    rec("of those, the code was right", "5", "%d" % cr)
    if mr + cr:
        chi = (abs(mr - cr) - 1) ** 2 / (mr + cr)
        rec("McNemar chi-square", "40.0", "%.1f" % chi, "continuity corrected")
    rec("patients in the paired comparison", "426", "%d" % len(both))

    # ---- registry frame -------------------------------------------------------
    try:
        import fig6_bayes_strata as B
        reg = set(B.build()["person_id"].astype(str))
        inframe = d[d["omop_person_id"].isin(reg)]
        rec("reference charts also in the registry frame", "494", "%d" % len(inframe))
        rec("of those, determinate", "431",
            "%d" % int(inframe["ref3"].isin(["AUD+", "AUD-"]).sum()))
    except Exception as e:
        rec("registry frame overlap", "494", "not computed", str(e)[:40])

    out = pd.DataFrame(rows)
    print("=" * 104)
    print("WHAT THE MANUSCRIPT STATES, AND WHAT THE CURRENT REFERENCE STANDARD GIVES")
    print("=" * 104)
    print("  %-52s %-26s %-22s" % ("quantity", "manuscript", "recomputed"))
    for _, x in out.iterrows():
        flag = "" if str(x["manuscript_says"]) == str(x["recomputed"]) else "  <-- change"
        print("  %-52s %-26s %-22s%s"
              % (x["quantity"], x["manuscript_says"], x["recomputed"], flag))
        if x["note"]:
            print("  %-52s %s" % ("", x["note"]))
    out.to_csv("downstream/npj/out/manuscript_numbers.csv", index=False)

    if a.tex and Path(a.tex).exists():
        t = Path(a.tex).read_text()
        print()
        print("=" * 104)
        print("STALE TOKENS STILL PRESENT IN %s" % a.tex)
        print("=" * 104)
        for tok in ("496", "438", "433", "426", "60 discordant"):
            n = len(re.findall(r"(?<![\d.])%s(?![\d.])" % re.escape(tok), t))
            if n:
                print("  %-16s %d occurrence(s)" % (tok, n))

    print()
    print("=" * 104)
    print("OWNED ELSEWHERE, NOT RECOMPUTED HERE")
    print("=" * 104)
    for what, who in [
            ("quotation grounding rate, 444 of 496", "run_npj_provenance.py"),
            ("pairwise model agreement matrix", "f04_figure4.py"),
            ("prompt and temperature robustness", "analyze_goal5_robustness.py"),
            ("learned note baselines", "note_ml_baseline.py, "
                                       "note_bert_baseline.py"),
            ("structured-feature comparators", "structured_baselines.py"),
            ("registry treatment model", "fig6_bayes_strata.py")]:
        print("  %-46s %s" % (what, who))
    print("  Each reads the archived v1 reference standard and will fail until it is")
    print("  migrated to ref_standard.build(), which is the intended behaviour.")


if __name__ == "__main__":
    main()
