#!/usr/bin/env python3
"""Build one long table holding every rating by every rater, human and model.

Ratings currently live in four places with three different patient-identifier spaces,
which is why the second fellow's data went unused for months. This consolidates them
into a single long table, one row per patient per rater, and writes the recovered
identifier crosswalk alongside it so the recovery never has to be repeated.

  omop_person_id  the analysis identifier throughout
  rater           maaz, sai, desai, gawrieh, the eleven model keys, or gold
  rater_type      fellow, attending, model, composite
  label_6         the six-category rating as recorded
  label_3         the prespecified collapse
  source          the file the rating came from
  stage           when in the reference-standard process the rating was made

MISALIGNMENT GATE. The dual-review workbook contains the first fellow's own ratings
as well as the second fellow's. Those ratings also exist, independently, in the
patient roster. If the recovered crosswalk is correct the two must agree closely on
the linked patients; if rows were shifted they would not. The script refuses to write
anything unless that agreement clears a threshold, and prints the disagreements so
they can be inspected rather than assumed away.

Output is patient-level and stays out of any repository.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, "downstream")
import stats_core as sc  # noqa: E402

OUT = "output/raters"
WB = "resources/3-3 Maaz and Sai inter-rater reliability.xlsx"
L6 = ["Definitely Yes", "Probably Yes", "Uncertain", "Probably No", "Definitely No",
      "Not Enough Data"]
YES = {"Definitely Yes", "Probably Yes"}
NO = {"Definitely No", "Probably No"}
# the linked patients on which the workbook's own first-fellow column must reproduce
# the roster's first-fellow column
GATE = 0.95


def three(x):
    if pd.isna(x):
        return np.nan
    return "AUD+" if x in YES else "AUD-" if x in NO else "Indeterminate"


def norm(s):
    return sc.normalize_labels(s, L6, strict=False)


def recover_crosswalk():
    """Map the old nDepth patient identifiers through the notes shown to reviewers."""
    ds = pd.ExcelFile(WB).parse("Document Sets")[
        ["GLOBAL_PATIENT_ID", "MULTIMEDIA_CONTENT_ID"]].dropna()
    ds["gpid"] = pd.to_numeric(ds["GLOBAL_PATIENT_ID"], errors="coerce").astype("Int64")
    ds["rid"] = pd.to_numeric(ds["MULTIMEDIA_CONTENT_ID"], errors="coerce").astype("Int64")
    mr = pd.read_csv("resources/mapped_report_ids.csv")
    r2p = dict(zip(pd.to_numeric(mr["REPORT_ID"], errors="coerce").astype("Int64"),
                   pd.to_numeric(mr["OMOP_PERSON_ID"], errors="coerce").astype("Int64")))
    votes, nnotes = defaultdict(set), defaultdict(int)
    for g, r in zip(ds["gpid"], ds["rid"]):
        p = r2p.get(r)
        if pd.notna(g) and p is not None and pd.notna(p):
            votes[int(g)].add(int(p))
            nnotes[int(g)] += 1
    rows = [{"nDepth_global_patient_id": g, "omop_person_id": next(iter(s)),
             "supporting_notes": nnotes[g]}
            for g, s in votes.items() if len(s) == 1]
    bad = {g: s for g, s in votes.items() if len(s) > 1}
    cw = pd.DataFrame(rows)
    # The old identifier space is document-set based, so one patient can appear under
    # several identifiers carrying different note bundles, and a reviewer can then have
    # given different answers to what is the same person. Those are marked and excluded
    # rather than collapsed by majority, because the ratings answer different evidence.
    n_per_person = cw.groupby("omop_person_id")["nDepth_global_patient_id"].transform(
        "size")
    cw["one_to_one"] = n_per_person == 1
    return cw, bad


def main():
    os.makedirs(OUT, exist_ok=True)
    xw, conflicts = recover_crosswalk()
    print("crosswalk recovered: %d identifiers, %d ambiguous" % (len(xw), len(conflicts)))
    if conflicts:
        raise SystemExit("ambiguous identifiers present, resolve before building")

    ros = pd.read_csv("resources/patient_roster.csv", dtype={"omop_person_id": str})
    # The reference standard is derived rather than read from a file. Reading a cached
    # gold CSV made this table depend on a build of the reference standard that predated
    # the ratings it is supposed to summarise, so a new attending rating never reached
    # the analyses that consume rater_long. The whole roster is requested here; the
    # analysis cohort is applied by each findings script through the same module.
    import sys as _sys
    _sys.path.insert(0, "downstream/npj")
    import ref_standard as _R
    ref, _ = _R.build(cohort="all")
    ref["omop_person_id"] = ref["omop_person_id"].astype(str)
    pan = pd.read_csv("output/canonical_panel.csv", dtype={"omop_person_id": str})
    ia = pd.ExcelFile(WB).parse("Interagreement ")
    ia = ia.rename(columns={c: str(c).strip() for c in ia.columns})
    ia = ia[["GLOBAL_PATIENT_ID", "MAAZ", "SAI"]].dropna()
    ia["nDepth_global_patient_id"] = pd.to_numeric(ia["GLOBAL_PATIENT_ID"],
                                                   errors="coerce").astype("Int64")
    ia = ia.drop_duplicates("nDepth_global_patient_id")
    ia = ia.merge(xw, on="nDepth_global_patient_id", how="inner", validate="one_to_one")
    n_before = len(ia)
    amb = ia[~ia["one_to_one"]]
    print("  review rows on patients holding several old identifiers: %d, covering %d"
          " patients. Excluded, because those ratings were made on different note"
          " bundles." % (len(amb), amb["omop_person_id"].nunique()))
    ia = ia[ia["one_to_one"]].copy()
    print("  second-fellow rows retained: %d of %d" % (len(ia), n_before))
    ia["omop_person_id"] = ia["omop_person_id"].astype("int64").astype(str)

    # ---------------------------------------------------------------- misalignment gate
    chk = ia.merge(ros[["omop_person_id", "ndepth_maaz_rating"]], on="omop_person_id",
                   how="inner", validate="one_to_one")
    a, b = norm(chk["MAAZ"]), norm(chk["ndepth_maaz_rating"])
    ok = pd.notna(a) & pd.notna(b)
    agree = float((a[ok] == b[ok]).mean())
    print("\nMISALIGNMENT GATE")
    print("  the workbook's own first-fellow rating against the roster's, on %d linked"
          % int(ok.sum()))
    print("  exact six-category agreement: %.4f  (threshold %.2f)" % (agree, GATE))
    if agree < GATE:
        d = chk.loc[ok][a[ok].values != b[ok].values]
        print(d[["omop_person_id", "MAAZ", "ndepth_maaz_rating"]].head(20).to_string())
        raise SystemExit("GATE FAILED: the crosswalk does not reproduce the first "
                         "fellow's own ratings, so rows are misaligned")
    mism = chk.loc[ok][a[ok].values != b[ok].values]
    print("  passed. %d of %d disagree, listed below for inspection:"
          % (len(mism), int(ok.sum())))
    for _, r in mism.iterrows():
        print("    %s  workbook %-16s roster %s"
              % (r["omop_person_id"], r["MAAZ"], r["ndepth_maaz_rating"]))

    # ------------------------------------------------------------------------ assemble
    parts = []

    def add(df, pid, lab, rater, rtype, source, stage):
        t = pd.DataFrame({"omop_person_id": df[pid].astype(str),
                          "label_6": norm(df[lab])})
        t = t[t["label_6"].notna()].copy()
        t["rater"], t["rater_type"] = rater, rtype
        t["source"], t["stage"] = source, stage
        if t["omop_person_id"].duplicated().any():
            raise SystemExit("duplicate patient rows for rater %s" % rater)
        parts.append(t)

    add(ros, "omop_person_id", "ndepth_maaz_rating", "maaz", "fellow",
        "resources/patient_roster.csv", "primary review, all charts")
    add(ia, "omop_person_id", "SAI", "sai", "fellow", WB,
        "independent dual review")
    add(ref[ref["desai_label"].notna()], "omop_person_id", "desai_label", "desai",
        "attending", ref["gold_version"].iloc[0],
        "attending second reading")
    add(ref[ref["gawrieh_label"].notna()], "omop_person_id", "gawrieh_label", "gawrieh",
        "attending", ref["gold_version"].iloc[0],
        "adjudication of charts where the first two readings differed")
    add(ref[ref["gold_label"].notna()], "omop_person_id", "gold_label", "gold",
        "composite", ref["gold_version"].iloc[0],
        "composite reference standard")
    for m, g in pan.groupby("model", sort=False):
        add(g, "omop_person_id", "label_canonical", m, "model",
            "output/canonical_panel.csv", "frozen prompt, one pass")

    tab = pd.concat(parts, ignore_index=True)
    tab["label_3"] = tab["label_6"].map(three)
    tab = tab[["omop_person_id", "rater", "rater_type", "label_6", "label_3",
               "source", "stage"]].sort_values(["omop_person_id", "rater"])
    if tab.duplicated(["omop_person_id", "rater"]).any():
        raise SystemExit("duplicate patient-rater pairs")
    if not set(tab["label_6"]).issubset(set(L6)):
        raise SystemExit("unexpected label value")

    print("\nTABLE")
    print("  %d rows, %d patients, %d raters" % (len(tab), tab["omop_person_id"].nunique(),
                                                 tab["rater"].nunique()))
    print(tab.groupby(["rater_type", "rater"]).size().to_string())

    # ------------------------------------------------------------- round-trip the kappas
    print("\nROUND TRIP")
    print("  Each value is computed twice: once from the assembled table, and once")
    print("  from the original files joined independently. Any row shifted during")
    print("  assembly would move the two apart.")
    w = tab.pivot(index="omop_person_id", columns="rater", values="label_6")
    src = {
        "maaz": ros.set_index("omop_person_id")["ndepth_maaz_rating"],
        "sai": ia.set_index("omop_person_id")["SAI"],
        "gold": ref.set_index("omop_person_id")["gold_label"],
        "desai": ref.set_index("omop_person_id")["desai_label"],
        "aud_qwen36_27b": pan[pan["model"] == "aud_qwen36_27b"].set_index(
            "omop_person_id")["label_canonical"],
    }
    bad = []
    for x, y in (("maaz", "sai"), ("aud_qwen36_27b", "sai"),
                 ("aud_qwen36_27b", "maaz"), ("aud_qwen36_27b", "gold"),
                 ("maaz", "desai"), ("aud_qwen36_27b", "desai")):
        s = w[[x, y]].dropna()
        got = sc.cohens_kappa(s[x], s[y], L6, weights="linear")
        d = pd.DataFrame({x: norm(src[x]), y: norm(src[y])}).dropna()
        want = sc.cohens_kappa(d[x], d[y], L6, weights="linear")
        flag = "ok" if abs(got - want) < 1e-9 and len(s) == len(d) else "MISMATCH"
        if flag != "ok":
            bad.append("%s vs %s" % (x, y))
        print("  %-34s table n=%3d k=%.4f   sources n=%3d k=%.4f   %s"
              % ("%s vs %s" % (x, y), len(s), got, len(d), want, flag))
    if bad:
        raise SystemExit("round trip failed for: %s" % ", ".join(bad))
    print("  every pair reproduces exactly, so no rows were shifted.")

    # Both tables cover every sampled patient, including the twelve outside the analysis
    # cohort, because a rating record should hold every rating that was made. Which rows
    # are eligible for analysis is marked rather than enforced here, and the mark comes
    # from the same module that defines the cohort, so a script that forgets to filter
    # produces a visibly wrong denominator instead of a quietly wrong one.
    member = dict(zip(ref["omop_person_id"].astype(str), ref["in_analysis_cohort"]))
    tab["in_analysis_cohort"] = tab["omop_person_id"].astype(str).map(member)
    w["in_analysis_cohort"] = pd.Series(w.index.astype(str).map(member), index=w.index)
    if tab["in_analysis_cohort"].isna().any() or w["in_analysis_cohort"].isna().any():
        raise SystemExit("a rated patient is absent from the roster, so cohort "
                         "membership could not be marked")
    n_pat = tab["omop_person_id"].nunique()
    n_in = tab.loc[tab["in_analysis_cohort"], "omop_person_id"].nunique()
    print()
    print("  analysis cohort marked: %d of %d patients carry ratings and are eligible, "
          "%d are not" % (n_in, n_pat, n_pat - n_in))
    for r_, gg in tab.groupby("rater"):
        if gg["rater_type"].iloc[0] != "model":
            print("    %-10s %4d ratings, %4d inside the analysis cohort"
                  % (r_, len(gg), int(gg["in_analysis_cohort"].sum())))

    tab.to_csv(f"{OUT}/rater_long.csv", index=False)
    xw.to_csv(f"{OUT}/ndepth_v1_crosswalk.csv", index=False)
    w.to_csv(f"{OUT}/rater_wide.csv")
    print("\n  wrote %s/rater_long.csv   %d rows" % (OUT, len(tab)))
    print("  wrote %s/rater_wide.csv   one column per rater" % OUT)
    print("  wrote %s/ndepth_v1_crosswalk.csv   %d recovered identifiers" % (OUT, len(xw)))
    print("  all three are patient level and must stay out of any repository")


if __name__ == "__main__":
    main()
