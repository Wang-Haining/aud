#!/usr/bin/env python3
"""The verification-independent standard, and what the readers achieve against it.

The full standard refers a chart to a second reader when any of three models differs from
the first hepatologist, and the index test is one of those three. On a chart the escalation
never referred, the index test agreed with the first hepatologist by construction, so its
performance against the full standard is optimistic by an amount the full standard cannot
show. The restricted standard removes that: over the 186 charts the second reader read, it
takes the first hepatologist's label where the two agreed and the adjudicated label where
they differed. No model influenced the selection of any chart in it.

Until 14 August the numbers this produces were in the Results with no script on disk that
computed them. This is that script, and it prints what it reproduces so the two can be
compared.

It also writes the four rates each reader achieves, in the same shape as
predictive_value_inputs.csv, so the predictive-value panel can draw the full and the
restricted standard on one axis. That panel currently draws the full standard alone and
concludes the index test leads at every prevalence; on the restricted rates the diagnosis
code has the higher positive predictive value throughout, and the index test's advantage
is in sensitivity and negative predictive value.

Weights are the registry stratum sizes over the charts contributing to each estimate, so
the restricted standard carries its own weights rather than the full standard's.

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

import ref_standard as RS                                          # noqa: E402

OUT = "downstream/npj/out/restricted"
RATERS = "output/raters/rater_long.csv"
PANEL = "output/canonical_panel.csv"
ROSTER = "resources/patient_roster.csv"
PRIMARY = "aud_qwen36_27b"

# Registry stratum sizes, the denominator every design weight divides into.
REG_CODED, REG_UNCODED = 4210, 10196
POS = {"Definitely Yes", "Probably Yes"}
NEG = {"Definitely No", "Probably No"}


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    z, p = 1.959963985, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def three(s):
    return np.where(s.isin(POS), "AUD+", np.where(s.isin(NEG), "AUD-", "Indeterminate"))


def build(a):
    ref, man = RS.build()
    assert len(ref) == man["cohort"]["analysed"] == 487
    ref["omop_person_id"] = ref["omop_person_id"].astype(str)

    r = pd.read_csv(a.raters, dtype={"omop_person_id": str})
    r = r[r["in_analysis_cohort"].astype(str).str.lower().eq("true")]
    w = r.pivot_table(index="omop_person_id", columns="rater", values="label_3",
                      aggfunc="first").reset_index()

    p = pd.read_csv(a.panel, dtype={"omop_person_id": str}, low_memory=False)
    p = p[p["model"].eq(PRIMARY)][["omop_person_id", "label_canonical"]]

    ros = pd.read_csv(a.roster, dtype={"omop_person_id": str})

    d = (ref[["omop_person_id", "gold_binary", "gold_source"]]
         .merge(w, on="omop_person_id", how="left")
         .merge(p, on="omop_person_id", how="left")
         .merge(ros[["omop_person_id", "has_aud_icd"]], on="omop_person_id", how="left"))

    d["llm3"] = three(d["label_canonical"])
    d["code3"] = np.where(d["has_aud_icd"].fillna(0).astype(int).eq(1), "AUD+", "AUD-")
    d["gold3"] = np.where(d["gold_binary"].eq("AUD+"), "AUD+",
                          np.where(d["gold_binary"].eq("AUD-"), "AUD-", "Indeterminate"))
    d["coded"] = d["has_aud_icd"].fillna(0).astype(int).eq(1)

    # The restricted standard: only the charts the second reader read, taking the first
    # hepatologist's label where the two agreed and the adjudicated label where they did
    # not. The adjudicated label is the gold label, because a disagreement between the two
    # first-pass readers is exactly what the second trigger referred.
    read = d["sai"].notna() & d["maaz"].notna()
    agree = read & d["maaz"].eq(d["sai"])
    d["restricted"] = np.where(agree, d["maaz"],
                               np.where(read, d["gold3"], None))
    return d, man


def rates(d, gold_col, subset, label):
    """The four rates each reader achieves, sampled and weighted, on one standard."""
    g = d[subset].copy()
    pos = g[g[gold_col].eq("AUD+")]
    neg = g[g[gold_col].eq("AUD-")]
    # Weights are the registry stratum size over the charts of that stratum contributing
    # to this estimate, so each standard carries its own.
    det = pd.concat([pos, neg])
    n_c, n_u = int(det["coded"].sum()), int((~det["coded"]).sum())
    wc = REG_CODED / n_c if n_c else np.nan
    wu = REG_UNCODED / n_u if n_u else np.nan
    det_w = np.where(det["coded"], wc, wu)
    det = det.assign(w=det_w)
    posw, negw = det[det[gold_col].eq("AUD+")], det[det[gold_col].eq("AUD-")]

    out = []
    for name, col in (("LLM", "llm3"), ("AUD diagnosis code", "code3")):
        tp = int(pos[col].eq("AUD+").sum())
        fn = int(pos[col].eq("AUD-").sum())
        ip = int(pos[col].eq("Indeterminate").sum())
        tn = int(neg[col].eq("AUD-").sum())
        fp = int(neg[col].eq("AUD+").sum())
        into = int(neg[col].eq("Indeterminate").sum())
        s = wilson(tp, len(pos))
        sp = wilson(tn, len(neg))
        wsens = (posw.loc[posw[col].eq("AUD+"), "w"].sum() / posw["w"].sum()
                 if len(posw) else np.nan)
        wspec = (negw.loc[negw[col].eq("AUD-"), "w"].sum() / negw["w"].sum()
                 if len(negw) else np.nan)
        out.append({"standard": label, "reader": name,
                    "n_pos": len(pos), "n_neg": len(neg),
                    "tp": tp, "fn": fn, "indeterminate_on_pos": ip,
                    "tn": tn, "fp": fp, "indeterminate_on_neg": into,
                    "sens": s[0], "sens_lo": s[1], "sens_hi": s[2],
                    "spec": sp[0], "spec_lo": sp[1], "spec_hi": sp[2],
                    "weighted_sens": wsens, "weighted_spec": wspec,
                    "tpr": tp / len(pos) if len(pos) else np.nan,
                    "fnr": fn / len(pos) if len(pos) else np.nan,
                    "fpr": fp / len(neg) if len(neg) else np.nan,
                    "tnr": tn / len(neg) if len(neg) else np.nan})
    return pd.DataFrame(out)


def curves(t, grid):
    """Positive and negative predictive value across assumed prevalence."""
    rows = []
    for _, r in t.iterrows():
        for pv in grid:
            ppv_d = r["tpr"] * pv + r["fpr"] * (1 - pv)
            npv_d = r["tnr"] * (1 - pv) + r["fnr"] * pv
            rows.append({"standard": r["standard"], "reader": r["reader"],
                         "prevalence": pv,
                         "ppv": r["tpr"] * pv / ppv_d if ppv_d else np.nan,
                         "npv": r["tnr"] * (1 - pv) / npv_d if npv_d else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raters", default=RATERS)
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--roster", default=ROSTER)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d, man = build(a)
    read = d["restricted"].notna()
    print("charts the second reader read            %d" % int(read.sum()))
    print("  restricted standard, determinate       %d"
          % int(d.loc[read, "restricted"].isin(["AUD+", "AUD-"]).sum()))
    print("    AUD present                          %d"
          % int(d.loc[read, "restricted"].eq("AUD+").sum()))
    print("    AUD absent                           %d"
          % int(d.loc[read, "restricted"].eq("AUD-").sum()))
    print("    indeterminate                        %d"
          % int(d.loc[read, "restricted"].eq("Indeterminate").sum()))
    print()

    full = rates(d, "gold3", d["gold3"].isin(["AUD+", "AUD-"]), "full")
    rest = rates(d, "restricted",
                 d["restricted"].isin(["AUD+", "AUD-"]).fillna(False), "restricted")
    t = pd.concat([full, rest], ignore_index=True)
    t.to_csv(os.path.join(a.out, "restricted_rates.csv"), index=False)

    show = t[["standard", "reader", "n_pos", "n_neg", "tp", "fn", "indeterminate_on_pos",
              "tn", "fp", "indeterminate_on_neg", "sens", "spec",
              "weighted_sens", "weighted_spec"]].copy()
    for c in ("sens", "spec", "weighted_sens", "weighted_spec"):
        show[c] = show[c].round(4)
    print(show.to_string(index=False))
    print()

    print("what the Results states, against what this reproduces")
    r = rest.set_index("reader")
    claims = [
        ("LLM identifies of the positive", "94 of 96",
         "%d of %d" % (r.loc["LLM", "tp"], r.loc["LLM", "n_pos"])),
        ("LLM identifies of the negative", "57 of 65",
         "%d of %d" % (r.loc["LLM", "tn"], r.loc["LLM", "n_neg"])),
        ("LLM weighted sensitivity", "0.984",
         "%.3f" % r.loc["LLM", "weighted_sens"]),
        ("LLM weighted specificity", "0.870",
         "%.3f" % r.loc["LLM", "weighted_spec"]),
        ("code weighted sensitivity", "0.673",
         "%.3f" % r.loc["AUD diagnosis code", "weighted_sens"]),
        ("code weighted specificity", "0.972",
         "%.3f" % r.loc["AUD diagnosis code", "weighted_spec"]),
    ]
    for what, said, got in claims:
        mark = "ok" if said == got else "DIFFERS"
        print("  %-32s manuscript %-9s script %-9s %s" % (what, said, got, mark))
    print()

    grid = np.round(np.arange(0.01, 0.505, 0.005), 4)
    cv = curves(t, grid)
    cv.to_csv(os.path.join(a.out, "predictive_value_curves.csv"), index=False)

    print("positive predictive value, at three prevalences")
    print("%-11s %-20s %8s %8s %8s" % ("standard", "reader", "0.10", "0.25", "0.411"))
    for (st, rd), g in cv.groupby(["standard", "reader"]):
        vals = []
        for pv in (0.10, 0.25, 0.41):
            row = g.iloc[(g["prevalence"] - pv).abs().argmin()]
            vals.append(row["ppv"])
        print("%-11s %-20s %8.3f %8.3f %8.3f" % (st, rd, *vals))
    print()
    print("negative predictive value, at the same three")
    for (st, rd), g in cv.groupby(["standard", "reader"]):
        vals = []
        for pv in (0.10, 0.25, 0.41):
            row = g.iloc[(g["prevalence"] - pv).abs().argmin()]
            vals.append(row["npv"])
        print("%-11s %-20s %8.3f %8.3f %8.3f" % (st, rd, *vals))

    print("\nwrote %s/restricted_rates.csv and predictive_value_curves.csv" % a.out)


if __name__ == "__main__":
    main()
