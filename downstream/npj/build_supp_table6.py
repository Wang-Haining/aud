#!/usr/bin/env python3
"""Build the comparator table, and the frame the comparator panel of Figure 1 draws from.

The table was written by hand on the previous cohort, so every figure in it was stale, and its
LLM row carried the specificity a since-corrected scoring convention had produced. Everything
is now derived from the csvs the baselines emit, so it cannot drift again.

One frame serves both outputs. Supplementary Table 6 is a rendering of it and panel g of
Figure 1 is a plot of it, which is the only way to guarantee that the picture and the table
carry the same numbers. Rows are ordered by Youden's J so the ladder the text describes is the
order the reader meets, and each row records whether the comparator reads notes or only the
structured record, since that distinction is the finding.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "downstream")
sys.path.insert(0, "downstream/npj")

OUT_TEX = Path("downstream/npj/out/supp_table6.tex")
OUT_CSV = Path("downstream/npj/out/supp_table6.csv")

# csv label -> table label, short label for the figure, what the comparator reads.
# The rule-based term match was scored here until 24 August and is no longer a
# reported comparator; its scorer is in _archive/rule_baseline_2026-08-24/.
NOTES, STRUCT = "notes", "structured record"
SPEC = [
    ("the instrument", "LLM (Qwen3.6-27B)", "LLM (Qwen3.6-27B)", NOTES),
    ("AUD diagnosis code alone", "ICD F10.x code alone", "ICD F10.x", STRUCT),
    ("any alcohol-attributed code", "Any alcohol-attributed code",
     "Any alcohol code", STRUCT),
    ("AUD code or alcoholic liver disease", "AUD code or alcoholic liver disease",
     "AUD or liver-disease code", STRUCT),
    ("structured model, with the AUD code among the features",
     "Gradient boosting over structured data, with the AUD code",
     "Gradient boosting, with code", STRUCT),
    ("structured model, without the AUD code",
     "Gradient boosting over structured data, without the AUD code",
     "Gradient boosting, no code", STRUCT),
]



def rows():
    """Every comparator on one frame, ordered by Youden's J."""
    comp = pd.read_csv("downstream/npj_baselines/comparators.csv").set_index("comparator")
    ml = pd.read_csv("output/npj_baselines/note_ml_row.csv").iloc[0]

    out = []
    for key, label, short, reads in SPEC:
        r = comp.loc[key]
        out.append(dict(r, comparator=label, short=short, reads=reads))
    out.append(dict(ml, comparator="TF-IDF logistic regression, out of fold",
                    short="TF-IDF logistic regression", reads=NOTES, n=425))

    # The fine-tuned clinical encoder writes one row per window aggregation. The
    # mean is the primary one, for the reason its script gives: a maximum over a
    # window count that runs from 1 to 1,619 is not comparable between patients.
    bert = pd.read_csv("output/npj_baselines/note_bert_row.csv")
    br = bert[bert["aggregation"] == "mean"].iloc[0]
    out.append(dict(br[["sens", "sens_lo", "sens_hi", "spec", "spec_lo",
                        "spec_hi", "ppv", "npv", "kappa"]],
                    comparator="Fine-tuned Bio\\_ClinicalBERT, out of fold",
                    short="Fine-tuned Bio_ClinicalBERT", reads=NOTES, n=425))

    df = pd.DataFrame(out)
    df["youden"] = (df["sens"] + df["spec"]) / 100 - 1
    return df.sort_values("youden", ascending=False).reset_index(drop=True)


def main():
    df = rows()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # The table keeps the order the paper introduces the comparators in, not the ranking, so
    # that a reader following the text finds each one where the text put it.
    order = [s[1] for s in SPEC] + ["TF-IDF logistic regression, out of fold",
                                    "Fine-tuned Bio\\_ClinicalBERT, out of fold"]
    t = df.set_index("comparator")
    lines = []
    for k, name in enumerate(order):
        r = t.loc[name]
        lines.append("%s & %.2f (%.2f--%.2f) & %.2f (%.2f--%.2f) & %.2f & %.2f & %.3f \\\\"
                     % (name, r["sens"] / 100, r["sens_lo"] / 100, r["sens_hi"] / 100,
                        r["spec"] / 100, r["spec_lo"] / 100, r["spec_hi"] / 100,
                        r["ppv"] / 100, r["npv"] / 100, r["kappa"]))
        if k == 0:
            lines.append("\\midrule")
    OUT_TEX.write_text("\n".join(lines) + "\n")

    print("wrote %s and %s" % (OUT_TEX, OUT_CSV))
    print()
    print("\n".join(lines))
    print()
    print("ranked by Youden's J, which is the order panel g draws:")
    for _, r in df.iterrows():
        print("  %-28s J %.3f   sens %.3f  spec %.3f   reads %s"
              % (r["short"], r["youden"], r["sens"] / 100, r["spec"] / 100, r["reads"]))


if __name__ == "__main__":
    main()
