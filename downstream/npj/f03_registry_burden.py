#!/usr/bin/env python3
"""Panel d of Figure 3: how many registry patients carry AUD, by what identifies them.

Panels a to c of this figure do not use the LLM. Chart review supplies P(true AUD |
stratum), the registry supplies the treatment rate, and a Bayesian misclassification model
combines them. Panel d is where the index test enters, and it enters through the sampling
design rather than through new inference.

The 487 reviewed charts are a stratified probability sample of the 14,406-patient registry,
drawn within two strata by whether an AUD diagnosis code appears at any time. Selection
probabilities are therefore known, and the Horvitz-Thompson estimator carries any reader
label from the sample to the registry:

    N_hat = sum over sampled patients of  w_i * 1(reader calls the patient AUD+)
    w_i   = registry size of the sampling stratum / number sampled from it

This is the same weight the weighted column of Fig. 1d uses. No note outside the 487 is
read, and none exists: notes were obtained for the sampled patients only.

CONVENTION. Every arm counts an indeterminate output as not positive, on all 487 charts.
That is the number of patients a reader would actually hand back, and it keeps one
denominator across the arms. Restricting to the 424 charts with a determinate reference
raises both reader arms and is reported as a sensitivity analysis, not as the estimate.

WHAT THE PANEL DOES NOT CLAIM. It carries the LLM measured performance to the registry. It
does not observe the LLM at registry scale.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "downstream")
sys.path.insert(0, str(HERE))

import fig6_bayes_strata as B                                       # noqa: E402
import ref_standard as R                                            # noqa: E402

NB, SEED = 4000, 20260813
OUT = "downstream/npj/out/f03_registry"
MODEL_RATER, FELLOW_RATER = "aud_qwen36_27b", "maaz"


def load():
    g, _ = R.build()
    r = pd.read_csv("output/raters/rater_long.csv", dtype=str)
    w3 = r.pivot_table(index="omop_person_id", columns="rater", values="label_3",
                       aggfunc="first")
    ros = pd.read_csv("resources/patient_roster.csv", dtype={"omop_person_id": str})
    d = g[["omop_person_id", "gold_three"]].merge(
        ros[["omop_person_id", "has_aud_icd"]], on="omop_person_id")
    d["LLM"] = d["omop_person_id"].map(w3[MODEL_RATER])
    d["Hepatologist"] = d["omop_person_id"].map(w3[FELLOW_RATER])
    d["Chart review"] = d["gold_three"]
    if d["LLM"].isna().any():
        raise SystemExit("%d sampled patients have no model result" % int(d["LLM"].isna().sum()))
    return d


def carry(frame, arms, registry_by_stratum, n_registry, col="U"):
    """Horvitz-Thompson count per arm, with a paired stratified bootstrap.

    col names the stratum variable, and it has to be the same variable the registry counts
    were tabulated on. Until 14 August the sample side used the roster's has_aud_icd while
    the registry side used the indicator recomputed from the parquet. Those disagree on 14
    of the 485 sampled patients inside the frame, so the ratio was not a valid design
    weight. The parquet indicator is now used on both sides, and the roster indicator is
    reported beside it.
    """
    n = frame.groupby(col).size()
    w = frame[col].map(
        {u: registry_by_stratum[u] / n.get(u, 1) for u in registry_by_stratum}).to_numpy()
    u = frame[col].to_numpy()
    strata = [np.flatnonzero(u == v) for v in np.unique(u)]
    rng = np.random.default_rng(SEED)
    idx = [np.concatenate([rng.choice(v, len(v), True) for v in strata]) for _ in range(NB)]
    rows, draws = [], {}
    for a in arms:
        pos = (frame[a] == "AUD+").to_numpy()
        draws[a] = np.array([float((w[i] * pos[i]).sum()) for i in idx])
        point = float((w * pos).sum())
        rows.append({"arm": a, "estimator": "Horvitz-Thompson", "n_sampled": len(frame),
                     "count": point, "lo": float(np.percentile(draws[a], 2.5)),
                     "hi": float(np.percentile(draws[a], 97.5)),
                     "fraction": point / n_registry})
    out = pd.DataFrame(rows)
    dd = draws["LLM"] - draws["Chart review"]
    paired = {"difference": float((out.loc[out["arm"] == "LLM", "count"].iloc[0]
                                   - out.loc[out["arm"] == "Chart review", "count"].iloc[0])),
              "lo": float(np.percentile(dd, 2.5)), "hi": float(np.percentile(dd, 97.5))}
    return out, paired


def build():
    d = load()
    reg = B.build()
    n_registry = len(reg)
    by_u = reg["U"].value_counts().to_dict()
    by_s = reg["S"].value_counts().sort_index().to_dict()
    arms = ["Chart review", "LLM", "Hepatologist"]

    # The stratum variable has to match the one the registry counts are tabulated on, so
    # the parquet indicator is attached here. Two sampled patients carry no cirrhosis code
    # and are therefore not in the frame at all; they cannot be weighted to it.
    umap = dict(zip(reg["person_id"].astype("int64"), reg["U"].astype(int)))
    d["U"] = pd.to_numeric(d["omop_person_id"], errors="coerce").map(umap)
    dropped = int(d["U"].isna().sum())
    d = d[d["U"].notna()].copy()
    d["U"] = d["U"].astype(int)
    disc = int((d["U"] != d["has_aud_icd"]).sum())
    print("  %d sampled charts are outside the registry frame and are dropped" % dropped)
    print("  %d of the remaining %d disagree between the roster and the parquet indicator"
          % (disc, len(d)))

    primary, paired = carry(d, arms, by_u, n_registry, col="U")
    primary["basis"] = "all %d sampled charts in the frame" % len(d)
    alt, _ = carry(d, arms, by_u, n_registry, col="has_aud_icd")
    alt["basis"] = "sensitivity, roster has_aud_icd as the stratum"
    det = d[d["Chart review"].isin(["AUD+", "AUD-"])].copy()
    sens, paired_det = carry(det, arms, by_u, n_registry, col="U")
    sens["basis"] = "the %d charts with a determinate reference" % len(det)

    enumerated = pd.DataFrame([
        {"arm": "AUD diagnosis code at $t_0$", "estimator": "enumerated", "n_sampled": np.nan,
         "count": float(by_s[0]), "lo": np.nan, "hi": np.nan,
         "fraction": by_s[0] / n_registry, "basis": "the whole registry"},
        {"arm": "AUD diagnosis code at any time", "estimator": "enumerated",
         "n_sampled": np.nan, "count": float(by_u[1]), "lo": np.nan, "hi": np.nan,
         "fraction": by_u[1] / n_registry, "basis": "the whole registry"},
    ])

    bs = pd.read_csv("output/fig6_registry/bayes_strata3.csv")
    # n_with_aud_total shares the prefix and is the sum of the other three, so summing the
    # prefix would count the registry twice. It is read as the total, with its interval,
    # and the three stratum rows are kept separately.
    nw = bs[bs["parameter"].str.startswith("n_with_aud_")
            & (bs["parameter"] != "n_with_aud_total")]
    tr = bs[bs["parameter"] == "n_with_aud_total"]
    model_total = float(tr["mean"].iloc[0]) if len(tr) else float(nw["mean"].sum())
    model_ci = ((float(tr["ci_low"].iloc[0]), float(tr["ci_high"].iloc[0]))
                if len(tr) else None)

    return dict(primary=primary, sensitivity=sens, alt_stratum=alt, enumerated=enumerated,
                paired=paired, paired_determinate=paired_det,
                model_total=model_total, model_ci=model_ci, n_registry=n_registry,
                model_by_stratum=nw)


def main():
    os.makedirs(OUT, exist_ok=True)
    r = build()
    tab = pd.concat([r["enumerated"], r["primary"]], ignore_index=True)
    tab.to_csv(os.path.join(OUT, "registry_burden.csv"), index=False)
    pd.concat([r["sensitivity"], r["alt_stratum"]], ignore_index=True).to_csv(
        os.path.join(OUT, "registry_burden_determinate.csv"), index=False)
    pd.DataFrame([dict(comparison="LLM minus chart review", basis="all sampled charts",
                       **r["paired"]),
                  dict(comparison="LLM minus chart review", basis="determinate only",
                       **r["paired_determinate"])]
                 ).to_csv(os.path.join(OUT, "registry_burden_paired.csv"), index=False)

    N = r["n_registry"]
    print("=" * 92)
    print("REGISTRY AUD BURDEN, %d PATIENTS, BY WHAT IDENTIFIES IT" % N)
    print("=" * 92)
    for _, x in tab.iterrows():
        ci = ("" if np.isnan(x["lo"]) else "({:,.0f} to {:,.0f})".format(x["lo"], x["hi"]))
        print("  %-32s %8.0f  %6.1f%%  %-24s %s"
              % (x["arm"], x["count"], 100 * x["fraction"], ci, x["basis"]))
    p = r["paired"]
    print("  LLM minus chart review, paired  %+8.0f  (%+.0f to %+.0f)"
          % (p["difference"], p["lo"], p["hi"]))
    print()
    print("  sensitivity, determinate charts only")
    for _, x in r["sensitivity"].iterrows():
        print("    %-30s %8.0f  %6.1f%%  (%.0f to %.0f)"
              % (x["arm"], x["count"], 100 * x["fraction"], x["lo"], x["hi"]))
    print()
    print("  the Bayesian misclassification model behind panels a to c gives "
          "{:,.0f} ({:.1f}%)".format(r["model_total"], 100 * r["model_total"] / N))
    print("  as the sum of its three stratum counts. No joint interval is available: only "
          "summaries were written.")
    print()
    print("wrote %s/registry_burden.csv and two companions" % OUT)


if __name__ == "__main__":
    main()
