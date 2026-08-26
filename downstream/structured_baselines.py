#!/usr/bin/env python3
"""Could structured data alone have found these patients?

The manuscript compares the instrument with the AUD diagnosis code. A reviewer will ask
whether that is too easy a comparator, and whether something better built from
structured data would have done as well without reading a single note. Two comparators
answer that.

Rule baseline. Any alcohol-attributed diagnosis code, not only F10: alcohol-associated
liver disease, alcoholic pancreatitis, alcoholic gastritis, alcoholic neuropathy and
the rest of the set the registry analysis already uses. This is what an experienced
analyst would actually write, and it is a fair comparator rather than a straw man.

Learned baseline. Gradient boosting over everything structured that exists for these
patients: age, sex, race, ethnicity, hospital system, nineteen Charlson conditions, the
comorbidity index, encounter volume, and the alcohol-code flags. Fitted with nested
cross-validation so the reported performance is out-of-fold and carries no tuning
optimism.

The learned baseline is run twice. With the AUD code among the features it shows the
ceiling for structured data when the code is available, which is close to circular
since the code is most of the signal. Without it, the model has to find AUD from
everything else, which is the question that matters: the patients the instrument
recovers are precisely those with no AUD code.

Everything is scored against the same reference standard, on the same determinate
charts, with the same metrics as the instrument, so the numbers drop straight into the
performance forest.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

# XGBoost's hist method reduces gradient histograms in thread order, so a run
# with a different thread count can choose different split points. The thread
# count has to be fixed before numpy or xgboost is imported, not after.
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import os  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict  # noqa: E402

sys.path.insert(0, "downstream")
import fig6_bayes_strata as B  # noqa: E402
import stats_core as sc  # noqa: E402

warnings.filterwarnings("ignore")
PRIMARY = "aud_qwen36_27b"
L6 = ["Definitely Yes", "Probably Yes", "Uncertain", "Probably No", "Definitely No",
      "Not Enough Data"]
YES = {"Definitely Yes", "Probably Yes"}
NO = {"Definitely No", "Probably No"}
C3 = ["AUD+", "AUD-", "Indeterminate"]
OUT = "downstream/npj_baselines"
SEED = 20260809


def t3(x):
    return "AUD+" if x in YES else "AUD-" if x in NO else "Indeterminate"


def wil(k, n):
    if n == 0:
        return np.nan, np.nan, np.nan
    lo, hi = sc.wilson_ci(int(k), int(n))
    return 100 * k / n, 100 * lo, 100 * hi


def report(name, truth, pred, extra="", indeterminate=None):
    """truth is boolean over determinate charts; pred is boolean; `indeterminate` marks
    predictions that stated no direction.

    A comparator that always states a direction passes `indeterminate=None`. For a reader
    that may decline, those patients are counted as neither a correct positive nor a correct
    negative: they leave the numerator of both sensitivity and specificity while staying in
    the denominators. Folding them into the negative column instead would let a reader raise
    its specificity by declining, which is the failure mode this paper measures in the
    diagnosis code.
    """
    truth, pred = np.asarray(truth, bool), np.asarray(pred, bool)
    und = np.zeros(len(truth), bool) if indeterminate is None else np.asarray(
        indeterminate, bool)
    # Both denominators are the full reference group. tp and tn count only the answers
    # that were both directional and correct; fn and fp collect everything else on their
    # side of the table, so an indeterminate answer counts against the reader exactly once
    # and cannot shrink the denominator it sits in.
    tp = int((truth & pred & ~und).sum())
    fn = int(truth.sum()) - tp
    tn = int((~truth & ~pred & ~und).sum())
    fp = int((~truth).sum()) - tn
    se = wil(tp, tp + fn)
    sp = wil(tn, tn + fp)
    pv = wil(tp, tp + fp)
    nv = wil(tn, tn + fn)
    # Kappa is computed on the charts where the reader stated a direction, since a
    # two-category kappa has nowhere to put a third answer; the count that entered it is
    # reported alongside.
    kk = ~und
    k = sc.cohens_kappa(np.where(truth[kk], "AUD+", "AUD-"),
                        np.where(pred[kk], "AUD+", "AUD-"), ["AUD+", "AUD-"], weights=None)
    print("  %-46s n=%3d  sens %5.1f (%.1f-%.1f)  spec %5.1f (%.1f-%.1f)  "
          "PPV %5.1f  NPV %5.1f  kappa %.3f%s"
          % (name, len(truth), se[0], se[1], se[2], sp[0], sp[1], sp[2],
             pv[0], nv[0], k, extra))
    return {"comparator": name, "n": len(truth), "sens": se[0], "sens_lo": se[1],
            "sens_hi": se[2], "spec": sp[0], "spec_lo": sp[1], "spec_hi": sp[2],
            "ppv": pv[0], "npv": nv[0], "kappa": k, "tp": tp, "fp": fp, "tn": tn,
            "fn": fn}


def main():
    os.makedirs(OUT, exist_ok=True)
    # Derived rather than read, for the same reason as note_baseline.py.
    import sys as _sys
    _sys.path.insert(0, "downstream/npj")
    import ref_standard as _R
    ref, _ = _R.build()
    ref["omop_person_id"] = ref["omop_person_id"].astype(str)
    ros = pd.read_csv("resources/patient_roster.csv", dtype={"omop_person_id": str})
    pan = pd.read_csv("output/canonical_panel.csv", dtype={"omop_person_id": str})
    pan = pan[pan["model"] == PRIMARY][["omop_person_id", "label_canonical"]]
    feat = pd.read_parquet("output/features_structured.parquet")
    feat["omop_person_id"] = feat["person_id"].astype("int64").astype(str)

    # alcohol-attributed codes other than the AUD code itself, from the registry frame
    m = B.build(0)
    co = pd.read_parquet(B.D + "/condition_occurrence.parquet",
                         columns=["person_id", "condition_source_value"])
    co["icd"] = co["condition_source_value"].astype(str).str.extract(r"\^\^([^\^]+)\^")[0]
    co.loc[co["icd"].isna(), "icd"] = co.loc[co["icd"].isna(),
                                             "condition_source_value"].astype(str)
    ic = co["icd"].astype(str)
    any_alc = set(co.loc[ic.str.match(B.ALC_SIGNAL, na=False), "person_id"].astype("int64"))
    ald = set(co.loc[ic.str.match(B.NARROW_SIGNAL, na=False), "person_id"].astype("int64"))
    del co, ic
    nvis = m.set_index("person_id")["nvis"] if "nvis" in m.columns else None

    d = (ref.merge(ros[["omop_person_id", "has_aud_icd", "n_notes"]], on="omop_person_id")
            .merge(pan, on="omop_person_id").merge(feat, on="omop_person_id"))
    d = d[d["gold_label"].notna()].copy()
    pid = d["omop_person_id"].astype("int64")
    d["any_alcohol_code"] = pid.isin(any_alc).astype(int)
    d["alcoholic_liver_disease"] = pid.isin(ald).astype(int)
    d["n_visits"] = pid.map(nvis).fillna(0) if nvis is not None else 0
    d["gold3"] = sc.normalize_labels(d["gold_label"], L6, strict=False).map(t3)
    d["model3"] = sc.normalize_labels(d["label_canonical"], L6, strict=False).map(t3)

    det = d[d["gold3"].isin(["AUD+", "AUD-"])].reset_index(drop=True)
    y = (det["gold3"] == "AUD+").to_numpy()
    print("determinate charts: %d, of which reference positive %d" % (len(det), int(y.sum())))

    rows = []
    print("\n" + "=" * 100)
    print("EVERY COMPARATOR ON THE SAME CHARTS")
    print("=" * 100)
    rows.append(report("the instrument", y, (det["model3"] == "AUD+").to_numpy(),
                       indeterminate=(det["model3"] == "Indeterminate").to_numpy()))
    rows.append(report("AUD diagnosis code alone", y, det["has_aud_icd"] == 1))
    rows.append(report("any alcohol-attributed code", y, det["any_alcohol_code"] == 1))
    rows.append(report("AUD code or alcoholic liver disease", y,
                       (det["has_aud_icd"] == 1) | (det["alcoholic_liver_disease"] == 1)))

    # ------------------------------------------------------------------ learned model
    base = ["age", "CCI", "n_visits", "n_notes", "any_alcohol_code",
            "alcoholic_liver_disease"]
    charl = [c for c in feat.columns if c not in
             ("person_id", "omop_person_id", "age", "gender", "race", "ethnicity",
              "care_site_id", "CCI")]
    X = det[base + charl].copy()
    for c in ("gender", "race", "ethnicity"):
        X = pd.concat([X, pd.get_dummies(det[c].astype(str), prefix=c)], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    from xgboost import XGBClassifier
    grid = {"max_depth": [2, 3], "n_estimators": [150, 300],
            "learning_rate": [0.05, 0.1]}
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED)

    print("\n" + "=" * 100)
    print("GRADIENT BOOSTING OVER STRUCTURED DATA, NESTED CROSS-VALIDATION")
    print("=" * 100)
    for label, cols in (("with the AUD code among the features", X.assign(
                             aud_code=(det["has_aud_icd"] == 1).astype(int))),
                        ("without the AUD code", X)):
        gs = GridSearchCV(XGBClassifier(eval_metric="logloss", random_state=SEED,
                                        tree_method="hist", n_jobs=1),
                          grid, cv=inner, scoring="roc_auc", n_jobs=1)
        p = cross_val_predict(gs, cols, y, cv=outer, method="predict_proba",
                              n_jobs=1)[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, p)
        # threshold at the reference prevalence so it is not tuned to the metric
        thr = float(np.quantile(p, 1 - y.mean()))
        rows.append(report("structured model, %s" % label, y, p >= thr,
                           extra="  AUROC %.3f" % auc))

    # -------------------------------------------------- where it actually matters
    print("\n" + "=" * 100)
    print("RESTRICTED TO PATIENTS WITH NO AUD DIAGNOSIS CODE")
    print("=" * 100)
    print("  These are the patients the instrument exists to recover. A structured")
    print("  comparator that cannot find them has not answered the question.")
    unc = (det["has_aud_icd"] == 0).to_numpy()
    print("  n = %d, of which reference positive %d" % (int(unc.sum()), int(y[unc].sum())))
    report("the instrument", y[unc], (det["model3"] == "AUD+").to_numpy()[unc],
           indeterminate=(det["model3"] == "Indeterminate").to_numpy()[unc])
    report("any alcohol-attributed code", y[unc], (det["any_alcohol_code"] == 1).to_numpy()[unc])
    gs = GridSearchCV(XGBClassifier(eval_metric="logloss", random_state=SEED,
                                    tree_method="hist", n_jobs=1), grid, cv=inner,
                      scoring="roc_auc", n_jobs=1)
    p = cross_val_predict(gs, X, y, cv=outer, method="predict_proba", n_jobs=1)[:, 1]
    thr = float(np.quantile(p, 1 - y.mean()))
    report("structured model, without the AUD code", y[unc], (p >= thr)[unc])

    pd.DataFrame(rows).to_csv(f"{OUT}/comparators.csv", index=False)
    print("\n  wrote %s/comparators.csv" % OUT)


if __name__ == "__main__":
    main()
