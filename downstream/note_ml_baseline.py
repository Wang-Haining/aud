#!/usr/bin/env python3
"""A supervised text classifier on the same notes, to complete the comparator set.

Four comparators now cover two questions crossed two ways. On structured data, a code rule
and a gradient-boosted model; on the notes, a keyword rule and, here, a supervised text
model. The instrument's advantage over the keyword rule could be attributed to a hand
rule being crude rather than to the reasoning, so a model that learns its own features
from the same text is the comparator that answers that.

Design. Bag of word and bigram counts, sublinear term frequency, inverse document
frequency, on the concatenated notes of each patient. L2 regularised logistic regression.
Five-fold outer cross-validation for the estimate, three-fold inner for the regularisation
strength, so no fold sees its own tuning. The operating threshold is fixed at the
reference prevalence rather than tuned, which is how the structured comparators were run.

This model has 433 labelled patients and tens of thousands of features, so it is expected
to be regularised heavily. That is the point of including it: it shows what supervision on
this much text and this many labels can reach.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud, under Slurm.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "downstream")
import stats_core as sc  # noqa: E402

L3 = ["AUD+", "Indeterminate", "AUD-"]
CACHE = "output/npj_baselines/notes_full.pkl"
SEED = 20260810
CS = [0.01, 0.1, 1.0, 10.0]


def wil(k, n):
    if not n:
        return float("nan"), float("nan")
    z, p = 1.959964, k / n
    dd = 1 + z * z / n
    c = (p + z * z / (2 * n)) / dd
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out", default="output/npj_baselines")
    a = ap.parse_args()

    notes = pickle.loads(Path(a.cache).read_bytes())
    r = pd.read_csv("output/raters/rater_long.csv", dtype=str)
    w3 = r.pivot_table(index="omop_person_id", columns="rater", values="label_3",
                       aggfunc="first")
    # rater_long carries every rating that was made, over all 499 sampled patients,
    # and marks which of them are in the analysis cohort. Without that filter this
    # baseline would be fitted and scored on twelve patients the paper excludes.
    inc = r[r["in_analysis_cohort"].astype(str).isin(["True", "true", "1"])]
    keep = set(inc["omop_person_id"].astype(str))
    g = w3["gold"].dropna()
    g = g[g.index.astype(str).isin(keep)]
    g = g[g != "Indeterminate"]
    ids = [i for i in g.index if i in notes]
    y = (g.loc[ids] == "AUD+").to_numpy().astype(int)
    X = [notes[i] for i in ids]
    print("labelled patients with notes: %d, of which %d positive" % (len(ids), y.sum()))

    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pred = np.zeros(len(y), int)
    chosen = []
    for tr, te in outer.split(X, y):
        Xtr = [X[i] for i in tr]
        prev = y[tr].mean()
        best, best_s = None, -1
        inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
        for C in CS:
            s = []
            for itr, ite in inner.split(Xtr, y[tr]):
                v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                    min_df=3, max_features=200000, lowercase=True)
                A = v.fit_transform([Xtr[i] for i in itr])
                B = v.transform([Xtr[i] for i in ite])
                m = LogisticRegression(C=C, max_iter=2000, class_weight=None)
                m.fit(A, y[tr][itr])
                p = m.predict_proba(B)[:, 1]
                s.append(((p >= prev).astype(int) == y[tr][ite]).mean())
            if np.mean(s) > best_s:
                best_s, best = np.mean(s), C
        chosen.append(best)
        v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=3,
                            max_features=200000, lowercase=True)
        A = v.fit_transform(Xtr)
        B = v.transform([X[i] for i in te])
        m = LogisticRegression(C=best, max_iter=2000)
        m.fit(A, y[tr])
        pred[te] = (m.predict_proba(B)[:, 1] >= prev).astype(int)
    print("regularisation chosen per outer fold: %s" % chosen)

    prev = y.mean()  # reporting only, below this point
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    kap = sc.cohens_kappa(pd.Series(np.where(y == 1, "AUD+", "AUD-")),
                          pd.Series(np.where(pred == 1, "AUD+", "AUD-")),
                          ["AUD+", "AUD-"], weights=None)
    sl, sh = wil(tp, tp + fn)
    pl, ph = wil(tn, tn + fp)
    print()
    print("  out of fold, threshold fixed at the reference prevalence %.3f" % prev)
    print("    sensitivity %.2f (%.2f-%.2f)" % (sens, sl, sh))
    print("    specificity %.2f (%.2f-%.2f)" % (spec, pl, ph))
    print("    PPV %.2f   NPV %.2f   kappa %.3f" % (ppv, npv, kap))

    ros = pd.read_csv("resources/patient_roster.csv", dtype={"omop_person_id": str})
    code = ros.set_index("omop_person_id")["has_aud_icd"].reindex(ids).to_numpy()
    unc = (code == 0)
    pos = unc & (y == 1)
    print()
    print("  restricted to the %d patients with no AUD diagnosis code, of whom %d carry a "
          "reference diagnosis" % (int(unc.sum()), int(pos.sum())))
    k = int((pred[pos] == 1).sum())
    lo, hi = wil(k, int(pos.sum()))
    print("    identified %d of %d  %.3f (%.3f-%.3f)"
          % (k, int(pos.sum()), k / pos.sum(), lo, hi))

    row = {"comparator": "supervised text model, out of fold",
           "sens": 100 * sens, "sens_lo": 100 * sl, "sens_hi": 100 * sh,
           "spec": 100 * spec, "spec_lo": 100 * pl, "spec_hi": 100 * ph,
           "ppv": 100 * ppv, "npv": 100 * npv, "kappa": kap}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(Path(a.out) / "note_ml_row.csv", index=False)
    print()
    print("  wrote %s/note_ml_row.csv" % a.out)


if __name__ == "__main__":
    main()
