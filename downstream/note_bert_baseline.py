#!/usr/bin/env python3
"""A fine-tuned clinical encoder on the same notes, to complete the comparator set.

The comparator set crosses two questions two ways: on the structured record a code
rule and a gradient-boosted model, on the notes a keyword rule and a learned model.
The learned note reader so far is a bag-of-words logistic regression. A reviewer
will ask whether a pretrained clinical encoder closes the gap, so this fits one.

Design. Bio_ClinicalBERT, pretrained on MIMIC-III notes. Each patient's
concatenated notes are split into non-overlapping 512-token windows; every window
inherits the patient's label. Window counts run from 1 to 1,619 with a median of
60, so two things follow. In training, at most TRAIN_CAP windows per patient enter
the loss, taken at even spacing through the note set, so that one long chart does
not dominate. At inference every window is scored and three aggregations are
reported: the mean window probability, the mean of the three largest, and the
maximum. The mean is primary, because a maximum over a number of windows that
varies by three orders of magnitude is not comparable between patients.

The decision rule is a quantile of the training fold's own score distribution, taken
at the prevalence of that same training fold. No test-fold label enters the
threshold. This is the rule Supplementary Table 17 gives for the structured learned
comparator, and it does not require the aggregation to be calibrated.

Five-fold outer cross-validation for the estimate, three-fold inner for the
learning rate, so no fold sees its own tuning. Same folds, seed, cache and label
filter as note_ml_baseline.py.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud, under Slurm, on a GPU.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

sys.path.insert(0, "downstream")
import stats_core as sc  # noqa: E402

CACHE = "output/npj_baselines/notes_full.pkl"
SEED = 20260810
MODEL = "emilyalsentzer/Bio_ClinicalBERT"
LRS = [2e-5, 5e-5]
EPOCHS = 3
BATCH = 16
MAXLEN = 512
TRAIN_CAP = 32
AGGS = ("mean", "top3", "max")


def wil(k, n):
    if not n:
        return float("nan"), float("nan")
    z, p = 1.959964, k / n
    dd = 1 + z * z / n
    c = (p + z * z / (2 * n)) / dd
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return max(0.0, c - h), min(1.0, c + h)


def agg(ps, how):
    ps = np.asarray(ps, dtype=float)
    if how == "mean":
        return float(ps.mean())
    if how == "max":
        return float(ps.max())
    if how == "top3":
        return float(np.sort(ps)[-3:].mean())
    raise ValueError(how)


class Chunks(Dataset):
    def __init__(self, ids, labels):
        self.ids, self.labels = ids, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "label": self.labels[i]}


def collate(batch, pad):
    n = max(len(b["input_ids"]) for b in batch)
    ii = torch.full((len(batch), n), pad, dtype=torch.long)
    am = torch.zeros((len(batch), n), dtype=torch.long)
    for k, b in enumerate(batch):
        L = len(b["input_ids"])
        ii[k, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
        am[k, :L] = 1
    return ii, am, torch.tensor([b["label"] for b in batch], dtype=torch.long)


def windows(tok, text, maxlen):
    body = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
    room = maxlen - 2
    return [[tok.cls_token_id] + body[s:s + room] + [tok.sep_token_id]
            for s in range(0, max(len(body), 1), room)]


def thin(cs, cap):
    """At most cap windows, evenly spaced through the note set."""
    if cap <= 0 or len(cs) <= cap:
        return cs
    idx = np.linspace(0, len(cs) - 1, cap).round().astype(int)
    return [cs[i] for i in sorted(set(idx.tolist()))]


def fit_score(tok, dev, chunks, y, tr, te, lr, tag):
    """Fine-tune on tr, return per-window probabilities for tr and for te."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2).to(dev)
    flat_ids, flat_y = [], []
    for i in tr:
        for c in thin(chunks[i], TRAIN_CAP):
            flat_ids.append(c)
            flat_y.append(int(y[i]))
    dl = DataLoader(Chunks(flat_ids, flat_y), batch_size=BATCH, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    steps = max(1, len(dl) * EPOCHS)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    model.train()
    for ep in range(EPOCHS):
        for ii, am, yy in dl:
            ii, am, yy = ii.to(dev), am.to(dev), yy.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ii, attention_mask=am, labels=yy).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
    print("      %s: %d training windows, %d steps" % (tag, len(flat_ids), steps),
          flush=True)

    model.eval()

    # probs closes over `model`, which is deleted below once the two calls that use it
    # have returned, to free the GPU before the next fold. A static checker cannot see
    # that ordering and reads the closure's `model` as possibly unbound, so the read is
    # marked here rather than the delete being moved, which is what frees the memory.
    def probs(idxs, cap):
        out = []
        with torch.no_grad():
            for i in idxs:
                cs = thin(chunks[i], cap)
                dl2 = DataLoader(Chunks(cs, [0] * len(cs)), batch_size=64,
                                 shuffle=False,
                                 collate_fn=lambda b: collate(b, tok.pad_token_id))
                ps = []
                for ii, am, _ in dl2:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lo = model(input_ids=ii.to(dev),  # noqa: F821
                                   attention_mask=am.to(dev)).logits.float()
                    ps.append(torch.softmax(lo, -1)[:, 1].cpu().numpy())
                out.append(np.concatenate(ps))
        return out

    # training-fold scores use the same thinning as training, so the threshold is
    # taken on the distribution the model was fitted to; test uses every window
    ptr = probs(tr, TRAIN_CAP)
    pte = probs(te, 0)
    del model
    torch.cuda.empty_cache()
    return ptr, pte


def metrics(y, pred, label):
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
    print("  %s" % label)
    print("    sensitivity %.2f (%.2f-%.2f)" % (sens, sl, sh))
    print("    specificity %.2f (%.2f-%.2f)" % (spec, pl, ph))
    print("    PPV %.2f   NPV %.2f   kappa %.3f" % (ppv, npv, kap))
    return {"sens": 100 * sens, "sens_lo": 100 * sl, "sens_hi": 100 * sh,
            "spec": 100 * spec, "spec_lo": 100 * pl, "spec_hi": 100 * ph,
            "ppv": 100 * ppv, "npv": 100 * npv, "kappa": kap,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out", default="output/npj_baselines")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    notes = pickle.loads(Path(a.cache).read_bytes())
    r = pd.read_csv("output/raters/rater_long.csv", dtype=str)
    w3 = r.pivot_table(index="omop_person_id", columns="rater", values="label_3",
                       aggfunc="first")
    inc = r[r["in_analysis_cohort"].astype(str).isin(["True", "true", "1"])]
    keep = set(inc["omop_person_id"].astype(str))
    g = w3["gold"].dropna()
    g = g[g.index.astype(str).isin(keep)]
    g = g[g != "Indeterminate"]
    ids = [i for i in g.index if i in notes]
    y = (g.loc[ids] == "AUD+").to_numpy().astype(int)
    X = [notes[i] for i in ids]
    print("labelled patients with notes: %d, of which %d positive"
          % (len(ids), y.sum()), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    chunks = [windows(tok, x, MAXLEN) for x in X]
    nch = np.array([len(c) for c in chunks])
    ncap = np.array([len(thin(c, TRAIN_CAP)) for c in chunks])
    print("windows per patient: median %d, mean %.1f, max %d, total %d"
          % (np.median(nch), nch.mean(), nch.max(), nch.sum()), flush=True)
    print("after the training cap of %d: total %d" % (TRAIN_CAP, ncap.sum()),
          flush=True)
    if a.dry:
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device %s, cohort prevalence %.3f (reported only, never used as a "
          "threshold level)" % (dev, float(y.mean())), flush=True)

    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pred = {k: np.zeros(len(y), int) for k in AGGS}
    score = {k: np.zeros(len(y)) for k in AGGS}
    chosen = []
    for fold, (tr, te) in enumerate(outer.split(X, y), 1):
        best, best_s = None, -1.0
        inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
        for lr in LRS:
            accs = []
            for itr, ite in inner.split([X[i] for i in tr], y[tr]):
                gi = [tr[i] for i in itr]
                gj = [tr[i] for i in ite]
                prev = float(y[gi].mean())
                ptr, pte = fit_score(tok, dev, chunks, y, gi, gj, lr,
                                     "fold %d lr %.0e inner" % (fold, lr))
                str_ = np.array([agg(p, "mean") for p in ptr])
                ste = np.array([agg(p, "mean") for p in pte])
                thr = float(np.quantile(str_, 1 - prev))
                accs.append((((ste >= thr).astype(int)) == y[gj]).mean())
            m = float(np.mean(accs))
            print("    fold %d lr %.0e inner accuracy %.3f" % (fold, lr, m),
                  flush=True)
            if m > best_s:
                best_s, best = m, lr
        chosen.append(best)
        prev = float(y[tr].mean())
        print("  fold %d training-fold prevalence %.3f" % (fold, prev),
              flush=True)
        ptr, pte = fit_score(tok, dev, chunks, y, list(tr), list(te), best,
                             "fold %d final" % fold)
        for k in AGGS:
            str_ = np.array([agg(p, k) for p in ptr])
            ste = np.array([agg(p, k) for p in pte])
            thr = float(np.quantile(str_, 1 - prev))
            score[k][te] = ste
            pred[k][te] = (ste >= thr).astype(int)
        print("  fold %d done, lr %.0e" % (fold, best), flush=True)

    print()
    print("learning rate chosen per outer fold: %s" % chosen)

    prev = float(y.mean())  # reporting only, below this point
    ros = pd.read_csv("resources/patient_roster.csv",
                      dtype={"omop_person_id": str})
    code = ros.set_index("omop_person_id")["has_aud_icd"].reindex(ids).to_numpy()
    unc = (code == 0)
    pos = unc & (y == 1)

    rows = []
    for k in AGGS:
        row = metrics(y, pred[k], "out of fold, %s window aggregation, "
                                  "prevalence-quantile rule" % k)
        kk = int((pred[k][pos] == 1).sum())
        lo, hi = wil(kk, int(pos.sum()))
        print("    of the %d uncoded patients with a reference diagnosis, "
              "identified %d  %.3f (%.3f-%.3f)"
              % (int(pos.sum()), kk, kk / pos.sum(), lo, hi))
        row.update({"comparator": "clinical encoder, out of fold",
                    "model": MODEL, "aggregation": k,
                    "uncoded_recovered": kk, "uncoded_positive": int(pos.sum()),
                    "uncoded_rate": 100 * kk / pos.sum(),
                    "primary": k == "mean"})
        rows.append(row)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "note_bert_row.csv", index=False)
    np.save(out / "note_bert_scores.npy",
            np.vstack([score[k] for k in AGGS]))
    (out / "note_bert_meta.json").write_text(json.dumps(
        {"model": MODEL, "seed": SEED, "lrs": LRS, "epochs": EPOCHS,
         "batch": BATCH, "maxlen": MAXLEN, "train_cap": TRAIN_CAP,
         "aggregations": list(AGGS), "chosen_lr": chosen,
         "n": int(len(y)), "n_pos": int(y.sum()),
         "windows_total": int(nch.sum()), "windows_median": int(np.median(nch)),
         "windows_max": int(nch.max()),
         "windows_total_capped": int(ncap.sum())}, indent=2))
    print()
    print("  wrote %s/note_bert_row.csv" % a.out)


if __name__ == "__main__":
    main()
