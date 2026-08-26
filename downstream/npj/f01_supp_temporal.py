#!/usr/bin/env python3
"""Where in time the evidence sits, relative to the index date.

The note extract runs 540 days either side of each patient's index date, and 56 per cent
of the notes in it are dated after that date. Two sentences in the paper depend on which
side the evidence falls on: the Figure 3 caption says the model reads notes already
present at t0, and the Discussion says the extract was drawn around the index
diagnosis and so cannot support a claim about earlier detection. Both cannot hold.

This file measures it rather than arguing it. For every chart it splits the note text into
documents, reads each document's own Note Date, and locates the quotation the model cited.
The question it answers is narrow and is the one that matters: for the patients the model
called positive, was the text it quoted available at the index date, or only afterwards.

It does not re-run the model. A rerun restricted to $t \\le t_0$ is the definitive
analysis; this establishes how much would change and whether the rerun is worth its cost.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, "downstream")
sys.path.insert(0, "downstream/npj")

import ref_standard as RS                                          # noqa: E402
import fig6_bayes_strata as B                                      # noqa: E402

OUT = "downstream/npj/out/temporal"
NOTES = "output/npj_baselines/notes_full.pkl"
WINDOW = "resources/windowed_report_ids.csv"
PANEL = "output/canonical_panel.csv"
PRIMARY = "aud_qwen36_27b"
POS = {"Definitely Yes", "Probably Yes"}

DOC_SPLIT = re.compile(r"<text\s+title='NEW DOCUMENT'>")
DATE_RE = re.compile(r"Note Date:\s*(\d{2})-(\d{2})-(\d{4})")


def norm(s):
    """Whitespace and case folded, so a quotation that crossed a line break still matches."""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def documents(blob):
    """(date, normalised text) for every document in one patient's note blob."""
    out = []
    for chunk in DOC_SPLIT.split(blob)[1:]:
        m = DATE_RE.search(chunk)
        d = None
        if m:
            mm, dd, yyyy = m.groups()
            try:
                d = pd.Timestamp(int(yyyy), int(mm), int(dd))
            except ValueError:
                d = None
        out.append((d, norm(chunk)))
    return out


def build(a):
    ref, man = RS.build()
    assert len(ref) == man["cohort"]["analysed"] == 487
    ref["pid"] = pd.to_numeric(ref["omop_person_id"], errors="coerce").astype("int64")

    reg = B.build()
    reg["pid"] = reg["person_id"].astype("int64")

    p = pd.read_csv(a.panel, dtype={"omop_person_id": str}, low_memory=False)
    p = p[p["model"].eq(PRIMARY)][["omop_person_id", "label_canonical",
                                   "supporting_quote"]].copy()
    p["pid"] = pd.to_numeric(p["omop_person_id"], errors="coerce").astype("int64")

    d = (ref[["omop_person_id", "pid", "gold_binary"]]
         .merge(reg[["pid", "t0"]], on="pid", how="left")
         .merge(p[["pid", "label_canonical", "supporting_quote"]], on="pid", how="left"))
    d["llm_pos"] = d["label_canonical"].isin(POS)

    # Exposure comes from the windowed manifest, which dates 7,946 notes across 403 of
    # the sampled patients. The note text dates only about half its documents, so the
    # manifest is the better source for how many notes sit on each side of the index date.
    wr = pd.read_csv(a.window, parse_dates=['note_date', 'index_date'])
    wr = wr[wr['in_window'].eq(1.0) & wr['note_date'].notna()].copy()
    wr['pid'] = pd.to_numeric(wr['omop_person_id'], errors='coerce').astype('int64')
    wr['pre'] = wr['note_date'] <= wr['index_date']
    ex = wr.groupby('pid').agg(w_notes=('pre', 'size'), w_pre=('pre', 'sum'))

    notes = pd.read_pickle(a.notes)
    rows = []
    for _, r in d.iterrows():
        blob = notes.get(str(r["omop_person_id"]))
        rec = {"pid": r["pid"], "gold": r["gold_binary"], "llm_pos": bool(r["llm_pos"]),
               "t0": r["t0"], "has_notes": blob is not None,
               "n_docs": 0, "n_dated": 0, "n_pre": 0, "quote_found": False,
               "quote_first": pd.NaT}
        if blob is not None:
            docs = documents(blob)
            rec["n_docs"] = len(docs)
            dated = [(dt, tx) for dt, tx in docs if dt is not None]
            rec["n_dated"] = len(dated)
            if pd.notna(r["t0"]):
                rec["n_pre"] = sum(1 for dt, _ in dated if dt <= r["t0"])
            q = norm(r["supporting_quote"]) if pd.notna(r["supporting_quote"]) else ""
            if len(q) >= 12:
                hits = [dt for dt, tx in dated if q in tx]
                if hits:
                    rec["quote_found"] = True
                    rec["quote_first"] = min(hits)
        rows.append(rec)
    out = pd.DataFrame(rows).merge(ex, on="pid", how="left")
    out["quote_pre_t0"] = out["quote_first"].notna() & out["t0"].notna() & (
        out["quote_first"] <= out["t0"])
    out["quote_offset_days"] = (out["quote_first"] - out["t0"]).dt.days
    return out, man


def report(d, man):
    n = len(d)
    print("analysis cohort                       %d" % n)
    print("  note blob present                   %d" % int(d["has_notes"].sum()))
    print("  has an index date                   %d" % int(d["t0"].notna().sum()))
    print()
    print("documents per chart, median %.0f; dated %.0f%% of them"
          % (d["n_docs"].median(), 100 * d["n_dated"].sum() / max(1, d["n_docs"].sum())))
    print()

    w = d[d["w_notes"].notna()]
    print("note-window manifest, the better-covered source")
    print("  charts with dated in-window notes    %d of %d" % (len(w), n))
    print("  notes in window, total               %d" % int(w["w_notes"].sum()))
    print("  of those at or before t0             %d  (%.1f%%)"
          % (int(w["w_pre"].sum()), 100 * w["w_pre"].sum() / w["w_notes"].sum()))
    print("  charts with no note at or before t0  %d  (%.1f%%)"
          % (int(w["w_pre"].eq(0).sum()), 100 * w["w_pre"].eq(0).mean()))
    print("  median notes per chart, all %.0f, pre-t0 %.0f"
          % (w["w_notes"].median(), w["w_pre"].median()))
    print()

    have = d[d["t0"].notna() & d["n_dated"].gt(0)]
    print("of the %d charts with an index date and at least one dated document:" % len(have))
    none_pre = int(have["n_pre"].eq(0).sum())
    print("  no document at or before t0      %d  (%.1f%%)"
          % (none_pre, 100 * none_pre / len(have)))
    print("  median documents in the extract     %.0f" % have["n_docs"].median())
    print("  median at or before t0           %.0f" % have["n_pre"].median())
    print("  share of dated documents pre-t0  %.1f%%"
          % (100 * have["n_pre"].sum() / have["n_dated"].sum()))
    print()

    q = d[d["quote_found"]]
    print("the cited quotation was located in a dated document for %d of %d charts (%.0f%%)"
          % (len(q), n, 100 * len(q) / n))
    if len(q):
        print("  first appears at or before t0    %d  (%.1f%%)"
              % (int(q["quote_pre_t0"].sum()), 100 * q["quote_pre_t0"].mean()))
        print("  offset in days, median %.0f, IQR %.0f to %.0f"
              % (q["quote_offset_days"].median(),
                 q["quote_offset_days"].quantile(.25),
                 q["quote_offset_days"].quantile(.75)))
    print()

    qp = q[q["llm_pos"]]
    print("restricted to the %d charts the model called positive and whose quotation was "
          "located:" % len(qp))
    if len(qp):
        print("  quotation available at t0        %d  (%.1f%%)"
              % (int(qp["quote_pre_t0"].sum()), 100 * qp["quote_pre_t0"].mean()))
        print("  quotation only afterwards           %d  (%.1f%%)"
              % (int((~qp["quote_pre_t0"]).sum()), 100 * (~qp["quote_pre_t0"]).mean()))
    print()

    # The claim the Figure 3 caption rests on: patients the code missed and the model found.
    ros = pd.read_csv("resources/patient_roster.csv")
    ros["pid"] = pd.to_numeric(ros["omop_person_id"], errors="coerce").astype("Int64")
    m = d.merge(ros[["pid", "has_aud_icd"]], on="pid", how="left")
    rec = m[m["gold"].eq("AUD+") & m["has_aud_icd"].fillna(0).astype(int).eq(0)
            & m["llm_pos"]]
    found = rec[rec["quote_found"]]
    print("the recovery claim: %d uncoded reference-positive charts the model called positive"
          % len(rec))
    print("  quotation located                   %d" % len(found))
    if len(found):
        print("  available at t0                  %d  (%.1f%%)"
              % (int(found["quote_pre_t0"].sum()), 100 * found["quote_pre_t0"].mean()))
        print("  only afterwards                     %d"
              % int((~found["quote_pre_t0"]).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--notes", default=NOTES)
    ap.add_argument("--window", default=WINDOW)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d, man = build(a)
    d.to_csv(os.path.join(a.out, "temporal_exposure.csv"), index=False)
    report(d, man)
    print("\nwrote %s/temporal_exposure.csv" % a.out)


if __name__ == "__main__":
    main()
