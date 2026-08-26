#!/usr/bin/env python3
"""Finish the grounding review: the unresolved quotations, and the abstentions.

Part 1. The quotations that could not be settled from a single 260-character window.
Some of those windows landed on the wrong part of the chart, so this retrieves the
three best candidate passages per quotation and widens them, and it aligns each
comma-separated fragment of the quotation separately, because the model elides across
non-contiguous spans with an ellipsis and a single contiguous comparison punishes that.

Part 2. The abstentions. Thirty classifications returned an explanatory statement, such
as "no mention of alcohol use", rather than a quotation. Nothing can be located for
them because the statement is the model's own prose. That is only a good outcome if
those charts really are silent about alcohol, which nobody has checked. Two independent
checks are run:

  a  what the reference standard says for the same patients. If the human reviewers
     also found nothing, the model's abstention agrees with a human reading.
  b  whether the chart contains any alcohol term at all. This is blunt but objective,
     and a chart full of alcohol terms where the model reported none would be a
     serious finding rather than a good one.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

sys.path.insert(0, "downstream")

PRIMARY = "aud_qwen36_27b"
WIDE = 700
ALCOHOL = re.compile(
    r"\b(alcohol\w*|etoh|eth\s?anol|drink\w*|drank|beer|wine|liquor|vodka|whisk\w+|"
    r"binge\w*|intoxicat\w+|sober\w*|sobriety|abstinen\w+|withdrawal|ciwa|"
    r"delirium tremens|dts?\b|audit-?c|cage|aa\b|alcoholics anonymous|"
    r"naltrexone|acamprosate|disulfiram|antabuse)\b", re.I)


def norm(t):
    t = unicodedata.normalize("NFKC", str(t)).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", t)).strip()


# Prose the model writes *about* the note rather than text taken *from* it. On the
# abstention branch there is nothing to quote, so the schema gets an explanation
# instead. These are not failed quotations and must not be scored as if they were.
ABSENCE = re.compile(
    r"\b(no (mention|evidence|documentation|reference|record|indication|report)"
    r"|not (mentioned|documented|reported|recorded|present|available)"
    r"|nothing (in|about)|absent from|does not (mention|document|contain)"
    r"|there (is|are) no|none (documented|mentioned|reported)"
    r"|insufficient (data|information|documentation)|no alcohol)\b", re.I)


def is_absence_statement(quote):
    return bool(ABSENCE.search(str(quote)))


def fragments(quote):
    """Split on the ellipses the model uses to join non-adjacent spans."""
    parts = re.split(r"\s*(?:\.\.\.|…)\s*", str(quote))
    return [p.strip() for p in parts if len(norm(p).split()) >= 3]


def locate(frag, note, n=1):
    """Best n passages for one fragment, with their similarity."""
    fn = norm(frag)
    raw = re.sub(r"\s+", " ", str(note))
    toks = norm(note).split()
    if not fn or not toks:
        return []
    f = fn.split()
    k = max(4, len(f))
    if k >= len(toks):
        return [(raw[:WIDE], SequenceMatcher(None, fn, " ".join(toks)).ratio())]
    freq = Counter(toks)
    anchors = sorted({w for w in f if len(w) > 2}, key=lambda w: freq.get(w, 0))[:6]
    starts = set()
    for w in anchors:
        if not freq.get(w):
            continue
        for i, tok in enumerate(toks):
            if tok == w:
                starts.add(max(0, min(i - k // 2, len(toks) - k)))
            if len(starts) > 4000:
                break
    if not starts:
        return []
    cand = sorted(starts, key=lambda s: -len(set(toks[s:s + k]) & set(f)))[:400]
    scored = []
    for s in cand:
        scored.append((s, SequenceMatcher(None, fn, " ".join(toks[s:s + k])).ratio()))
    scored.sort(key=lambda x: -x[1])
    out = []
    nn = norm(raw)
    for s, r in scored[:n]:
        probe = " ".join(toks[s:s + min(6, k)])
        pos = nn.find(probe)
        frac = pos / max(len(nn), 1) if pos >= 0 else s / max(len(toks), 1)
        c = int(frac * len(raw))
        out.append((raw[max(0, c - WIDE // 3):c + WIDE].strip(), r))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mapping", default="resources/mapped_report_ids.csv")
    ap.add_argument("--notes-dir", nargs="+", default=[
        "/N/project/depot/hw56/aud/cirrhosis_regv20/redacted_txt",
        "/N/project/depot/hw56/aud/cirrhosis_regv20/redacted_pdf-text"])
    ap.add_argument("--out", default="output/npj_goal6/grounding_finish.txt")
    a = ap.parse_args()

    gd = pd.read_csv("output/npj_goal6/grounding_detail.csv", dtype=str)
    gd = gd[gd["method"].str.contains("Qwen", na=False)] if "method" in gd else gd
    ref = pd.read_csv("resources/gold_reference_v1_2026-08-03.csv",
                      dtype={"omop_person_id": str})
    pan = pd.read_csv("output/canonical_panel.csv", dtype={"omop_person_id": str})
    pan = pan[pan["model"] == PRIMARY]
    have = set(ref.loc[ref["gold_label"].notna(), "omop_person_id"])

    col = [c for c in gd.columns if "ground" in c.lower()][0]
    ung = gd[~gd[col].astype(str).str.lower().isin(["true", "1"])]
    ids = [i for i in ung["omop_person_id"].astype(str).unique() if i in have]
    print("ungrounded within the 487 analysis cohort: %d" % len(ids), flush=True)

    cache = Path("output/npj_goal6/notes_cache.pkl")
    if cache.exists():
        notes = pickle.loads(cache.read_bytes())
        print("note corpus from cache", flush=True)
    else:
        print("loading notes for %d patients only" % len(ids), flush=True)
        m = pd.read_csv(a.mapping, dtype=str)
        m = m[m["OMOP_PERSON_ID"].astype(str).isin(ids)]
        roots = [Path(p) for p in a.notes_dir]
        notes = {}
        for pid, sub in m.groupby("OMOP_PERSON_ID"):
            parts = []
            for rid in sub["REPORT_ID"].dropna().unique():
                for r in roots:
                    p = r / f"{rid}.txt"
                    if p.exists():
                        parts.append(p.read_text(encoding="utf-8", errors="replace"))
                        break
            notes[str(pid)] = "\n".join(parts)
        print("loaded %d patients" % len(notes), flush=True)

    q = pan.set_index("omop_person_id")["supporting_quote"]
    g = ref.set_index("omop_person_id")["gold_label"]
    lines, quoting, abstaining = [], [], []
    for pid in ids:
        quote = q.get(pid)
        no_quote = pd.isna(quote) or not str(quote).strip()
        (abstaining if (no_quote or is_absence_statement(quote)
                        or not fragments(quote)) else quoting).append(pid)

    # ---------------------------------------------------------------- part 1
    print("\n" + "=" * 74)
    print("PART 1  QUOTATIONS, ALIGNED FRAGMENT BY FRAGMENT")
    print("=" * 74)
    summary = []
    for pid in quoting:
        note = notes.get(pid, "")
        frs = fragments(q.get(pid))
        best = []
        for f in frs:
            hits = locate(f, note, n=1)
            best.append(hits[0][1] if hits else 0.0)
        worst = min(best) if best else 0.0
        summary.append({"omop_person_id": pid, "fragments": len(frs),
                        "worst_fragment_similarity": round(worst, 3),
                        "reference": g.get(pid)})
        lines.append("--- %s  reference=%s  fragments=%d  worst=%.3f ---"
                     % (pid, g.get(pid), len(frs), worst))
        for f, b in zip(frs, best):
            hits = locate(f, notes.get(pid, ""), n=1)
            lines.append("  FRAGMENT (%.3f): %s" % (b, f[:200]))
            lines.append("  PASSAGE: %s" % (hits[0][0][:900] if hits else "(nothing)"))
        lines.append("")
    s = pd.DataFrame(summary)
    print("  quotations with fragments: %d" % len(s))
    if len(s):
        print("  worst-fragment similarity, distribution:")
        print(s["worst_fragment_similarity"].describe()[
            ["min", "25%", "50%", "75%", "max"]].round(3).to_string())
        print("\n  every fragment above 0.90: %d of %d"
              % (int((s["worst_fragment_similarity"] >= 0.90).sum()), len(s)))
        print("  worst fragment below 0.75, needs a human: %d"
              % int((s["worst_fragment_similarity"] < 0.75).sum()))
        print(s.sort_values("worst_fragment_similarity").head(8).to_string(index=False))

    # ---------------------------------------------------------------- part 2
    print("\n" + "=" * 74)
    print("PART 2  THE ABSTENTIONS: WAS THE CHART REALLY SILENT")
    print("=" * 74)
    rows = []
    for pid in abstaining:
        note = notes.get(pid, "")
        hits = ALCOHOL.findall(note)
        rows.append({"omop_person_id": pid, "reference": g.get(pid),
                     "alcohol_terms_in_chart": len(hits),
                     "distinct_terms": len({h.lower() for h in hits}),
                     "chart_chars": len(note)})
    b = pd.DataFrame(rows)
    n_stmt = sum(1 for p in abstaining
                 if not pd.isna(q.get(p)) and str(q.get(p)).strip())
    print("  classifications with no quotable content: %d" % len(b))
    print("    returned an explanatory statement: %d" % n_stmt)
    print("    returned nothing at all:           %d" % (len(abstaining) - n_stmt))
    if len(b):
        print("\n  what the human reference said for the same patients:")
        print(b["reference"].value_counts().to_string())
        print("\n  alcohol terms found in the chart:")
        print("    none at all            %d" % int((b["alcohol_terms_in_chart"] == 0).sum()))
        print("    1 to 4 mentions        %d"
              % int(b["alcohol_terms_in_chart"].between(1, 4).sum()))
        print("    5 or more mentions     %d" % int((b["alcohol_terms_in_chart"] >= 5).sum()))
        loud = b[b["alcohol_terms_in_chart"] >= 5].sort_values(
            "alcohol_terms_in_chart", ascending=False)
        if len(loud):
            print("\n  charts with 5 or more alcohol terms where the model reported none:")
            print(loud.to_string(index=False))
        b.to_csv("output/npj_goal6/abstention_check.csv", index=False)

    Path(a.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n  wrote %s   PHI, stays on the cluster" % a.out)
    print("  wrote output/npj_goal6/abstention_check.csv   counts only, no chart text")


if __name__ == "__main__":
    main()
