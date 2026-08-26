#!/usr/bin/env python3
"""Every interval the pipeline publishes, as a list the manuscript check can read.

The manuscript prints about two hundred triples of the form value (low to high). Four of
them were stale on 24 August and nothing saw them: the kappa gate covers kappas only, and
a triple inside a caption falls outside every check the manuscript side has.

Column conventions in this tree, applied in this order:

  X_mean, X_lo, X_hi     the value column carries the _mean suffix
  X, X_lo, X_hi          the value column is the name without the suffix
  X, ci_low, ci_high     the value column immediately precedes the low bound
  X, lo, hi              the same, for the bare pair

Values are written unrounded. The manuscript side rounds each triple to the number of
decimals the sentence prints, and tries the value as written and multiplied by one
hundred, so a proportion published here matches a percentage printed there.

Writes resources/published_intervals.json.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import csv
import json
import os

ROOTS = ["downstream/npj/out", "output/fig6_registry"]
OUT = "resources/published_intervals.json"
SKIP = ("_archive", "_prerefit", "_pre_arviz", "failed_attempts", "canonical_panel_history")
LOWS = {"ci_low": "ci_high", "lo": "hi", "low": "high"}


def triples(header):
    """(value column, low column, high column) for every interval this header carries."""
    out = []
    for i, h in enumerate(header):
        hi, base = None, None
        if h in LOWS and LOWS[h] in header:
            hi = LOWS[h]
        elif h.endswith("_lo") and h[:-3] + "_hi" in header:
            hi, base = h[:-3] + "_hi", h[:-3]
        elif h.endswith("_low") and h[:-4] + "_high" in header:
            hi, base = h[:-4] + "_high", h[:-4]
        if hi is None:
            continue
        val = None
        for cand in ((base + "_mean") if base else None, base,
                     header[i - 1] if i else None):
            if cand and cand in header and cand not in (h, hi):
                val = cand
                break
        if val:
            out.append((val, h, hi))
    return out


def main():
    rows_out, files = [], 0
    for root in ROOTS:
        for d, _, names in os.walk(root):
            if any(s in d for s in SKIP):
                continue
            for n in sorted(names):
                if not n.endswith(".csv"):
                    continue
                p = os.path.join(d, n)
                try:
                    rows = list(csv.DictReader(open(p, newline="")))
                except Exception:
                    continue
                if not rows:
                    continue
                cols = triples(list(rows[0].keys()))
                if not cols:
                    continue
                files += 1
                for r in rows:
                    for val, lo, hi in cols:
                        try:
                            v, a, b = float(r[val]), float(r[lo]), float(r[hi])
                        except (TypeError, ValueError):
                            continue
                        rows_out.append([v, a, b, "%s::%s" % (p, val)])
    json.dump({"triples": rows_out}, open(OUT, "w"), indent=0)
    print("%d files carry intervals, %d triples" % (files, len(rows_out)))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
