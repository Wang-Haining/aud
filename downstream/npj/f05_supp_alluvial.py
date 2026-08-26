#!/usr/bin/env python3
"""Supplementary figure: which extracted trigger carried each cited evidence category.

The trigger-to-evidence flows for the primary model, on the 487 charts of the analysis
cohort. This was panel d of the auditability figure. It answers the same question as
panel c of that figure in aggregate, so it moves here rather than occupying a main
panel; it is kept because it shows which signal the model was acting on, which panel c
does not.

The flows are computed here from the canonical panel restricted to the analysis cohort,
rather than read from f5_evidence_paths.csv. That file is written by
run_npj_provenance.py over all 499 rostered patients and holds aggregated counts, so it
cannot be filtered to the 487 after the fact.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "downstream")
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.path import Path as MPath                          # noqa: E402
import matplotlib.patches as mpatches                              # noqa: E402

import ref_standard as RS                                          # noqa: E402

OUT = "downstream/npj/out/figure5"
PANEL = "output/canonical_panel.csv"
PRIMARY_TAG = "aud_qwen36_27b"
INK, MUTED, RULE = "#1f1e1d", "#55534e", "#B9B7B2"
WONG = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
NONE_TRIGGER = "no standalone trigger"


def parse_list(s):
    if pd.isna(s):
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(s)
            return [str(x).strip().lower() for x in v] if isinstance(v, list) else []
        except Exception:
            continue
    return []


def flows(a):
    ref, man = RS.build()
    keep = set(ref["omop_person_id"].astype(str))
    p = pd.read_csv(a.panel, dtype={"omop_person_id": str})
    p = p[p["model"].eq(PRIMARY_TAG) & p["omop_person_id"].astype(str).isin(keep)]
    assert len(p) == man["cohort"]["analysed"], "the panel is not the analysis cohort"

    c = Counter()
    ltrig, levid = Counter(), Counter()
    for _, r in p.iterrows():
        tr = sorted(set(parse_list(r["triggers_present"]))) or [NONE_TRIGGER]
        ev = sorted(set(parse_list(r["evidence_categories"])))
        for t in tr:
            ltrig[t] += 1
            for e in ev:
                c[(t, e)] += 1
        for e in ev:
            levid[e] += 1
    return c, ltrig, levid, len(p), man


def draw(ax, c, ltrig, levid, n, top_e=8):
    trigs = [t for t, _ in ltrig.most_common()]
    evids = [e for e, _ in levid.most_common(top_e)]
    c = {k: v for k, v in c.items() if k[1] in evids}

    def stack(keys, weights):
        y, out = 0.0, {}
        for k in keys:
            h = weights[k] / sum(weights[j] for j in keys)
            out[k] = (y, y + h)
            y += h + 0.012
        return out, y

    wl = {t: sum(v for (a, b), v in c.items() if a == t) for t in trigs}
    trigs = [t for t in trigs if wl[t] > 0]
    wr = {e: sum(v for (a, b), v in c.items() if b == e) for e in evids}
    L, hl = stack(trigs, wl)
    R, hr = stack(evids, wr)
    sc = 1.0 / max(hl, hr)
    L = {k: (v[0] * sc, v[1] * sc) for k, v in L.items()}
    R = {k: (v[0] * sc, v[1] * sc) for k, v in R.items()}

    col = {t: WONG[i % len(WONG)] for i, t in enumerate(trigs)}
    lcur = {t: L[t][0] for t in trigs}
    rcur = {e: R[e][0] for e in evids}
    for t in trigs:
        for e in evids:
            v = c.get((t, e), 0)
            if not v:
                continue
            hl_ = (L[t][1] - L[t][0]) * v / wl[t]
            hr_ = (R[e][1] - R[e][0]) * v / wr[e]
            y0, y1 = lcur[t], lcur[t] + hl_
            z0, z1 = rcur[e], rcur[e] + hr_
            lcur[t], rcur[e] = y1, z1
            verts = [(0.10, y0), (0.42, y0), (0.58, z0), (0.90, z0),
                     (0.90, z1), (0.58, z1), (0.42, y1), (0.10, y1), (0.10, y0)]
            codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                     MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                     MPath.CLOSEPOLY]
            ax.add_patch(mpatches.PathPatch(MPath(verts, codes), facecolor=col[t],
                                            alpha=0.34, lw=0, zorder=2))
    for t in trigs:
        ax.add_patch(mpatches.Rectangle((0.075, L[t][0]), 0.025, L[t][1] - L[t][0],
                                        facecolor=col[t], lw=0, zorder=3))
        ax.text(0.065, (L[t][0] + L[t][1]) / 2, "%s  %d" % (t, ltrig[t]), ha="right",
                va="center", fontsize=5.4, color=INK)
    for e in evids:
        ax.add_patch(mpatches.Rectangle((0.90, R[e][0]), 0.025, R[e][1] - R[e][0],
                                        facecolor="#8D8B87", lw=0, zorder=3))
        ax.text(0.935, (R[e][0] + R[e][1]) / 2, "%s  %d" % (e, levid[e]), ha="left",
                va="center", fontsize=5.4, color=INK)
    ax.set_xlim(-0.30, 1.32)
    ax.set_ylim(-0.03, 1.03)
    ax.axis("off")
    ax.text(0.0875, 1.045, "Trigger extracted", ha="center", va="bottom", fontsize=6.0,
            color=MUTED)
    ax.text(0.9125, 1.045, "Evidence category cited", ha="center", va="bottom",
            fontsize=6.0, color=MUTED)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    c, ltrig, levid, n, man = flows(a)
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
        "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 300})
    fig = plt.figure(figsize=(7.09, 4.0))
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.88])
    draw(ax, c, ltrig, levid, n)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "figure5_alluvial.%s" % ext))
    plt.close(fig)

    print("wrote %s/figure5_alluvial.pdf and .png" % a.out)
    print("cohort %d classifications of the primary model" % n)
    print("\ntriggers extracted")
    for t, v in ltrig.most_common():
        print("  %-34s %4d" % (t, v))
    print("\nevidence categories cited")
    for e, v in levid.most_common(12):
        print("  %-34s %4d" % (e, v))
    print("\nlargest flows")
    for (t, e), v in sorted(c.items(), key=lambda kv: -kv[1])[:12]:
        print("  %-26s -> %-24s %4d" % (t, e, v))


if __name__ == "__main__":
    main()
