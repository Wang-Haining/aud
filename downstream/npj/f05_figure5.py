#!/usr/bin/env python3
"""Assemble the auditability figure: can a reader check any single classification?

  a  how often the automatic quote-to-note match places the quotation, four models
  b  the audit trail of all 487 classifications, from the automatic match to the
     quotations read against the chart by hand
  c  which evidence categories the primary model cites, by reference-standard label

The figure reads as one sequence rather than one cut: a asks how far the automatic
check gets, b accounts for everything it did not settle, and c says what the model was
citing when it did. The trigger-to-evidence alluvial answers the same question as c in
aggregate and moves to a supplementary figure.

Two quantities are separated and must not be merged. A classification that returns a
quotation makes a claim about the note and can be checked against it. A classification
of indeterminacy has nothing to quote and returns an explanatory statement instead; it
is not a failed quotation and is not scored as one.

The chain this file replaces read resources/gold_reference_v1_2026-08-03.csv, deleted
when the standard closed, and render_fig4_auditability.py rebuilt the reference standard
a fourth time from chart_scores.csv. Both are gone; ref_standard.build() is the source.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "downstream")
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch                                    # noqa: E402

import ref_standard as RS                                          # noqa: E402
from grounding_finish import fragments, is_absence_statement       # noqa: E402

OUT = "downstream/npj/out/figure5"
DETAIL = "output/npj_goal6/grounding_detail.csv"
PANEL = "output/canonical_panel.csv"
ABSTAIN = "output/npj_goal6/abstention_check.csv"
PRIMARY, PRIMARY_TAG = "Qwen3.6-27B", "aud_qwen36_27b"

INK, MUTED, RULE = "#1f1e1d", "#55534e", "#B9B7B2"
BLUE, SKY, AMBER, VERM, GREY = "#0072B2", "#56B4E9", "#E69F00", "#D55E00", "#AAAAAA"
LABELS = ["Definitely Yes", "Probably Yes", "Uncertain",
          "Probably No", "Definitely No", "Not Enough Data"]
SHORT = {"Definitely Yes": "Def yes", "Probably Yes": "Prob yes",
         "Uncertain": "Uncertain", "Probably No": "Prob no",
         "Definitely No": "Def no", "Not Enough Data": "Not enough data"}
LCOL = {"Definitely Yes": BLUE, "Probably Yes": SKY, "Uncertain": AMBER,
        "Probably No": "#C9C7C2", "Definitely No": "#8D8B87",
        "Not Enough Data": "#E3E1DD"}
TOPCATS = ["social history", "hpi narrative", "discharge diagnosis",
           "problem list", "assessment/plan", "structured field",
           "discharge summary", "progress note", "hospital course",
           "consultation note"]


def parse_cats(s):
    if pd.isna(s):
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            return [c.lower().strip() for c in loader(s)]
        except Exception:
            continue
    return []


def load(a):
    ref, man = RS.build()
    ref["pid"] = ref["omop_person_id"].astype(str)
    keep = set(ref["pid"])

    gd = pd.read_csv(a.detail, dtype=str)
    gd["pid"] = gd["omop_person_id"].astype(str)
    gd = gd[gd["pid"].isin(keep)]

    pan = pd.read_csv(a.panel, dtype={"omop_person_id": str})
    pan["pid"] = pan["omop_person_id"].astype(str)
    qw = pan[pan["model"].eq(PRIMARY_TAG)].set_index("pid")

    # A classification that made no quotation claim cannot be placed and is not a
    # failure to place one. The primary model returns that kind 30 times; the panel
    # used to count all of them against its match rate.
    TAG = {"GPT-OSS-120B": "aud", "Qwen3.6-27B": "aud_qwen36_27b",
           "GPT-OSS-20B": "aud_gptoss_20b", "Nemotron-3-120B-FP8": "aud_nemotron_fp8"}

    def statement_test(tag):
        q = pan[pan["model"].eq(tag)].set_index("pid")["supporting_quote"]

        def f(pid):
            v = q.get(pid)
            return (pd.isna(v) or not str(v).strip() or is_absence_statement(v)
                    or not fragments(v))
        return f

    rates = []
    for meth, s in gd.groupby("method"):
        ok = s["fuzzy_grounded_090"].astype(str).str.lower().isin(["true", "1"])
        unres = s.loc[~ok, "pid"].tolist()
        is_stmt = statement_test(TAG[meth])
        stmt = sum(1 for p in unres if is_stmt(p))
        left = len(unres) - stmt
        # H.W. read the primary model's residual against the charts and located every
        # one. No other model's residual was read, so its status is unknown.
        hand, unknown = (left, 0) if meth == PRIMARY else (0, left)
        claims = len(s) - stmt
        rates.append({"method": meth, "n": len(s), "grounded": int(ok.sum()),
                      "rate": 100 * float(ok.mean()),
                      "unresolved": int((~ok).sum()),
                      "no_claim": stmt, "located_by_hand": hand,
                      "unread": unknown, "quotation_claims": claims,
                      "located_of_claims": 100.0 * (int(ok.sum()) + hand) / claims})
    rates = pd.DataFrame(rates).sort_values("rate")
    assert rates["n"].nunique() == 1, "the four models do not share one denominator"

    s = gd[gd["method"].eq(PRIMARY)]
    ok = s["fuzzy_grounded_090"].astype(str).str.lower().isin(["true", "1"])
    unresolved = s.loc[~ok, "pid"].tolist()
    quote = qw["supporting_quote"]

    def is_statement(pid):
        v = quote.get(pid)
        return (pd.isna(v) or not str(v).strip() or is_absence_statement(v)
                or not fragments(v))

    stmt = [p for p in unresolved if is_statement(p)]
    hand = [p for p in unresolved if not is_statement(p)]
    # Every quotation the automatic step failed to place was read against the chart by
    # one of us and located. The count is carried as data rather than as a caption
    # claim, so a repeat review that finds one changes the figure.
    not_found = 0

    trail = {"n": len(s), "auto": int(ok.sum()), "unresolved": len(unresolved),
             "statement": len(stmt), "hand": len(hand), "not_found": not_found,
             "quoting": len(s) - len(stmt)}

    # Evidence categories by reference-standard label, on the same cohort.
    r = ref[["pid", "gold_label"]].merge(
        qw.reset_index()[["pid", "evidence_categories"]], on="pid", how="inner")
    rows = []
    for _, x in r.iterrows():
        for c in parse_cats(x["evidence_categories"]):
            if c in TOPCATS:
                rows.append({"category": c, "reference": x["gold_label"]})
    ct = (pd.DataFrame(rows).groupby(["category", "reference"]).size()
          .unstack(fill_value=0).reindex(columns=LABELS, fill_value=0)
          .reindex(TOPCATS).fillna(0).astype(int))

    ab = pd.read_csv(a.abstain, dtype={"omop_person_id": str})
    ab = ab[ab["omop_person_id"].astype(str).isin(set(stmt))]
    return rates, trail, ct, ab, man, stmt, hand


def draw_rate(ax, rates):
    """Every classification of each model, in four states.

    The axis runs from zero to the full cohort, so no baseline is cut. A single
    automatic rate put the primary model third of four, because a classification
    carrying a stated absence rather than a quotation counted against it. The states
    are separated here, and the one model whose residual was read is the only one for
    which the unplaced quotations have a verdict.
    """
    y = np.arange(len(rates))
    order = [("grounded", BLUE, "Located by the automatic match"),
             ("located_by_hand", SKY, "Located by reading the chart"),
             ("unread", AMBER, "Unplaced, not read"),
             ("no_claim", "#D9D7D2", "No quotation claim")]
    for i, r in enumerate(rates.itertuples()):
        left = 0
        for key, colour, _ in order:
            w = int(getattr(r, key))
            if w:
                ax.barh(i, w, left=left, height=0.62, color=colour,
                        edgecolor="white", linewidth=0.4, zorder=2)
            left += w
    ax.set_yticks(y)
    ax.set_yticklabels(rates["method"], fontsize=5.6)
    for t in ax.get_yticklabels():
        if t.get_text() == PRIMARY:
            t.set_fontweight("bold")
    n = int(rates["n"].iloc[0])
    ax.set_xlim(0, n)
    ax.set_xticks([0, n // 4, n // 2, 3 * n // 4, n])
    ax.set_ylim(-0.6, len(rates) - 0.4)
    ax.set_xlabel("Classifications")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=c, edgecolor="white", linewidth=0.4, label=lab)
               for _, c, lab in order]
    ax.legend(handles=handles, fontsize=4.9, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), ncol=2, handlelength=1.1,
              handleheight=0.9, columnspacing=0.8, borderpad=0.2)
    ax.set_title("Where every classification ended up\n"
                 "n = %d per model" % n, fontsize=6.5, color="#333333", pad=4)


def draw_trail(ax, t):
    """The audit trail as a stepped bar: what happened to all 487 classifications."""
    steps = [
        ("All classifications", t["n"], INK, 0),
        ("Quotation placed automatically", t["auto"], BLUE, 1),
        ("Not placed automatically", t["unresolved"], VERM, 1),
        ("An explanatory statement, nothing to place", t["statement"], "#9C9A96", 2),
        ("A quotation, read against the chart", t["hand"], VERM, 2),
        ("Located in the chart", t["hand"] - t["not_found"], BLUE, 3),
        ("Not found in the chart", t["not_found"], VERM, 3),
    ]
    y = np.arange(len(steps))[::-1]
    W = float(t["n"])
    for yy, (lab, n, c, depth) in zip(y, steps):
        x0 = depth * 0.0
        ax.barh(yy, n, left=x0, height=0.56, color=c, edgecolor="white",
                linewidth=0.5, zorder=2)
        ax.text(n + W * 0.012, yy, "%d" % n, va="center", fontsize=5.6, color=INK)
        # Depth is carried by the ink, not by an indent: a right-aligned label cannot
        # be indented, and a leading marker reads as an out-dent.
        ax.text(-W * 0.02, yy, lab, va="center", ha="right", fontsize=5.5,
                color=INK if depth == 0 else MUTED)
    ax.set_yticks([])
    ax.set_xlim(0, W * 1.10)
    ax.set_ylim(-0.6, len(steps) - 0.4)
    ax.set_xlabel("Classifications")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Every classification accounted for\n"
                 "%d of %d quotation claims were located"
                 % (t["quoting"] - t["not_found"], t["quoting"]),
                 fontsize=6.5, color="#333333", pad=4)


def draw_evidence(ax, ct):
    y = np.arange(len(ct))
    left = np.zeros(len(ct))
    for lab in LABELS:
        v = ct[lab].to_numpy(float)
        ax.barh(y, v, left=left, height=0.62, color=LCOL[lab], edgecolor="white",
                linewidth=0.4, label=SHORT[lab], zorder=2)
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([("HPI narrative" if c == "hpi narrative"
                         else c[0].upper() + c[1:]) for c in ct.index], fontsize=5.6)
    ax.invert_yaxis()
    ax.set_xlabel("Citations")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", fontsize=5.0, frameon=False, ncol=3,
              handlelength=0.9, handletextpad=0.4, columnspacing=1.0)
    ax.set_title("What the primary model cited, by reference-standard label\n"
                 "a classification may cite more than one category, so citations "
                 "exceed classifications", fontsize=6.5, color="#333333", pad=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detail", default=DETAIL)
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--abstain", default=ABSTAIN)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rates, trail, ct, ab, man, stmt, hand = load(a)

    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 3, "ytick.major.size": 3, "legend.frameon": False,
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 300})

    fig = plt.figure(figsize=(7.09, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.30], hspace=0.46, wspace=0.62,
                          left=0.135, right=0.960, top=0.895, bottom=0.085)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, :])
    draw_rate(axa, rates)
    draw_trail(axb, trail)
    draw_evidence(axc, ct)
    for ax, lab, dx in ((axa, "a", -0.34), (axb, "b", -0.62), (axc, "c", -0.145)):
        ax.text(dx, 1.02, lab, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="left")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "figure5.%s" % ext))
    plt.close(fig)

    # Findings 5 quotes the audit trail and the match rate. Until 15 August this script
    # printed them and wrote nothing, so a number in the text could not be checked against
    # the run that drew the panel beside it. Two scripts in that state produced published
    # numbers that had drifted; this one is now written down as well.
    rates.to_csv(os.path.join(a.out, "panel_a_match_rates.csv"), index=False)
    pd.DataFrame([trail]).to_csv(os.path.join(a.out, "panel_b_audit_trail.csv"),
                                 index=False)
    print("wrote %s/figure5.pdf and .png, and the panel tables beside them" % a.out)
    print("\ncohort: %d analysed of %d rostered"
          % (man["cohort"]["analysed"], man["cohort"]["roster_rows"]))
    print("\npanel a, automatic quote-to-note match")
    print(rates.to_string(index=False, float_format=lambda x: "%.1f" % x))
    print("\npanel b, the audit trail")
    for k in ("n", "auto", "unresolved", "statement", "hand", "not_found", "quoting"):
        print("  %-12s %d" % (k, trail[k]))
    print("  located, of the classifications that made a quotation claim: %d of %d"
          % (trail["quoting"] - trail["not_found"], trail["quoting"]))
    print("\nthe explanatory statements, alcohol terms in the chart (n = %d of %d "
          "matched in the abstention check)" % (len(ab), trail["statement"]))
    if len(ab):
        n0 = int((ab["alcohol_terms_in_chart"] == 0).sum())
        n14 = int(ab["alcohol_terms_in_chart"].between(1, 4).sum())
        n5 = int((ab["alcohol_terms_in_chart"] >= 5).sum())
        print("  none at all %d, one to four %d, five or more %d" % (n0, n14, n5))
        loud = ab[ab["alcohol_terms_in_chart"] >= 5].sort_values(
            "alcohol_terms_in_chart", ascending=False)
        for _, r in loud.iterrows():
            print("    %d terms in a chart of %s characters"
                  % (int(r["alcohol_terms_in_chart"]),
                     "{:,d}".format(int(r["chart_chars"]))))
        print("  reference-standard label of these charts: %s"
              % ab["reference"].value_counts().to_dict())
    print("\npanel c, evidence categories by reference label")
    print(ct.to_string())


if __name__ == "__main__":
    main()
