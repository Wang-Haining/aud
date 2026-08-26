#!/usr/bin/env python3
"""Findings 01: the participant flow diagram and Table 1.

The two belong in one script because they are one derivation. Table 1 describes whichever
patients survive the flow, so a filter applied in one and not the other is a defect that no
amount of separate checking would catch. Here the cohort is derived once and both outputs
are built from it, and the script refuses to finish if a count in the diagram fails to match
the corresponding denominator in the table.

The diagram is a CONSORT-style flow. CONSORT was written for randomised trials and the
form is used well beyond them, which is what a reader expects to see: how patients moved
from the sampling frame through the language model and the reference standard to the
analysed set.

Reads the reference standard from ref_standard.build(), never from a cached copy, so the
three charts still awaiting a third reading are visible here and are labelled.

Outputs, under --out:
    flow_counts.csv     every number in the diagram, named, one row each
    flow.tex            the diagram as TikZ, for the manuscript
    flow.pdf, flow.png  the diagram as a figure, for slides
    table1.tex          Table 1
    table1.csv          the same, machine readable
    supp_sdoh.csv       the documentation audit behind the SDOH rows

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "downstream")
sys.path.insert(0, str(HERE))
import fig6_bayes_strata as B  # noqa: E402
import npj_style as S  # noqa: E402
import ref_standard as R  # noqa: E402

OUT = "downstream/npj/out/f01_flow_table1"
MODEL = "aud_qwen36_27b"


def registry_frame():
    """Registry denominators, from the same builder the registry model uses."""
    m = B.build()
    ever = m["aud"].notna()
    return {"registry_indexed": len(m),
            "registry_coded": int(ever.sum()),
            "registry_uncoded": int((~ever).sum())}


def collect(g: pd.DataFrame, man: dict, ros: pd.DataFrame) -> list[dict]:
    """Every count in the diagram, computed once, in the order it is read."""
    c = man["counts"]
    reg = registry_frame()
    read = g["gold_label"].notna()
    three = g["gold_three"]

    rows = [
        ("registry_indexed", "Cirrhosis registry, patients with an index date",
         reg["registry_indexed"]),
        ("registry_coded", "AUD diagnosis code recorded at any time",
         reg["registry_coded"]),
        ("registry_uncoded", "No AUD diagnosis code", reg["registry_uncoded"]),
        ("sampled", "Sampled for chart review", len(g)),
        ("sampled_coded", "Sampled, AUD code present",
         int((ros["has_aud_icd"] == 1).sum())),
        ("sampled_uncoded", "Sampled, no AUD code",
         int((ros["has_aud_icd"] == 0).sum())),
        ("notes_total", "Notes retrieved", int(ros["n_notes"].sum())),
        ("index_test_applied", "Notes read by the language model", len(g)),
        ("ref_read_once", "Read by the hepatology fellow", c["read_by_maaz"]),
        ("ref_unread", "No chart reading available, excluded",
         int(len(g) - c["read_by_maaz"])),
        ("ref_read_twice", "Read a second time by an attending", c["read_twice"]),
        ("ref_single", "Single reading only", int(c["read_by_maaz"] - c["read_twice"])),
        ("ref_concordant", "Two readings, same category", c["concordant_bucket"]),
        ("ref_discordant", "Two readings, different category", c["discordant_bucket"]),
        ("ref_third_done", "Third reading complete", c["third_reading_present"]),
        ("ref_third_pending", "Third reading outstanding, provisional label",
         c["third_reading_outstanding"]),
        ("ref_established", "Reference standard obtained by chart review",
         int(read.sum())),
        ("ref_pos", "Reference: AUD present", int((three == "AUD+").sum())),
        ("ref_neg", "Reference: AUD absent", int((three == "AUD-").sum())),
        ("ref_indet", "Reference: indeterminate, excluded from accuracy",
         int((three == "Indeterminate").sum())),
        ("analysed_accuracy", "Analysed for accuracy",
         int(((three == "AUD+") | (three == "AUD-")).sum())),
    ]
    return [{"key": k, "label": lab, "n": int(n)} for k, lab, n in rows]


def reconcile(rows: list[dict], g: pd.DataFrame, man: dict):
    """Refuse to emit a diagram whose numbers do not add up."""
    v = {r["key"]: r["n"] for r in rows}
    checks = [
        ("registry splits into coded and uncoded",
         v["registry_coded"] + v["registry_uncoded"] == v["registry_indexed"]),
        ("the sample splits into coded and uncoded",
         v["sampled_coded"] + v["sampled_uncoded"] == v["sampled"]),
        ("the language model read every sampled patient's notes",
         v["index_test_applied"] == v["sampled"]),
        ("read plus unread accounts for the sample",
         v["ref_read_once"] + v["ref_unread"] == v["sampled"]),
        ("second readings plus single readings account for those read",
         v["ref_read_twice"] + v["ref_single"] == v["ref_read_once"]),
        ("concordant plus discordant accounts for the second readings",
         v["ref_concordant"] + v["ref_discordant"] == v["ref_read_twice"]),
        ("third readings done plus outstanding accounts for the discordant",
         v["ref_third_done"] + v["ref_third_pending"] == v["ref_discordant"]),
        ("the three reference categories account for the established set",
         v["ref_pos"] + v["ref_neg"] + v["ref_indet"] == v["ref_established"]),
        ("the analysed set is the two determinate categories",
         v["ref_pos"] + v["ref_neg"] == v["analysed_accuracy"]),
        ("the established reference matches the non-missing gold labels",
         v["ref_established"] == int(g["gold_label"].notna().sum())),
        ("the provisional count matches the manifest",
         v["ref_third_pending"] == len(man["provisional_review_ids"])),
    ]
    bad = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print("  %-62s %s" % (name, "ok" if ok else "FAILED"))
    if bad:
        raise SystemExit("flow counts do not reconcile: %s" % "; ".join(bad))


def tikz(rows: list[dict], provisional: list[str]) -> str:
    v = {r["key"]: r["n"] for r in rows}
    note = (r"\\ \footnotesize provisional: %s" % ", ".join(provisional)) if provisional \
        else ""
    return r"""\begin{tikzpicture}[
  font=\small,
  box/.style={draw, rounded corners=2pt, align=center, inner sep=5pt,
              text width=64mm, minimum height=9mm},
  side/.style={box, text width=52mm},
  ->, >=latex, node distance=8mm]
  \node[box] (reg)   {Cirrhosis registry with an index date\\ $n=%(registry_indexed)s$};
  \node[box, below=of reg] (samp)
        {Sampled for chart review\\ $n=%(sampled)s$
         (%(sampled_coded)s with an AUD code, %(sampled_uncoded)s without)};
  \node[box, below=of samp] (idx)
        {Notes read by the language model\\ $n=%(index_test_applied)s$};
  \node[box, below=of idx] (ref)
        {Reference standard established by chart review\\ $n=%(ref_established)s$};
  \node[side, right=10mm of ref] (refx)
        {Excluded, no chart reading available\\ $n=%(ref_unread)s$};
  \node[box, below=of ref] (det)
        {Determinate reference\\ AUD present $n=%(ref_pos)s$,
         AUD absent $n=%(ref_neg)s$};
  \node[side, right=10mm of det] (detx)
        {Excluded, indeterminate reference\\ $n=%(ref_indet)s$};
  \node[box, below=of det] (an)
        {Analysed for accuracy\\ $n=%(analysed_accuracy)s$%(note)s};
  \draw (reg) -- (samp);
  \draw (samp) -- (idx);
  \draw (idx) -- (ref);
  \draw (ref) -- (refx);
  \draw (ref) -- (det);
  \draw (det) -- (detx);
  \draw (det) -- (an);
\end{tikzpicture}
""" % dict(v, note=note)


def figure(rows: list[dict], provisional: list[str], out: str, stem="flow"):
    """Draw the flow. `provisional` empty draws the manuscript version.

    Two versions are written on every run. The manuscript version states only what the
    reader needs, since a note about which charts are awaiting a third reading describes
    the state of our worklist rather than the study. The internal version keeps the note,
    so the working copy never loses track of it. The guard against that note being dropped
    and then forgotten is ref_standard.py --strict, not this annotation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    v = {r["key"]: r["n"] for r in rows}
    # Each box says what happened to the patients at that step, in the words a clinician
    # uses. The reporting-guideline terms were carried here until 25 August; "index test"
    # in particular names the thing being evaluated to nobody outside diagnostic accuracy
    # reporting, and here that thing is the language model reading notes.
    main = [
        ("Cirrhosis registry with an index date", f"n = {v['registry_indexed']:,}", None),
        ("Sampled for chart review",
         f"n = {v['sampled']} ({v['sampled_coded']} with an AUD diagnosis code, "
         f"{v['sampled_uncoded']} without)", None),
        ("Notes read by the language model", f"n = {v['index_test_applied']}", None),
        ("Reference standard obtained", f"n = {v['ref_established']}",
         "clinician chart review reached a verdict"),
        ("Determinate reference",
         f"AUD present {v['ref_pos']}, AUD absent {v['ref_neg']}", None),
        ("Analysed for accuracy", f"n = {v['analysed_accuracy']}", None),
    ]
    # An exclusion box reading zero invites the question of why it is drawn at all, so a
    # branch that excluded nobody is left off.
    side = {}
    if v["ref_unread"]:
        side[3] = ("Excluded, no chart reading", f"n = {v['ref_unread']}")
    if v["ref_indet"]:
        side[4] = ("Excluded, indeterminate reference", f"n = {v['ref_indet']}")

    # This script set no rcParams, so the diagram came out in matplotlib's defaults:
    # DejaVu Sans at 10 pt, and Type 3 fonts, which Nature Portfolio does not accept.
    # npj_style gives the paper's face, its type size and pdf.fonttype 42.
    S.apply_npj_style()
    fig, ax = plt.subplots(figsize=S.mm(S.W_DOUBLE, 218.0))
    ax.set_xlim(0, 10)
    # The lower bound leaves a clear band under the last box for the provisional note,
    # which otherwise prints across it.
    ax.set_ylim(-1.2 if provisional else -0.2, len(main) * 2)
    ax.axis("off")
    ys = [len(main) * 2 - 1 - 2 * i for i in range(len(main))]
    heights = [1.34 if gl else 1.1 for _, _, gl in main]
    for i, ((title, sub, gloss), y) in enumerate(zip(main, ys)):
        h = heights[i]
        ax.add_patch(FancyBboxPatch((0.4, y - h / 2), 5.6, h,
                                    boxstyle="round,pad=0.06,rounding_size=0.08",
                                    fc="white", ec="#3c3c3c", lw=1.1))
        ax.text(3.2, y + (0.32 if gloss else 0.16), title, ha="center", va="center",
                fontsize=9.5)
        ax.text(3.2, y + (-0.06 if gloss else -0.24), sub, ha="center", va="center",
                fontsize=9.5, color="#4a4a4a")
        if gloss:
            ax.text(3.2, y - 0.4, gloss, ha="center", va="center", fontsize=8,
                    style="italic", color="#77756f")
        if i:
            # The arrow starts at the bottom edge of the box above, which is not a fixed
            # offset once a box has grown to hold a gloss.
            ax.add_patch(FancyArrowPatch((3.2, ys[i - 1] - heights[i - 1] / 2),
                                         (3.2, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=11,
                                         color="#3c3c3c", lw=1.1))
        if i in side:
            st, ss = side[i]
            ax.add_patch(FancyBboxPatch((6.6, y - 0.5), 3.1, 1.0,
                                        boxstyle="round,pad=0.06,rounding_size=0.08",
                                        fc="#f4f4f2", ec="#8a8a8a", lw=1.0))
            ax.text(8.15, y + 0.15, st, ha="center", va="center", fontsize=8.5)
            ax.text(8.15, y - 0.2, ss, ha="center", va="center", fontsize=8.5,
                    color="#4a4a4a")
            ax.add_patch(FancyArrowPatch((6.0, y), (6.6, y), arrowstyle="-|>",
                                         mutation_scale=10, color="#8a8a8a", lw=1.0))
    if provisional:
        ax.text(3.2, -0.75, "provisional reference on %d chart(s): %s"
                % (len(provisional), ", ".join(provisional)),
                ha="center", va="center", fontsize=8, color="#8a3a2a")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "%s.%s" % (stem, ext)), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--gold", default=R.OUT_CSV)
    ap.add_argument("--cohort", default=R.COHORT, choices=["era_a", "all"])
    ap.add_argument("--skip-table1", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    g, man = R.build(cohort=a.cohort)
    g.to_csv(a.gold, index=False)
    # The roster carries every sampled patient; the cohort filter lives in ref_standard, so
    # the roster is cut down to whatever it returned rather than filtered again here.
    ros = pd.read_csv("resources/patient_roster.csv", dtype={"omop_person_id": str})
    ros = ros[ros["omop_person_id"].isin(set(g["omop_person_id"]))].copy()
    co = man["cohort"]
    print("cohort %s: %d of %d sampled patients (%s)"
          % (co["name"], co["analysed"], co["roster_rows"], co["definition"]))

    print("=" * 78)
    print("PARTICIPANT FLOW")
    print("=" * 78)
    rows = collect(g, man, ros)
    for r in rows:
        print("  %-52s %8s" % (r["label"], format(r["n"], ",")))

    print()
    print("=" * 78)
    print("RECONCILIATION")
    print("=" * 78)
    reconcile(rows, g, man)

    pd.DataFrame(rows).to_csv(os.path.join(a.out, "flow_counts.csv"), index=False)
    prov = man["provisional_review_ids"]
    open(os.path.join(a.out, "flow.tex"), "w").write(tikz(rows, []))
    figure(rows, [], a.out, stem="flow")
    figure(rows, prov, a.out, stem="flow_internal")
    print()
    print("wrote flow_counts.csv, flow.tex, flow.pdf, flow.png (for the manuscript)")
    print("     and flow_internal.pdf, flow_internal.png (with the provisional note)")

    if a.skip_table1:
        return
    print()
    print("=" * 78)
    print("TABLE 1")
    print("=" * 78)
    cmd = [sys.executable, "downstream/build_table1.py", "--out", a.out,
           "--gold", a.gold]
    p = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    if p.returncode:
        sys.stderr.write(p.stderr)
        raise SystemExit("Table 1 failed; the flow outputs above are still written")

    # The point of building both here is that they cannot disagree, so check the counts
    # themselves rather than anything about the layout: the reference-standard rows of
    # Table 1 must reproduce the three terminal numbers of the flow.
    t1 = pd.read_csv(os.path.join(a.out, "table1.csv"), index_col=0)
    overall = t1.columns[-1]
    want = {r["key"]: r["n"] for r in rows}
    pairs = [("AUD present", "ref_pos"), ("AUD absent", "ref_neg"),
             ("Indeterminate", "ref_indet")]
    seen = {}
    for label, key in pairs:
        hits = [i for i in t1.index if str(i).strip() == label]
        if not hits:
            raise SystemExit("Table 1 has no %r row to check against the flow" % label)
        seen[key] = int(str(t1.at[hits[-1], overall]).split()[0])
    bad = [(k, seen[k], want[k]) for _, k in pairs if seen[k] != want[k]]
    for _, k in pairs:
        print("  %-12s flow %4d   Table 1 %4d   %s"
              % (k, want[k], seen[k], "ok" if seen[k] == want[k] else "MISMATCH"))
    if bad:
        raise SystemExit("Table 1 and the flow disagree: %s" % bad)
    if sum(seen.values()) != want["ref_established"]:
        raise SystemExit("Table 1 reference rows sum to %d, the flow establishes %d"
                         % (sum(seen.values()), want["ref_established"]))
    print("Table 1 reproduces the flow's reference standard, %d patients"
          % want["ref_established"])

    if prov:
        print()
        print("!" * 74)
        print("Both outputs rest on a provisional reference for %s." % ", ".join(prov))
        print("Rerun this script once chart/ratings/gawrieh.json holds those charts.")
        print("!" * 74)


if __name__ == "__main__":
    main()
