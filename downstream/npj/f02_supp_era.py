#!/usr/bin/env python3
"""Transportability across calendar era, on the reviewed charts.

The paper asks whether the measurement holds across hospital system, across model and
across generation settings. It did not ask whether it holds over time. This file asks it.

The quantity is not the registry-scale coding trend of Supplementary Figure 5. That trend
describes the record. This describes the reading: agreement between the index test and the
reference standard, and sensitivity of the index test against sensitivity of the diagnosis
code, in each calendar band.

The bands are the ones `fig6_bayes_strata` already fits on, 2015--2016, 2017--2018 and
2019--2020, so the paper carries one calendar definition rather than two. They hold 131,
153 and 201 of the 487 charts. Two charts carry no cirrhosis code and so have no $t_0$;
they fall outside every band and are reported as such.

The window matters for the comparator. ICD-10-CM replaced ICD-9-CM in October 2015, inside
the first band, and registry coding of AUD rose over the same period, so the code is known
to move on this axis and the index test is not.

Panel a uses all charts in a band, because agreement is defined on three categories and
indeterminate is one of them. Panel b uses the reference-positive charts in a band, because
that is the denominator of a sensitivity.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import argparse
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
import matplotlib.pyplot as plt                                    # noqa: E402

import npj_style as S                                                    # noqa: E402

import ref_standard as RS                                          # noqa: E402
import fig6_bayes_strata as B                                      # noqa: E402
import stats_core as sc                                            # noqa: E402

OUT = "downstream/npj/out/figure2"
PANEL = "output/canonical_panel.csv"
PRIMARY_TAG = "aud_qwen36_27b"
ROSTER = "resources/patient_roster.csv"

INK, MUTED, RULE = "#1f1e1d", "#55534e", "#B9B7B2"
# The index test carries the index colour and the diagnosis code the code colour, as
# in npj_style. Until 24 August this module had the two the other way round, which is
# the palette the paper used before the orange convention.
C_INDEX, C_CODE = "#D55E00", "#0072B2"
ORD3 = ["AUD-", "Uncertain", "AUD+"]
NED = "Not Enough Data"
# En dashes, not the LaTeX double hyphen: these strings are drawn by matplotlib.
BANDS = {0: "2015–2016", 1: "2017–2018", 2: "2019–2020"}
POS = {"Definitely Yes", "Probably Yes"}
N_BOOT, SEED = 2000, 20260814


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    z, p = 1.959963985, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def boot_kappa(a, b):
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(a), len(a))
        try:
            out.append(sc.cohens_kappa(a[i], b[i], ORD3, None))
        except Exception:
            continue
    return np.nanpercentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def load(a):
    ref, man = RS.build()
    assert len(ref) == man["cohort"]["analysed"] == 487, "reference standard is not the 487"
    ref = ref[["omop_person_id", "gold_binary", "gold_three", "gold_label"]].copy()
    ref["pid"] = pd.to_numeric(ref["omop_person_id"], errors="coerce").astype("int64")

    reg = B.build()
    reg["pid"] = reg["person_id"].astype("int64")

    p = pd.read_csv(a.panel, dtype={"omop_person_id": str})
    p = p[p["model"].eq(PRIMARY_TAG)][["omop_person_id", "label_canonical"]].copy()
    p["pid"] = pd.to_numeric(p["omop_person_id"], errors="coerce").astype("int64")

    ros = pd.read_csv(a.roster)
    ros["pid"] = pd.to_numeric(ros["omop_person_id"], errors="coerce").astype("Int64")

    d = (ref.merge(reg[["pid", "t0", "period"]], on="pid", how="left")
            .merge(p[["pid", "label_canonical"]], on="pid", how="left")
            .merge(ros[["pid", "has_aud_icd"]], on="pid", how="left"))

    # The three-category label the agreement panel scores. gold_three carries the reference
    # side; the index test's six-grade output collapses the same way the rest of the paper
    # collapses it.
    # Primary definition: three categories with not enough data excluded. The
    # reference side has to come from gold_label rather than gold_binary, because
    # gold_binary already merges uncertain and not enough data into one value.
    def _c3(s):
        return pd.Series(np.where(s.isin(POS), "AUD+",
                         np.where(s.isin({"Definitely No", "Probably No"}), "AUD-",
                         np.where(s.eq("Uncertain"), "Uncertain", None))),
                         index=s.index).where(s.notna() & s.ne(NED))

    d["ref3"] = _c3(d["gold_label"])
    d["llm3"] = _c3(d["label_canonical"])
    d["code_pos"] = d["has_aud_icd"].astype("Int64").fillna(0).astype(int).eq(1)
    d["ref_pos"] = d["gold_binary"].eq("AUD+")
    return d, man


def rows(d):
    out = []
    for k, name in BANDS.items():
        g = d[d["period"].eq(k)]
        kp = g[g["llm3"].notna() & g["ref3"].notna()]
        kap = sc.cohens_kappa(kp["llm3"].to_numpy(), kp["ref3"].to_numpy(), ORD3, None)
        lo, hi = boot_kappa(kp["llm3"].to_numpy(), kp["ref3"].to_numpy())
        pos = g[g["ref_pos"]]
        ls = wilson(int(pos["llm3"].eq("AUD+").sum()), len(pos))
        cs = wilson(int(pos["code_pos"].sum()), len(pos))
        out.append({"band": name, "n": len(kp), "n_band": len(g), "n_pos": len(pos),
                    "kappa": kap, "k_lo": lo, "k_hi": hi,
                    "llm_sens": ls[0], "llm_lo": ls[1], "llm_hi": ls[2],
                    "code_sens": cs[0], "code_lo": cs[1], "code_hi": cs[2],
                    "gain": ls[0] - cs[0]})
    return pd.DataFrame(out)


def draw_kappa(ax, t):
    y = np.arange(len(t))[::-1]
    ax.hlines(y, t["k_lo"], t["k_hi"], color=C_INDEX, lw=1.5, zorder=2)
    ax.plot(t["kappa"], y, "o", ms=5, color=C_INDEX, zorder=3)
    for yy, r in zip(y, t.itertuples()):
        ax.text(1.02, yy, "%.3f" % r.kappa, va="center", ha="left", fontsize=6,
                color=INK, transform=ax.get_yaxis_transform())
    ax.set_yticks(y)
    ax.set_yticklabels(["%s\n%d charts" % (r.band, r.n) for r in t.itertuples()],
                       fontsize=6.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Cohen's $\\kappa$ against the reference standard", fontsize=6.6)
    ax.set_ylim(-0.6, len(t) - 0.4)
    ax.tick_params(axis="x", labelsize=6.2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.grid(axis="x", color=RULE, lw=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)


def draw_sens(ax, t):
    y = np.arange(len(t))[::-1]
    off = 0.16
    for col, lo, hi, c, lab in (("llm_sens", "llm_lo", "llm_hi", C_INDEX, "LLM"),
                                ("code_sens", "code_lo", "code_hi", C_CODE, "Diagnosis code")):
        dy = off if c == C_INDEX else -off
        ax.hlines(y + dy, t[lo], t[hi], color=c, lw=1.5, zorder=2)
        ax.plot(t[col], y + dy, "o", ms=5, color=c, zorder=3, label=lab)
    for yy, r in zip(y, t.itertuples()):
        ax.text(1.02, yy, "$+$%.1f" % (100 * r.gain), va="center", ha="left",
                fontsize=6, color=INK, transform=ax.get_yaxis_transform())
    ax.set_yticks(y)
    ax.set_yticklabels(["%s\n%d with AUD" % (r.band, r.n_pos) for r in t.itertuples()],
                       fontsize=6.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Sensitivity", fontsize=6.6)
    ax.set_ylim(-0.6, len(t) - 0.4)
    ax.tick_params(axis="x", labelsize=6.2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.grid(axis="x", color=RULE, lw=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    # The dots sit against the right edge, so the left half of the panel is empty and the
    # legend does not have to crowd the row labels.
    ax.legend(loc="center left", fontsize=6, frameon=False, handletextpad=0.4,
              borderaxespad=0.4)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--roster", default=ROSTER)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d, man = load(a)
    t = rows(d)

    # The stack here ran Arial, Helvetica, DejaVu Sans. Neither of the first two is on
    # Quartz, so this figure alone came out in DejaVu while the rest of the paper is in
    # Liberation Sans, which is metric-compatible with Arial. npj_style carries the full
    # stack and npj Digital Medicine's own type size, so the figure is styled from there.
    S.apply_npj_style()
    matplotlib.rcParams.update({
        "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 300})
    fig = plt.figure(figsize=S.mm(S.W_DOUBLE, 62.2))
    gs = fig.add_gridspec(1, 2, wspace=0.62, left=0.115, right=0.945,
                          top=0.90, bottom=0.215)
    axa, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    draw_kappa(axa, t)
    draw_sens(axb, t)
    for ax, lab in ((axa, "a"), (axb, "b")):
        ax.text(-0.30, 1.12, lab, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top", ha="left")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "figure2_era.%s" % ext))
    plt.close(fig)
    t.to_csv(os.path.join(a.out, "figure2_era.csv"), index=False)

    n_band = int(d["period"].notna().sum())
    print("wrote %s/figure2_era.pdf and .csv" % a.out)
    print("charts in a band %d of %d; %d carry no cirrhosis code and so no $t_0$"
          % (n_band, len(d), len(d) - n_band))
    print()
    show = t.copy()
    for c in ("kappa", "k_lo", "k_hi", "llm_sens", "llm_lo", "llm_hi",
              "code_sens", "code_lo", "code_hi", "gain"):
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    print()
    print("index test, kappa range across bands  %.3f to %.3f"
          % (t["kappa"].min(), t["kappa"].max()))
    print("index test, sensitivity range         %.3f to %.3f"
          % (t["llm_sens"].min(), t["llm_sens"].max()))
    print("diagnosis code, sensitivity range     %.3f to %.3f"
          % (t["code_sens"].min(), t["code_sens"].max()))
    print("paired gain range, points             %.1f to %.1f"
          % (100 * t["gain"].min(), 100 * t["gain"].max()))


if __name__ == "__main__":
    main()
