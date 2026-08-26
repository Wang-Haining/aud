#!/usr/bin/env python3
"""Render Figure 2: Performance and why diagnosis codes are not enough (5 panels).

Panels:
  2a  6x6 confusion matrix, Qwen3.6-27B vs reference
  2b  Forest: model / fellow / ICD x {kappa, sens, spec, PPV, NPV}, CIs
  2c  Alluvial: ICD F10.x -> reference standard, recovered cases highlighted
  2d  McNemar paired discordance table
  2e  PPV/NPV vs prevalence 0.01-0.50, ICD vs model, plausible band shaded
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
import numpy as np
import pandas as pd

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

GOLD = ""   # the reference standard is complete; nothing pending
PANELS = Path("downstream/npj_goal6_outputs/panels")

# Charts carrying only the first reading. Read from the reference-standard manifest so
# that it moves when the reference standard does, rather than being typed in here.
import json as _json  # noqa: E402
_MAN = _json.loads(Path("resources/gold_manifest_v2.json").read_text())
UNREVIEWED = _MAN["by_source"].get("maaz_unreviewed", 0)
CATS = ["Definitely Yes", "Probably Yes", "Uncertain",
        "Probably No", "Definitely No", "Not Enough Data"]
CAT_SHORT = ["Def Yes", "Prob Yes", "Uncert", "Prob No", "Def No", "NED"]

WONG_BLUE = "#0072B2"
WONG_ORANGE = "#E69F00"
WONG_VERMILLION = "#D55E00"
WONG_GREEN = "#009E73"
WONG_SKY = "#56B4E9"
WONG_PURPLE = "#CC79A7"


# ── Confusion matrix, six categories or the three-category collapse ─────────
# Which of the six grades each collapsed class holds. Uncertain and not enough data are both
# indeterminate, and they are not adjacent on the scale, which is why the collapse is stated
# as a mapping rather than as a partition of a sorted list.
COLLAPSE = {"Definitely Yes": "AUD+", "Probably Yes": "AUD+",
            "Probably No": "AUD-", "Definitely No": "AUD-",
            "Uncertain": "Indeterminate", "Not Enough Data": "Indeterminate"}
CATS3 = ["AUD+", "Indeterminate", "AUD-"]
CAT3_SHORT = ["AUD+", "Indet.", "AUD\u2212"]


def _confusion_matrix(cats, collapse=False):
    """The primary model's counts, as a square matrix over `cats`."""
    df = pd.read_csv(PANELS / "f1_confusion_6class.csv")
    qw = df[df["method"] == "Qwen3.6-27B"]
    n = len(cats)
    mat = np.zeros((n, n), dtype=int)
    for _, r in qw.iterrows():
        ref, pred = r["reference_label"], r["prediction_label"]
        if collapse:
            ref, pred = COLLAPSE[ref], COLLAPSE[pred]
        mat[cats.index(ref), cats.index(pred)] += int(r["n"])
    return mat


def _draw_matrix(ax, mat, short, fontsize):
    """Counts and row percentages on a Blues field. One implementation, two callers."""
    n = mat.shape[0]
    row_sums = mat.sum(axis=1, keepdims=True)
    pct = np.where(row_sums > 0, mat / row_sums * 100, 0)
    ax.imshow(mat, cmap="Blues", aspect="equal", origin="upper")
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if v == 0:
                continue
            c = "white" if v > mat.max() * 0.6 else "black"
            ax.text(j, i, f"{v}\n({pct[i, j]:.0f}%)", ha="center", va="center",
                    fontsize=fontsize, color=c)
    ax.set_xticks(range(n))
    ax.set_xticklabels(short, rotation=45 if n > 3 else 0, ha="right" if n > 3 else "center",
                       fontsize=5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short, fontsize=5)
    ax.set_xlabel("LLM (Qwen3.6-27B)", fontsize=6)
    ax.set_ylabel("Reference standard", fontsize=6)


def draw_confusion(ax):
    """Six categories. Unchanged entry point."""
    _draw_matrix(ax, _confusion_matrix(CATS), CAT_SHORT, 4.5)


def draw_confusion_3class(ax, note=False):
    """The collapse used by every analysis in the main text.

    `note` prints the denominator under the axes. It is off by default because the caption
    states it, and because the note hangs far enough below the axes to land in the panel
    beneath when the cell is short.
    """
    mat = _confusion_matrix(CATS3, collapse=True)
    _draw_matrix(ax, mat, CAT3_SHORT, 6.0)
    if note:
        ax.text(0.5, -0.30, ("N = %d %s" % (int(mat.sum()), GOLD)).strip(),
                transform=ax.transAxes, ha="center", fontsize=5, color="#666666")


# ── Panel 2b: Forest plot ────────────────────────────────────────────────────
def draw_forest(ax):
    df = pd.read_csv(PANELS / "f1_metric_forest.csv")

    metrics = ["linear_weighted_kappa", "sensitivity", "specificity", "PPV", "NPV"]
    metric_labels = ["Kappa (linear-weighted)", "Sensitivity", "Specificity", "PPV", "NPV"]
    method_order = ["Qwen3.6-27B", "Nemotron-3-120B-FP8", "GPT-OSS-120B",
                    "GPT-OSS-20B", "Maaz", "ICD F10.x"]

    rows = []
    for mi, metric in enumerate(metrics):
        sub = df[df["metric"] == metric]
        for method in method_order:
            r = sub[sub["method"] == method]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            rows.append({
                "value": r["value"],
                "ci_lo": r["ci_low"],
                "ci_hi": r["ci_high"],
                "metric_group": mi,
                "method": method,
            })

    # Build y positions with gaps between metric groups
    plot_rows = []
    y = 0
    prev_group = -1
    for r in rows:
        if r["metric_group"] != prev_group:
            if prev_group >= 0:
                y += 1.8
            else:
                y += 1.0
            prev_group = r["metric_group"]
        plot_rows.append((y, r))
        y += 1

    n_rows = y
    ax.set_xlim(0.76, 1.02)
    ax.set_ylim(n_rows + 1.5, -0.5)
    ax.axvline(1.0, color="#cccccc", lw=0.3, zorder=0)

    current_group = -1
    for yp, r in plot_rows:
        if r["metric_group"] != current_group:
            current_group = r["metric_group"]
            ax.text(0.76, yp - 0.9, metric_labels[current_group],
                    fontsize=5.5, fontweight="bold", va="bottom",
                    clip_on=False)

        is_fellow = r["method"] == "Maaz"
        is_icd = r["method"] == "ICD F10.x"
        is_primary = r["method"] == "Qwen3.6-27B"

        color = WONG_BLUE if is_primary else (
            "#999999" if is_fellow else (
                WONG_VERMILLION if is_icd else "#444444"))
        alpha = 0.4 if is_fellow else 1.0
        ms = 5 if is_primary else 3.5

        display_name = "Fellow" if is_fellow else r["method"]
        ax.text(0.005, yp, display_name, fontsize=4.5, va="center",
                color=color, alpha=alpha, transform=ax.get_yaxis_transform(),
                clip_on=False)

        ax.plot(r["value"], yp, "o", color=color, markersize=ms,
                alpha=alpha, zorder=3)
        ax.plot([r["ci_lo"], r["ci_hi"]], [yp, yp], "-", color=color,
                lw=0.8, alpha=alpha, zorder=2)

    ax.set_yticks([])
    ax.set_xlabel("Value (95% CI)", fontsize=6)
    ax.spines["left"].set_visible(False)

    # Warnings — below the plot
    warn1 = ("Selection-panel members agree with the reference by construction\n"
             "on %d unreviewed charts %s" % (UNREVIEWED, GOLD)).rstrip()
    warn2 = ("Fellow: reference incorporates fellow ratings for %d patients"
             % UNREVIEWED)
    ax.text(0.0, 1.0, warn1, fontsize=3.8, color=WONG_VERMILLION,
            va="bottom", transform=ax.transAxes, clip_on=False)
    ax.text(0.0, -0.06, warn2, fontsize=3.8, color="#999999",
            va="top", transform=ax.transAxes, clip_on=False)


# ── Panel 2c: Alluvial ICD -> Reference ──────────────────────────────────────
def _band(ax, x0, y0_top, y0_bot, x1, y1_top, y1_bot, color, alpha=0.35):
    verts = [
        (x0, y0_top),
        (x0 + 0.35, y0_top), (x1 - 0.35, y1_top), (x1, y1_top),
        (x1, y1_bot),
        (x1 - 0.35, y1_bot), (x0 + 0.35, y0_bot), (x0, y0_bot),
        (x0, y0_top),
    ]
    codes = [MPath.MOVETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.CLOSEPOLY]
    patch = mpatches.PathPatch(MPath(verts, codes), facecolor=color,
                               edgecolor="none", alpha=alpha, zorder=1)
    ax.add_patch(patch)


def draw_alluvial(ax):
    df = pd.read_csv(PANELS / "f2_alluvial.csv")
    qw = df[df["method"] == "Qwen3.6-27B"]

    # Aggregate ICD -> Reference flows
    flows = {}
    for _, r in qw.iterrows():
        key = (r["icd"], r["reference"])
        flows[key] = flows.get(key, 0) + int(r["n"])

    total = sum(flows.values())
    gap = 0.02

    # Left column: ICD
    icd_p = sum(v for (i, _), v in flows.items() if i == "+")
    icd_n = sum(v for (i, _), v in flows.items() if i == "-")

    # Right column: Reference
    ref_p = sum(v for (_, r), v in flows.items() if r == "+")
    ref_n = sum(v for (_, r), v in flows.items() if r == "-")
    ref_i = sum(v for (_, r), v in flows.items() if r == "I")

    scale = 1.0 / total
    x_left, x_right = 0.0, 1.0
    bar_w = 0.12

    # Left node positions (top to bottom: ICD+, ICD-)
    l_top = 1.0
    l_icd_p_h = icd_p * scale
    l_icd_n_h = icd_n * scale
    left_nodes = {
        "+": (l_top - l_icd_p_h, l_top),
        "-": (l_top - l_icd_p_h - gap - l_icd_n_h, l_top - l_icd_p_h - gap),
    }

    # Right node positions (top to bottom: Ref+, Ref-, Ref I)
    r_top = 1.0
    r_p_h = ref_p * scale
    r_n_h = ref_n * scale
    r_i_h = ref_i * scale
    right_nodes = {
        "+": (r_top - r_p_h, r_top),
        "-": (r_top - r_p_h - gap - r_n_h, r_top - r_p_h - gap),
        "I": (r_top - r_p_h - gap - r_n_h - gap - r_i_h,
              r_top - r_p_h - gap - r_n_h - gap),
    }

    # Draw left bars
    for key, (bot, top) in left_nodes.items():
        rect = mpatches.FancyBboxPatch(
            (x_left, bot), bar_w, top - bot,
            boxstyle="round,pad=0.005", facecolor=WONG_SKY if key == "+" else "#BBBBBB",
            edgecolor="#666666", lw=0.5, zorder=2)
        ax.add_patch(rect)
        n = icd_p if key == "+" else icd_n
        ax.text(x_left + bar_w / 2, (bot + top) / 2,
                f"{'AUD code' if key == '+' else 'No AUD code'}\nn={n}",
                ha="center", va="center", fontsize=4.5, fontweight="bold", zorder=3)

    # Draw right bars
    ref_labels = {"+": "AUD+", "-": "AUD−", "I": "Indet."}
    ref_colors = {"+": WONG_GREEN, "-": "#BBBBBB", "I": "#DDDDDD"}
    ref_ns = {"+": ref_p, "-": ref_n, "I": ref_i}
    for key, (bot, top) in right_nodes.items():
        rect = mpatches.FancyBboxPatch(
            (x_right - bar_w, bot), bar_w, top - bot,
            boxstyle="round,pad=0.005", facecolor=ref_colors[key],
            edgecolor="#666666", lw=0.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x_right - bar_w / 2, (bot + top) / 2,
                f"Ref {ref_labels[key]}\nn={ref_ns[key]}",
                ha="center", va="center", fontsize=4.5, fontweight="bold", zorder=3)

    # Draw flows
    flow_order = [("+", "+"), ("+", "-"), ("+", "I"),
                  ("-", "+"), ("-", "-"), ("-", "I")]

    # Track cursor positions within each node
    l_cursor = {k: v[1] for k, v in left_nodes.items()}
    r_cursor = {k: v[1] for k, v in right_nodes.items()}

    for icd_key, ref_key in flow_order:
        n = flows.get((icd_key, ref_key), 0)
        if n == 0:
            continue
        h = n * scale

        l_top_pos = l_cursor[icd_key]
        l_bot_pos = l_top_pos - h
        l_cursor[icd_key] = l_bot_pos

        r_top_pos = r_cursor[ref_key]
        r_bot_pos = r_top_pos - h
        r_cursor[ref_key] = r_bot_pos

        is_recovered = (icd_key == "-" and ref_key == "+")
        color = WONG_ORANGE if is_recovered else "#AAAAAA"
        alpha = 0.6 if is_recovered else 0.2

        _band(ax, x_left + bar_w, l_top_pos, l_bot_pos,
              x_right - bar_w, r_top_pos, r_bot_pos, color, alpha)

    # Recovered annotation
    recovered_n = flows.get(("-", "+"), 0)
    # Both numbers come from the flows, so the annotation cannot carry a count from
    # this cohort beside one from the previous cohort.
    # qw, not df: the panel file holds one block of rows per model, and summing over df
    # adds every model's recoveries together.
    recovered_by_model = int(qw[(qw["icd"] == "-") & (qw["reference"] == "+")
                               & (qw["prediction"] == "+")]["n"].sum())
    ax.annotate("%d with no AUD code,\nreference AUD+\n(%d recovered\nby the LLM)"
                % (recovered_n, recovered_by_model),
                xy=(0.5, right_nodes["+"][0] + 0.02),
                fontsize=4.5, color=WONG_VERMILLION, ha="center", va="bottom",
                fontweight="bold")

    ax.set_xlim(-0.05, 1.05)
    # The indeterminate node reaches y = -0.039, so a heading at -0.03 is drawn under
    # a box. The axis gains room below the nodes and the headings sit in it.
    ax.set_ylim(-0.13, 1.08)
    ax.axis("off")
    # "ICD negative" was the old label and it overstated what the record holds. The
    # absence of a code is silence, not a clinician stating the patient has no AUD.
    ax.text(0.0, -0.085, "AUD diagnosis code in the structured record",
            fontsize=5, ha="left")
    ax.text(1.0, -0.085, "Reference standard", fontsize=5, ha="right")


# ── Panel 2d: McNemar ────────────────────────────────────────────────────────
def draw_mcnemar(ax):
    df = pd.read_csv(PANELS / "f2_mcnemar.csv")
    qw = df[df["method"] == "Qwen3.6-27B"].iloc[0]
    a = int(qw["model_correct_icd_wrong"])  # 55
    b = int(qw["model_wrong_icd_correct"])  # 5
    n_common = int(qw["n_common_binary"])
    p = qw["p_value"]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.50, 0.95, "Paired discordance", ha="center", va="top",
            fontsize=6, fontweight="bold")
    ax.text(0.50, 0.88, f"N = {n_common} patients with binary classification {GOLD}",
            ha="center", va="top", fontsize=5, color="#666666")

    # Table headers
    x0, x1, x2 = 0.30, 0.55, 0.80
    y_head = 0.72
    y_r1 = 0.58
    y_r2 = 0.44

    ax.text(x1, y_head + 0.08, "ICD correct", ha="center", fontsize=5, fontweight="bold")
    ax.text(x2, y_head + 0.08, "ICD wrong", ha="center", fontsize=5, fontweight="bold")
    ax.text(x0 - 0.10, y_r1, "Model\ncorrect", ha="center", va="center",
            fontsize=5, fontweight="bold")
    ax.text(x0 - 0.10, y_r2, "Model\nwrong", ha="center", va="center",
            fontsize=5, fontweight="bold")

    # Cells
    cells = [
        (x1, y_r1, "—", "#F5F5F5"),
        (x2, y_r1, str(a), "#FFF3E0"),
        (x1, y_r2, str(b), "#E3F2FD"),
        (x2, y_r2, "—", "#F5F5F5"),
    ]
    cw, ch = 0.20, 0.10
    for cx, cy, txt, bg in cells:
        rect = mpatches.FancyBboxPatch(
            (cx - cw / 2, cy - ch / 2), cw, ch,
            boxstyle="round,pad=0.01", facecolor=bg, edgecolor="#AAAAAA", lw=0.4)
        ax.add_patch(rect)
        fs = 7 if txt.isdigit() else 6
        fw = "bold" if txt.isdigit() else "normal"
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=fs, fontweight=fw)

    # Result text
    ax.text(0.50, 0.28, f"McNemar χ² = {qw['statistic']:.1f}",
            ha="center", fontsize=6, fontweight="bold")
    ax.text(0.50, 0.20, f"p = {p:.1e}", ha="center", fontsize=5.5)
    ax.text(0.50, 0.12,
            f"Model correct when ICD wrong: {a}\n"
            f"Model wrong when ICD correct: {b}\n"
            f"Ratio: {a / b:.0f}:1",
            ha="center", va="top", fontsize=5, color="#444444")


# ── Panel 2e: Prevalence curve ───────────────────────────────────────────────
def draw_prevalence(ax):
    df = pd.read_csv(PANELS / "f2_predictive_value_curve.csv")
    qw = df[df["method"] == "Qwen3.6-27B"]
    ppv_q = qw[qw["metric"] == "PPV"].sort_values("prevalence")
    npv_q = qw[qw["metric"] == "NPV"].sort_values("prevalence")

    # ICD: compute from sens/spec
    icd_sens = 0.83984375
    icd_spec = 0.903954802259887
    prev = np.linspace(0.01, 0.50, 100)
    ppv_icd = icd_sens * prev / (icd_sens * prev + (1 - icd_spec) * (1 - prev))
    npv_icd = icd_spec * (1 - prev) / (icd_spec * (1 - prev) + (1 - icd_sens) * prev)

    # Plausible AUD prevalence band
    ax.axvspan(0.10, 0.25, color="#F0F0F0", alpha=0.8, zorder=0,
               label="Plausible AUD\nprevalence range")

    # Model curves
    ax.plot(ppv_q["prevalence"], ppv_q["value"], "-", color=WONG_BLUE, lw=1.2,
            label="LLM PPV")
    ax.plot(npv_q["prevalence"], npv_q["value"], "--", color=WONG_BLUE, lw=1.2,
            label="LLM NPV")

    # ICD curves
    ax.plot(prev, ppv_icd, "-", color=WONG_VERMILLION, lw=1.2, label="ICD PPV")
    ax.plot(prev, npv_icd, "--", color=WONG_VERMILLION, lw=1.2, label="ICD NPV")

    ax.set_xlabel("Prevalence", fontsize=6)
    ax.set_ylabel("Predictive value", fontsize=6)
    ax.set_xlim(0, 0.52)
    ax.set_ylim(0.5, 1.01)
    ax.legend(fontsize=4.5, loc="center right", frameon=False)

    ax.text(0.175, 0.52, "plausible\nrange", ha="center", fontsize=4,
            color="#888888", style="italic")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    figures = Path("downstream/npj_goal6_outputs/figures")
    figures.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.205, 9.0))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.85],
                          hspace=0.35, wspace=0.4)

    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2:])
    ax_c = fig.add_subplot(gs[1, :2])
    ax_d = fig.add_subplot(gs[1, 2:4])
    ax_e = fig.add_subplot(gs[1, 4:])

    panels = [ax_a, ax_b, ax_c, ax_d, ax_e]
    for i, ax in enumerate(panels):
        label = chr(ord("a") + i) + ")"
        ax.text(-0.08, 1.05, label, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top")

    draw_confusion(ax_a)
    draw_forest(ax_b)
    draw_alluvial(ax_c)
    draw_mcnemar(ax_d)
    draw_prevalence(ax_e)

    fig.text(0.50, 0.005,
             f"Performance against the reference standard {GOLD}.",
             ha="center", fontsize=5, color="#555555")

    stem = figures / "Fig2_headline"
    fig.savefig(str(stem) + ".png", dpi=300, facecolor="white")
    fig.savefig(str(stem) + ".svg", facecolor="white")
    plt.close(fig)

    gold_count = sum(1 for line in open(str(stem) + ".svg") if GOLD in line)
    print(f"saved={stem}.png,svg  gold_pending_grep={gold_count}")


if __name__ == "__main__":
    main()
