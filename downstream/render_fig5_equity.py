#!/usr/bin/env python3
"""Render Figure 5: Equitable performance, unequal coverage.

Panels:
  5a  Conditional-sensitivity parity ratios (4 MedACE candidates)
  5b  Parity in the rate of indeterminate output (4 MedACE candidates)
  5c  Mechanism: indeterminate-rate delta against documentation availability

Data sources (pre-computed):
  downstream/npj_goal2_outputs/parity_strata.csv
  downstream/npj_goal2_outputs/parity_precision_summary.csv
  downstream/npj_goal6_outputs/panels/f3_documentation_coverage.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Nature Portfolio rcParams
# ---------------------------------------------------------------------------
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
        "legend.title_fontsize": 7,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 0.8,
        "lines.markersize": 3,
        "legend.frameon": False,
        "legend.borderpad": 0.3,
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GOLD = ""   # the reference standard is complete; nothing pending

MODELS = [
    "Qwen3.6-27B",
    "Nemotron-3-120B-FP8",
    "GPT-OSS-120B",
    "GPT-OSS-20B",
]
MODEL_MARKERS = {
    "Qwen3.6-27B": "o",
    "Nemotron-3-120B-FP8": "s",
    "GPT-OSS-120B": "D",
    "GPT-OSS-20B": "^",
}
MODEL_SHORT = {
    "Qwen3.6-27B": "Qwen-27B",
    "Nemotron-3-120B-FP8": "Nemotron-120B",
    "GPT-OSS-120B": "GPT-120B",
    "GPT-OSS-20B": "GPT-20B",
}

CONCLUSION_COLORS = {
    "equivalent": "#009E73",
    "disparity": "#D55E00",
    "no_conclusion": "#404040",
    "no_conclusion_imprecise": "#BBBBBB",
}
CONCLUSION_ALPHA = {
    "equivalent": 1.0,
    "disparity": 1.0,
    "no_conclusion": 0.65,
    "no_conclusion_imprecise": 0.35,
}

# A stratifying variable is reported only when the record carries it for at least this
# share of the cohort. Below it, a stratum is mostly an artefact of who was asked, and
# the comparison is between a small selected group and everybody else. The fields that
# do not clear the bar are kept for reference in the supplement, not reported here.
# This drops food security (8.3%), insurance (11.4%), transportation barrier (24.0%),
# incarceration history (5.0%), and disability, which is not separately assessed at all
# but derived from the employment field (6.6% positive). It also drops the area
# deprivation quintile, which is not available for a single patient in the cohort.
MIN_DOCUMENTED = 0.30
REPORTED_AXES = {"gender", "age_bin", "race", "ethnicity", "cci_bin", "housing",
                 "labor_force_status", "ruca_category"}

# Axes grouped conceptually; strata ordered for readability
AXIS_GROUPS_ALL = [
    (
        "Demographics",
        [
            ("gender", ["Man", "Woman"]),
            ("age_bin", ["18-49", "50-64", "65+"]),
            ("race", ["Black", "White", "Others"]),
            ("ethnicity", ["Hispanic", "Others"]),
        ],
    ),
    (
        "Clinical",
        [
            ("cci_bin", ["0-1", "2-3", "4+"]),
        ],
    ),
    (
        "Social determinants",
        [
            ("housing", ["stable", "unstable", "institutional", "homeless", "not_documented"]),
            (
                "labor_force_status",
                ["employed", "unemployed", "not_in_labor_force", "not_documented"],
            ),
            ("food_security", ["secure", "insecure", "not_documented"]),
            (
                "insurance",
                [
                    "private",
                    "medicare",
                    "medicaid",
                    "dual_medicaid_medicare",
                    "other",
                    "uninsured",
                    "not_documented",
                ],
            ),
            (
                "disability_status",
                [
                    "documented_disabled_in_employment_field",
                    "unknown_not_separately_assessed",
                ],
            ),
            ("transportation_barrier", ["yes", "no", "not_documented"]),
            ("incarceration_history", ["yes", "no", "not_documented"]),
        ],
    ),
    (
        "Geographic",
        [
            ("ruca_category", ["urban", "suburban", "rural", "not_available"]),
        ],
    ),
]

AXIS_GROUPS = [(g, [(a, s) for a, s in axes if a in REPORTED_AXES])
               for g, axes in AXIS_GROUPS_ALL]
AXIS_GROUPS = [(g, axes) for g, axes in AXIS_GROUPS if axes]

AXIS_LABELS = {
    "gender": "Gender",
    "age_bin": "Age",
    "race": "Race",
    "ethnicity": "Ethnicity",
    "cci_bin": "Charlson comorbidity",
    "housing": "Housing",
    "labor_force_status": "Labor force status",
    "food_security": "Food security",
    "insurance": "Insurance",
    "disability_status": "Disability",
    "transportation_barrier": "Transport barrier",
    "incarceration_history": "Incarceration history",
    "ruca_category": "Urban–rural (RUCA)",
    "adi_quintile": "ADI quintile",
}

STRATUM_DISPLAY = {
    "Man": "Man",
    "Woman": "Woman",
    "18-49": "18–49",
    "50-64": "50–64",
    "65+": "65+",
    "Black": "Black",
    "White": "White",
    "Others": "Others",
    "Hispanic": "Hispanic",
    "0-1": "0–1",
    "2-3": "2–3",
    "4+": "4+",
    "stable": "Stable",
    "unstable": "Unstable",
    "institutional": "Institutional",
    "homeless": "Homeless",
    "not_documented": "Not documented",
    "employed": "Employed",
    "unemployed": "Unemployed",
    "not_in_labor_force": "Not in labor force",
    "secure": "Secure",
    "insecure": "Insecure",
    "private": "Private",
    "medicare": "Medicare",
    "medicaid": "Medicaid",
    "dual_medicaid_medicare": "Dual Medicaid/Medicare",
    "other": "Other",
    "uninsured": "Uninsured",
    "documented_disabled_in_employment_field": "Documented disabled",
    "unknown_not_separately_assessed": "Unknown / not assessed",
    "yes": "Yes",
    "no": "No",
    "urban": "Urban",
    "suburban": "Suburban",
    "rural": "Rural",
    "not_available": "Not available",
}

# Which "not documented" stratum to use per axis for the mechanism scatter
NOT_DOC_STRATUM = {
    "labor_force_status": "not_documented",
    "disability_status": "unknown_not_separately_assessed",
    "housing": "not_documented",
    "food_security": "not_documented",
    "insurance": "not_documented",
    "transportation_barrier": "not_documented",
    "incarceration_history": "not_documented",
}


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def build_forest_layout():
    """Build y-positions for strata and header rows.

    Returns rows, headers, total_height.
    headers: list of dict(y, label, level='group'|'axis').
    """
    rows = []
    headers = []
    y = 0.0

    for gi, (group_name, axes) in enumerate(AXIS_GROUPS):
        if gi > 0:
            y += 0.6
        headers.append({"y": y, "label": group_name, "level": "group"})
        y += 0.85

        for ai, (axis_name, strata) in enumerate(axes):
            if ai > 0:
                y += 0.35
            headers.append({"y": y, "label": AXIS_LABELS[axis_name], "level": "axis"})
            y += 0.7

            for stratum in strata:
                rows.append(
                    {
                        "y": y,
                        "axis": axis_name,
                        "stratum": stratum,
                        "label": STRATUM_DISPLAY.get(stratum, stratum),
                    }
                )
                y += 1.0

    return rows, headers, y


def draw_forest_panel(
    ax,
    parity: pd.DataFrame,
    metric: str,
    rows: list[dict],
    methods: list[str],
    y_max: float,
    *,
    show_all_models: bool = True,
    title: str = "",
):
    """Draw one forest panel with model dots and CIs."""
    sub = parity[(parity["metric"] == metric) & (parity["axis"] != "care_site_id")]

    ax.axvspan(-0.05, 0.05, color="#009E73", alpha=0.07, zorder=0)
    ax.axvline(0, color="#888888", linewidth=0.4, zorder=1)

    offsets = np.linspace(-0.28, 0.28, len(methods)) if len(methods) > 1 else [0.0]

    for row in rows:
        for mi, method in enumerate(methods):
            match = sub[
                (sub["method"] == method)
                & (sub["axis"] == row["axis"])
                & (sub["stratum"] == row["stratum"])
            ]
            if match.empty:
                continue
            r = match.iloc[0]
            yy = row["y"] + offsets[mi]
            delta = r["delta_stratum_minus_complement"]
            ci_lo = r["delta_primary_ci95_low"]
            ci_hi = r["delta_primary_ci95_high"]
            conclusion = r["conclusion"]

            color = CONCLUSION_COLORS.get(conclusion, "#BBBBBB")
            alpha = CONCLUSION_ALPHA.get(conclusion, 0.35)
            marker = MODEL_MARKERS.get(method, "o") if show_all_models else "o"
            ms = 3.0 if conclusion not in ("no_conclusion_imprecise",) else 2.0
            lw_ci = 0.5 if conclusion not in ("no_conclusion_imprecise",) else 0.3

            if not np.isnan(delta):
                ax.plot(
                    [ci_lo, ci_hi],
                    [yy, yy],
                    color=color,
                    linewidth=lw_ci,
                    alpha=alpha,
                    zorder=2,
                    solid_capstyle="round",
                )
                ax.scatter(
                    delta,
                    yy,
                    marker=marker,
                    s=ms**2,
                    color=color,
                    alpha=alpha,
                    zorder=3,
                    linewidths=0.3,
                    edgecolors="white" if alpha > 0.5 else "none",
                )

    ax.set_ylim(-0.8, y_max + 0.3)
    ax.invert_yaxis()
    ax.set_xlim(-0.35, 0.35)
    ax.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    ax.set_xticklabels(["−.3", "", "−.1", "0", ".1", "", ".3"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    if title:
        ax.set_xlabel("Δ stratum − complement", fontsize=5.5)
        ax.set_title(title, loc="left", fontsize=7)


def draw_labels(ax, rows, headers, parity, y_max):
    """Draw stratum labels, n, and header rows."""
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, y_max + 0.3)
    ax.invert_yaxis()
    ax.axis("off")

    ref = parity[
        (parity["method"] == MODELS[0])
        & (parity["metric"] == "conditional_sensitivity")
        & (parity["axis"] != "care_site_id")
    ]

    for h in headers:
        if h["level"] == "group":
            ax.text(
                0.02, h["y"], h["label"],
                fontsize=6, fontweight="bold", color="#222222", va="center",
            )
        else:
            ax.text(
                0.08, h["y"], h["label"],
                fontsize=5.5, fontweight="bold", color="#555555", va="center",
            )

    for row in rows:
        match = ref[(ref["axis"] == row["axis"]) & (ref["stratum"] == row["stratum"])]
        n = int(match.iloc[0]["n_cohort"]) if not match.empty else "?"
        ax.text(0.16, row["y"], row["label"], fontsize=5.5, va="center")
        ax.text(0.97, row["y"], f"n={n}", fontsize=5, va="center", ha="right", color="#666666")


def draw_mechanism(ax, parity, doc_df):
    """Panel 5c: indeterminate-rate delta against documentation availability."""
    # The primary model takes the index colour and the other three the Tier 2 foreground
    # grey of npj_style, because the panel has no legend and its four former hues could
    # not be decoded. Marker shape separates the models. Corrected 24 August; the hues it
    # replaces were #0072B2 for the primary, which is the diagnosis code's colour, plus
    # #009E73 and #CC79A7, both of which the palette review removed.
    C_INDEX, C_FORE = "#D55E00", "#404040"
    model_colors = {m: (C_INDEX if m == "Qwen3.6-27B" else C_FORE) for m in MODELS}
    doc_rates = doc_df.set_index("variable")["documentation_or_positive_rate"]

    xs_all, ys_all = [], []
    for axis_name, stratum in NOT_DOC_STRATUM.items():
        if axis_name not in doc_rates.index:
            continue
        dr = doc_rates[axis_name]
        for method in MODELS:
            match = parity[
                (parity["method"] == method)
                & (parity["metric"] == "abstention_rate")
                & (parity["axis"] == axis_name)
                & (parity["stratum"] == stratum)
            ]
            if match.empty:
                continue
            r = match.iloc[0]
            delta = r["delta_stratum_minus_complement"]
            ci_lo = r["delta_primary_ci95_low"]
            ci_hi = r["delta_primary_ci95_high"]
            ax.scatter(
                dr, delta,
                marker=MODEL_MARKERS[method],
                s=12,
                color=model_colors[method],
                alpha=0.8,
                zorder=3,
                linewidths=0.3,
                edgecolors="white",
            )
            ax.plot(
                [dr, dr],
                [ci_lo, ci_hi],
                color=model_colors[method],
                linewidth=0.4,
                alpha=0.5,
                zorder=2,
            )
            xs_all.append(dr)
            ys_all.append(delta)

    label_cfg = {
        "housing": {"off": (8, -10), "arrow": False},
        "labor_force_status": {"off": (8, -10), "arrow": False},
        "food_security": {"off": (8, 12), "arrow": True},
        "insurance": {"off": (-8, -16), "arrow": True},
        "disability_status": {"off": (8, 6), "arrow": True},
        "transportation_barrier": {"off": (8, -10), "arrow": False},
        "incarceration_history": {"off": (-8, -16), "arrow": True},
    }
    label_map = {
        "labor_force_status": "Labor force",
        "disability_status": "Disability",
        "housing": "Housing",
        "food_security": "Food security",
        "insurance": "Insurance",
        "transportation_barrier": "Transport",
        "incarceration_history": "Incarceration",
    }
    arrow_kw = dict(arrowstyle="-", color="#B9B7B2", linewidth=0.4)
    placed = set()
    for axis_name in NOT_DOC_STRATUM:
        if axis_name not in doc_rates.index or axis_name in placed:
            continue
        dr = doc_rates[axis_name]
        mean_delta = parity[
            (parity["method"].isin(MODELS))
            & (parity["metric"] == "abstention_rate")
            & (parity["axis"] == axis_name)
            & (parity["stratum"] == NOT_DOC_STRATUM[axis_name])
        ]["delta_stratum_minus_complement"].mean()
        lbl = label_map.get(axis_name, axis_name)
        cfg = label_cfg.get(axis_name, {"off": (5, 3), "arrow": False})
        kw = dict(
            textcoords="offset points",
            xytext=cfg["off"],
            fontsize=5,
            color="#55534E",
        )
        if cfg["arrow"]:
            kw["arrowprops"] = arrow_kw
        ax.annotate(lbl, (dr, mean_delta), **kw)
        placed.add(axis_name)

    ax.axhline(0, color="#B9B7B2", linewidth=0.4, zorder=0)
    ax.set_xlabel("Documentation availability (fraction of cohort)")
    ax.set_ylabel("Indeterminate rate, Δ (not documented − complement)")
    ax.set_xlim(-0.02, 0.50)
    ax.set_ylim(-0.06, 0.28)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata", default="downstream/npj_goal2_outputs/parity_strata.csv")
    ap.add_argument(
        "--precision", default="downstream/npj_goal2_outputs/parity_precision_summary.csv"
    )
    ap.add_argument(
        "--docs", default="downstream/npj_goal6_outputs/panels/f3_documentation_coverage.csv"
    )
    ap.add_argument("--figures", default="downstream/npj_goal6_outputs/figures")
    args = ap.parse_args()

    parity = pd.read_csv(args.strata)
    precision = pd.read_csv(args.precision)
    doc_df = pd.read_csv(args.docs)
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)

    # ---- restrict to the reported axes and verify the denominator ----
    # These counts go into the Results text and the caption, so they are asserted here.
    # If the axis rule changes, this fails rather than letting the figure and the text
    # drift apart.
    precision = precision[precision["axis"].isin(REPORTED_AXES)]
    parity = parity[parity["axis"].isin(REPORTED_AXES)]
    prec_conclusion = precision["conclusion"].copy()
    prec_conclusion = prec_conclusion.where(
        ~prec_conclusion.str.startswith("no_conclusion"), "no_conclusion"
    )
    counts = prec_conclusion.value_counts()
    n_eq = int(counts.get("equivalent", 0))
    n_disp = int(counts.get("disparity", 0))
    n_nc = int(counts.get("no_conclusion", 0))
    n_total = len(precision)
    n_axes = precision["axis"].nunique()
    n_strata = parity.groupby("axis")["stratum"].nunique().sum()
    # Raw row counts of the precision summary on the reported axes. They are a guard on
    # the inputs, not the counts the manuscript reports: the file carries 107 rows for the
    # 64 evaluations of the four candidate models, so a recount for the text has to
    # deduplicate on (method, metric, axis) first. Values below are on the closed
    # 487-chart standard, after run_parity.py was regated on 14 August.
    assert n_total == 226, f"Expected 226 rows, got {n_total}"
    assert n_eq == 14, f"Expected 14 equivalent, got {n_eq}"
    assert n_disp == 10, f"Expected 10 disparity, got {n_disp}"
    assert n_nc == 202, f"Expected 202 no conclusion, got {n_nc}"
    assert n_axes == 8, f"Expected 8 axes, got {n_axes}"
    assert n_strata == 26, f"Expected 26 strata, got {n_strata}"

    # ---- build layout ----
    rows, headers, y_max = build_forest_layout()

    # ---- figure ----
    # Twenty-six strata rather than forty-five, so the panel is shorter. The mechanism
    # panel that used to sit below is built as its own supplementary figure, because it
    # is drawn mostly from fields the axis rule keeps for reference rather than reports.
    fig = plt.figure(figsize=(7.205, 5.60))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[8.4, 0.7],
        width_ratios=[1.5, 1.0, 1.0],
        hspace=0.14,
        wspace=0.08,
    )

    ax_lab = fig.add_subplot(gs[0, 0])
    ax_5a = fig.add_subplot(gs[0, 1])
    ax_5b = fig.add_subplot(gs[0, 2])

    ax_lg = fig.add_subplot(gs[1, 1:3])

    # ---- panel labels ----
    ax_5a.text(-0.08, 1.02, "a)", transform=ax_5a.transAxes, fontsize=8,
               fontweight="bold", va="top")
    ax_5b.text(-0.08, 1.02, "b)", transform=ax_5b.transAxes, fontsize=8,
               fontweight="bold", va="top")

    # ---- labels ----
    draw_labels(ax_lab, rows, headers, parity, y_max)

    # ---- 5a: conditional sensitivity ----
    draw_forest_panel(
        ax_5a,
        parity,
        "conditional_sensitivity",
        rows,
        MODELS,
        y_max,
        title="Conditional sensitivity",
    )

    # ---- 5b: rate of indeterminate output ----
    draw_forest_panel(
        ax_5b,
        parity,
        "abstention_rate",
        rows,
        MODELS,
        y_max,
        title="Indeterminate rate",
    )

    # ---- legend ----
    ax_lg.axis("off")
    handles = []
    for method in MODELS:
        handles.append(
            Line2D(
                [],
                [],
                marker=MODEL_MARKERS[method],
                color="grey",
                markersize=3.5,
                linestyle="none",
                label=MODEL_SHORT[method],
            )
        )
    handles.append(Line2D([], [], color="white", linewidth=0))
    handles.append(
        Line2D(
            [],
            [],
            marker="s",
            color=CONCLUSION_COLORS["equivalent"],
            markersize=4,
            linestyle="none",
            markeredgecolor="white",
            markeredgewidth=0.3,
            label="Equivalent",
        )
    )
    handles.append(
        Line2D(
            [],
            [],
            marker="s",
            color=CONCLUSION_COLORS["disparity"],
            markersize=4,
            linestyle="none",
            markeredgecolor="white",
            markeredgewidth=0.3,
            label="Disparity",
        )
    )
    handles.append(
        Line2D(
            [],
            [],
            marker="s",
            color=CONCLUSION_COLORS["no_conclusion"],
            markersize=4,
            linestyle="none",
            label="No conclusion",
        )
    )
    handles.append(
        Line2D(
            [],
            [],
            marker="s",
            color=CONCLUSION_COLORS["no_conclusion_imprecise"],
            markersize=4,
            linestyle="none",
            label="Imprecise",
        )
    )
    # Laid out along the row rather than down it. With 26 strata the figure is short,
    # and a stacked legend runs into the annotation below it.
    handles = [h for h in handles if h.get_label() not in ("", "_nolegend_")]
    ax_lg.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        fontsize=5,
        handletextpad=0.4,
        labelspacing=0.5,
        columnspacing=1.1,
        borderpad=0.5,
        title="Models & conclusions",
        title_fontsize=5.5,
    )

    # ---- denominator annotation ----
    fig.text(
        0.03,
        0.012,
        (
            f"Of {n_total} method × metric × axis evaluations across "
            f"4 models: {n_eq} equivalent, {n_disp} disparity, {n_nc} no conclusion.\n"
            f"Performance against the reference standard {GOLD}.  "
            "Equivalence band (green shading): 90% CI within ±0.05.\n"
            "Registry-scale sensitivity of the AUD diagnosis code by patient subgroup "
            "is shown in c."
        ),
        fontsize=5,
        color="#555555",
        va="bottom",
    )

    # ---- export ----
    stem = figures / "Fig5_equity_coverage"
    fig.savefig(str(stem) + ".png", dpi=300, facecolor="white")
    fig.savefig(str(stem) + ".svg", facecolor="white")
    plt.close(fig)

    # ---- supplementary: the mechanism panel, on all seven fields ----
    # This is drawn from the unfiltered data on purpose. Five of the seven fields do not
    # clear the reporting bar, so the panel cannot support a claim in the Results; the
    # relation it shows is between documentation availability and indeterminate output,
    # given here so the reader can see it rather than take the rule on trust.
    fig2 = plt.figure(figsize=(4.9, 3.1))
    ax2 = fig2.add_subplot(111)
    draw_mechanism(ax2, pd.read_csv(args.strata), doc_df)
    fig2.tight_layout()
    stem2 = figures / "FigS2_documentation_mechanism"
    fig2.savefig(str(stem2) + ".png", dpi=300, facecolor="white")
    fig2.savefig(str(stem2) + ".pdf", facecolor="white")
    plt.close(fig2)
    print("saved=%s.png,pdf" % stem2)

    gold_count = sum(1 for _ in open(str(stem) + ".svg") if GOLD in _)
    print(
        f"saved={stem}.png,svg  "
        f"gold_pending_grep={gold_count}  "
        f"equivalent={n_eq}  disparity={n_disp}  no_conclusion={n_nc}  "
        f"total={n_total}"
    )


if __name__ == "__main__":
    main()
