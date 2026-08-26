#!/usr/bin/env python3
"""Precision-gated npj equity screen against the pending reference standard.

Every observed stratum is retained.  Conclusions use the pre-specified rule in
``downstream/npj_parity_precision_protocol.md``; there is no minimum-n filter.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats_core as sc


GOLD_MARKER = "[[GOLD-PENDING]]"
ESTIMAND = f"performance against the reference standard {GOLD_MARKER}"

MODEL_METHODS = {
    "Qwen3.6-27B": "aud_qwen36_27b",
    "Nemotron-3-120B-FP8": "aud_nemotron_fp8",
    "GPT-OSS-120B": "aud",
    "GPT-OSS-20B": "aud_gptoss_20b",
}


def collapse3(value, cfg: dict) -> str:
    if value in set(cfg["aud_pos"]):
        return "+"
    if value in set(cfg["aud_neg"]):
        return "-"
    if value in set(cfg["abstain"]):
        return "I"
    return "missing"


def _stable_seed(*parts: object) -> int:
    text = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big") % (2**32)


def _components(frame: pd.DataFrame, pred_col: str, metric: str) -> list[tuple[int, int]]:
    ref = frame["reference_3"].to_numpy()
    pred = frame[pred_col].to_numpy()
    if metric == "conditional_sensitivity":
        mask = (ref == "+") & (pred != "I")
        return [(int((pred[mask] == "+").sum()), int(mask.sum()))]
    if metric == "conditional_specificity":
        mask = (ref == "-") & (pred != "I")
        return [(int((pred[mask] == "-").sum()), int(mask.sum()))]
    if metric == "binary_balanced_performance":
        pos = (ref == "+") & (pred != "I")
        neg = (ref == "-") & (pred != "I")
        return [
            (int((pred[pos] == "+").sum()), int(pos.sum())),
            (int((pred[neg] == "-").sum()), int(neg.sum())),
        ]
    if metric == "abstention_rate":
        mask = ref != "missing"
        return [(int((pred[mask] == "I").sum()), int(mask.sum()))]
    raise ValueError(metric)


def _component_point(components: list[tuple[int, int]]) -> float:
    if not components or any(n == 0 for _, n in components):
        return float("nan")
    return float(np.mean([k / n for k, n in components]))


def _component_interval(
    components: list[tuple[int, int]], *, alpha: float
) -> tuple[float, float]:
    if not components or any(n == 0 for _, n in components):
        return float("nan"), float("nan")
    intervals = [sc.wilson_ci(k, n, alpha=alpha) for k, n in components]
    return (
        float(np.mean([interval[0] for interval in intervals])),
        float(np.mean([interval[1] for interval in intervals])),
    )


def _component_bootstrap(
    components: list[tuple[int, int]], *, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    if not components or any(n == 0 for _, n in components):
        return np.full(n_boot, np.nan)
    draws = []
    for k, n in components:
        # Exact empirical patient bootstrap for a 0/1 rate.
        draws.append(rng.binomial(n, k / n, size=n_boot) / n)
    return np.mean(np.vstack(draws), axis=0)


def _quantile(values: np.ndarray, ci_level: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    alpha = 1.0 - ci_level
    return tuple(
        map(float, np.quantile(finite, [alpha / 2, 1 - alpha / 2]))
    )


def _contrast_row(
    subgroup: pd.DataFrame,
    complement: pd.DataFrame,
    *,
    method: str,
    pred_col: str,
    metric: str,
    axis: str,
    level: str,
    n_cohort: int,
    cfg: dict,
) -> dict:
    parity = cfg["parity"]
    n_boot = int(parity["n_boot"])
    rng = np.random.default_rng(_stable_seed(sc.SEED, method, metric, axis, level))

    sub_components = _components(subgroup, pred_col, metric)
    cmp_components = _components(complement, pred_col, metric)
    sub_point = _component_point(sub_components)
    cmp_point = _component_point(cmp_components)
    contrast = sub_point - cmp_point

    sub_boot = _component_bootstrap(sub_components, n_boot=n_boot, rng=rng)
    cmp_boot = _component_bootstrap(cmp_components, n_boot=n_boot, rng=rng)
    diff_boot = sub_boot - cmp_boot
    finite = np.isfinite(diff_boot)
    finite_fraction = float(finite.mean())

    boot_sub95 = _quantile(sub_boot, float(parity["ci_level"]))
    boot_diff95 = _quantile(diff_boot, float(parity["ci_level"]))
    boot_diff90 = _quantile(diff_boot, float(parity["equivalence_ci_level"]))
    score_sub95 = _component_interval(
        sub_components, alpha=1 - float(parity["ci_level"])
    )
    score_cmp95 = _component_interval(
        cmp_components, alpha=1 - float(parity["ci_level"])
    )
    score_sub90 = _component_interval(
        sub_components, alpha=1 - float(parity["equivalence_ci_level"])
    )
    score_cmp90 = _component_interval(
        cmp_components, alpha=1 - float(parity["equivalence_ci_level"])
    )

    score_diff95 = (
        score_sub95[0] - score_cmp95[1],
        score_sub95[1] - score_cmp95[0],
    )
    score_diff90 = (
        score_sub90[0] - score_cmp90[1],
        score_sub90[1] - score_cmp90[0],
    )
    primary95 = (
        min(boot_diff95[0], score_diff95[0]),
        max(boot_diff95[1], score_diff95[1]),
    )
    primary90 = (
        min(boot_diff90[0], score_diff90[0]),
        max(boot_diff90[1], score_diff90[1]),
    )

    estimable = bool(
        len(subgroup)
        and len(complement)
        and not any(n == 0 for _, n in sub_components + cmp_components)
        and finite_fraction >= 0.95
        and np.isfinite(primary95).all()
    )
    half_width = (
        float((primary95[1] - primary95[0]) / 2) if estimable else float("nan")
    )
    informative = bool(
        estimable
        and half_width <= float(parity["informative_ci_half_width"])
    )
    margin = float(parity["equivalence_margin_abs"])
    equivalent = bool(
        estimable and primary90[0] >= -margin and primary90[1] <= margin
    )

    finite_diff = diff_boot[finite]
    if len(finite_diff):
        lower = (np.count_nonzero(finite_diff <= 0) + 1) / (len(finite_diff) + 1)
        upper = (np.count_nonzero(finite_diff >= 0) + 1) / (len(finite_diff) + 1)
        p_value = float(min(1.0, 2 * min(lower, upper)))
    else:
        p_value = float("nan")

    ref = subgroup["reference_3"]
    pred = subgroup[pred_col]
    return {
        "method": method,
        "metric": metric,
        "axis": axis,
        "stratum": level,
        "n_cohort": int(n_cohort),
        "n_reference": int(len(subgroup)),
        "n_reference_complement": int(len(complement)),
        "reference_positive_n": int(ref.eq("+").sum()),
        "reference_negative_n": int(ref.eq("-").sum()),
        "reference_abstention_n": int(ref.eq("I").sum()),
        "prediction_abstention_n": int(pred.eq("I").sum()),
        "metric_denominator_components": ";".join(
            str(n) for _, n in sub_components
        ),
        "complement_denominator_components": ";".join(
            str(n) for _, n in cmp_components
        ),
        "stratum_metric": sub_point,
        "stratum_bootstrap_ci95_low": boot_sub95[0],
        "stratum_bootstrap_ci95_high": boot_sub95[1],
        "stratum_wilson_ci95_low": score_sub95[0],
        "stratum_wilson_ci95_high": score_sub95[1],
        "complement_metric": cmp_point,
        "delta_stratum_minus_complement": contrast,
        "delta_bootstrap_ci95_low": boot_diff95[0],
        "delta_bootstrap_ci95_high": boot_diff95[1],
        "delta_score_ci95_low": score_diff95[0],
        "delta_score_ci95_high": score_diff95[1],
        "delta_primary_ci95_low": primary95[0],
        "delta_primary_ci95_high": primary95[1],
        "delta_primary_ci95_half_width": half_width,
        "delta_primary_ci90_low": primary90[0],
        "delta_primary_ci90_high": primary90[1],
        "finite_bootstrap_fraction": finite_fraction,
        "bootstrap_p_value": p_value,
        "estimable": estimable,
        "precision_informative": informative,
        "equivalence_90ci_within_5pp": equivalent,
        "documentation_availability_contrast": level
        in {
            "not_documented",
            "unknown_not_separately_assessed",
            "not_available",
        },
        "rule_version": parity["rule_version"],
        "estimand_label": ESTIMAND,
    }


def _load_frame(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    labels = cfg["labels_6"]
    roster = pd.read_csv(
        cfg["paths"]["roster"], dtype={"omop_person_id": "string"}
    )
    if len(roster) != 499 or roster["omop_person_id"].nunique() != 499:
        raise RuntimeError("roster grain check failed")
    roster["icd_pred"] = np.where(
        roster["has_aud_icd"].astype(int).eq(1), "+", "-"
    )

    gold = pd.read_csv(
        cfg["paths"]["gold"], dtype={"omop_person_id": "string"}
    )
    # The roster is 499; the reference standard covers the 487 of them that fall
    # in the analysis cohort. The two counts are separate and the join is a left
    # join, so a rostered patient outside the cohort carries no reference label.
    if len(gold) != 487 or gold["omop_person_id"].nunique() != 487:
        raise RuntimeError("reference-standard grain check failed")
    gold["reference_label"] = sc.normalize_labels(
        gold[cfg["raters"]["gold"]["col"]], labels, strict=True
    )

    features = pd.read_parquet(cfg["paths"]["features"]).rename(
        columns={"person_id": "omop_person_id"}
    )
    features["omop_person_id"] = features["omop_person_id"].astype("string")
    sdoh = pd.read_parquet(cfg["paths"]["sdoh"]).rename(
        columns={"person_id": "omop_person_id"}
    )
    sdoh["omop_person_id"] = sdoh["omop_person_id"].astype("string")
    geo = pd.read_parquet(cfg["paths"]["geo"])
    geo["omop_person_id"] = geo["omop_person_id"].astype("string")

    panel = pd.read_csv(
        cfg["paths"]["canonical_panel"],
        usecols=["omop_person_id", "model", "label_canonical"],
        dtype={"omop_person_id": "string"},
    )
    wanted = set(MODEL_METHODS.values())
    selected = panel[panel["model"].isin(wanted)].copy()
    selected["label_canonical"] = sc.normalize_labels(
        selected["label_canonical"], labels, strict=True
    )
    counts = selected.groupby("model")["omop_person_id"].agg(["size", "nunique"])
    if set(counts.index) != wanted or not (counts.eq(499).all().all()):
        raise RuntimeError(f"canonical candidate coverage check failed:\n{counts}")
    wide = selected.pivot(
        index="omop_person_id", columns="model", values="label_canonical"
    ).reset_index()
    tag_to_col = {}
    for method, tag in MODEL_METHODS.items():
        col = f"pred_{tag}"
        wide[col] = wide[tag].map(lambda value: collapse3(value, cfg))
        tag_to_col[method] = col
    wide = wide[["omop_person_id", *tag_to_col.values()]]

    frame = (
        roster[["omop_person_id", "icd_pred"]]
        .merge(
            gold[["omop_person_id", "reference_label"]],
            on="omop_person_id",
            how="left",
            validate="one_to_one",
        )
        .merge(wide, on="omop_person_id", how="left", validate="one_to_one")
        .merge(features, on="omop_person_id", how="left", validate="one_to_one")
        .merge(sdoh, on="omop_person_id", how="left", validate="one_to_one")
        .merge(geo, on="omop_person_id", how="left", validate="one_to_one")
    )
    if len(frame) != 499:
        raise RuntimeError("joined analysis frame is not 499 rows")

    frame["reference_3"] = frame["reference_label"].map(
        lambda value: collapse3(value, cfg)
    )
    frame["age_bin"] = pd.cut(
        frame["age"],
        [-np.inf, 49, 64, np.inf],
        labels=["18-49", "50-64", "65+"],
    ).astype("string")
    frame["cci_bin"] = pd.cut(
        frame["CCI"],
        [-np.inf, 1, 3, np.inf],
        labels=["0-1", "2-3", "4+"],
    ).astype("string")
    frame["care_site_id"] = frame["care_site_id"].astype("Int64").astype("string")
    analysis = frame[frame["reference_label"].notna()].copy()
    if len(analysis) != 487:
        raise RuntimeError(f"reference-available N is {len(analysis)}, expected 487")
    tag_to_col["ICD"] = "icd_pred"
    return frame, analysis, tag_to_col


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-yaml", default="downstream/data.yaml")
    ap.add_argument("--out", default="downstream/npj_goal2_outputs")
    args = ap.parse_args()
    cfg = sc.load_yaml_with_scope(args.data_yaml, "ref_gold")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    full, analysis, pred_cols = _load_frame(cfg)
    axes = [
        "race",
        "gender",
        "ethnicity",
        "age_bin",
        "cci_bin",
        "care_site_id",
        *cfg["subgroups"]["sdoh"],
        *cfg["subgroups"]["population_characteristics"],
        *cfg["subgroups"]["geo"],
    ]
    axes = list(dict.fromkeys(axes))
    missing_axes = [axis for axis in axes if axis not in full.columns]
    if missing_axes:
        raise RuntimeError(f"missing subgroup axes: {missing_axes}")
    for axis in axes:
        full[axis] = full[axis].astype("string").fillna("not_available")
        analysis[axis] = analysis[axis].astype("string").fillna("not_available")

    rows = []
    metrics_by_method = {
        method: [
            "conditional_sensitivity",
            "conditional_specificity",
            "binary_balanced_performance",
            "abstention_rate",
        ]
        for method in MODEL_METHODS
    }
    metrics_by_method["ICD"] = [
        "conditional_sensitivity",
        "conditional_specificity",
        "binary_balanced_performance",
    ]
    for axis in axes:
        levels = sorted(analysis[axis].unique().tolist())
        for level in levels:
            # Counted on the analysis frame, not on the sampled frame. The estimates below
            # are computed on patients with a reference label, so a stratum size taken from
            # the larger frame would not be the denominator of anything reported.
            cohort_n = int(analysis[axis].eq(level).sum())
            sub = analysis[analysis[axis].eq(level)]
            complement = analysis[~analysis[axis].eq(level)]
            for method, metrics in metrics_by_method.items():
                for metric in metrics:
                    rows.append(
                        _contrast_row(
                            sub,
                            complement,
                            method=method,
                            pred_col=pred_cols[method],
                            metric=metric,
                            axis=axis,
                            level=level,
                            n_cohort=cohort_n,
                            cfg=cfg,
                        )
                    )
    table = pd.DataFrame(rows)
    table["q_value"] = np.nan
    for _, index in table.groupby(["method", "metric"], sort=False).groups.items():
        index = list(index)
        finite_index = [i for i in index if pd.notna(table.at[i, "bootstrap_p_value"])]
        if finite_index:
            table.loc[finite_index, "q_value"] = sc.bh_fdr(
                table.loc[finite_index, "bootstrap_p_value"].to_numpy()
            )

    conclusions = []
    for row in table.itertuples(index=False):
        if not row.estimable or not row.precision_informative:
            conclusion = "no_conclusion_imprecise"
        elif row.equivalence_90ci_within_5pp:
            conclusion = "equivalent"
        elif (
            (row.delta_primary_ci95_low > 0 or row.delta_primary_ci95_high < 0)
            and pd.notna(row.q_value)
            and row.q_value < 0.05
        ):
            conclusion = "disparity"
        else:
            conclusion = "no_conclusion"
        conclusions.append(conclusion)
    table["conclusion"] = conclusions
    table.to_csv(out / "parity_strata.csv", index=False)
    table[table["conclusion"].isin(["equivalent", "disparity"])].to_csv(
        out / "conclusive_strata.csv", index=False
    )

    summary = (
        table.groupby(["method", "metric", "axis", "conclusion"], dropna=False)
        .size()
        .rename("n_strata")
        .reset_index()
    )
    summary["estimand_label"] = ESTIMAND
    summary.to_csv(out / "parity_precision_summary.csv", index=False)

    # The q values were computed above and then discarded, so the multiplicity correction
    # existed in the code and in no output. The full table is written here.
    table.to_csv(out / "parity_table.csv", index=False)

    coverage_rows = []
    for axis in axes:
        for level, cohort_sub in analysis.groupby(axis, dropna=False):
            ref_sub = analysis[analysis[axis].eq(level)]
            coverage_rows.append(
                {
                    "axis": axis,
                    "stratum": level,
                    "n_cohort": int(len(cohort_sub)),
                    "n_reference_available": int(len(ref_sub)),
                    "reference_positive_n": int(ref_sub["reference_3"].eq("+").sum()),
                    "reference_negative_n": int(ref_sub["reference_3"].eq("-").sum()),
                    "reference_abstention_n": int(ref_sub["reference_3"].eq("I").sum()),
                    "documentation_availability_contrast": str(level)
                    in {
                        "not_documented",
                        "unknown_not_separately_assessed",
                        "not_available",
                    },
                    "estimand_label": ESTIMAND,
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(out / "axis_coverage.csv", index=False)

    methods = []
    for method, tag in MODEL_METHODS.items():
        methods.append(
            {
                "method": method,
                "canonical_model_tag": tag,
                "role": "primary-decision candidate; Haining has not selected a primary",
                "metrics": ";".join(metrics_by_method[method]),
                "estimand_label": ESTIMAND,
            }
        )
    methods.append(
        {
            "method": "ICD",
            "canonical_model_tag": "not_applicable",
            "role": "coded comparator",
            "metrics": ";".join(metrics_by_method["ICD"]),
            "estimand_label": ESTIMAND,
        }
    )
    pd.DataFrame(methods).to_csv(out / "method_manifest.csv", index=False)

    checks = {
        "rows": len(table),
        "axes": len(axes),
        "observed_axis_strata": len(coverage),
        "cohort_n": len(analysis),
        "reference_available_n": len(analysis),
        "all_rows_gold_pending": bool(table["estimand_label"].str.contains(
            r"\[\[GOLD-PENDING\]\]", regex=True
        ).all()),
        "old_minimum_n_gate_present": False,
        "employment_attempted": "labor_force_status" in axes,
        "transportation_barrier_attempted": "transportation_barrier" in axes,
        "ruca_attempted": "ruca_category" in axes,
        "adi_reported_not_available": bool(
            (table["axis"].eq("adi_quintile") & table["stratum"].eq("not_available")).any()
        ),
    }
    pd.DataFrame(
        [{"check": key, "value": value} for key, value in checks.items()]
    ).to_csv(out / "validation_checks.csv", index=False)
    if not all(
        [
            checks["all_rows_gold_pending"],
            checks["employment_attempted"],
            checks["transportation_barrier_attempted"],
            checks["ruca_attempted"],
            checks["adi_reported_not_available"],
        ]
    ):
        raise RuntimeError(f"parity validation failed: {checks}")

    print(f"analysis_cohort={len(analysis)} axes={len(axes)} rows={len(table)}")
    print("count_floor=absent all_observed_strata_reported=true")
    print(
        "methods=Qwen,Nemotron-FP8,GPT-OSS-120B,GPT-OSS-20B,ICD; "
        "primary_not_selected=true"
    )
    print(
        "employment_attempted=true transportation_barrier_attempted=true "
        "ruca_attempted=true adi_not_available_reported=true"
    )
    print(f"conclusions={table['conclusion'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
