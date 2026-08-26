#!/usr/bin/env python3
"""Analyze the repaired five-condition Goal 5 sensitivity experiment.

Patient-level inputs and the extended rater table remain under ``output/`` and
must never enter git.  Only identifier-free aggregate tables and the report are
written to ``downstream/npj_goal5_outputs``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import downstream.stats_core as sc


# The run roster and the reference standard are two different counts. Every model
# scored all 499 rostered patients; the reference standard covers the 487 that fall
# in the analysis cohort. Conditions are compared on the 487.
N_ANALYSIS = 487
N_FULL = 499
N_BOOT = 2000
GOLD_TAG = "[[GOLD-PENDING]]"

CONDITIONS = [
    ("v1_t0p0", "v1 prompt, T=0.0", "canonical"),
    ("v1_t0p3", "v1 prompt, T=0.3", "variant"),
    ("v1_t0p7", "v1 prompt, T=0.7", "variant"),
    ("paraphrase_a_t0p0", "paraphrase A, T=0.0", "variant"),
    ("paraphrase_b_t0p0", "paraphrase B, T=0.0", "variant"),
]



# --- primary agreement definition ------------------------------------------
# Three categories with not enough data excluded, matching every other agreement
# statistic in the paper. Operates on object arrays because this file bootstraps with a
# precomputed index matrix; the pair is dropped inside each resample, so the interval
# covers the whole procedure rather than a fixed subset.
ORD3_PRIMARY = ["AUD-", "Uncertain", "AUD+"]
_POS3 = ("Definitely Yes", "Probably Yes")
_NEG3 = ("Definitely No", "Probably No")


def _c3(v):
    v = np.asarray(v, dtype=object)
    out = np.full(v.shape, None, dtype=object)
    out[np.isin(v, _POS3)] = "AUD+"
    out[np.isin(v, _NEG3)] = "AUD-"
    out[v == "Uncertain"] = "Uncertain"
    return out


def kappa3(a, b):
    a3, b3 = _c3(a), _c3(b)
    keep = (a3 != None) & (b3 != None)          # noqa: E711
    if keep.sum() < 2:
        return float("nan")
    return sc.cohens_kappa(a3[keep], b3[keep], ORD3_PRIMARY, None)


def pairs3(a, b):
    a3, b3 = _c3(a), _c3(b)
    return int(((a3 != None) & (b3 != None)).sum())   # noqa: E711
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--out", type=Path, default=Path("downstream/npj_goal5_outputs")
    )
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_condition_labels(root: Path, labels: list[str]) -> dict[str, pd.DataFrame]:
    panel = pd.read_csv(root / "output/canonical_panel.csv", dtype={"omop_person_id": str})
    qwen = panel.loc[
        panel["model"].eq("aud_qwen36_27b"),
        ["omop_person_id", "label_v1", "label_canonical"],
    ].copy()
    if len(qwen) != N_FULL or qwen["omop_person_id"].nunique() != N_FULL:
        raise RuntimeError("frozen Qwen baseline does not contain the full roster")
    if not qwen["label_v1"].equals(qwen["label_canonical"]):
        raise RuntimeError("Qwen canonical labels no longer equal frozen v1")
    qwen["label"] = sc.normalize_labels(qwen["label_v1"], labels, strict=True)
    conditions = {"v1_t0p0": qwen[["omop_person_id", "label"]]}

    for condition, _, source in CONDITIONS:
        if source == "canonical":
            continue
        patient_root = root / "output/npj_goal5" / f"qwen36_{condition}" / "per_patient"
        files = sorted(patient_root.glob("*.json"))
        if len(files) != N_FULL:
            raise RuntimeError(f"{condition}: patient JSON count is incomplete")
        rows = []
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "ok" or not payload.get("extraction"):
                raise RuntimeError(f"{condition}: residual non-result in {path.name}")
            rows.append(
                {
                    "omop_person_id": str(payload["omop_person_id"]),
                    "label": payload["extraction"]["classification"],
                }
            )
        frame = pd.DataFrame(rows)
        if frame["omop_person_id"].nunique() != N_FULL:
            raise RuntimeError(f"{condition}: duplicate patient identifier")
        frame["label"] = sc.normalize_labels(frame["label"], labels, strict=True)
        conditions[condition] = frame

    expected_ids = set(conditions["v1_t0p0"]["omop_person_id"])
    for condition, frame in conditions.items():
        if set(frame["omop_person_id"]) != expected_ids:
            raise RuntimeError(f"{condition}: roster differs from frozen baseline")
    return conditions


def load_gold(root: Path, labels: list[str]) -> pd.DataFrame:
    gold = pd.read_csv(
        root / "resources/gold_reference_v2.csv",
        dtype={"omop_person_id": str},
        keep_default_na=False,
    )
    if len(gold) != N_ANALYSIS or gold["omop_person_id"].nunique() != N_ANALYSIS:
        raise RuntimeError("gold file does not contain the analysis cohort")
    analysis = gold.loc[gold["gold_label"].str.strip().ne(""),
                        ["omop_person_id", "gold_label"]].copy()
    if len(analysis) != N_ANALYSIS:
        raise RuntimeError(f"reference-standard analysis set is {len(analysis)}, not {N_ANALYSIS}")
    analysis["gold_label"] = sc.normalize_labels(
        analysis["gold_label"], labels, strict=True
    )
    return analysis


def three_class(label: str, cfg: dict) -> str:
    if label in set(cfg["aud_pos"]):
        return "AUD+"
    if label in set(cfg["aud_neg"]):
        return "AUD-"
    if label in set(cfg["abstain"]):
        return "Indeterminate"
    raise ValueError(f"unmapped label: {label}")


def atomic_extend_rater_long(
    root: Path, conditions: dict[str, pd.DataFrame], cfg: dict
) -> tuple[int, Path]:
    path = root / "output/raters/rater_long.csv"
    current = pd.read_csv(path, dtype={"omop_person_id": str})
    # The check is that every column this function uses is present. It was an equality
    # test, which broke when the reference-standard rebuild added in_analysis_cohort to
    # rater_long; an added column is not a reason to refuse to run.
    required = {"omop_person_id", "rater", "rater_type", "label_6", "label_3", "source", "stage"}
    if not required.issubset(current.columns):
        raise RuntimeError(f"rater_long is missing {sorted(required - set(current.columns))}")
    if current.duplicated(["omop_person_id", "rater"]).any():
        raise RuntimeError("input rater_long has duplicate patient-rater cells")

    goal5_raters = {f"goal5_qwen_{condition}" for condition, _, _ in CONDITIONS}
    current = current.loc[~current["rater"].isin(goal5_raters)].copy()
    appended = []
    display = {condition: label for condition, label, _ in CONDITIONS}
    for condition, frame in conditions.items():
        source = (
            "output/canonical_panel.csv"
            if condition == "v1_t0p0"
            else f"output/npj_goal5/qwen36_{condition}/per_patient"
        )
        stage = (
            "Goal 5 frozen canonical baseline"
            if condition == "v1_t0p0"
            else "Goal 5 native-context sensitivity; complete repaired coverage"
        )
        part = frame.copy()
        part["rater"] = f"goal5_qwen_{condition}"
        part["rater_type"] = "model"
        part["label_6"] = part["label"]
        part["label_3"] = part["label"].map(lambda value: three_class(value, cfg))
        part["source"] = source
        part["stage"] = stage
        # rater_long gained in_analysis_cohort when the reference standard was rebuilt.
        # It is a property of the patient, not of the rater, so it is carried over from
        # the rows already present rather than recomputed here.
        for extra in [c for c in current.columns if c not in part.columns]:
            m = (current.drop_duplicates("omop_person_id")
                 .set_index("omop_person_id")[extra])
            part[extra] = part["omop_person_id"].map(m)
            if part[extra].isna().any():
                raise RuntimeError(f"{extra} is missing for some patients")
        appended.append(part[list(current.columns)])
        if len(part) != N_FULL:
            raise RuntimeError(f"{display[condition]} rater rows are incomplete")

    combined = pd.concat([current, *appended], ignore_index=True)
    if combined.duplicated(["omop_person_id", "rater"]).any():
        raise RuntimeError("extended rater_long has duplicate patient-rater cells")
    if sum(combined["rater"].isin(goal5_raters)) != len(CONDITIONS) * N_FULL:
        raise RuntimeError("extended rater_long Goal 5 grain failed")

    archive = path.parent / "archive" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(path, archive / path.name)
    tmp = path.with_suffix(".csv.goal5.tmp")
    combined.to_csv(tmp, index=False)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return len(combined), archive


def repair_summary(root: Path) -> pd.DataFrame:
    path = root / "output/npj_goal5/repair_provenance.csv"
    repair = pd.read_csv(path, dtype={"omop_person_id": str})
    if len(repair) != 13 or repair.duplicated(["condition", "omop_person_id"]).any():
        raise RuntimeError("repair provenance is not the expected 13 unique cells")
    summary = (
        repair.groupby(["condition", "repair_temperature", "top_p"], as_index=False)
        .size()
        .rename(columns={"size": "charts_repaired"})
    )
    return summary


def validate_manuscript_outputs(out: Path) -> None:
    for path in sorted(out.glob("*")):
        if not path.is_file() or path.suffix not in {".csv", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"(?<![\d.])499(?!\d)", text):
            raise RuntimeError(f"manuscript-bound output exposes the full-run count: {path}")
        if "n_chunks" in text or "mean chunks" in text.lower():
            raise RuntimeError(f"prohibited chunking content in {path}")


def finalize_template_manifest(out: Path) -> None:
    """Mark the preregistered prompt design complete without full-roster counts."""
    path = out / "template_manifest.csv"
    manifest = pd.read_csv(path, keep_default_na=False)
    if manifest["template_id"].tolist() != ["v1", "paraphrase_a", "paraphrase_b"]:
        raise RuntimeError("unexpected Goal 5 template manifest")
    manifest = manifest.drop(columns=["bytes"], errors="ignore")
    manifest["execution_status"] = "completed"
    manifest.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = (root / args.out).resolve() if not args.out.is_absolute() else args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = sc.load_yaml_with_scope(str(root / "downstream/data.yaml"))
    labels = list(cfg["labels_6"])
    conditions = load_condition_labels(root, labels)
    gold = load_gold(root, labels)

    analysis = gold.rename(columns={"gold_label": "gold"}).copy()
    for condition, _, _ in CONDITIONS:
        analysis = analysis.merge(
            conditions[condition].rename(columns={"label": condition}),
            on="omop_person_id",
            how="left",
            validate="one_to_one",
        )
    if len(analysis) != N_ANALYSIS or analysis.isna().any().any():
        raise RuntimeError("the common reference-standard analysis set is incomplete")

    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, N_ANALYSIS, size=(args.n_boot, N_ANALYSIS))
    gold_values = analysis["gold"].to_numpy(dtype=object)
    baseline_values = analysis["v1_t0p0"].to_numpy(dtype=object)
    baseline_point = kappa3(gold_values, baseline_values)
    baseline_boot = np.asarray(
        [
            kappa3(gold_values[idx], baseline_values[idx])
            for idx in indices
        ]
    )

    condition_rows = []
    bucket_rows = []
    pairwise_rows = []
    baseline_bucket = np.asarray([three_class(x, cfg) for x in baseline_values], dtype=object)
    for order, (condition, display, _) in enumerate(CONDITIONS, start=1):
        values = analysis[condition].to_numpy(dtype=object)
        point = kappa3(gold_values, values)
        boot = np.asarray(
            [
                kappa3(gold_values[idx], values[idx])
                for idx in indices
            ]
        )
        delta_boot = boot - baseline_boot
        ci_low, ci_high = np.nanpercentile(boot, [2.5, 97.5])
        delta = point - baseline_point
        delta_low, delta_high = np.nanpercentile(delta_boot, [2.5, 97.5])
        bucket = np.asarray([three_class(x, cfg) for x in values], dtype=object)
        abstentions = int(np.isin(values, cfg["abstain"]).sum())
        # Under the primary agreement definition the two non-committal grades part
        # company: Not Enough Data leaves the kappa denominator, Uncertain is scored.
        n_ned = int((values == "Not Enough Data").sum())
        n_unc = int((values == "Uncertain").sum())
        bucket_changes = int((bucket != baseline_bucket).sum())
        condition_rows.append(
            {
                "order": order,
                "condition": condition,
                "display": display,
                "n": pairs3(gold_values, values),
                "n_labelled": N_ANALYSIS,
                "kappa_w": point,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "paired_difference_from_baseline": delta,
                "paired_ci_low": delta_low,
                "paired_ci_high": delta_high,
                "bucket_changes_from_baseline": bucket_changes,
                "abstentions": abstentions,
                "abstention_rate": abstentions / N_ANALYSIS,
                "not_enough_data": n_ned,
                "uncertain": n_unc,
                "estimand": "performance against the reference standard",
                "status": GOLD_TAG,
            }
        )
        counts = Counter(zip(baseline_bucket, bucket))
        for source_bucket in ["AUD+", "AUD-", "Indeterminate"]:
            for target_bucket in ["AUD+", "AUD-", "Indeterminate"]:
                bucket_rows.append(
                    {
                        "condition": condition,
                        "display": display,
                        "from_bucket": source_bucket,
                        "to_bucket": target_bucket,
                        "count": counts[(source_bucket, target_bucket)],
                    }
                )

    for condition_a, display_a, _ in CONDITIONS:
        for condition_b, display_b, _ in CONDITIONS:
            kappa = sc.cohens_kappa(
                analysis[condition_a].to_numpy(dtype=object),
                analysis[condition_b].to_numpy(dtype=object),
                labels,
                "linear",
            )
            pairwise_rows.append(
                {
                    "condition_a": condition_a,
                    "display_a": display_a,
                    "condition_b": condition_b,
                    "display_b": display_b,
                    "n": N_ANALYSIS,
                    "kappa_w": kappa,
                }
            )

    condition_table = pd.DataFrame(condition_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    bucket = pd.DataFrame(bucket_rows)
    repair = repair_summary(root)

    condition_table.to_csv(out / "condition_reference_performance.csv", index=False)
    pairwise.to_csv(out / "pairwise_kappa.csv", index=False)
    bucket.to_csv(out / "bucket_stability.csv", index=False)
    repair.to_csv(out / "repair_summary.csv", index=False)

    matrix = pairwise.pivot(index="display_a", columns="display_b", values="kappa_w")
    order = [display for _, display, _ in CONDITIONS]
    matrix = matrix.reindex(index=order, columns=order)
    matrix.to_csv(out / "pairwise_kappa_matrix.csv")
    pd.DataFrame(N_ANALYSIS, index=order, columns=order).to_csv(
        out / "pairwise_n_matrix.csv"
    )

    # Figure 4a must use the same reference-standard cohort as the rest of the
    # manuscript-bound figure.  Recompute the 11-model matrix on those patients
    # rather than carrying forward the registry-wide matrix.
    canonical = pd.read_csv(
        root / "output/canonical_panel.csv", dtype={"omop_person_id": str}
    )
    canonical = canonical[canonical["omop_person_id"].isin(set(gold["omop_person_id"]))]
    model_wide = canonical.pivot(
        index="omop_person_id", columns="model", values="label_canonical"
    )
    if model_wide.shape != (N_ANALYSIS, 11) or model_wide.isna().any().any():
        raise RuntimeError("11-model reference-standard matrix is incomplete")
    # Display names and column order come from downstream/data.yaml, the registry of
    # models. They came from downstream/npj_goal1_outputs until 14 August 2026. That
    # directory is a product of the reference standard as it stood on 9 August, before
    # the standard closed; see downstream/_archive/2026-08-14/README.md.
    model_to_display = {spec["model"]: name for name, spec in cfg["models"].items()}
    if set(model_wide.columns) != set(model_to_display):
        raise RuntimeError("data.yaml model names do not match canonical panel")
    model_order = list(model_to_display)
    model_matrix = pd.DataFrame(index=model_order, columns=model_order, dtype=float)
    for model_a in model_order:
        for model_b in model_order:
            model_matrix.loc[model_a, model_b] = sc.cohens_kappa(
                model_wide[model_a], model_wide[model_b], labels, "linear"
            )
    model_matrix = model_matrix.rename(index=model_to_display, columns=model_to_display)
    model_matrix.index.name = "rater_a"
    model_matrix.to_csv(out / "model_pairwise_kappa_496.csv")

    rater_rows, rater_archive = atomic_extend_rater_long(root, conditions, cfg)
    largest = condition_table.iloc[
        condition_table["paired_difference_from_baseline"].abs().argmax()
    ]
    nonbaseline_changes = condition_table.loc[
        condition_table["condition"].ne("v1_t0p0"), "bucket_changes_from_baseline"
    ]
    report = f"""# Goal 5 prompt and temperature sensitivity

All five conditions had complete schema-valid coverage before analysis. The
reported cohort is the {N_ANALYSIS} patients with a reference-standard label;
all reference-standard results remain {GOLD_TAG} pending final estimand wording.

The largest paired change from the frozen baseline was {largest['display']}:
{largest['paired_difference_from_baseline']:+.4f} (95% paired bootstrap interval
{largest['paired_ci_low']:+.4f} to {largest['paired_ci_high']:+.4f}). Across the
four sensitivity conditions, three-category bucket changes ranged from
{int(nonbaseline_changes.min())} to {int(nonbaseline_changes.max())} patients.
Repair restored complete coverage before any statistic was computed; the
patient-level repair ledger and attempt counts remain in the protected analysis
tree and are not manuscript outputs.

The frozen baseline used the July configuration, whereas the four sensitivity
conditions used the native-context corrected configuration. These contrasts are
therefore descriptive sensitivity analyses with configuration confounding, not
a clean factorial estimate of prompt or temperature effects.
"""
    (out / "goal5_report.md").write_text(report, encoding="utf-8")
    finalize_template_manifest(out)
    validate_manuscript_outputs(out)

    print(
        f"GOAL5_ANALYSIS_OK conditions=5 analysis_n={N_ANALYSIS} "
        f"bootstrap={args.n_boot} rater_rows={rater_rows}"
    )
    print(f"RATER_LONG_ARCHIVE={rater_archive}")
    print(
        "LARGEST_PAIRED_DIFFERENCE "
        f"condition={largest['condition']} value={largest['paired_difference_from_baseline']:.6f} "
        f"ci=[{largest['paired_ci_low']:.6f},{largest['paired_ci_high']:.6f}]"
    )
    print(
        "GOLD_PENDING_COUNT="
        + str(sum(path.read_text(encoding="utf-8").count(GOLD_TAG)
                  for path in out.glob("*.*") if path.is_file()))
    )


if __name__ == "__main__":
    main()
