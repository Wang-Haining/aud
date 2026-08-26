#!/usr/bin/env python3
"""Quote-grounding and evidence-path material for the primary-decision set.

Patient-level match details stay under ignored ``output/`` with mode 600.  The
tracked outputs contain only aggregate counts and rates.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


MODEL_METHODS = {
    "Qwen3.6-27B": "aud_qwen36_27b",
    "Nemotron-3-120B-FP8": "aud_nemotron_fp8",
    "GPT-OSS-120B": "aud",
    "GPT-OSS-20B": "aud_gptoss_20b",
}
LABEL = "descriptive grounding/provenance; primary model not selected"


def parse_list(value) -> list[str]:
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (TypeError, ValueError):
        pass
    return [item.strip() for item in str(value).split(";") if item.strip()]


def normalize(text) -> str:
    text = html.unescape(str(text or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def light_fuzzy(quote: str, normalized_notes: str, threshold: float = 0.90) -> float:
    q = normalize(quote)
    if not q:
        return 0.0
    q_tokens = q.split()
    note_tokens = normalized_notes.split()
    anchor = max(q_tokens, key=len)
    q_positions = [i for i, token in enumerate(q_tokens) if token == anchor]
    note_positions = [i for i, token in enumerate(note_tokens) if token == anchor]
    best = 0.0
    width = max(1, len(q_tokens))
    for pos in note_positions:
        for qpos in q_positions:
            start = max(0, pos - qpos)
            for candidate_width in range(max(1, width - 2), width + 3):
                candidate = " ".join(note_tokens[start : start + candidate_width])
                best = max(best, SequenceMatcher(None, q, candidate).ratio())
                if best >= threshold:
                    return best
    return best


def load_patient_notes(mapping_path: str, notes_dirs: list[str]) -> dict[str, str]:
    mapping = pd.read_csv(
        mapping_path, dtype={"OMOP_PERSON_ID": "string", "REPORT_ID": "string"}
    )
    roots = [Path(path) for path in notes_dirs]
    cache: dict[str, str] = {}
    for report_id in mapping["REPORT_ID"].dropna().drop_duplicates():
        cache[report_id] = ""
        for root in roots:
            path = root / f"{report_id}.txt"
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    cache[report_id] = text
                break
    notes = {}
    for person_id, sub in mapping.groupby("OMOP_PERSON_ID", sort=False):
        parts = [
            cache[report_id]
            for report_id in sub["REPORT_ID"].dropna().drop_duplicates()
            if cache.get(report_id)
        ]
        notes[str(person_id)] = normalize("\n".join(parts))
    return notes


def canon_item(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value).strip().casefold())
    aliases = {
        "assessment & plan": "assessment/plan",
        "active problems / assessment & plan": "assessment/plan",
        "hpi": "hpi narrative",
    }
    return aliases.get(value, value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="output/canonical_panel.csv")
    ap.add_argument("--mapping", default="resources/mapped_report_ids.csv")
    ap.add_argument(
        "--notes-dir",
        nargs="+",
        default=[
            "/N/project/depot/hw56/aud/cirrhosis_regv20/redacted_txt",
            "/N/project/depot/hw56/aud/cirrhosis_regv20/redacted_pdf-text",
        ],
    )
    ap.add_argument("--out", default="downstream/npj_goal6_outputs/panels")
    ap.add_argument(
        "--detail-output", default="output/npj_goal6/grounding_detail.csv"
    )
    ap.add_argument(
        "--qwen-spotcheck",
        default=(
            "transfer_staging/branch_outputs/downstream/"
            "grounding_spotcheck_assessments.csv"
        ),
    )
    args = ap.parse_args()

    panel = pd.read_csv(args.panel, dtype={"omop_person_id": "string"})
    tag_to_name = {tag: name for name, tag in MODEL_METHODS.items()}
    work = panel[panel["model"].isin(tag_to_name)].copy()
    work["method"] = work["model"].map(tag_to_name)
    counts = work.groupby("method")["omop_person_id"].agg(["size", "nunique"])
    if set(counts.index) != set(MODEL_METHODS) or not counts.eq(499).all().all():
        raise RuntimeError(f"candidate grounding coverage failed:\n{counts}")

    notes = load_patient_notes(args.mapping, args.notes_dir)
    if len(notes) != 499:
        raise RuntimeError(f"note-map patient count is {len(notes)}, expected 499")

    details = []
    path_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    distribution_rows = []
    for row in work.itertuples(index=False):
        person_id = str(row.omop_person_id)
        quote = "" if pd.isna(row.supporting_quote) else str(row.supporting_quote)
        note_text = notes.get(person_id, "")
        qnorm = normalize(quote)
        exact = bool(qnorm and qnorm in note_text)
        similarity = 1.0 if exact else light_fuzzy(quote, note_text)
        fuzzy = bool(exact or similarity >= 0.90)
        details.append(
            {
                "omop_person_id": person_id,
                "method": row.method,
                "label_canonical": row.label_canonical,
                "supporting_quote": quote,
                "exact_grounded": exact,
                "fuzzy_grounded_090": fuzzy,
                "similarity": similarity,
                "match_method": (
                    "normalized_exact"
                    if exact
                    else "light_fuzzy_090" if fuzzy else "ungrounded"
                ),
            }
        )
        triggers = sorted({canon_item(x) for x in parse_list(row.triggers_present)})
        evidence = sorted({canon_item(x) for x in parse_list(row.evidence_categories)})
        triggers = triggers or ["(no standalone trigger)"]
        evidence = evidence or ["(no evidence category)"]
        for trigger in triggers:
            for category in evidence:
                path_counts[(row.method, "trigger_to_evidence")][
                    (trigger, category)
                ] += 1
        for category in evidence:
            path_counts[(row.method, "evidence_to_class")][
                (category, row.label_canonical)
            ] += 1
        for dimension, items in (("trigger", triggers), ("evidence_category", evidence)):
            for item in items:
                distribution_rows.append(
                    {
                        "method": row.method,
                        "dimension": dimension,
                        "item": item,
                    }
                )

    detail = pd.DataFrame(details)
    detail_path = Path(args.detail_output)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False)
    os.chmod(detail_path, 0o600)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rates = (
        detail.groupby("method", sort=False)
        .agg(
            n=("fuzzy_grounded_090", "size"),
            exact_n=("exact_grounded", "sum"),
            exact_rate=("exact_grounded", "mean"),
            fuzzy_n=("fuzzy_grounded_090", "sum"),
            fuzzy_rate=("fuzzy_grounded_090", "mean"),
        )
        .reset_index()
    )
    rates["ungrounded_n"] = rates["n"] - rates["fuzzy_n"]
    rates["primary_model_selected"] = False
    rates["estimand_label"] = LABEL
    rates.to_csv(out / "f5_grounding_rates.csv", index=False)

    distributions = (
        pd.DataFrame(distribution_rows)
        .groupby(["method", "dimension", "item"])
        .size()
        .rename("n")
        .reset_index()
    )
    distributions["pct_patients"] = distributions["n"] / 499
    distributions["estimand_label"] = LABEL
    distributions.to_csv(out / "f5_evidence_distributions.csv", index=False)

    paths = []
    for (method, layer), counter in path_counts.items():
        for (source, target), n in counter.items():
            paths.append(
                {
                    "method": method,
                    "layer": layer,
                    "source": source,
                    "target": target,
                    "n": n,
                    "estimand_label": LABEL,
                }
            )
    pd.DataFrame(paths).to_csv(out / "f5_evidence_paths.csv", index=False)

    summary_rows = []
    for row in rates.itertuples(index=False):
        summary_rows.append(
            {
                "method": row.method,
                "category": "fuzzy_below_0.90",
                "n": int(row.ungrounded_n),
                "denominator": int(row.n),
                "scope": "all candidate outputs",
                "estimand_label": LABEL,
            }
        )
    spot_path = Path(args.qwen_spotcheck)
    if spot_path.exists():
        spot = pd.read_csv(spot_path)
        for category, n in spot["manual_assessment"].value_counts().items():
            summary_rows.append(
                {
                    "method": "Qwen3.6-27B",
                    "category": str(category),
                    "n": int(n),
                    "denominator": int(len(spot)),
                    "scope": "manual sample of 12 Qwen matcher misses; not all misses",
                    "estimand_label": LABEL,
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        out / "f5_ungrounded_summary.csv", index=False
    )

    print("candidate_methods=4 rows=1996 patient_detail_mode=600")
    for row in rates.itertuples(index=False):
        print(
            f"{row.method}: exact={row.exact_n}/{row.n} "
            f"fuzzy090={row.fuzzy_n}/{row.n} ungrounded={row.ungrounded_n}"
        )


if __name__ == "__main__":
    main()
