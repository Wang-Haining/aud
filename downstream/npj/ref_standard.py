#!/usr/bin/env python3
"""The reference standard, derived in one place so every findings script agrees.

Three humans read charts. Maaz read all of them. Desai read a subset a second time. Gawrieh
read the charts where Maaz and Desai landed in different buckets, and his reading decides
those. The rule is therefore an escalation, and the label a patient carries depends on how
far up it went:

    Maaz only                            -> Maaz's label, unverified
    Maaz and Desai in the same bucket     -> Desai's label, verified
    Maaz and Desai in different buckets   -> Gawrieh's label, verified

The escalation can leave a chart in the last row with no adjudicating reading, and for a
period three charts did. Those rows take Desai's label, are marked provisional, and are
listed by review id in the manifest, so no result can rest on one without it being visible.
The rule is not neutral and is not presented as one: Desai's bucket was the modal
adjudication outcome on the charts already resolved at the time, which is what makes it the
least assuming stand-in rather than a correct one.

The rule holds no rows at present. Every discordant chart now carries an adjudicating
reading, `provisional_review_ids` in the manifest is empty, and `--strict` passes. Nothing
needs editing if it recurs: an adjudicating rating for a discordant chart takes precedence
over the stand-in by construction, so dropping the file into chart/ratings/ and rerunning
moves those rows from provisional to adjudicated.

Import `build()` from a findings script, or run this file to write the CSV and the manifest.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, "downstream")
import stats_core as sc  # noqa: E402

GOLD_VERSION = "v2_2026-08-11"
OUT_CSV = "resources/gold_reference_v2.csv"
OUT_MANIFEST = "resources/gold_manifest_v2.json"

# The analysis cohort is the 487 patients carrying an identifier from the earlier data era,
# which is the cohort the group's AASLD abstract describes. Keeping the two papers on the
# same patients removes an inconsistency a reader would otherwise have to resolve, and it
# costs nothing measurable: the twelve patients it drops include the only three with no
# chart reading at all, and the model classifies the other nine without error.
COHORT = "era_a"

L6 = ["Definitely Yes", "Probably Yes", "Uncertain", "Probably No", "Definitely No",
      "Not Enough Data"]
POS = {"Definitely Yes", "Probably Yes"}
NEG = {"Definitely No", "Probably No"}

# v1 wrote "abstain" in gold_binary. Scripts already read that vocabulary, so it is kept
# unchanged and the manuscript's wording is carried in a separate column.
BINARY_INDET = "abstain"
THREE = {"AUD+": "AUD+", "AUD-": "AUD-", BINARY_INDET: "Indeterminate"}

SRC_MAAZ = "maaz_unreviewed"
SRC_DESAI = "desai_concordant_bucket"
SRC_GAWRIEH = "gawrieh_adjudicated"
SRC_PROVISIONAL = "provisional_desai_pending_gawrieh"
SRC_MISSING = "missing_human_reference"


def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bucket(label) -> str | None:
    """Collapse a six-category label to the three buckets the escalation rule compares."""
    if label is None or pd.isna(label):
        return None
    if label in POS:
        return "AUD+"
    if label in NEG:
        return "AUD-"
    return BINARY_INDET


def _load_json_ratings(path, who: str) -> pd.DataFrame:
    raw = json.load(open(path))
    rows = [{"review_id": rid, "omop_person_id": str(rec["omop_person_id"]),
             f"{who}_label": rec["classification"],
             f"{who}_comment": rec.get("comment", "")}
            for rid, rec in raw.items()]
    out = pd.DataFrame(rows)
    out[f"{who}_label"] = sc.normalize_labels(out[f"{who}_label"], L6, strict=False)
    if out[f"{who}_label"].isna().any():
        bad = out.loc[out[f"{who}_label"].isna(), "review_id"].tolist()
        raise SystemExit("unmappable %s labels on %s" % (who, bad))
    return out


def build(roster_path="resources/patient_roster.csv",
          chart_roster="chart/data/roster.csv",
          desai_path="chart/ratings/desai.json",
          gawrieh_path="chart/ratings/gawrieh.json",
          cohort=COHORT) -> tuple[pd.DataFrame, dict]:
    """Return the reference standard and a manifest describing how it was derived.

    `cohort` is "era_a" for the 487 patients who carry an identifier from the earlier data
    era, which is the cohort the group's AASLD abstract describes, or "all" for the full
    499. Restricting here rather than in each findings script is deliberate: the filter
    cannot then be applied in one place and forgotten in another.
    """
    ros = pd.read_csv(roster_path, dtype={"omop_person_id": str})
    n_roster = len(ros)
    if cohort not in ("era_a", "all"):
        raise SystemExit("unknown cohort %r; use 'era_a' or 'all'" % cohort)
    # Membership is decided once, here, and travels as a column. Asking for the whole
    # roster therefore still tells a caller which patients are in the analysis sample,
    # so no other file has to re-derive the definition and risk disagreeing with it.
    ros["in_analysis_cohort"] = ros["gpids_era_a"].notna()
    if cohort == "era_a":
        ros = ros[ros["in_analysis_cohort"]].copy()
    g = ros[["omop_person_id", "in_analysis_cohort", "ndepth_maaz_rating"]].copy()
    g["maaz_label"] = sc.normalize_labels(g.pop("ndepth_maaz_rating"), L6, strict=False)

    link = pd.read_csv(chart_roster, dtype=str)[["review_id", "omop_person_id"]]
    if link["review_id"].duplicated().any() or link["omop_person_id"].duplicated().any():
        raise SystemExit("the chart roster does not map review IDs to patients one to one")

    des = _load_json_ratings(desai_path, "desai")
    gaw = _load_json_ratings(gawrieh_path, "gawrieh")
    for frame, who in ((des, "desai"), (gaw, "gawrieh")):
        unknown = set(frame["review_id"]) - set(link["review_id"])
        if unknown:
            raise SystemExit("%s rated review IDs absent from the chart roster: %s"
                             % (who, sorted(unknown)))

    g = (g.merge(link, on="omop_person_id", how="left")
          .merge(des.drop(columns=["omop_person_id"]), on="review_id", how="left")
          .merge(gaw.drop(columns=["omop_person_id"]), on="review_id", how="left"))
    if len(g) != len(ros):
        raise SystemExit("the merge changed the roster size, %d to %d" % (len(ros), len(g)))

    for who in ("maaz", "desai", "gawrieh"):
        g[f"{who}_bucket"] = g[f"{who}_label"].map(bucket)

    read_twice = g["desai_label"].notna()
    concordant = read_twice & (g["maaz_bucket"] == g["desai_bucket"])
    discordant = read_twice & (g["maaz_bucket"] != g["desai_bucket"])
    has_third = g["gawrieh_label"].notna()

    # Gawrieh should only ever have read a discordant chart. If he has rated a concordant
    # one, the escalation rule was not what produced his worklist and the rule below would
    # silently discard a senior reading.
    stray = g.loc[has_third & ~discordant, "review_id"].dropna().tolist()
    if stray:
        raise SystemExit("Gawrieh rated charts that are not Maaz-Desai discordant: %s"
                         % sorted(stray))

    g["gold_label"] = g["maaz_label"]
    g["gold_source"] = SRC_MAAZ
    g["gold_verified"] = False
    g["gold_provisional"] = False

    g.loc[concordant, "gold_label"] = g.loc[concordant, "desai_label"]
    g.loc[concordant, "gold_source"] = SRC_DESAI
    g.loc[concordant, "gold_verified"] = True

    settled = discordant & has_third
    g.loc[settled, "gold_label"] = g.loc[settled, "gawrieh_label"]
    g.loc[settled, "gold_source"] = SRC_GAWRIEH
    g.loc[settled, "gold_verified"] = True

    pending = discordant & ~has_third
    g.loc[pending, "gold_label"] = g.loc[pending, "desai_label"]
    g.loc[pending, "gold_source"] = SRC_PROVISIONAL
    g.loc[pending, "gold_verified"] = False
    g.loc[pending, "gold_provisional"] = True

    g.loc[g["gold_label"].isna(), "gold_source"] = SRC_MISSING

    g["gold_binary"] = g["gold_label"].map(bucket)
    g["gold_three"] = g["gold_binary"].map(THREE)
    g["gold_version"] = GOLD_VERSION

    manifest = {
        "gold_version": GOLD_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cohort": {
            "name": cohort,
            "definition": "carries an identifier from the earlier data era, the cohort the "
                          "group's AASLD abstract describes" if cohort == "era_a"
                          else "every sampled patient",
            "roster_rows": int(n_roster),
            "analysed": int(len(ros)),
            "dropped": int(n_roster - len(ros)),
        },
        "inputs": {
            "patient_roster": {"path": roster_path, "sha256": sha256(roster_path),
                               "rows": int(n_roster)},
            "desai": {"path": desai_path, "sha256": sha256(desai_path),
                      "ratings": int(len(des))},
            "gawrieh": {"path": gawrieh_path, "sha256": sha256(gawrieh_path),
                        "ratings": int(len(gaw))},
        },
        "counts": {
            "patients": int(len(g)),
            "read_by_maaz": int(g["maaz_label"].notna().sum()),
            "read_twice": int(read_twice.sum()),
            "concordant_bucket": int(concordant.sum()),
            "discordant_bucket": int(discordant.sum()),
            "third_reading_present": int(settled.sum()),
            "third_reading_outstanding": int(pending.sum()),
        },
        "by_source": {k: int(v) for k, v in g["gold_source"].value_counts().items()},
        "by_reference": {str(k): int(v) for k, v in
                         g["gold_three"].value_counts(dropna=False).items()},
        "provisional_review_ids": sorted(g.loc[pending, "review_id"].dropna().tolist()),
        # Emitted whether or not any row is provisional, so the rule that would apply is
        # on the record either way. provisional_review_ids above is the field that says
        # whether it applied to this build; it is empty when nothing is provisional.
        "stand_in_rule": {
            "rule": "Desai's label stands in until Gawrieh rates a discordant chart",
            "basis": "Desai's bucket was the modal adjudication outcome on the charts "
                     "resolved at the time the rule was written",
            "reporting": "no result may rest on a provisional row without saying so; the "
                         "rows are listed in provisional_review_ids",
        },
    }
    return g, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--manifest", default=OUT_MANIFEST)
    ap.add_argument("--cohort", default=COHORT, choices=["era_a", "all"])
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any row is still provisional; use for the "
                         "final submission run")
    a = ap.parse_args()

    g, man = build(cohort=a.cohort)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    g.to_csv(a.out, index=False)
    json.dump(man, open(a.manifest, "w"), indent=2)

    c, co = man["counts"], man["cohort"]
    print("reference standard %s, cohort %s" % (man["gold_version"], co["name"]))
    print("  %s" % co["definition"])
    print("  roster %d, analysed %d, dropped %d"
          % (co["roster_rows"], co["analysed"], co["dropped"]))
    print("  patients %d, read by Maaz %d, read a second time %d"
          % (c["patients"], c["read_by_maaz"], c["read_twice"]))
    print("  same bucket %d, different bucket %d"
          % (c["concordant_bucket"], c["discordant_bucket"]))
    print("  third reading present %d, outstanding %d"
          % (c["third_reading_present"], c["third_reading_outstanding"]))
    print()
    for k, v in sorted(man["by_source"].items(), key=lambda kv: -kv[1]):
        print("  %-38s %4d" % (k, v))
    print()
    for k, v in man["by_reference"].items():
        print("  reference %-14s %4d" % (k, v))
    print()
    print("wrote %s and %s" % (a.out, a.manifest))

    n = c["third_reading_outstanding"]
    if n:
        print()
        print("!" * 74)
        print("%d row(s) are PROVISIONAL: %s" % (n, ", ".join(man["provisional_review_ids"])))
        print("They carry Desai's label because Gawrieh has not read them yet.")
        print("Every downstream table built from this file inherits that. Rerun once")
        print("chart/ratings/gawrieh.json contains those charts; no code changes.")
        print("!" * 74)
        if a.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
