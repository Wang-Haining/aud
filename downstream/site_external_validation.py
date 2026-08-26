#!/usr/bin/env python3
"""External validation of the instrument across the three contributing hospital systems.

Design. The reviewed patients come from three health systems that contribute notes to the
same regional exchange. Each system has its own clinicians, its own documentation culture
and its own note templates. Performance is reported at each system separately:

  IUH        the academic health system, the largest contributor
  Eskenazi   the safety-net system, external validation site 1
  Other      the remaining contributing hospitals, external validation site 2

Institution comes from the data provider's own report on the delivered notes
(resources/institution_report.xlsx), not from an OMOP care_site inference. A patient is
assigned to the system that wrote most of that patient's delivered notes.

Metrics. Linear-weighted kappa against the reference standard on the six-level scale, with
a nonparametric bootstrap interval. Weighted kappa depends on case mix, so the table also
reports sensitivity and specificity against the binary reference, which do not, and the
prevalence of reference-confirmed AUD at each site. The diagnosis code is scored against
the same reference at the same sites.

Handling of the indeterminate category. Patients whose reference label is indeterminate
have no truth to score against and are dropped from sensitivity and specificity. An
indeterminate instrument output on a patient with a determinate reference label is kept and
counted as a failure to identify, which is the conservative reading for a case-finding
tool. The rate at which the instrument returns an indeterminate output is reported
separately for each site, because that rate is itself a transportability property. Weighted
kappa uses all six levels and all patients with an available reference label.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "downstream")
import stats_core as sc  # noqa: E402

OUT = "downstream/npj_goal7_outputs"
SITES = ["IUH", "Eskenazi", "Other"]
SITE_ROLE = {
    "IUH": "primary site",
    "Eskenazi": "external validation site 1",
    "Other": "external validation site 2",
}
POS = "AUD+"
NEG = "AUD-"
YES = "Yes-side"


def load(args):
    cfg = sc.load_yaml_with_scope(args.data_yaml, "ref_gold")
    labels = list(cfg["labels_6"])
    display = {spec["model"]: name for name, spec in cfg["models"].items()}
    panel = pd.read_csv(args.panel, dtype={"omop_person_id": str})
    ref = pd.read_csv(args.reference, dtype={"omop_person_id": str})
    inst = pd.read_csv(args.institution, dtype={"person_id": str})
    inst = inst.rename(columns={"person_id": "omop_person_id"})
    ref["gold_label"] = sc.normalize_labels(ref["gold_label"], labels, strict=True)
    panel["label_canonical"] = sc.normalize_labels(panel["label_canonical"], labels,
                                                   strict=True)
    ref = ref.merge(inst[["omop_person_id", "institution"]], on="omop_person_id", how="left")
    if ref["institution"].isna().any():
        raise RuntimeError("%d reviewed patients have no institution"
                           % int(ref["institution"].isna().sum()))
    return labels, display, panel, ref


def binmetrics(y_true, y_pred):
    """Sensitivity, specificity and their Wilson intervals, on the binary reference."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int((y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    out = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    for nm, k, n in (("sens", tp, tp + fn), ("spec", tn, tn + fp),
                     ("ppv", tp, tp + fp), ("npv", tn, tn + fn)):
        if n == 0:
            out[nm] = out[nm + "_lo"] = out[nm + "_hi"] = np.nan
            continue
        lo, hi = sc.wilson_ci(k, n)
        out[nm], out[nm + "_lo"], out[nm + "_hi"] = 100 * k / n, 100 * lo, 100 * hi
    return out


def kappa_boot(y_true, y_pred, labels, n_boot, rng):
    """Bootstrap draws of linear-weighted kappa. Returned so sites can be differenced."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            draws[b] = sc.cohens_kappa(y_true[idx], y_pred[idx], labels, weights="linear")
        except Exception:
            draws[b] = np.nan
    return draws


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-yaml", default="downstream/data.yaml")
    ap.add_argument("--panel", default="output/canonical_panel.csv")
    ap.add_argument("--reference", default="resources/gold_reference_v2.csv")
    ap.add_argument("--institution", default="output/fairness/patient_institution.csv")
    ap.add_argument("--primary", default="aud_qwen36_27b")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(20260807)

    labels, display, panel, ref = load(args)
    print("=" * 78)
    print("SITE COMPOSITION OF THE REVIEWED SAMPLE")
    print("=" * 78)
    av = ref[ref["gold_label"].notna()]
    print("  reviewed patients %d ; reference available %d" % (len(ref), len(av)))
    for s in SITES:
        g = av[av["institution"] == s]
        d = g[g["gold_binary"].isin([POS, NEG])]
        prev = 100 * (d["gold_binary"] == POS).mean() if len(d) else np.nan
        print("  %-9s %-28s n = %3d  determinate %3d  reference AUD prevalence %5.1f%%"
              % (s, SITE_ROLE[s], len(g), len(d), prev))

    # ---------------------------------------------------------------- per model, per site
    print("\n" + "=" * 78)
    print("WEIGHTED KAPPA AGAINST THE REFERENCE STANDARD, BY SITE")
    print("=" * 78)
    rows, draws_store = [], {}
    for model, one in panel.groupby("model", sort=False):
        one = one.merge(ref[["omop_person_id", "gold_label", "gold_binary", "institution"]],
                        on="omop_person_id", how="left", validate="one_to_one")
        one = one[one["gold_label"].notna() & one["label_canonical"].notna()]
        for s in SITES + ["ALL"]:
            g = one if s == "ALL" else one[one["institution"] == s]
            if len(g) < 20:
                continue
            k = sc.cohens_kappa(g["gold_label"], g["label_canonical"], labels,
                                weights="linear")
            d = kappa_boot(g["gold_label"].to_numpy(), g["label_canonical"].to_numpy(),
                           labels, args.n_boot, rng)
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            draws_store[(model, s)] = d
            rows.append({"model": model, "display_name": display.get(model, model),
                         "site": s, "site_role": SITE_ROLE.get(s, "all sites pooled"),
                         "n": len(g), "kappa_w": k, "ci_low": lo, "ci_high": hi,
                         "exact_agreement": float(
                             g["gold_label"].eq(g["label_canonical"]).mean())})
    km = pd.DataFrame(rows)
    wide = km.pivot(index="display_name", columns="site", values="kappa_w")
    wide = wide.reindex(columns=["ALL"] + SITES).sort_values("ALL", ascending=False)
    print(wide.round(3).to_string())

    print("\n  rank stability across sites, Spearman correlation with the pooled ranking")
    base = wide["ALL"].rank(ascending=False)
    for s in SITES:
        ok = wide[s].notna()
        r = base[ok].corr(wide.loc[ok, s].rank(ascending=False), method="spearman")
        print("    %-9s rho = %.3f  (%d models)" % (s, r, int(ok.sum())))

    # ------------------------------------------------------------- primary model detail
    print("\n" + "=" * 78)
    print("PRIMARY MODEL %s : SITE-BY-SITE PERFORMANCE" % display.get(args.primary))
    print("=" * 78)
    pm = panel[panel["model"] == args.primary].merge(
        ref[["omop_person_id", "gold_label", "gold_binary", "institution"]],
        on="omop_person_id", how="left", validate="one_to_one")
    pm = pm[pm["gold_label"].notna() & pm["label_canonical"].notna()]
    det = []
    for s in SITES + ["ALL"]:
        g = pm if s == "ALL" else pm[pm["institution"] == s]
        d = g[g["gold_binary"].isin([POS, NEG])]
        r = km[(km["model"] == args.primary) & (km["site"] == s)].iloc[0]
        b = binmetrics(d["gold_binary"] == POS, d["aud_binary_canonical"] == YES)
        abst = 100 * (d["aud_binary_canonical"] == "abstain").mean()
        print("  %-9s kappa n=%3d %.3f (%.3f-%.3f)  |  binary n=%3d  sens %5.1f%% "
              "(%.1f-%.1f)  spec %5.1f%% (%.1f-%.1f)  indeterminate %4.1f%%"
              % (s, len(g), r["kappa_w"], r["ci_low"], r["ci_high"], len(d),
                 b["sens"], b["sens_lo"], b["sens_hi"],
                 b["spec"], b["spec_lo"], b["spec_hi"], abst))
        det.append({"site": s, "site_role": SITE_ROLE.get(s, "all sites pooled"),
                    "instrument": display.get(args.primary), "n_kappa": len(g),
                    "n_binary": len(d), "kappa_w": r["kappa_w"],
                    "kappa_ci_low": r["ci_low"], "kappa_ci_high": r["ci_high"],
                    "indeterminate_pct": abst, **b})

    print("\n  difference in weighted kappa relative to the primary site")
    for s in SITES[1:]:
        d = draws_store[(args.primary, s)] - draws_store[(args.primary, "IUH")]
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        pt = (km[(km["model"] == args.primary) & (km["site"] == s)]["kappa_w"].iloc[0]
              - km[(km["model"] == args.primary) & (km["site"] == "IUH")]["kappa_w"].iloc[0])
        print("    %-9s %+0.3f  (95%% CI %+0.3f to %+0.3f)" % (s, pt, lo, hi))
        det.append({"site": "%s minus IUH" % s, "site_role": "difference",
                    "instrument": display.get(args.primary), "n_kappa": np.nan,
                    "kappa_w": pt, "kappa_ci_low": lo, "kappa_ci_high": hi})

    # ------------------------------------------------------------------- the code baseline
    print("\n" + "=" * 78)
    print("THE DIAGNOSIS CODE AGAINST THE SAME REFERENCE, BY SITE")
    print("=" * 78)
    code = []
    for s in SITES + ["ALL"]:
        g = pm if s == "ALL" else pm[pm["institution"] == s]
        d = g[g["gold_binary"].isin([POS, NEG])]
        b = binmetrics(d["gold_binary"] == POS, d["has_aud_icd"].astype(bool))
        print("  %-9s n=%3d  sens %5.1f%% (%.1f-%.1f)  spec %5.1f%% (%.1f-%.1f)"
              % (s, len(d), b["sens"], b["sens_lo"], b["sens_hi"],
                 b["spec"], b["spec_lo"], b["spec_hi"]))
        code.append({"site": s, "site_role": SITE_ROLE.get(s, "all sites pooled"),
                     "instrument": "AUD diagnosis code", "n_kappa": np.nan,
                     "n_binary": len(d), "kappa_w": np.nan, "kappa_ci_low": np.nan,
                     "kappa_ci_high": np.nan, "indeterminate_pct": 0.0, **b})

    print("\n  sensitivity gained by reading the notes, in percentage points")
    for s in SITES + ["ALL"]:
        a = [r for r in det if r["site"] == s][0]
        c = [r for r in code if r["site"] == s][0]
        print("    %-9s %+5.1f points  (%.1f%% versus %.1f%%)"
              % (s, a["sens"] - c["sens"], a["sens"], c["sens"]))

    pd.DataFrame(km).to_csv(Path(args.out) / "site_kappa_all_models.csv", index=False)
    pd.DataFrame(det + code).to_csv(Path(args.out) / "site_performance.csv", index=False)
    print("\n  wrote %s/site_kappa_all_models.csv" % args.out)
    print("  wrote %s/site_performance.csv" % args.out)


if __name__ == "__main__":
    main()
