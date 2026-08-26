#!/usr/bin/env python3
"""Assemble every quantity the external-validation figure needs, in four tidy files.

  site_fig_kappa.csv    weighted kappa for all 11 models at each site
  site_fig_primary.csv  the primary model and the diagnosis code, site by site
  site_fig_paired.csv   the paired discordance between the two, site by site
  site_fig_stratum.csv  instrument sensitivity inside each code-status stratum, by site

Institution comes from the data provider's report on the delivered notes. Run on Quartz
from /N/project/depot/hw56/aud/medace_aud.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "downstream")
sys.path.insert(0, "downstream/npj")
import fig6_bayes_strata as B  # noqa: E402
import stats_core as sc  # noqa: E402

OUT = "downstream/npj_goal7_outputs"
SITES = ["IUH", "Eskenazi", "Other"]
SNAMES = B.SNAMES
PRIMARY = "aud_qwen36_27b"

# --- primary agreement definition ------------------------------------------
# Three categories, not enough data excluded rather than placed on the scale.
ORD3_PRIMARY = ["AUD-", "Uncertain", "AUD+"]
_POS3 = {"Definitely Yes", "Probably Yes"}
_NEG3 = {"Definitely No", "Probably No"}


def collapse3(s):
    s = pd.Series(s)
    out = pd.Series(np.where(s.isin(_POS3), "AUD+",
                    np.where(s.isin(_NEG3), "AUD-",
                    np.where(s.eq("Uncertain"), "Uncertain", None))), index=s.index)
    return out.where(s.notna() & s.ne("Not Enough Data"))


def kappa3(a, b):
    a, b = collapse3(a), collapse3(b)
    m = a.notna() & b.notna()
    if m.sum() < 2:
        return float("nan")
    return sc.cohens_kappa(a[m], b[m], ORD3_PRIMARY, None)
# ---------------------------------------------------------------------------

NB = 2000   # the resample count used everywhere in the paper


def wil(k, n):
    if n == 0:
        return np.nan, np.nan, np.nan
    lo, hi = sc.wilson_ci(int(k), int(n))
    return 100 * k / n, 100 * lo, 100 * hi


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20260807)
    cfg = sc.load_yaml_with_scope("downstream/data.yaml", "ref_gold")
    labels = list(cfg["labels_6"])
    display = {s["model"]: n for n, s in cfg["models"].items()}

    m = B.build(0)
    inst = pd.read_csv("output/fairness/patient_institution.csv")
    inst["person_id"] = pd.to_numeric(inst["person_id"], errors="coerce").astype("Int64")
    m = m.merge(inst[["person_id", "institution"]], on="person_id", how="left")

    panel = pd.read_csv("output/canonical_panel.csv", dtype={"omop_person_id": str})
    # The reference standard comes from ref_standard.build(), the one place it is derived.
    # This file previously read resources/gold_reference_v1_2026-08-03.csv, which was moved
    # to resources/_archive when the standard closed on 13 August, so the script could not
    # run at all. The v1 file predates the second-reader escalation and the final three
    # adjudications: it holds a different set of labels on the same 487 charts.
    import ref_standard as R
    ref, _ = R.build()
    ref["omop_person_id"] = ref["omop_person_id"].astype(str)
    ref["gold_label"] = sc.normalize_labels(ref["gold_label"], labels, strict=True)
    panel["label_canonical"] = sc.normalize_labels(panel["label_canonical"], labels,
                                                   strict=True)
    # Institution comes straight from the provider's report so that every reviewed patient
    # keeps a site. The code-status stratum S needs the registry frame, and two reviewed
    # patients have no qualifying cirrhosis record there, so S is missing for those two and
    # they appear in panels a to c but not in the stratum panel.
    inst_key = inst.copy()
    inst_key["omop_person_id"] = inst_key["person_id"].astype(str)
    s_key = m[["person_id", "S"]].copy()
    s_key["omop_person_id"] = s_key["person_id"].astype(str)
    ref = ref.merge(inst_key[["omop_person_id", "institution"]], on="omop_person_id",
                    how="left")
    ref = ref.merge(s_key[["omop_person_id", "S"]], on="omop_person_id", how="left")
    if ref["institution"].isna().any():
        raise RuntimeError("a reviewed patient has no institution")

    # ------------------------------------------------------------------ kappa, all models
    rows = []
    for model, one in panel.groupby("model", sort=False):
        one = one.merge(ref[["omop_person_id", "gold_label", "gold_binary", "institution",
                             "S"]], on="omop_person_id", how="left", validate="one_to_one")
        one = one[one["gold_label"].notna() & one["label_canonical"].notna()]
        for s in ["ALL"] + SITES:
            g = one if s == "ALL" else one[one["institution"] == s]
            if len(g) < 20:
                continue
            # Primary definition: the charts both readers placed, not enough data
            # excluded. The denominator is therefore smaller than the labelled count.
            ka, kb = collapse3(g["gold_label"]), collapse3(g["label_canonical"])
            keep = ka.notna() & kb.notna()
            ka, kb = ka[keep].to_numpy(), kb[keep].to_numpy()
            if len(ka) < 20:
                continue
            k = sc.cohens_kappa(ka, kb, ORD3_PRIMARY, None)
            d = np.array([sc.cohens_kappa(ka[i], kb[i], ORD3_PRIMARY, None)
                          for i in (rng.integers(0, len(ka), len(ka)) for _ in range(NB))])
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            rows.append({"model": model, "display_name": display.get(model, model),
                         "site": s, "n": len(ka), "n_labelled": len(g),
                         "kappa_w": k, "ci_low": lo, "ci_high": hi,
                         "is_primary": model == PRIMARY})
    pd.DataFrame(rows).to_csv(f"{OUT}/site_fig_kappa.csv", index=False)

    # -------------------------------------------------- primary model and the code, by site
    pm = panel[panel["model"] == PRIMARY].merge(
        ref[["omop_person_id", "gold_label", "gold_binary", "institution", "S"]],
        on="omop_person_id", how="left", validate="one_to_one")
    pm = pm[pm["gold_binary"].isin(["AUD+", "AUD-"]) & pm["institution"].notna()]
    prim, paired = [], []
    for s in ["ALL"] + SITES:
        g = pm if s == "ALL" else pm[pm["institution"] == s]
        truth = (g["gold_binary"] == "AUD+").to_numpy()
        mod = (g["aud_binary_canonical"] == "Yes-side").to_numpy()
        icd = g["has_aud_icd"].astype(bool).to_numpy()
        for nm, pred in (("instrument", mod), ("AUD diagnosis code", icd)):
            se = wil((truth & pred).sum(), truth.sum())
            sp = wil((~truth & ~pred).sum(), (~truth).sum())
            prim.append({"site": s, "instrument": nm, "n": len(g),
                         "n_pos": int(truth.sum()), "sens": se[0], "sens_lo": se[1],
                         "sens_hi": se[2], "spec": sp[0], "spec_lo": sp[1],
                         "spec_hi": sp[2]})
        # paired discordance among all determinate patients
        b = int((mod == truth).sum() - ((mod == truth) & (icd == truth)).sum())
        c = int((icd == truth).sum() - ((mod == truth) & (icd == truth)).sum())
        p = (sum(math.comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c) * 2
             if b + c else 1.0)
        ga = np.array([(truth[i] & mod[i]).sum() / max(truth[i].sum(), 1)
                       - (truth[i] & icd[i]).sum() / max(truth[i].sum(), 1)
                       for i in (rng.integers(0, len(g), len(g)) for _ in range(NB))])
        lo, hi = np.percentile(100 * ga, [2.5, 97.5])
        paired.append({"site": s, "n": len(g), "instrument_only_correct": b,
                       "code_only_correct": c, "p_exact": min(p, 1.0),
                       "sens_gain": prim[-2]["sens"] - prim[-1]["sens"],
                       "gain_lo": lo, "gain_hi": hi})
    pd.DataFrame(prim).to_csv(f"{OUT}/site_fig_primary.csv", index=False)
    pd.DataFrame(paired).to_csv(f"{OUT}/site_fig_paired.csv", index=False)

    # ------------------------------------------------ instrument inside each code stratum
    # The diagnosis code is carried through this panel as well. Its sensitivity inside the
    # record-silent stratum is the sharpest comparison the study can make: that stratum is
    # defined by the structured record holding no alcohol code at t0, so the code can only
    # reach those patients by acquiring one later, after the index date. The LLM reaches
    # them from notes that already exist at t0.
    st = []
    for s in SITES + ["ALL"]:
        for k in range(3):
            sel = pm["institution"].notna() if s == "ALL" else (pm["institution"] == s)
            g = pm[sel & (pm["S"].eq(k)) & (pm["gold_binary"] == "AUD+")]
            v = wil((g["aud_binary_canonical"] == "Yes-side").sum(), len(g))
            st.append({"site": s, "stratum": SNAMES[k], "instrument": "LLM",
                       "n": len(g), "sens": v[0], "ci_low": v[1], "ci_high": v[2]})
            c = wil(int(g["has_aud_icd"].sum()), len(g))
            st.append({"site": s, "stratum": SNAMES[k], "instrument": "AUD diagnosis code",
                       "n": len(g), "sens": c[0], "ci_low": c[1], "ci_high": c[2]})
    pd.DataFrame(st).to_csv(f"{OUT}/site_fig_stratum.csv", index=False)

    print("=" * 78)
    print("PAIRED COMPARISON, INSTRUMENT AGAINST THE DIAGNOSIS CODE, BY SITE")
    print("=" * 78)
    for r in paired:
        print("  %-9s n=%3d  instrument only correct %3d, code only correct %2d, "
              "exact p = %.2e" % (r["site"], r["n"], r["instrument_only_correct"],
                                  r["code_only_correct"], r["p_exact"]))
        print("            sensitivity gain %+5.1f points (95%% CI %+.1f to %+.1f)"
              % (r["sens_gain"], r["gain_lo"], r["gain_hi"]))
    print("\n  wrote four files to %s" % OUT)


if __name__ == "__main__":
    main()
