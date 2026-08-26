#!/usr/bin/env python3
"""Figure 6, registry scale, three-stratum exposure.

Why this replaces the binary version. A binary "AUD code present or absent" exposure
merges two groups that are clinically different:

  stratum 2  alcohol aetiology is coded but no AUD diagnosis code. Chart-confirmed AUD is
             87.5%, statistically the same as stratum 1 (87.1%). An AUD code adds no
             detection information here; the record already names alcohol. The gap is a
             documentation and care-pathway gap.
  stratum 3  the structured record carries no alcohol code of any kind. Chart-confirmed
             AUD is 26.8%. This is the only stratum where reading notes recovers
             information that no structured rule can reproduce, and it holds 70.6% of the
             registry.

Model
  A | S = s, period = k  ~  Bernoulli(theta[s, k])          s in {1, 2, 3}
  Y | A, S, X            ~  Bernoulli(expit(b0 + bA A + b2 S2 + b3 S3 + bX'X))

S1 is the reference. The latent binary A is marginalised analytically for registry-only
patients. Sampling reuses run_chain from fig6_bayes_me_v2 through a local log posterior
with the same structure, generalised from one code column to two stratum dummies.

Verification: setting COLLAPSE=1 merges strata 2 and 3 into one "no AUD code" group and
must reproduce the binary primary analysis, risk ratio 2.64 (1.76-3.96).

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "downstream")
# ref_standard lives in downstream/npj, and this file is run directly by
# jobs_bayes_all.sh, so downstream/npj is not on the path unless it is put there.
sys.path.insert(0, "downstream/npj")
import fig6_bayes_me_v2 as M  # noqa: E402

D = M.D
OUT = "output/fig6_registry"
MIN_DATE = pd.Timestamp("2005-01-01")
AUD_CODE = r"^(F10|303|305\.0)"
CIRR_CODE = r"^(K70|K74|571)"
ALC_SIGNAL = (r"^(K70|571\.0|571\.1|571\.2|571\.3|K85\.2|577\.0|G62\.1|357\.5|"
              r"I42\.6|425\.5|K29\.2|535\.3|E24\.4|291|E52|265\.2|T51|980|Z71\.4|Z72\.1|"
              r"V11\.3|V79\.1|F10)")
AUD_DRUGS = {1714319, 19043959, 735850}
EXCLUDE_DRUG = {909841}
POS = {"definitely yes", "probably yes"}
NEG = {"definitely no", "probably no"}
AB = {"uncertain", "not enough data"}
PNAMES = list(M.PNAMES)
SNAMES = ["AUD code", "alcohol coded only", "record silent"]


def cat(x):
    x = (x or "").strip().lower()
    return "AUD+" if x in POS else "AUD-" if x in NEG else "Ind" if x in AB else ""


NARROW_SIGNAL = r"^(K70|571\.0|571\.1|571\.2|571\.3)"


def build(collapse=0, narrow=0, acamp=0, ever_rx=0, u_source="parquet"):
    # The reference standard comes from ref_standard.build(), the one place it is derived.
    # Until 13 August this function assembled its own labels from resources/chart_scores.csv
    # and from patient_roster.csv:ndepth_maaz_rating. That path read a 499-patient roster, so
    # it carried 12 patients outside the analysis cohort; it never saw the charts referred on
    # the second-reader trigger, because chart_scores.csv holds no SAI record at all; and it
    # never saw the final three adjudications. It disagreed with the canonical standard on 3
    # of the shared 487 and gave a validated arm of 431 where the canonical standard gives 423.
    import ref_standard as R
    g, _ = R.build()
    ref = {int(pid): cat(lab) for pid, lab in
           zip(g["omop_person_id"].astype(str), g["gold_label"])
           if str(pid).strip().isdigit()}

    pe = pd.read_parquet(D + "/person.parquet",
                         columns=["person_id", "year_of_birth", "gender_concept_id"])
    pe["person_id"] = pe["person_id"].astype("int64")
    co = pd.read_parquet(D + "/condition_occurrence.parquet",
                         columns=["person_id", "condition_source_value", "condition_start_date"])
    co["person_id"] = co["person_id"].astype("int64")
    co["condition_start_date"] = pd.to_datetime(co["condition_start_date"], errors="coerce")
    co = co[co["condition_start_date"].notna() & (co["condition_start_date"] >= MIN_DATE)]
    co["icd"] = co["condition_source_value"].astype(str).str.extract(r"\^\^([^\^]+)\^")[0]
    co.loc[co["icd"].isna(), "icd"] = co.loc[co["icd"].isna(), "condition_source_value"].astype(str)
    ic = co["icd"].astype(str)
    f_cirr = (co[ic.str.match(CIRR_CODE, na=False)]
              .groupby("person_id")["condition_start_date"].min())
    f_aud = co[ic.str.match(AUD_CODE, na=False)].groupby("person_id")["condition_start_date"].min()
    sig = NARROW_SIGNAL if narrow else ALC_SIGNAL
    f_alc = co[ic.str.match(sig, na=False)].groupby("person_id")["condition_start_date"].min()
    # narrow alcoholic liver disease, the same definition fig6_bayes_me_v2 adjusts for.
    # Needed only for the collapse verification, where the covariate set must match v2.
    f_ald = co[ic.str.match(r"^(K70|571\.0|571\.1|571\.2|571\.3)", na=False)].groupby(
        "person_id")["condition_start_date"].min()
    del co, ic

    m = pe.copy()
    m["t0"] = m["person_id"].map(f_cirr)
    m = m[m["t0"].notna()].copy()
    m["aud"] = m["person_id"].map(f_aud)
    m["alc"] = m["person_id"].map(f_alc)
    c1 = m["aud"].notna() & (m["aud"] <= m["t0"])
    c2 = (~c1) & m["alc"].notna() & (m["alc"] <= m["t0"])
    m["S"] = np.where(c1, 0, np.where(c2, 1, 2))
    if collapse:
        m["S"] = np.where(c1, 0, 1)
    fa = m["person_id"].map(f_ald)
    m["alcld"] = (fa.notna() & (fa <= m["t0"])).astype(float)
    m["age"] = m["t0"].dt.year - m["year_of_birth"]
    m["age"] = m["age"].fillna(m["age"].median())
    m["male"] = (m["gender_concept_id"] == 8507).astype(float)

    vo = pd.read_parquet(D + "/visit_occurrence.parquet",
                         columns=["person_id", "visit_start_date"])
    vo["person_id"] = vo["person_id"].astype("int64")
    vo["visit_start_date"] = pd.to_datetime(vo["visit_start_date"], errors="coerce")
    vo = vo[vo["visit_start_date"] >= MIN_DATE]
    j = vo.merge(m[["person_id", "t0"]], on="person_id", how="inner")
    m["nvis"] = m["person_id"].map(
        j[j["visit_start_date"] <= j["t0"]].groupby("person_id").size()).fillna(0)
    del vo, j

    de = pd.read_parquet(D + "/drug_exposure.parquet",
                         columns=["person_id", "drug_concept_id",
                                  "drug_exposure_start_date"])
    de["person_id"] = de["person_id"].astype("int64")
    de["drug_concept_id"] = de["drug_concept_id"].astype("int64")
    anc = pd.read_parquet(D + "/concept_ancestor.parquet",
                          columns=["ancestor_concept_id", "descendant_concept_id"])
    anc["ancestor_concept_id"] = anc["ancestor_concept_id"].astype("int64")
    conc = AUD_DRUGS | set(anc.loc[anc["ancestor_concept_id"].isin(AUD_DRUGS),
                                   "descendant_concept_id"].astype("int64"))
    conc -= EXCLUDE_DRUG
    anc2 = anc
    del anc
    if acamp:
        # Naltrexone is also indicated for opioid use disorder. Acamprosate and disulfiram
        # have no other indication.
        seeds = {19043959, 735850}
        conc = seeds | set(anc2.loc[anc2["ancestor_concept_id"].isin(seeds),
                                    "descendant_concept_id"].astype("int64"))
        conc -= EXCLUDE_DRUG
    # Receipt is dated relative to the cirrhosis diagnosis. Counting exposure at any time
    # includes medication dispensed years before the diagnosis, and that is not evenly
    # distributed: 3.94% of the AUD-code stratum has medication only before t0 against
    # 0.22% of the record-silent stratum, so an undated outcome widens the contrast.
    de["d"] = pd.to_datetime(de["drug_exposure_start_date"], errors="coerce")
    de = de[de["d"].notna()]
    de = de[de["drug_concept_id"].isin(conc)]
    first = de.groupby("person_id")["d"].min().rename("first_rx")
    m["first_rx"] = m["person_id"].map(first)
    if ever_rx:
        m["Y"] = m["first_rx"].notna().astype(float)
    else:
        m["Y"] = ((m["first_rx"].notna()) & (m["first_rx"] >= m["t0"])).astype(float)
    m["rx_before_t0"] = ((m["first_rx"].notna()) & (m["first_rx"] < m["t0"])).astype(float)
    del de

    m["period"] = np.where(m["t0"].dt.year <= 2016, 0,
                           np.where(m["t0"].dt.year <= 2018, 1, 2))
    m["per1"] = (m["period"] == 1).astype(float)
    m["per2"] = (m["period"] == 2).astype(float)
    lab = m["person_id"].map(lambda p: ref.get(p, ""))
    # A chart the reviewers could not resolve carries no information about AUD status, so
    # it does not enter the validated likelihood. Those patients stay in the registry arm
    # with A latent, which keeps their treatment outcome and their stratum.
    m["Aobs"] = np.where(lab.eq("AUD+"), 1.0,
                         np.where(lab.eq("AUD-"), 0.0, np.nan))
    m["val"] = m["Aobs"].notna()
    # Charts were drawn stratified on whether an AUD diagnosis code appears at any time,
    # while the strata above are defined at t0. The two do not coincide: a patient can be
    # silent at t0 and coded later. Selection therefore depends on U within stratum, and U
    # is strongly related to chart-documented AUD, so theta has to be indexed by it.
    m["U"] = m["aud"].notna().astype(int)
    if u_source == "roster":
        # The design variable is the value the sampler used, not the value the current
        # extract reproduces. For sampled patients that is patient_roster.has_aud_icd.
        r = pd.read_csv("resources/patient_roster.csv")
        r["pid"] = pd.to_numeric(r["omop_person_id"], errors="coerce")
        r = r[r["gpids_era_a"].notna() & r["pid"].notna()]
        h = dict(zip(r["pid"].astype("int64"), r["has_aud_icd"].astype(int)))
        u = m["person_id"].astype("int64").map(h)
        n_moved = int(((u.notna()) & (u != m["U"])).sum())
        m["U"] = u.fillna(m["U"]).astype(int)
        print("u_source=roster: %d sampled patients moved cell" % n_moved)
    return m


def make_loglik(nS, nth):
    """Log posterior with nS strata. Stratum 0 is the reference in the outcome model.

    theta is indexed by stratum, period and sampling status U, because the charts were
    drawn on U. A validated patient contributes at its own cell, since that is the cell
    its selection came from. A registry patient contributes at the U-marginal for its
    stratum and period, weighted by the registry's own distribution of U. Marginalising
    there is deliberate: U can be set after t0 and after treatment, so carrying it into
    the outcome arm would condition on a post-treatment variable.
    """
    def loglik(par, d, nX):
        th = par[:nth]
        if np.any(th <= 1e-6) or np.any(th >= 1 - 1e-6):
            return -np.inf
        b0, bA = par[nth], par[nth + 1]
        bS = par[nth + 2:nth + 1 + nS]          # nS-1 stratum dummies
        bX = par[nth + 1 + nS:nth + 1 + nS + nX]
        ll = 0.0
        lv = b0 + d["Sv"] @ bS + d["Xv"] @ bX
        pv = M.expit(lv + bA * d["Av"])
        ll += np.sum(np.log(np.where(d["Yv"] == 1, pv, 1 - pv) + 1e-300))
        thv = th[d["thidx_v"]]
        ll += np.sum(np.where(d["Av"] == 1, np.log(thv), np.log(1 - thv)))
        lr = b0 + d["Sr"] @ bS + d["Xr"] @ bX
        p1, p0 = M.expit(lr + bA), M.expit(lr)
        yr = d["Yr"]
        q1 = np.where(yr == 1, p1, 1 - p1)
        q0 = np.where(yr == 1, p0, 1 - p0)
        # The marginal is constant within a stratum-period group, and there are at most
        # nine of those, so it is computed once per group and then indexed. Forming it per
        # patient would multiply a 14,000-row matrix on every likelihood evaluation.
        thr = (d["Wsp"] @ th)[d["spidx_r"]]
        ll += np.sum(np.log(thr * q1 + (1 - thr) * q0 + 1e-300))
        ll += -0.5 * ((b0 / 2.5) ** 2 + (bA / 2.5) ** 2 + np.sum((bS / 2.5) ** 2))
        ll += -0.5 * np.sum((bX / 1.0) ** 2)
        return ll
    return loglik



def run_cut_chain(loglik, d, nX, seed, n_outer, n_inner, burn_inner, nth, ncoef,
                  init, a_cell, n_cell, block_idx, warmup_outer=0):
    """Two-stage sampler for the cut posterior.

    In a single joint likelihood the outcome of an unvalidated patient contributes to theta
    through the marginalised mixture, so treatment data would help decide who has AUD and
    the estimated AUD prevalence would in turn drive the treatment contrast. The reference
    standard is meant to be the only source of theta, so the theta block is cut off from
    the outcome data.

    Stage one is closed form. Within a cell the validated charts are Bernoulli draws with a
    Beta(1,1) prior, so theta | charts is Beta(1 + a, 1 + n - a). Stage two fits the
    outcome coefficients by Metropolis with theta held at a draw from stage one. Outer
    iterations redraw theta and rerun a short inner chain from the previous endpoint, and
    the kept draws are pooled. This targets p(theta | charts) p(coefficients | theta, Y),
    which is the cut distribution.
    """
    rng = np.random.default_rng(seed)
    par = np.asarray(init, dtype=float).copy()
    step = np.full(ncoef, 0.2)
    bidx = np.asarray(block_idx) - nth
    Cb = np.eye(len(bidx)) * 0.04
    # adapt_buf collects the draws taken while the proposal is still being tuned. They are
    # used to form the block covariance and are then thrown away, so the returned chain
    # contains only draws from the adapted kernel.
    keep, adapt_buf, acc, tot = [], [], np.zeros(ncoef), 0
    adapt_at = n_outer // 4
    for out in range(n_outer):
        par[:nth] = rng.beta(1.0 + a_cell, 1.0 + n_cell - a_cell)
        cur = loglik(par, d, nX)
        for it in range(n_inner):
            for k in range(ncoef):
                prop = par.copy()
                prop[nth + k] += rng.normal(0, step[k])
                lp = loglik(prop, d, nX)
                if np.log(rng.random()) < lp - cur:
                    par, cur = prop, lp
                    acc[k] += 1
            tot += 1
            prop = par.copy()
            prop[nth + bidx] += rng.multivariate_normal(np.zeros(len(bidx)), 0.6 * Cb)
            lp = loglik(prop, d, nX)
            if np.log(rng.random()) < lp - cur:
                par, cur = prop, lp
            if it >= burn_inner:
                (keep if out >= warmup_outer else adapt_buf).append(par.copy())
        if out == adapt_at:
            r = acc / max(tot, 1)
            step *= np.clip(np.exp((r - 0.35) * 2.0), 0.5, 2.0)
            pool = adapt_buf if adapt_buf else keep
            K = np.cov(np.asarray(pool)[:, nth + bidx].T)
            if np.all(np.isfinite(K)):
                Cb = K + np.eye(len(bidx)) * 1e-6
    if not keep:
        raise SystemExit("warmup_outer discarded every draw")
    return np.asarray(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collapse", type=int, default=0,
                    help="1 merges strata 2 and 3, to reproduce the binary analysis")
    ap.add_argument("--variant", default="primary",
                    choices=["primary", "acamp_disulf", "theta_pooled", "narrow_signal"],
                    help="sensitivity variant, all under the three-stratum specification")
    ap.add_argument("--iters", type=int, default=16000)
    ap.add_argument("--burn", type=int, default=8000)
    ap.add_argument("--joint", type=int, default=0,
                    help="1 fits the single joint likelihood, in which the outcomes of "
                         "unvalidated patients also inform theta")
    ap.add_argument("--ever-rx", type=int, default=0,
                    help="1 counts medication at any time rather than at or after t0")
    # The cut sampler's lengths were hard coded, and --burn was read but never reached the
    # cut path, so a longer --iters only raised the number of outer draws. The three
    # lengths are now separate arguments, and the draws taken before the proposal adapts
    # are discarded rather than pooled with the rest.
    ap.add_argument("--outer", type=int, default=0,
                    help="outer iterations; 0 keeps the old rule, max(200, iters/40)")
    ap.add_argument("--inner", type=int, default=30,
                    help="inner Metropolis steps per outer iteration")
    ap.add_argument("--inner-burn", type=int, default=10,
                    help="inner steps discarded before any draw is kept")
    ap.add_argument("--warmup-frac", type=float, default=0.25,
                    help="fraction of outer iterations discarded; the proposal adapts at "
                         "the end of this fraction, so draws before it are not from the "
                         "adapted kernel")
    # The sampling indicator the charts were drawn on is the roster's has_aud_icd. It
    # disagrees with the indicator recomputed from the current parquet on 14 of the 485
    # sampled patients who are in the frame, all of them chart-confirmed AUD+ with no
    # treatment, and 10 of those sit in the record-silent stratum where U=0 holds only 33
    # positives. Refitting with the roster value is the sensitivity analysis for that.
    # The record starts on 1 January 2015 and t0 is the first cirrhosis code in it, so
    # t0 is left-censored at that date and the 2015 entrants have almost no record before
    # t0 in which an AUD code could appear. Stratum membership is defined at t0. Dropping
    # the boundary year leaves every remaining patient at least a year of record before
    # t0. The measurement is in downstream/npj/f03_extract_window.py.
    ap.add_argument("--min-t0-year", type=int, default=0,
                    help="drop patients whose t0 falls before this year; 2016 drops the "
                         "extract-boundary cohort")
    ap.add_argument("--u-source", default="parquet", choices=["parquet", "roster"],
                    help="which AUD-code-ever indicator defines the theta cells for "
                         "sampled patients")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    m = build(a.collapse, narrow=int(a.variant == "narrow_signal"),
              acamp=int(a.variant == "acamp_disulf"), ever_rx=a.ever_rx,
              u_source=a.u_source)
    if a.min_t0_year:
        before, val_before = len(m), int(m["val"].sum())
        m = m[m["t0"].dt.year >= a.min_t0_year].copy()
        print("t0 restricted to %d and later: %d of %d patients, %d of %d validated"
              % (a.min_t0_year, len(m), before, int(m["val"].sum()), val_before))
    if a.variant == "theta_pooled":
        M.PERIODS = [(2000, 2030)]
        M.PNAMES = ["all"]
        m["period"] = 0
    nS = int(m["S"].max()) + 1
    print("variant: %s" % a.variant)
    snames = SNAMES[:2] if a.collapse else SNAMES
    if a.collapse:
        snames = ["AUD code", "no AUD code"]
    print("registry with a cirrhosis T0: %d   validated: %d   events: %d"
          % (len(m), m["val"].sum(), m["Y"].sum()))
    for s in range(nS):
        g = m[m["S"] == s]
        print("  stratum %d %-20s n=%6d  validated %3d  events %4d"
              % (s + 1, snames[s], len(g), g["val"].sum(), int(g["Y"].sum())))

    # The three-stratum model stratifies on alcohol coding, so it must not also adjust for
    # it. The collapse verification must match fig6_bayes_me_v2 exactly, which does adjust.
    Xcols = ["age", "male", "lognvis", "per1", "per2"]
    if a.collapse:
        Xcols = ["age", "male", "alcld", "lognvis", "per1", "per2"]
    m["lognvis"] = np.log1p(m["nvis"])
    Xm = m[Xcols].to_numpy(float)
    mu, sd = Xm.mean(0), Xm.std(0)
    sd[sd == 0] = 1.0
    Xs = (Xm - mu) / sd
    Sd = np.zeros((len(m), nS - 1))
    for s in range(1, nS):
        Sd[:, s - 1] = (m["S"].to_numpy() == s).astype(float)
    v = m["val"].to_numpy(bool)

    # theta cells are (stratum, period, sampling status). Only the combinations that
    # occur are given a parameter: a patient carrying an AUD code at t0 has carried one at
    # some time by definition, so that stratum has no U=0 cell in the registry or in the
    # sample.
    key = list(zip(m["S"].to_numpy(int), m["period"].to_numpy(int), m["U"].to_numpy(int)))
    cells = sorted(set(key))
    cellmap = {c: i for i, c in enumerate(cells)}
    thidx = np.array([cellmap[k] for k in key])
    nth = len(cells)

    # For a registry patient theta is the U-marginal within its stratum and period, with
    # the weights taken from the registry itself, so they carry no uncertainty.
    sp = pd.DataFrame({"S": m["S"].to_numpy(int), "period": m["period"].to_numpy(int),
                       "U": m["U"].to_numpy(int)})
    frac = (sp.groupby(["S", "period"])["U"].value_counts(normalize=True)
              .rename("w").reset_index())
    Wsp = np.zeros((nS * len(M.PERIODS), nth))
    for _, r in frac.iterrows():
        Wsp[int(r["S"]) * len(M.PERIODS) + int(r["period"]),
            cellmap[(int(r["S"]), int(r["period"]), int(r["U"]))]] = r["w"]
    spidx = m["S"].to_numpy(int) * len(M.PERIODS) + m["period"].to_numpy(int)
    assert np.allclose(Wsp.sum(1)[np.unique(spidx)], 1.0), "marginal weights must sum to 1"

    d = {"Sv": Sd[v], "Yv": m["Y"].to_numpy(float)[v], "Xv": Xs[v],
         "Av": m["Aobs"].to_numpy(float)[v], "thidx_v": thidx[v],
         "Sr": Sd[~v], "Yr": m["Y"].to_numpy(float)[~v], "Xr": Xs[~v],
         "Wsp": Wsp, "spidx_r": spidx[~v]}
    nX = len(Xcols)

    print("\ntheta cells (stratum, period, an AUD code appears at any time)")
    for c, i in cellmap.items():
        n_all = int((thidx == i).sum())
        n_val = int((thidx[v] == i).sum())
        print("  S%d %-5s U=%d   registry n=%6d   validated n=%3d"
              % (c[0] + 1, M.PNAMES[c[1]], c[2], n_all, n_val))

    Aobs = m["Aobs"].to_numpy(float)
    a_cell = np.array([np.nansum(Aobs[v & (thidx == i)]) for i in range(nth)])
    n_cell = np.array([int((v & (thidx == i)).sum()) for i in range(nth)])
    print("\nvalidated charts per theta cell: %s" % n_cell.tolist())
    print("of which reference positive:      %s" % a_cell.astype(int).tolist())

    ll = make_loglik(nS, nth)
    ncoef = 1 + nS + nX          # b0, bA, (nS-1) stratum dummies, nX covariates
    init = np.concatenate([np.full(nth, 0.5), np.array([-4.0, 2.0]),
                           np.zeros(nS - 1), np.zeros(nX)])
    assert len(init) == nth + ncoef, "layout mismatch: %d vs %d" % (len(init), nth + ncoef)
    block = np.arange(nth, nth + 1 + nS)   # b0, bA and the stratum dummies lie on a ridge
    print("\nsampling 4 chains; %d theta + %d coefficients = %d parameters"
          % (nth, ncoef, nth + ncoef))
    if a.joint:
        chains = [M.run_chain_general(ll, d, nX, M.SEED + c, a.iters, a.burn,
                                      nth, ncoef, init, block_idx=block)
                  for c in range(4)]
    else:
        n_outer = a.outer if a.outer else max(200, a.iters // 40)
        warm = int(round(a.warmup_frac * n_outer))
        print("cut sampler: outer %d, inner %d, inner burn %d, warmup outer %d"
              % (n_outer, a.inner, a.inner_burn, warm))
        chains = [run_cut_chain(ll, d, nX, M.SEED + c, n_outer, a.inner, a.inner_burn,
                                nth, ncoef, init, a_cell, n_cell, block, warm)
                  for c in range(4)]
    print("sampler: %s" % ("single joint likelihood" if a.joint
                           else "two stage, theta cut off from the outcomes"))
    S = np.stack(chains)

    names = ([f"theta_S{c[0]+1}_{M.PNAMES[c[1]]}_U{c[2]}" for c in cells]
             + ["b0", "bA"] + ["b_S%d" % (s + 1) for s in range(1, nS)]
             + ["b_" + c for c in Xcols])
    rows = []
    print("\n%-22s %8s %8s %8s %8s %6s" % ("parameter", "mean", "2.5%", "97.5%", "R-hat", "ESS"))
    for k, nm in enumerate(names):
        ch = S[:, :, k]
        fl = ch.reshape(-1)
        ac = np.corrcoef(fl[:-1], fl[1:])[0, 1]
        ess = len(fl) * (1 - ac) / (1 + ac) if ac < 1 else np.nan
        print("%-22s %8.3f %8.3f %8.3f %8.3f %6.0f"
              % (nm, fl.mean(), np.percentile(fl, 2.5), np.percentile(fl, 97.5),
                 M.rhat(ch), ess))
        rows.append({"parameter": nm, "mean": fl.mean(),
                     "ci_low": np.percentile(fl, 2.5), "ci_high": np.percentile(fl, 97.5),
                     "rhat": M.rhat(ch), "ess": ess})

    # theta reported per stratum is the registry-weighted average over the period and
    # sampling-status cells that make up that stratum, so it describes the stratum as it
    # exists in the registry and not as it was sampled.
    print("\n" + "=" * 74)
    print("CHART-DOCUMENTED AUD BY STRATUM, weighted to the registry")
    print("=" * 74)
    Sarr = m["S"].to_numpy(int)
    wmat = []
    for s in range(nS):
        w = np.zeros(nth)
        for k, i in cellmap.items():
            if k[0] == s:
                w[i] = ((thidx == i) & (Sarr == s)).sum()
        w /= w.sum()
        vals = np.concatenate([S[c][:, :nth] @ w for c in range(S.shape[0])])
        chs = np.stack([S[c][:, :nth] @ w for c in range(S.shape[0])])
        n_s = int((Sarr == s).sum())
        print("  %-22s %5.1f%%  (%.1f to %.1f)   %6.0f patients (%.0f to %.0f)  R-hat %.3f"
              % (snames[s], 100 * vals.mean(), 100 * np.percentile(vals, 2.5),
                 100 * np.percentile(vals, 97.5), vals.mean() * n_s,
                 np.percentile(vals, 2.5) * n_s, np.percentile(vals, 97.5) * n_s,
                 M.rhat(chs)))
        rows.append({"parameter": "theta_registry_" + snames[s], "mean": 100 * vals.mean(),
                     "ci_low": 100 * np.percentile(vals, 2.5),
                     "ci_high": 100 * np.percentile(vals, 97.5),
                     "rhat": M.rhat(chs), "ess": np.nan})
        rows.append({"parameter": "n_with_aud_" + snames[s], "mean": vals.mean() * n_s,
                     "ci_low": np.percentile(vals, 2.5) * n_s,
                     "ci_high": np.percentile(vals, 97.5) * n_s,
                     "rhat": np.nan, "ess": np.nan})
        wmat.append(w)

    # Differences between strata, taken on the posterior draws. Reading them off the
    # marginal intervals would give the wrong width. The pairs come from the strata the
    # run actually has, so a collapsed run reports its one contrast instead of reaching
    # for a third stratum that does not exist.
    print()
    for i, j in itertools.combinations(range(nS), 2):
        di = np.concatenate([S[c][:, :nth] @ (wmat[i] - wmat[j])
                             for c in range(S.shape[0])])
        print("  %-22s minus %-22s %+6.1f points (%+.1f to %+.1f)  P(>0) = %.3f"
              % (snames[i], snames[j], 100 * di.mean(),
                 100 * np.percentile(di, 2.5), 100 * np.percentile(di, 97.5),
                 (di > 0).mean()))
        rows.append({"parameter": "theta_diff_%s_minus_%s" % (snames[i], snames[j]),
                     "mean": 100 * di.mean(), "ci_low": 100 * np.percentile(di, 2.5),
                     "ci_high": 100 * np.percentile(di, 97.5),
                     "rhat": np.nan, "ess": (di > 0).mean()})

    def rates(ch):
        b0, bA = ch[:, nth], ch[:, nth + 1]
        bS = ch[:, nth + 2:nth + 1 + nS]
        bX = ch[:, nth + 1 + nS:nth + 1 + nS + nX]
        base = b0[:, None] + bX @ Xs.T + bA[:, None]
        out = [M.expit(base).mean(1)]
        for s in range(nS - 1):
            out.append(M.expit(base + bS[:, s][:, None]).mean(1))
        return out
    pc = [rates(S[c]) for c in range(S.shape[0])]
    print("\n" + "=" * 74)
    print("TREATMENT AMONG PATIENTS WITH TRUE AUD, standardised, by stratum")
    print("=" * 74)
    pooled = []
    for s in range(nS):
        vals = np.concatenate([pc[c][s] for c in range(S.shape[0])])
        chs = np.stack([pc[c][s] for c in range(S.shape[0])])
        pooled.append(vals)
        print("  %-22s %6.2f%%  (%.2f to %.2f)   R-hat %.3f"
              % (snames[s], 100 * vals.mean(), 100 * np.percentile(vals, 2.5),
                 100 * np.percentile(vals, 97.5), M.rhat(chs)))
        rows.append({"parameter": "treated_" + snames[s], "mean": 100 * vals.mean(),
                     "ci_low": 100 * np.percentile(vals, 2.5),
                     "ci_high": 100 * np.percentile(vals, 97.5),
                     "rhat": M.rhat(chs), "ess": np.nan})
    print()
    for s in range(1, nS):
        rr = pooled[0] / pooled[s]
        rrc = np.stack([pc[c][0] / pc[c][s] for c in range(S.shape[0])])
        print("  risk ratio, AUD code versus %-20s %5.2f  (%.2f to %.2f)  "
              "P(>1) = %.4f  R-hat %.3f"
              % (snames[s], rr.mean(), np.percentile(rr, 2.5), np.percentile(rr, 97.5),
                 (rr > 1).mean(), M.rhat(rrc)))
        rows.append({"parameter": "rr_vs_" + snames[s], "mean": rr.mean(),
                     "ci_low": np.percentile(rr, 2.5), "ci_high": np.percentile(rr, 97.5),
                     "rhat": M.rhat(rrc), "ess": np.nan})
    tag = "collapsed" if a.collapse else ("strata3" if a.variant == "primary"
                                          else "strata3_" + a.variant)
    if a.joint:
        tag += "_joint"
    if a.ever_rx:
        tag += "_everrx"
    if a.u_source != "parquet":
        tag += "_u" + a.u_source
    if a.min_t0_year:
        tag += "_t0from%d" % a.min_t0_year
    if a.outer or a.inner != 30 or a.inner_burn != 10:
        tag += "_s%d_%d_%d" % (a.outer, a.inner, a.inner_burn)

    # A sidecar written at fit time, so a later reader does not have to trust that the
    # draws on disk came from the files that are on disk now. It records the commit, the
    # full hash of every input the fit reads, every argument, the seed, the shape of the
    # draw array and the hash of the array itself.
    import hashlib
    import json
    import subprocess

    def _h(p):
        hh = hashlib.sha256()
        with open(p, "rb") as f:
            for ch in iter(lambda: f.read(1 << 20), b""):
                hh.update(ch)
        return hh.hexdigest()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True).stdout.strip())
    except Exception:
        commit, dirty = "unknown", None
    np.save(os.path.join(OUT, "draws_%s.npy" % tag), S)
    man = {"tag": tag, "commit": commit, "working_tree_dirty": dirty,
           "args": vars(a), "seed_base": int(M.SEED), "n_chains": int(S.shape[0]),
           "draw_shape": list(S.shape), "parameter_names": names,
           "registry_n": int(len(m)), "validated_n": int(m["Aobs"].notna().sum()),
           "theta_cells": [list(map(int, c)) for c in cells],
           "n_cell": n_cell.tolist(), "a_cell": a_cell.astype(int).tolist(),
           "inputs": {p: _h(p) for p in
                      ["downstream/fig6_bayes_strata.py", "downstream/fig6_bayes_me_v2.py",
                       "downstream/npj/ref_standard.py", "resources/patient_roster.csv",
                       "downstream/npj/out/canonical_rating_sheet.csv"]
                      if os.path.exists(p)},
           "draws_sha256": _h(os.path.join(OUT, "draws_%s.npy" % tag))}
    with open(os.path.join(OUT, "fit_%s.json" % tag), "w") as f:
        json.dump(man, f, indent=2)
    print("  wrote %s/fit_%s.json" % (OUT, tag))
    # The draws were saved above, beside the sidecar. The registry AUD total is formed
    # inside each draw: adding the three stratum means gives a point estimate with no
    # interval, and adding the three marginal intervals gives the wrong width.
    tot = np.concatenate([
        np.sum([int((Sarr == s).sum()) * (S[c][:, :nth] @ wmat[s]) for s in range(nS)],
               axis=0) for c in range(S.shape[0])])
    print("\n  registry patients with AUD, formed inside each draw: %.0f (%.0f to %.0f)"
          % (tot.mean(), np.percentile(tot, 2.5), np.percentile(tot, 97.5)))
    rows.append({"parameter": "n_with_aud_total", "mean": float(tot.mean()),
                 "ci_low": float(np.percentile(tot, 2.5)),
                 "ci_high": float(np.percentile(tot, 97.5)),
                 "rhat": np.nan, "ess": np.nan})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "bayes_%s.csv" % tag), index=False)
    if a.collapse:
        print("\n  verification target, binary primary analysis: risk ratio 2.64 (1.76-3.96)")
    print("\n  wrote %s/bayes_%s.csv" % (OUT, tag))


if __name__ == "__main__":
    main()
