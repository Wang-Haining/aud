#!/usr/bin/env python3
"""Figure 6, registry scale. Bayesian misclassification model.

Design: main study plus internal validation subsample.

  main study   the 14,406-patient cirrhosis registry. Diagnosis-code status C and
               outcome Y observed. True AUD status A is latent.
  validation   the 487 chart-reviewed patients. C, A and Y all observed.

Model
  A | C = c, period = k  ~  Bernoulli(theta[c, k])
  Y | A, C, X            ~  Bernoulli(expit(b0 + bA*A + bC*C + bX'X))

The latent binary A is marginalised analytically for registry-only patients, so the
posterior is continuous. Sampling is adaptive component-wise random-walk Metropolis
with four chains; R-hat and effective sample size are reported.

theta is stratified by calendar period because the probe showed a downward trend in
theta[0] (42.9%, 39.5%, 31.8% across 2015-16, 2017-18, 2019-20; homogeneity P = 0.188).

Assumptions this model makes, all stated in the output:
  1. bC is common to both arms. An A-by-C interaction is not estimable: only 2 of 239
     validated AUD-negative patients received a drug.
  2. theta estimated in the validation sample transports to the registry within the
     same calendar period.
  3. Chart review is a perfect measure of AUD status.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
import os

import numpy as np
import pandas as pd

D = "/N/project/depot/hw56/aud/cirrhosis_regv20/data_tables_v1/parquet"
OUT = "output/fig6_registry"
MIN_DATE = pd.Timestamp("2005-01-01")
SEED = 42
# bounds are (exclusive lower, inclusive upper]
PERIODS = [(2000, 2016), (2016, 2018), (2018, 2030)]
PNAMES = ["2015-16", "2017-18", "2019+"]


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def build():
    m = pd.read_csv("output/feas/me_design_input.csv", parse_dates=["t0"])
    m = m[m["t0"].notna()].copy()
    pe = pd.read_parquet(D + "/person.parquet",
                         columns=["person_id", "year_of_birth", "gender_concept_id"])
    pe["person_id"] = pe["person_id"].astype("int64")
    m = m.merge(pe, on="person_id", how="left")
    m["age"] = m["t0"].dt.year - m["year_of_birth"]
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

    co = pd.read_parquet(D + "/condition_occurrence.parquet",
                         columns=["person_id", "condition_source_value", "condition_start_date"])
    co["person_id"] = co["person_id"].astype("int64")
    co["condition_start_date"] = pd.to_datetime(co["condition_start_date"], errors="coerce")
    co = co[co["condition_start_date"].notna() & (co["condition_start_date"] >= MIN_DATE)]
    co["icd"] = co["condition_source_value"].astype(str).str.extract(r"\^\^([^\^]+)\^")[0]
    co.loc[co["icd"].isna(), "icd"] = co.loc[co["icd"].isna(), "condition_source_value"].astype(str)
    # Alcoholic liver disease. K70 is ICD-10 only; 571.0 to 571.3 are its ICD-9
    # equivalents. A K70-only definition is confounded with calendar time because the
    # cohort spans the October 2015 transition.
    alc = co[co["icd"].astype(str).str.match(r"^(K70|571\.0|571\.1|571\.2|571\.3)",
                                             na=False)].groupby(
        "person_id")["condition_start_date"].min()
    del co
    fk = m["person_id"].map(alc)
    m["alcld"] = (fk.notna() & (fk <= m["t0"])).astype(float)

    m["age"] = m["age"].fillna(m["age"].median())
    m["period"] = -1
    for i, (a, b) in enumerate(PERIODS):
        m.loc[(m["t0"].dt.year > a) & (m["t0"].dt.year <= b), "period"] = i
    m.loc[m["t0"].dt.year <= PERIODS[0][0], "period"] = 0
    m = m[m["period"] >= 0].copy()
    m["per1"] = (m["period"] == 1).astype(float)
    m["per2"] = (m["period"] == 2).astype(float)
    m["Aobs"] = np.where(m["validated"] & m["A"].eq("AUD+"), 1.0,
                         np.where(m["validated"] & m["A"].isin(["AUD-", "Ind"]), 0.0, np.nan))
    m["val"] = m["Aobs"].notna()
    return m


def loglik(par, d, nX):
    """Log posterior. Validated and registry-only rows are pre-split in d."""
    nth = 2 * len(PERIODS)
    th = par[:nth]
    if np.any(th <= 1e-6) or np.any(th >= 1 - 1e-6):
        return -np.inf
    b0, bA, bC = par[nth], par[nth + 1], par[nth + 2]
    bX = par[nth + 3:nth + 3 + nX]
    ll = 0.0
    # validated rows: A observed
    lv = b0 + bC * d["Cv"] + d["Xv"] @ bX
    pv = expit(lv + bA * d["Av"])
    ll += np.sum(np.log(np.where(d["Yv"] == 1, pv, 1 - pv) + 1e-300))
    thv = th[d["thidx_v"]]
    ll += np.sum(np.where(d["Av"] == 1, np.log(thv), np.log(1 - thv)))
    # registry-only rows: A latent, marginalised
    lr = b0 + bC * d["Cr"] + d["Xr"] @ bX
    p1, p0 = expit(lr + bA), expit(lr)
    yr = d["Yr"]
    q1 = np.where(yr == 1, p1, 1 - p1)
    q0 = np.where(yr == 1, p0, 1 - p0)
    thr = th[d["thidx_r"]]
    ll += np.sum(np.log(thr * q1 + (1 - thr) * q0 + 1e-300))
    # priors
    ll += -0.5 * ((b0 / 2.5) ** 2 + (bA / 2.5) ** 2 + (bC / 2.5) ** 2)
    ll += -0.5 * np.sum((bX / 1.0) ** 2)
    return ll


def run_chain(d, nX, seed, n_iter, burn):
    """Component-wise Metropolis plus an adaptive joint update on (b0, bA, bC).

    b0 and bA lie on a ridge: A is latent for 97 percent of rows, so the intercept and
    the AUD main effect trade off. A joint proposal scaled by the running covariance of
    those three coordinates moves along the ridge instead of across it.
    """
    rng = np.random.default_rng(seed)
    nth = 2 * len(PERIODS)
    npar = nth + 3 + nX
    bidx = np.array([nth, nth + 1, nth + 2])
    par = np.concatenate([np.full(nth, 0.5), np.array([-4.0, 2.0, 1.0]), np.zeros(nX)])
    par[:nth] += rng.normal(0, 0.05, nth)
    cur = loglik(par, d, nX)
    step = np.concatenate([np.full(nth, 0.05), np.full(3 + nX, 0.2)])
    acc = np.zeros(npar)
    hist, bacc, bscale = [], 0, 0.6
    Cb = np.eye(3) * 0.04
    keep = []
    for it in range(n_iter):
        for k in range(npar):
            prop = par.copy()
            prop[k] += rng.normal(0, step[k])
            lp = loglik(prop, d, nX)
            if np.log(rng.random()) < lp - cur:
                par, cur = prop, lp
                acc[k] += 1
        for _ in range(4):
            prop = par.copy()
            prop[bidx] += rng.multivariate_normal(np.zeros(3), bscale * Cb)
            lp = loglik(prop, d, nX)
            if np.log(rng.random()) < lp - cur:
                par, cur = prop, lp
                bacc += 1
        hist.append(par[bidx].copy())
        if it < burn and (it + 1) % 200 == 0:
            r = acc / 200.0
            step *= np.exp((r - 0.35))
            step = np.clip(step, 1e-4, 3.0)
            acc[:] = 0
            br = bacc / 800.0
            bscale *= np.exp(br - 0.25)
            bscale = float(np.clip(bscale, 0.02, 5.0))
            bacc = 0
            H = np.array(hist[-min(len(hist), 2000):])
            if len(H) > 50:
                Cb = np.cov(H.T) + np.eye(3) * 1e-8
        if it >= burn and (it - burn) % 5 == 0:
            keep.append(par.copy())
    return np.array(keep)


def run_chain_general(logpost, d, nX, seed, n_iter, burn, nth, ncoef, init,
                      block_idx=None, theta_step=0.05, coef_step=0.2):
    """Adaptive component-wise Metropolis with a joint block update, any parameter layout.

    run_chain above hardcodes the binary layout (theta, b0, bA, bC, bX). This version takes
    the layout explicitly so a model with a different number of coefficients can reuse the
    same algorithm. The two must agree: the collapse check in fig6_bayes_strata.py runs the
    binary model through this function and must reproduce run_chain's result.

    logpost(par, d, nX) -> float. nth is the count of theta parameters, ncoef the count of
    coefficients after them, init the starting vector, block_idx the coordinates updated
    jointly with an adapted covariance.
    """
    rng = np.random.default_rng(seed)
    npar = nth + ncoef
    if block_idx is None:
        block_idx = np.arange(nth, npar)
    bidx = np.asarray(block_idx)
    nb = len(bidx)
    par = np.asarray(init, dtype=float).copy()
    par[:nth] += rng.normal(0, 0.05, nth)
    cur = logpost(par, d, nX)
    step = np.concatenate([np.full(nth, theta_step), np.full(ncoef, coef_step)])
    acc = np.zeros(npar)
    hist, bacc, bscale = [], 0, 0.6
    Cb = np.eye(nb) * 0.04
    keep = []
    for it in range(n_iter):
        for k in range(npar):
            prop = par.copy()
            prop[k] += rng.normal(0, step[k])
            lp = logpost(prop, d, nX)
            if np.log(rng.random()) < lp - cur:
                par, cur = prop, lp
                acc[k] += 1
        for _ in range(4):
            prop = par.copy()
            prop[bidx] += rng.multivariate_normal(np.zeros(nb), bscale * Cb)
            lp = logpost(prop, d, nX)
            if np.log(rng.random()) < lp - cur:
                par, cur = prop, lp
                bacc += 1
        hist.append(par[bidx].copy())
        if it < burn and (it + 1) % 200 == 0:
            r = acc / 200.0
            step *= np.exp(r - 0.35)
            step = np.clip(step, 1e-4, 3.0)
            acc[:] = 0
            br = bacc / 800.0
            bscale *= np.exp(br - 0.25)
            bscale = float(np.clip(bscale, 0.02, 5.0))
            bacc = 0
            H = np.array(hist[-min(len(hist), 2000):])
            if len(H) > 50:
                Cb = np.cov(H.T) + np.eye(nb) * 1e-8
        if it >= burn and (it - burn) % 5 == 0:
            keep.append(par.copy())
    return np.array(keep)


def rhat(chains):
    n = chains.shape[1]
    means = chains.mean(1)
    W = chains.var(1, ddof=1).mean()
    B = means.var(ddof=1) * n
    if W <= 0:
        return np.nan
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


def main():
    os.makedirs(OUT, exist_ok=True)
    m = build()
    print("registry with a cirrhosis T0 and a period: %d   validated: %d"
          % (len(m), m["val"].sum()))
    print("  outcome events: %d (%.2f%%)" % (m["Y"].sum(), 100 * m["Y"].mean()))

    Xcols = ["age", "male", "alcld", "lognvis", "per1", "per2"]
    m["lognvis"] = np.log1p(m["nvis"])
    Xm = m[Xcols].to_numpy(float)
    mu, sd = Xm.mean(0), Xm.std(0)
    sd[sd == 0] = 1.0
    Xs = (Xm - mu) / sd
    v = m["val"].to_numpy(bool)
    thidx = m["C"].to_numpy(int) * len(PERIODS) + m["period"].to_numpy(int)
    d = {"Cv": m["C"].to_numpy(float)[v], "Yv": m["Y"].to_numpy(float)[v],
         "Xv": Xs[v], "Av": m["Aobs"].to_numpy(float)[v], "thidx_v": thidx[v],
         "Cr": m["C"].to_numpy(float)[~v], "Yr": m["Y"].to_numpy(float)[~v],
         "Xr": Xs[~v], "thidx_r": thidx[~v], "Xall": Xs}
    nX = len(Xcols)

    print("\nsampling 4 chains, adaptive component-wise Metropolis ...")
    chains = [run_chain(d, nX, SEED + c, 16000, 8000) for c in range(4)]
    S = np.stack(chains)
    nth = 2 * len(PERIODS)
    names = (["theta_C0_" + p for p in PNAMES] + ["theta_C1_" + p for p in PNAMES]
             + ["b0", "bA", "bC"] + ["b_" + c for c in Xcols])
    print("  draws per chain: %d" % S.shape[1])

    rows = []
    print("\n%-20s %8s %8s %8s %8s %6s" % ("parameter", "mean", "2.5%", "97.5%", "R-hat", "ESS"))
    for k, nm in enumerate(names):
        ch = S[:, :, k]
        fl = ch.reshape(-1)
        r = rhat(ch)
        ac = np.corrcoef(fl[:-1], fl[1:])[0, 1]
        ess = len(fl) * (1 - ac) / (1 + ac) if ac < 1 else np.nan
        print("%-20s %8.3f %8.3f %8.3f %8.3f %6.0f"
              % (nm, fl.mean(), np.percentile(fl, 2.5), np.percentile(fl, 97.5), r, ess))
        rows.append({"parameter": nm, "mean": fl.mean(),
                     "ci_low": np.percentile(fl, 2.5), "ci_high": np.percentile(fl, 97.5),
                     "rhat": r, "ess": ess})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "bayes_parameters.csv"), index=False)

    # target quantities, standardised over the observed covariate distribution
    Xall = d["Xall"]
    def std_rates(ch):
        b0, bA, bC = ch[:, nth], ch[:, nth + 1], ch[:, nth + 2]
        bX = ch[:, nth + 3:nth + 3 + nX]
        base = b0[:, None] + bX @ Xall.T
        return (expit(base + (bA + bC)[:, None]).mean(1),
                expit(base + bA[:, None]).mean(1),
                expit(base + bC[:, None]).mean(1),
                expit(base).mean(1))
    per_chain = [std_rates(S[c]) for c in range(S.shape[0])]
    print("\n  derived-quantity convergence, standardised rates")
    for j, nm in enumerate(["P(Y|A=1,C=1)", "P(Y|A=1,C=0)"]):
        ch = np.stack([per_chain[c][j] for c in range(S.shape[0])])
        fl_ = ch.reshape(-1)
        ac = np.corrcoef(fl_[:-1], fl_[1:])[0, 1]
        print("    %-14s R-hat %.3f   ESS %.0f" % (nm, rhat(ch),
              len(fl_) * (1 - ac) / (1 + ac) if ac < 1 else np.nan))
    rrc = np.stack([per_chain[c][0] / per_chain[c][1] for c in range(S.shape[0])])
    ac = np.corrcoef(rrc.reshape(-1)[:-1], rrc.reshape(-1)[1:])[0, 1]
    print("    %-14s R-hat %.3f   ESS %.0f" % ("risk ratio", rhat(rrc),
          rrc.size * (1 - ac) / (1 + ac) if ac < 1 else np.nan))
    p11 = np.concatenate([per_chain[c][0] for c in range(S.shape[0])])
    p10 = np.concatenate([per_chain[c][1] for c in range(S.shape[0])])
    p01 = np.concatenate([per_chain[c][2] for c in range(S.shape[0])])
    p00 = np.concatenate([per_chain[c][3] for c in range(S.shape[0])])
    fl = S.reshape(-1, S.shape[2])
    rr, rd = p11 / p10, p11 - p10
    print("\n" + "=" * 72)
    print("TARGET QUANTITIES, true AUD, standardised over the covariate distribution")
    print("=" * 72)
    for nm, q in (("treated, code present", 100 * p11), ("treated, code absent", 100 * p10),
                  ("risk ratio", rr), ("risk difference, points", 100 * rd)):
        print("  %-26s %7.3f   95%% credible interval %7.3f to %7.3f"
              % (nm, q.mean(), np.percentile(q, 2.5), np.percentile(q, 97.5)))
    print("  P(risk ratio > 1) = %.4f" % (rr > 1).mean())
    print("\n  implied treatment rate if truly AUD-negative: %.2f%% (code absent), %.2f%% (present)"
          % (100 * p00.mean(), 100 * p01.mean()))
    print("  measured in the validation sample: 2 of 239 = 0.84%")
    pd.DataFrame({"p11": p11, "p10": p10, "rr": rr, "rd": rd}).to_csv(
        os.path.join(OUT, "bayes_posterior_draws.csv"), index=False)
    print("\n  wrote %s" % OUT)


if __name__ == "__main__":
    main()
