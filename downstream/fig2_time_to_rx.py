#!/usr/bin/env python3
"""Figure 2e/f on a time axis: cumulative incidence of AUD pharmacotherapy after t0.

Why this exists. The registry contrast in the paper is a binary outcome, first recorded
naltrexone, acamprosate or disulfiram at any time on or after t0. Observation time is not
equal across patients (t0 runs from 2015 to 2022, the extract ends on 10 July 2023) and
death competes with treatment in a cirrhosis cohort, so a binary outcome cannot say
whether a stratum's low rate is fewer prescriptions or less time in which to receive one.
This script puts the same contrast on a time axis.

What is kept. The cohort, the strata at t0, the covariates and the outcome concept set
come from fig6_bayes_strata.build(), so they are the ones the paper uses. The measurement
model is unchanged and is still cut: theta | charts is the closed-form Beta of equation
(6), drawn from every validated chart, and the treatment and death data never revise it.

What changes. The Bernoulli outcome model is replaced by two cause-specific
piecewise-exponential hazards, first AUD pharmacotherapy and death, each with its own
baseline on the intervals EDGES and its own coefficients for chart-documented AUD, the
strata and the covariates:

    lambda_c(t | A, S, X) = exp(alpha_c[j(t)] + bA_c A + bS_c[S] + bX_c'X),  c in {rx, death}

A validated patient contributes both hazards at its observed A. For a registry patient A
is marginalised at the U-marginal of its stratum and period, exactly as in
fig6_bayes_strata, so death enters the mixture too: it carries information about A, and
leaving it out would misweight the mixture whenever death depends on A.

Anchor. The registry fit's primary specification (fig6_bayes_strata.ANCHOR_DEFAULT,
"asuncoded" since 30 August) removes from the validated arm the charts whose note window
was centred on the AUD code that places them in their cell, and gives the registry patients
coded after t0 the theta of the never-coded cell of their stratum and period. --anchor
mirrors the fit: the cells are keyed on Uth, the sampling status the fit uses, exactly as
fig6_bayes_strata does, so stage one here is the paper's stage one under either anchor.

New users only. A patient whose first prescription precedes t0 cannot initiate after it,
so the 176 prior users leave the outcome arm. They stay in stage one; theta is the same
draw the paper uses.

Censoring. Death is ascertained to the end of the extract whether or not the patient is
still seen (1,325 deaths fall after the last record of any other kind), so the primary
rule censors at the administrative end. --censor last_record censors at the last dated
record instead, which is the rule under which a prescription could still be recorded.

Estimand. For stratum s and horizon t, the probability of first AUD pharmacotherapy by t
among patients with chart-documented AUD, death competing, standardised over the covariate
distribution of the whole registry:

    F_s(t) = (1/N) sum_i CIF_rx(t | A = 1, S = s, X_i)

formed inside each posterior draw, with ratios and differences taken within the draw as
for rho_s. At a horizon past the last follow-up of most patients the curve is carried by
the baseline hazard of a sparse interval; the curves are reported to 6 years.

Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "downstream")
sys.path.insert(0, "downstream/npj")
import fig6_bayes_me_v2 as M  # noqa: E402
import fig6_bayes_strata as FS  # noqa: E402

D = M.D
OUT = "output/fig2_time"
EDGES = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0])
HORIZONS = [round(x, 2) for x in np.arange(0.25, 6.0001, 0.25)]   # curve grid, years
REPORT = [1, 3, 5]
CRUDE_GRID = [round(x, 3) for x in np.arange(0.02, 6.0001, 0.02)]   # step curves


def followup(m, censor):
    """Add exit time T (years since t0) and status E (1 rx, 2 death, 0 censored)."""
    de = pd.read_parquet(D + "/death.parquet", columns=["person_id", "death_date"])
    de["person_id"] = de["person_id"].astype("int64")
    de["death_date"] = pd.to_datetime(de["death_date"], errors="coerce")
    m["death"] = m["person_id"].map(de.groupby("person_id")["death_date"].min())
    last = []
    for tab, col in (("visit_occurrence", "visit_start_date"),
                     ("condition_occurrence", "condition_start_date"),
                     ("drug_exposure", "drug_exposure_start_date")):
        x = pd.read_parquet(D + "/%s.parquet" % tab, columns=["person_id", col])
        x["person_id"] = x["person_id"].astype("int64")
        x[col] = pd.to_datetime(x[col], errors="coerce")
        last.append(x.groupby("person_id")[col].max().rename(tab))
        del x
    m["last_rec"] = m["person_id"].map(pd.concat(last, axis=1).max(axis=1))
    admin = m["last_rec"].max()
    cens = (m["last_rec"].clip(upper=admin) if censor == "last_record"
            else pd.Series(admin, index=m.index))
    rx = m["first_rx"].where(m["first_rx"] >= m["t0"])
    dth = m["death"].where(m["death"] >= m["t0"])
    end = pd.concat([rx, dth, cens], axis=1).min(axis=1)
    m["E"] = np.where(rx.notna() & (rx <= end), 1,
                      np.where(dth.notna() & (dth <= end), 2, 0))
    # A death on the day of t0 would give zero exposure and an event: half a day at risk.
    m["T"] = np.maximum((end - m["t0"]).dt.days.to_numpy(float), 0.5) / 365.25
    return m, admin


def exposure(T):
    """Time at risk in each interval, and the interval in which each patient exits."""
    lo, hi = EDGES[:-1], EDGES[1:]
    E = np.clip(T[:, None] - lo[None, :], 0, hi - lo)
    j = np.clip(np.searchsorted(EDGES, T, side="right") - 1, 0, len(lo) - 1)
    return E, j


def layout(nS, nX):
    J = len(EDGES) - 1
    P = J + 1 + (nS - 1) + nX            # per cause: baselines, bA, strata, covariates
    return J, P


def make_loglik(nS, nX, nth):
    J, P = layout(nS, nX)

    def unpack(b):
        return b[:J], b[J], b[J + 1:J + nS], b[J + nS:J + nS + nX]

    def cause_ll(b, S, X, Ex, jx, dx, a):
        al, bA, bS, bX = unpack(b)
        lin = S @ bS + X @ bX + bA * a
        return -(Ex @ np.exp(al)) * np.exp(lin) + dx * (al[jx] + lin)

    def loglik(par, d):
        th = par[:nth]
        if np.any(th <= 1e-6) or np.any(th >= 1 - 1e-6):
            return -np.inf
        br, bd = par[nth:nth + P], par[nth + P:nth + 2 * P]
        ll = 0.0
        a = d["Av"]
        ll += np.sum(cause_ll(br, d["Sv"], d["Xv"], d["Ev"], d["jv"], d["drv"], a)
                     + cause_ll(bd, d["Sv"], d["Xv"], d["Ev"], d["jv"], d["ddv"], a))
        l1 = (cause_ll(br, d["Sr"], d["Xr"], d["Er"], d["jr"], d["drr"], 1.0)
              + cause_ll(bd, d["Sr"], d["Xr"], d["Er"], d["jr"], d["ddr"], 1.0))
        l0 = (cause_ll(br, d["Sr"], d["Xr"], d["Er"], d["jr"], d["drr"], 0.0)
              + cause_ll(bd, d["Sr"], d["Xr"], d["Er"], d["jr"], d["ddr"], 0.0))
        thr = (d["Wsp"] @ th)[d["spidx_r"]]
        ll += np.sum(np.logaddexp(np.log(thr) + l1, np.log1p(-thr) + l0))
        for b in (br, bd):
            al, bA, bS, bX = unpack(b)
            ll += -0.5 * np.sum((al / 10.0) ** 2)
            ll += -0.5 * ((bA / 2.5) ** 2 + np.sum((bS / 2.5) ** 2))
            ll += -0.5 * np.sum(bX ** 2)
        return ll

    return loglik, unpack


def run_cut_chain(loglik, d, seed, n_outer, n_inner, burn_inner, nth, ncoef, init,
                  a_cell, n_cell, blocks, warmup_outer):
    """The two-stage cut sampler of fig6_bayes_strata, with one adapted block per cause.

    Stage one redraws theta from its closed form each outer iteration. Stage two runs a
    short Metropolis chain on the hazard coefficients with theta held there: one
    component-wise sweep, then a joint move within each cause, whose covariance is taken
    from the warm-up draws. Warm-up draws are discarded.
    """
    rng = np.random.default_rng(seed)
    par = np.asarray(init, dtype=float).copy()
    step = np.full(ncoef, 0.1)
    Cb = [np.eye(len(b)) * 0.01 for b in blocks]
    keep, buf, acc, tot = [], [], np.zeros(ncoef), 0
    adapt_at = max(warmup_outer - 1, 1)
    for out in range(n_outer):
        par[:nth] = rng.beta(1.0 + a_cell, 1.0 + n_cell - a_cell)
        cur = loglik(par, d)
        for it in range(n_inner):
            for k in range(ncoef):
                prop = par.copy()
                prop[nth + k] += rng.normal(0, step[k])
                lp = loglik(prop, d)
                if np.log(rng.random()) < lp - cur:
                    par, cur = prop, lp
                    acc[k] += 1
            tot += 1
            for b, C in zip(blocks, Cb):
                prop = par.copy()
                prop[nth + b] += rng.multivariate_normal(np.zeros(len(b)),
                                                         (2.38 ** 2 / len(b)) * C)
                lp = loglik(prop, d)
                if np.log(rng.random()) < lp - cur:
                    par, cur = prop, lp
            if it >= burn_inner:
                (keep if out >= warmup_outer else buf).append(par.copy())
        if out == adapt_at:
            r = acc / max(tot, 1)
            step *= np.clip(np.exp((r - 0.35) * 2.0), 0.3, 3.0)
            pool = np.asarray(buf)
            for i, b in enumerate(blocks):
                K = np.cov(pool[:, nth + b].T)
                if np.all(np.isfinite(K)):
                    Cb[i] = K + np.eye(len(b)) * 1e-8
    return np.asarray(keep)


def cif(al_r, lin_r, al_d, lin_d, grid=None):
    """Piecewise-exponential CIF of rx and death, per row of lin.

    Without grid the columns are EDGES[1:]. With grid the intervals are split at every
    grid point, each piece keeping its parent interval's baseline, and the columns are
    the grid points, so a curve can be drawn between the edges without changing the model.
    """
    if grid is None:
        cuts = EDGES
    else:
        cuts = np.union1d(EDGES[EDGES <= max(grid)], np.asarray(grid, float))
        cuts = np.union1d([0.0], cuts)
    par = np.clip(np.searchsorted(EDGES, cuts[:-1], side="right") - 1, 0, len(EDGES) - 2)
    al_r, al_d = al_r[par], al_d[par]
    dt = np.diff(cuts)
    lr = np.exp(al_r[None, :] + lin_r[:, None])
    ld = np.exp(al_d[None, :] + lin_d[:, None])
    L = lr + ld
    q = np.exp(-L * dt[None, :])
    S0 = np.concatenate([np.ones((len(lin_r), 1)), np.cumprod(q, axis=1)[:, :-1]], axis=1)
    Fr = np.cumsum(S0 * lr / L * (1 - q), axis=1)
    Fd = np.cumsum(S0 * ld / L * (1 - q), axis=1)
    if grid is not None:
        col = np.searchsorted(cuts[1:], np.asarray(grid, float))
        return Fr[:, col], Fd[:, col]
    return Fr, Fd


def aj(T, E, horizons):
    """Aalen-Johansen CIF of cause 1 and cause 2 at the horizons, no covariates."""
    o = np.argsort(T, kind="stable")
    T, E = T[o], E[o]
    n = len(T)
    surv, c1, c2, out, k = 1.0, 0.0, 0.0, {}, 0
    for u in np.unique(T):
        while k < len(horizons) and horizons[k] < u:
            out[horizons[k]] = (c1, c2)
            k += 1
        at = T == u
        nrisk = n - np.searchsorted(T, u, side="left")
        d1, d2 = int((at & (E == 1)).sum()), int((at & (E == 2)).sum())
        if nrisk and (d1 or d2):
            c1 += surv * d1 / nrisk
            c2 += surv * d2 / nrisk
            surv *= 1 - (d1 + d2) / nrisk
    while k < len(horizons):
        out[horizons[k]] = (c1, c2)
        k += 1
    return out


def ess_rhat(ch):
    """Bulk ESS and rank-normalised R-hat from ArviZ when available."""
    try:
        import arviz as az
        x = az.convert_to_dataset(ch[None] if ch.ndim == 1 else ch)
        return (float(az.ess(x, method="bulk").to_array().values.squeeze()),
                float(az.ess(x, method="tail").to_array().values.squeeze()),
                float(az.rhat(x).to_array().values.squeeze()))
    except ImportError:
        return np.nan, np.nan, M.rhat(ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--censor", default="admin", choices=["admin", "last_record"])
    ap.add_argument("--anchor", default=getattr(FS, "ANCHOR_DEFAULT", "keep"),
                    choices=["keep", "drop", "asuncoded"],
                    help="mirrors fig6_bayes_strata --anchor; the default is the fit's own")
    ap.add_argument("--outer", type=int, default=800)
    ap.add_argument("--inner", type=int, default=30)
    ap.add_argument("--inner-burn", type=int, default=10)
    ap.add_argument("--warmup-frac", type=float, default=0.25)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--chain", type=int, default=-1,
                    help="run one chain and save it; -1 runs all and summarises")
    ap.add_argument("--summarise", type=int, default=0,
                    help="1 skips sampling and summarises saved per-chain draws")
    ap.add_argument("--n-derived", type=int, default=2000,
                    help="posterior draws, thinned evenly, used for the curves")
    a = ap.parse_args()
    tag = "time_%s" % a.censor
    if a.anchor != getattr(FS, "ANCHOR_DEFAULT", "keep"):
        tag += "_anchor%s" % a.anchor
    os.makedirs(OUT, exist_ok=True)

    m = FS.build(anchor=a.anchor)
    if "Uth" not in m:
        m["Uth"] = m["U"]
    m, admin = followup(m, a.censor)
    nS = int(m["S"].max()) + 1
    Xcols = ["age", "male", "lognvis", "per1", "per2"]
    m["lognvis"] = np.log1p(m["nvis"])
    Xm = m[Xcols].to_numpy(float)
    mu, sd = Xm.mean(0), Xm.std(0)
    sd[sd == 0] = 1.0
    Xs_all = (Xm - mu) / sd
    Sd_all = np.zeros((len(m), nS - 1))
    for s in range(1, nS):
        Sd_all[:, s - 1] = (m["S"].to_numpy() == s).astype(float)

    # Stage one, on every validated chart, exactly as fig6_bayes_strata.
    key = list(zip(m["S"].to_numpy(int), m["period"].to_numpy(int), m["Uth"].to_numpy(int)))
    cells = sorted(set(key))
    cellmap = {c: i for i, c in enumerate(cells)}
    thidx = np.array([cellmap[k] for k in key])
    nth = len(cells)
    v_all = m["val"].to_numpy(bool)
    Aobs = m["Aobs"].to_numpy(float)
    a_cell = np.array([np.nansum(Aobs[v_all & (thidx == i)]) for i in range(nth)])
    n_cell = np.array([int((v_all & (thidx == i)).sum()) for i in range(nth)])
    sp = pd.DataFrame({"S": m["S"].to_numpy(int), "period": m["period"].to_numpy(int),
                       "U": m["Uth"].to_numpy(int)})
    frac = (sp.groupby(["S", "period"])["U"].value_counts(normalize=True)
              .rename("w").reset_index())
    nP = len(M.PERIODS)
    Wsp = np.zeros((nS * nP, nth))
    for _, r in frac.iterrows():
        Wsp[int(r["S"]) * nP + int(r["period"]),
            cellmap[(int(r["S"]), int(r["period"]), int(r["U"]))]] = r["w"]
    spidx_all = m["S"].to_numpy(int) * nP + m["period"].to_numpy(int)

    # Stage two, on new users.
    nu = ~m["rx_before_t0"].to_numpy(bool)
    T, E = m["T"].to_numpy(float), m["E"].to_numpy(int)
    Ex, jx = exposure(T)
    v = v_all & nu
    r = (~v_all) & nu
    d = {"Sv": Sd_all[v], "Xv": Xs_all[v], "Ev": Ex[v], "jv": jx[v],
         "drv": (E[v] == 1).astype(float), "ddv": (E[v] == 2).astype(float), "Av": Aobs[v],
         "Sr": Sd_all[r], "Xr": Xs_all[r], "Er": Ex[r], "jr": jx[r],
         "drr": (E[r] == 1).astype(float), "ddr": (E[r] == 2).astype(float),
         "Wsp": Wsp, "spidx_r": spidx_all[r]}
    nX = len(Xcols)
    J, P = layout(nS, nX)
    print("censor: %s, anchor: %s, administrative end %s" % (a.censor, a.anchor, admin.date()))
    print("theta cells %d, validated charts in stage one %d" % (nth, int(n_cell.sum())))
    print("new users %d of %d (validated %d); events rx %d, death %d"
          % (nu.sum(), len(m), v.sum(), (E[nu] == 1).sum(), (E[nu] == 2).sum()))
    for s in range(nS):
        g = nu & (m["S"].to_numpy() == s)
        print("  S%d %-20s n=%6d  rx %4d  death %5d  person-years %.0f"
              % (s + 1, FS.SNAMES[s], g.sum(), (E[g] == 1).sum(), (E[g] == 2).sum(), T[g].sum()))
    print("  events per interval, rx:    %s" % np.bincount(jx[nu & (E == 1)], minlength=J).tolist())
    print("  events per interval, death: %s" % np.bincount(jx[nu & (E == 2)], minlength=J).tolist())

    ll, unpack = make_loglik(nS, nX, nth)
    ncoef = 2 * P
    # Initial baselines from the crude interval rates.
    py = Ex[nu].sum(0)
    ar = np.log((np.bincount(jx[nu & (E == 1)], minlength=J) + 0.5) / py) - 1.0
    ad = np.log((np.bincount(jx[nu & (E == 2)], minlength=J) + 0.5) / py)
    init = np.concatenate([np.full(nth, 0.5),
                           ar, [1.5], np.zeros(nS - 1), np.zeros(nX),
                           ad, [0.2], np.zeros(nS - 1), np.zeros(nX)])
    blocks = [np.arange(0, P), np.arange(P, 2 * P)]   # coefficient indices, after theta
    n_outer = a.outer
    warm = int(round(a.warmup_frac * n_outer))

    chains_to_run = [] if a.summarise else (range(a.chains) if a.chain < 0 else [a.chain])
    for c in chains_to_run:
        t0 = time.time()
        ch = run_cut_chain(ll, d, M.SEED + c, n_outer, a.inner, a.inner_burn, nth, ncoef,
                           init, a_cell, n_cell, blocks, warm)
        np.save(os.path.join(OUT, "chain_%s_%d.npy" % (tag, c)), ch)
        print("chain %d: %d draws in %.0f s" % (c, len(ch), time.time() - t0), flush=True)
    if a.chain >= 0:
        return

    S = np.stack([np.load(os.path.join(OUT, "chain_%s_%d.npy" % (tag, c)))
                  for c in range(a.chains)])
    names = ([f"theta_S{c[0]+1}_{M.PNAMES[c[1]]}_U{c[2]}" for c in cells]
             + [f"{cz}_{n}" for cz in ("rx", "death")
                for n in (["alpha_%g-%g" % (EDGES[j], EDGES[j + 1]) for j in range(J)]
                          + ["bA"] + ["b_S%d" % (s + 1) for s in range(1, nS)]
                          + ["b_" + x for x in Xcols])])
    rows = []
    print("\n%-26s %8s %8s %8s %7s %7s %7s" % ("parameter", "mean", "2.5%", "97.5%",
                                                "R-hat", "ESSb", "ESSt"))
    for k, nm in enumerate(names):
        ch = S[:, :, k]
        if k < nth:
            # theta is held fixed across the retained draws of one inner chain.
            ch = ch[:, ::(a.inner - a.inner_burn)]
        eb, et, rh = ess_rhat(ch)
        fl = ch.reshape(-1)
        print("%-26s %8.3f %8.3f %8.3f %7.3f %7.0f %7.0f"
              % (nm, fl.mean(), *np.percentile(fl, [2.5, 97.5]), rh, eb, et))
        rows.append({"parameter": nm, "mean": fl.mean(), "ci_low": np.percentile(fl, 2.5),
                     "ci_high": np.percentile(fl, 97.5), "rhat": rh, "ess_bulk": eb,
                     "ess_tail": et})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "params_%s.csv" % tag), index=False)

    # Derived quantities on an even thinning of the pooled draws, chain by chain so that
    # R-hat can be reported for them too.
    per_chain = max(a.n_derived // a.chains, 1)
    pick = np.linspace(0, S.shape[1] - 1, per_chain).astype(int)
    Xs_reg = Xs_all                             # standardise over the whole registry
    Fr = np.zeros((a.chains, per_chain, nS, len(HORIZONS)))
    Fd = np.zeros_like(Fr)
    for c in range(a.chains):
        for i, t in enumerate(pick):
            par = S[c, t]
            al_r, bA_r, bS_r, bX_r = unpack(par[nth:nth + P])
            al_d, bA_d, bS_d, bX_d = unpack(par[nth + P:nth + 2 * P])
            for s in range(nS):
                Ss = np.zeros(nS - 1)
                if s:
                    Ss[s - 1] = 1.0
                lr = Xs_reg @ bX_r + Ss @ bS_r + bA_r
                ld = Xs_reg @ bX_d + Ss @ bS_d + bA_d
                fr, fd = cif(al_r, lr, al_d, ld, HORIZONS)
                Fr[c, i, s] = fr.mean(0)
                Fd[c, i, s] = fd.mean(0)

    def summ(x):
        fl = x.reshape(-1)
        return fl.mean(), np.percentile(fl, 2.5), np.percentile(fl, 97.5), M.rhat(x)

    out = []
    print("\n" + "=" * 78)
    print("CUMULATIVE INCIDENCE AMONG CHART-DOCUMENTED AUD, standardised, death competing")
    print("=" * 78)
    for s in range(nS):
        for k, h in enumerate(HORIZONS):
            for what, F in (("rx", Fr), ("death", Fd)):
                mn, lo, hi, rh = summ(F[:, :, s, k])
                out.append({"quantity": "cif_" + what, "stratum": FS.SNAMES[s], "t": h,
                            "mean": mn, "ci_low": lo, "ci_high": hi, "rhat": rh})
            if h in REPORT:
                o = out[-2]
                print("  S%d %-20s t=%gy  rx %.2f%% (%.2f-%.2f)   death %.1f%%"
                      % (s + 1, FS.SNAMES[s], h, 100 * o["mean"], 100 * o["ci_low"],
                         100 * o["ci_high"], 100 * out[-1]["mean"]))
    print("\ncontrasts against the AUD-code stratum, formed within each draw")
    for s in range(1, nS):
        for k, h in enumerate(HORIZONS):
            rr = Fr[:, :, 0, k] / Fr[:, :, s, k]
            rd = Fr[:, :, 0, k] - Fr[:, :, s, k]
            for q, x in (("rr", rr), ("rd", rd)):
                mn, lo, hi, rh = summ(x)
                out.append({"quantity": q, "stratum": FS.SNAMES[s], "t": h, "mean": mn,
                            "ci_low": lo, "ci_high": hi, "rhat": rh})
            if h in REPORT:
                a_, b_ = out[-2], out[-1]
                print("  S1 vs S%d t=%gy  RR %.2f (%.2f-%.2f)  RD %.2f points (%.2f-%.2f)"
                      % (s + 1, h, a_["mean"], a_["ci_low"], a_["ci_high"],
                         100 * b_["mean"], 100 * b_["ci_low"], 100 * b_["ci_high"]))

    # Posterior predictive check against the crude Aalen-Johansen curve of each stratum:
    # the model-implied registry curve mixes A = 1 and A = 0 by the stratum's own theta.
    print("\ncheck: model-implied vs crude cumulative incidence of rx, new users, by stratum")
    th_mean = S[:, :, :nth].reshape(-1, nth).mean(0)
    par = S.reshape(-1, S.shape[-1]).mean(0)
    al_r, bA_r, bS_r, bX_r = unpack(par[nth:nth + P])
    al_d, bA_d, bS_d, bX_d = unpack(par[nth + P:nth + 2 * P])
    for s in range(nS):
        g = nu & (m["S"].to_numpy() == s)
        thg = (Wsp @ th_mean)[spidx_all[g]]
        base_r = Sd_all[g] @ bS_r + Xs_all[g] @ bX_r
        base_d = Sd_all[g] @ bS_d + Xs_all[g] @ bX_d
        f1, _ = cif(al_r, base_r + bA_r, al_d, base_d + bA_d, HORIZONS)
        f0, _ = cif(al_r, base_r, al_d, base_d, HORIZONS)
        fm = (thg[:, None] * f1 + (1 - thg[:, None]) * f0).mean(0)
        crude = aj(T[g], E[g], HORIZONS)
        print("  S%d " % (s + 1) + "  ".join(
            "%gy %.2f/%.2f%%" % (h, 100 * fm[k], 100 * crude[h][0])
            for k, h in enumerate(HORIZONS) if h in REPORT))
        for k, h in enumerate(HORIZONS):
            out.append({"quantity": "ppc_model_rx", "stratum": FS.SNAMES[s], "t": h,
                        "mean": fm[k], "ci_low": np.nan, "ci_high": np.nan, "rhat": np.nan})
        # The crude curves are step functions, so they are written on a fine grid.
        fine = aj(T[g], E[g], CRUDE_GRID)
        for h in CRUDE_GRID:
            out.append({"quantity": "crude_aj_rx", "stratum": FS.SNAMES[s], "t": h,
                        "mean": fine[h][0], "ci_low": np.nan, "ci_high": np.nan, "rhat": np.nan})
            out.append({"quantity": "crude_aj_death", "stratum": FS.SNAMES[s], "t": h,
                        "mean": fine[h][1], "ci_low": np.nan, "ci_high": np.nan, "rhat": np.nan})
    # Numbers at risk under the curves: new users still under observation at each year.
    for s in range(nS):
        g = nu & (m["S"].to_numpy() == s)
        for t in range(0, 7):
            out.append({"quantity": "at_risk", "stratum": FS.SNAMES[s], "t": t,
                        "mean": int((T[g] >= t).sum()), "ci_low": np.nan, "ci_high": np.nan,
                        "rhat": np.nan})
    pd.DataFrame(out).to_csv(os.path.join(OUT, "cif_%s.csv" % tag), index=False)
    np.save(os.path.join(OUT, "derived_rx_%s.npy" % tag), Fr)
    json.dump({"args": vars(a), "n_cell": n_cell.tolist(), "a_cell": a_cell.tolist(),
               "edges": EDGES.tolist(), "horizons": HORIZONS,
               "admin_end": str(admin.date()), "n_new_users": int(nu.sum()),
               "n_validated_new_users": int(v.sum()), "seed_base": int(M.SEED)},
              open(os.path.join(OUT, "fit_%s.json" % tag), "w"), indent=1)
    print("\nwrote %s/{params,cif,fit}_%s.*" % (OUT, tag))


if __name__ == "__main__":
    main()
