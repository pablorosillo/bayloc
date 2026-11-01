#!/usr/bin/env python3
# =============================================================================
#  bayloc_synthetic.py
#
#  Runs the inference pipeline on trajectories from the synthetic generators
#  SG1-SG4 and writes a CSV with the SAME COLUMN SCHEMA as the real-data script
#  (bayloc_metrics.py), so the figure notebook can read it with the same
#  helpers.  This produces the synthetic results of the revised manuscript.
#
#  Usage
#  -----
#      python bayloc_synthetic.py <generator> <n_p> <n_per_p> [out_dir]
#
#      generator   SG1 | SG2 | SG3 | SG4
#      n_p         number of points on the noise grid p, spanning [0, 0.5]
#      n_per_p     replicate trajectories per value of p
#      out_dir     optional; default "results_synthetic"
#
#  Each job writes its own file, so one generator per job is the natural unit:
#
#  Settings used in the paper (20 values of p in [0, 0.5], 500 replicates):
#      python bayloc_synthetic.py SG1 20 500
#      python bayloc_synthetic.py SG2 20 500
#      python bayloc_synthetic.py SG3 20 500
#      python bayloc_synthetic.py SG4 20 500
#
#  The inference machinery (kernels, lambda grid, split, metrics, baselines)
#  is imported from bayloc_metrics.py in the same folder; the generators are
#  defined here.
#
#  Runtime is roughly 0.35 s per trajectory with all six kernels and five
#  baselines, i.e. about one hour per generator for the paper's settings.
# =============================================================================

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from tqdm import tqdm

# =============================================================================
#  SYNTHETIC GENERATORS
# =============================================================================
#  Each returns (days, locs, labels) with `days` sorted ordinal day numbers and
#  `locs` integer location codes, i.e. the same structure the real-data loader
#  produces.
#
#  SG1 and SG2 are the generators of the submitted manuscript.  Two
#  discrepancies between the submitted text and the submitted code are resolved
#  here and must also be corrected in the manuscript:
#    * SG2's code used a Pareto shape of a - 1 = 0.25 with a rejection cutoff of
#      100 days, whereas the text quoted an exponent of 1.25 and a cutoff of
#      180 days;
#    * SG1's residence intervals appear as (60, 15) in the caption of Fig. 2 but
#      as (18, 54) in the text accompanying Table II.
#
#  SG3 and SG4 are new, and answer Reviewer 2's request for scenarios that
#  violate the assumptions of the inference framework.
# =============================================================================


def sg1(p=0.15, P=5, dt1=18, dt2=54, n_days=730, rng=None):
    """SG1: commuting / seasonal alternation with FIXED residence intervals.

    The user posts every P days; nominal residence alternates between A (for
    dt1 posts) and B (for dt2 posts); each individual day is flipped to the
    other location with probability p.
    """
    rng = np.random.default_rng(rng)
    days, locs, t, cur = [], [], 0, 0
    while t <= n_days:
        for _ in range(dt1 if cur == 0 else dt2):
            if t > n_days:
                break
            days.append(t)
            locs.append((1 - cur) if rng.random() < p else cur)
            t += P
        cur = 1 - cur
    return np.array(days, np.int64), np.array(locs, np.int64), ["A", "B"]


def sg2(p=0.15, a=1.25, scale=0.2, cutoff=100, n_days=730, rng=None):
    """SG2: bursty activity separated by heavy-tailed inactivity periods.

    Inter-event times are Lomax(a - 1), rescaled and truncated at `cutoff`
    days.  At each post the user switches location with probability p.
    """
    rng = np.random.default_rng(rng)
    days, locs, t = [], [], 0
    cur = int(rng.random() > 0.5)
    while t <= n_days:
        days.append(t)
        locs.append(cur)
        if rng.random() < p:
            cur = 1 - cur
        gap = int(scale * rng.pareto(a - 1) + 1)
        while gap > cutoff:
            gap = int(scale * rng.pareto(a - 1) + 1)
        t += gap
    return np.array(days, np.int64), np.array(locs, np.int64), ["A", "B"]


def sg3(p=0.15, K=4, n_days=730, mean_gap=3.0, rng=None):
    """SG3 (new): more than two locations, with a routine that EVOLVES.

    Violates two assumptions at once.  There are K > 2 candidate locations, and
    the home location changes once part-way through the observation window, so
    the statistics of the early timeline do not extend to the late one.
    Excursions to secondary destinations occur at jittered onsets and with
    random durations, so there is no fixed period to lock onto.
    """
    rng = np.random.default_rng(rng)
    home_a, home_b = 0, 1
    switch = int(n_days * rng.uniform(0.35, 0.65))
    secondaries = np.arange(2, K) if K > 2 else np.array([1])

    exc, t = [], 0.0
    while t < n_days:
        t += rng.uniform(80, 120)
        if t >= n_days:
            break
        L = rng.uniform(7, 30)
        exc.append((t, t + L))
        t += L

    days, locs, t = [], [], 0.0
    while t <= n_days:
        day = int(t)
        loc = home_a if day < switch else home_b
        for (s, e) in exc:
            if s <= day < e:
                loc = int(rng.choice(secondaries))
                break
        if rng.random() < p:
            loc = int(rng.integers(K))
        days.append(day)
        locs.append(loc)
        t += max(1.0, rng.exponential(mean_gap))

    days, locs = np.array(days, np.int64), np.array(locs, np.int64)
    keep = np.concatenate([[True], np.diff(days) > 0])
    return days[keep], locs[keep], [f"L{i}" for i in range(K)]


def sg4(p=0.15, K=4, n_days=730, mean_gap=3.0, median_stay=45.0, jitter=0.6,
        rng=None):
    """SG4 (new): NON-PERIODIC random shifts between locations.

    There is no cycle and no fixed period.  Residence durations are drawn from
    a heavy-tailed (log-normal) distribution and the next location is chosen at
    random among the others, so nothing in the trajectory repeats on a
    schedule.  This is the hardest case for a method that relies on temporal
    regularity, and is the "irregular seasonal behaviour" asked for in review.
    """
    rng = np.random.default_rng(rng)
    days, locs, t = [], [], 0.0
    cur = int(rng.integers(K))
    next_shift = t + max(3.0, rng.lognormal(np.log(median_stay), jitter))
    while t <= n_days:
        if t >= next_shift:
            nxt = int(rng.integers(K - 1))
            cur = nxt if nxt < cur else nxt + 1      # force a genuine change
            next_shift = t + max(3.0, rng.lognormal(np.log(median_stay), jitter))
        loc = int(rng.integers(K)) if rng.random() < p else cur
        days.append(int(t))
        locs.append(loc)
        t += max(1.0, rng.exponential(mean_gap))

    days, locs = np.array(days, np.int64), np.array(locs, np.int64)
    keep = np.concatenate([[True], np.diff(days) > 0])
    return days[keep], locs[keep], [f"L{i}" for i in range(K)]


GENERATORS = {"SG1": sg1, "SG2": sg2, "SG3": sg3, "SG4": sg4}


# =============================================================================
#  EVALUATION
# =============================================================================
#  The inference machinery is imported from the real-data script, so kernels,
#  the lambda grid, the three-way split, the inclusion thresholds and the
#  metrics are identical to those applied to the real data.
# =============================================================================

SYN_KERNELS = [("PL", 0, None), ("EXP", 0, None), ("GAUSS", 0, None),
               ("STREXP", 0, 0.5), ("TPL", 0, 365.0), ("LOGNORM", 0, 2.0)]

SYN_BASELINES = ("MostFrequent", "LastObserved", "NearestObserved",
                 "Markov", "MarkovCT")


def evaluate_synthetic(R, gen_fn, p, seed, scheme="random",
                       kernels=SYN_KERNELS, baselines=SYN_BASELINES):
    """Generate one trajectory and run the full pipeline on it.

    Returns a list of long-format rows, one per method, with the same columns
    the real-data script writes.
    """
    days, locs, labels = gen_fn(p=p, rng=seed)
    n = len(days)
    if n < R.MIN_OBS:
        return []

    rs = np.random.default_rng(seed)
    tr_i, va_i, te_i = R.three_way_split(days, n, scheme, rs)
    if len(tr_i) < 8 or len(va_i) < 2 or len(te_i) < 2:
        return []

    tr_days, tr_locs = days[tr_i], locs[tr_i]
    va_days, va_locs = days[va_i], locs[va_i]
    te_days, te_locs = days[te_i], locs[te_i]

    # candidate set from the TRAINING fold only, as on real data
    seen = np.unique(tr_locs)
    remap = -np.ones(len(labels), dtype=int)
    remap[seen] = np.arange(len(seen))
    n_loc = len(seen)
    if n_loc < R.MIN_LOCATIONS:
        return []
    vm, tm = remap[va_locs] >= 0, remap[te_locs] >= 0
    if vm.sum() < 2 or tm.sum() < 2:
        return []
    tr_l = remap[tr_locs]
    va_d, va_l = va_days[vm], remap[va_locs[vm]]
    te_d, te_l = te_days[tm], remap[te_locs[tm]]

    # p_nominal is the generator's switching parameter; p is what the
    # trajectory actually realised.  They differ substantially, and Table I of
    # the manuscript notes that p means different things across generators, so
    # both are recorded.
    cnt = np.bincount(locs).astype(float)
    span = int(days[-1] - days[0] + 1)
    gm, gmean, g90 = R.gap_geometry(tr_days, te_d)

    # the remaining covariates are computed exactly as `covariates()` does on
    # real data, so the two CSVs share a column schema and the notebook
    # helpers work on either without modification
    ent = -np.sum((cnt / span) * np.log(cnt / span))
    p_empty = (span - n) / span
    if p_empty > 0:
        ent -= p_empty * np.log(p_empty)
    p_loc = cnt / cnt.sum()
    p_loc = p_loc[p_loc > 0]

    base = dict(p_nominal=p,
                p=float(np.mean(locs[1:] != locs[:-1])) if n > 1 else 0.0,
                n_obs=n, Nc=int(len(np.unique(locs))), n_loc_train=n_loc,
                span=span, coverage=n / max(1, span),
                modal_frac=float(cnt.max() / n), scheme=scheme,
                entropy_nats=float(ent),
                entropy_loc=float(-(p_loc * np.log(p_loc)).sum()),
                median_intertweet=float(np.median(np.diff(days))) if n > 1 else np.nan,
                n_unseen_test=int((~tm).sum()),
                gap_median=gm, gap_mean=gmean, gap_p90=g90)

    out = []
    for (name, gamma, shape) in kernels:
        m = R.MemoryModel(tr_days, tr_l, n_loc, name, gamma, shape, None)
        m.set_query(va_d)
        nll = np.array([m.nll(va_l, lam) for lam in R.LAMBDA_GRID])
        k = int(np.argmin(nll))
        lam_star = float(R.LAMBDA_GRID[k])
        logZ = R.log_evidence(nll, R.LAMBDA_GRID)
        post = R.lambda_posterior(nll, R.LAMBDA_GRID)
        psum = R.posterior_summaries(post, R.LAMBDA_GRID)
        if 1 <= k <= len(nll) - 2:
            h = R.LAMBDA_GRID[k + 1] - R.LAMBDA_GRID[k - 1]
            d2 = (nll[k + 1] - 2 * nll[k] + nll[k - 1]) / (h / 2) ** 2
        else:
            d2 = np.nan
        m.set_query(te_d)
        out.append({**base, "method": f"memory-{name}", "gamma": gamma,
                    "shape": shape if shape is not None else np.nan,
                    "window": -1, "posterior_avg": False,
                    "lambda_star": lam_star, "log_evidence": logZ,
                    "min_negll": float(nll[k]), "second_derivative": float(d2),
                    "at_lower_bound": k == 0,
                    "at_upper_bound": k == len(R.LAMBDA_GRID) - 1,
                    "interior": 0 < k < len(R.LAMBDA_GRID) - 1,
                    **psum, **R.evaluate(te_l, m.predictive(lam_star))})

    for bn in baselines:
        try:
            P = R.BASELINES[bn](tr_days, tr_l, te_d, n_loc)
        except Exception:
            continue
        out.append({**base, "method": bn, "gamma": np.nan, "shape": np.nan,
                    "window": -1, "posterior_avg": False,
                    "lambda_star": np.nan, "log_evidence": np.nan,
                    "min_negll": np.nan, "second_derivative": np.nan,
                    "at_lower_bound": False,
                    "at_upper_bound": False, "interior": False,
                    "post_mean": np.nan, "post_sd": np.nan,
                    "post_mass_lower": np.nan, "post_mass_upper": np.nan,
                    **R.evaluate(te_l, P)})
    return out


# =============================================================================
#  MAIN
# =============================================================================


def main():
    if len(sys.argv) < 4:
        sys.exit("usage: python bayloc_synthetic.py <generator> <n_p> "
                 "<n_per_p> [out_dir]\n"
                 f"       generator is one of {list(GENERATORS)}")

    gname = sys.argv[1]
    n_p = int(sys.argv[2])
    n_per_p = int(sys.argv[3])
    out_dir = sys.argv[4] if len(sys.argv) > 4 else "results_synthetic"

    if gname not in GENERATORS:
        sys.exit(f"unknown generator {gname!r}; choose from {list(GENERATORS)}")

    # fixed settings; edit here rather than passing more arguments
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
    P_MIN, P_MAX = 0.0, 0.5
    SCHEME = "random"
    SEED = 0

    sys.path.insert(0, CODE_DIR)
    import bayloc_metrics as R

    os.makedirs(out_dir, exist_ok=True)
    fn = GENERATORS[gname]
    p_grid = np.linspace(P_MIN, P_MAX, n_p)

    print(f"generator  : {gname}")
    print(f"p grid     : {n_p} points in [{P_MIN}, {P_MAX}]")
    print(f"replicates : {n_per_p} per value of p")
    print(f"lambda grid: {len(R.LAMBDA_GRID)} points, "
          f"[{R.LAMBDA_GRID[0]}, {R.LAMBDA_GRID[-1]}]")
    print(f"total trajectories: {n_p * n_per_p}")

    # one file per job, named after the generator and the sweep size, so
    # reruns with different settings do not silently append to each other
    csv_name = os.path.join(
        out_dir, f"synthetic_metrics_{gname}_np{n_p}_nrep{n_per_p}.csv")
    if os.path.isfile(csv_name):
        print(f"WARNING: {csv_name} exists; rows will be appended")
    header_written = os.path.isfile(csv_name)

    buffer, BUF = [], 400
    done = 0
    gi = list(GENERATORS).index(gname)

    for ip, p in tqdm(enumerate(p_grid)):
        for r in range(n_per_p):
            # seed is a deterministic function of (generator, p index,
            # replicate), so the sweep is reproducible and independent jobs
            # never generate the same trajectory
            s_ = SEED + 1000000 * (gi + 1) + 1000 * ip + r
            for row in evaluate_synthetic(R, fn, float(p), s_, scheme=SCHEME):
                buffer.append({**row, "generator": gname,
                               "uid": f"{gname}_{ip}_{r}"})
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{n_p * n_per_p} trajectories", end="\r")
            if len(buffer) >= BUF:
                pd.DataFrame(buffer).to_csv(csv_name, mode="a", index=False,
                                            header=not header_written)
                header_written = True
                buffer = []

    if buffer:
        pd.DataFrame(buffer).to_csv(csv_name, mode="a", index=False,
                                    header=not header_written)
    print(f"\ndone -> {csv_name}")


if __name__ == "__main__":
    main()
