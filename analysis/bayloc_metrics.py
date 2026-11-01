#!/usr/bin/env python3
# =============================================================================
#  bayloc_metrics.py
#
#  Metric computation for "Inferring missing locations in human mobility data
#  using memory effects" -- REVISION 1.
#
#  This is the script that produced every real-data result of the revised
#  manuscript (country level and NUTS3).  It replaces the analysis script of
#  the submitted version and keeps its command-line interface:
#
#      python bayloc_metrics.py <n_file> <delta> [chunk_index]
#
#      n_file       e.g. 'a0' -- first character selects the condition file,
#                   second character selects the chunk (as before)
#      delta        number of timelines per chunk
#      chunk_index  optional; overrides n_file[1] so that a single condition
#                   can be split across an arbitrary number of nodes
#
#  Example (unchanged from the original usage):
#      python bayloc_metrics.py a0 10000
#      python bayloc_metrics.py a1 10000
#
#  Example (more nodes than the single digit of n_file allows):
#      for i in $(seq 0 49); do
#          sbatch --wrap "python bayloc_metrics.py a0 2000 $i"
#      done
#
# -----------------------------------------------------------------------------
#  WHAT IS NEW WITH RESPECT TO THE SUBMITTED SCRIPT
# -----------------------------------------------------------------------------
#
#  Bug fixes (these change published numbers -- see comments at each site):
#    B1  log-evidence is computed with logsumexp instead of np.exp(-NLL).
#        exp(-x) underflows to exactly 0.0 for x > 745.13 in float64, and
#        Fig. 3a of the submitted manuscript shows [-log L]_min reaching ~1000.
#        Those users returned -inf and were then silently dropped downstream,
#        so Table II was computed on a NON-RANDOM subsample that excluded the
#        longest, least predictable timelines.
#    B2  `forced_train` is removed.  It re-appended rare-location observations
#        that were already in `train`, doubling their kernel weight.
#    B3  the candidate location set and all counts are built from the TRAINING
#        fold only.  The submitted code built `all_locations` and
#        `location_counts` from the full timeline, including held-out days.
#    B4  the lambda grid is logarithmic and includes the exact boundaries, so
#        that "lambda* is interior" is well defined and resolvable below 0.29
#        (the submitted linear grid had spacing 0.285).
#
#  New quantities (each answers a specific referee request):
#    N1  three-way TRAIN / VALIDATION / TEST split.  lambda is fitted on VAL,
#        all reported metrics come from TEST.            [R1.2, R4.2]
#    N2  baselines: MostFrequent, LastObserved, NearestObserved, Markov,
#        MarkovCT (gap-aware).                           [R1.3, R2.4, R3, R4.1]
#    N3  stratification covariates: p, n_obs, Nc, span, coverage, modal
#        fraction, entropy.                              [R2.3, R4.8]
#    N4  gap geometry: distance from each evaluated day to its nearest
#        training observation.                           [R2.2, R4.3]
#    N5  extra kernels: stretched exponential, truncated power law,
#        log-normal.                                     [R4.6]
#    N6  posterior-averaged prediction over lambda.      [R2.6]
#    N7  temporal-window ablation.                       [R2.2]
#    N8  block hold-out (contiguous calendar gaps).      [R2.2, R4.3]
#    N9  macro-AUROC, balanced accuracy, log-loss, Brier, in addition to
#        micro-AUROC and T.                              [R2.3, R3, R4.8]
#    N10 posterior summaries of lambda (mean, s.d., and the mass sitting on
#        each boundary).  These say HOW WELL lambda is identified, which the
#        submitted script only probed locally via `second_derivative`, and
#        they give a graded alternative to the binary "saved trajectories"
#        count of Fig. 3c.                               [R2.6, and Fig. 3c]
#
#  Output format change: the submitted script wrote ONE WIDE ROW per timeline
#  with 6 blocks of columns.  With 9 kernels x 3 windows x 2 split geometries
#  plus 5 baselines that would be several hundred columns.  This script writes
#  LONG format: one row per (timeline, method, configuration).  The companion
#  notebook pivots it back to wide where needed.
#
#  Also: the full fpr/tpr arrays are no longer stored.  They were written into
#  CSV cells as multi-line strings, which is what forced the fragile
#  `'nan' in str(...)` filtering in the notebook.  Only scalar AUROC is kept.
# =============================================================================

from __future__ import annotations

import ast
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# =============================================================================
#  CONFIGURATION
# =============================================================================

# --- paths -------------------------------------------------------------------
# The Twitter timelines cannot be shared (data protection). Point these at your
# own files with the environment variables BAYLOC_TIMELINES and BAYLOC_OUTPUT.
# Input format: one timeline per line, a Python list of ('YYYY-MM-DD', label).
TIMELINE_TEMPLATE = os.environ.get(
    "BAYLOC_TIMELINES", "data/timelines_country_condition_{cond}.txt")
OUTPUT_DIR = os.environ.get("BAYLOC_OUTPUT", "results")

# --- lambda grid (B4) --------------------------------------------------------
# Logarithmic spacing across four orders of magnitude, with the exact lower and
# upper boundaries included so that interior/boundary status is exact.
# LAMBDA_0 = 0 is the exact "most frequent location" limit; LAMBDA_INF is the
# "nearest observed day" limit (Proposition 1 of the revised Methods).
LAMBDA_0 = 0.0
LAMBDA_INF = 200.0  # value used for every result in the paper
N_LAMBDA = 1000  # interior points; total grid size is N_LAMBDA + 2


def build_lambda_grid() -> np.ndarray:
    """Exact boundaries plus a log-spaced interior."""
    interior = np.geomspace(1e-3, LAMBDA_INF * 0.999, N_LAMBDA)
    return np.concatenate([[LAMBDA_0], interior, [LAMBDA_INF]])


LAMBDA_GRID = build_lambda_grid()

# --- split -------------------------------------------------------------------
FRAC_VAL = 0.25   # selects lambda
FRAC_TEST = 0.25  # reported once, never used for fitting
# remaining 50% is TRAIN, matching the training fraction of the submitted paper

SPLIT_SCHEMES = ["random", "block"]  # N8
BLOCK_DAYS = 14                      # contiguous calendar block length

# --- kernels -----------------------------------------------------------------
# (name, gamma, shape)
#   gamma = 1 multiplies the kernel by the location's visitation count
#           (the "preferential attachment" variant reported in the SI)
#   gamma = 0 is the plain kernel of Eqs. (3)-(5)
#   shape   is the second parameter of the two-parameter families (N5)
MODELS = [
    ("PL",      1, None),   # model 1 in the submitted script
    ("EXP",     1, None),   # model 2
    ("GAUSS",   1, None),   # model 3
    ("PL",      0, None),   # model 4  <- main text
    ("EXP",     0, None),   # model 5  <- main text
    ("GAUSS",   0, None),   # model 6  <- main text
    ("STREXP",  0, 0.5),    # new, R4.6
    ("TPL",     0, 365.0),  # new, R4.6
    ("LOGNORM", 0, 2.0),    # new, R4.6
]

# --- window ablation (N7) ----------------------------------------------------
# None means "use the whole timeline", as in the submitted manuscript.
WINDOWS = [None, 15, 30, 90, 120]

# --- filters -----------------------------------------------------------------
MIN_OBS = 30       # fewer points cannot identify a temporal scale by grid search
MIN_LOCATIONS = 2  # a single-location timeline is trivially imputed

# --- misc --------------------------------------------------------------------
SEED = 20260101
EPS = 1e-12

# Floor on individual log-probabilities inside the composite likelihood.
# Matches the `-100` penalty used in the submitted script.
LOGP_FLOOR = -100.0


# =============================================================================
#  DATA LOADING
# =============================================================================

def read_timelines_from_file(file_path, limit=None):
    """Read timelines and collapse each day to its modal location.

    Identical collapsing rule to the submitted script: for each calendar day,
    take the most frequent location among that day's posts, breaking ties by
    the LAST occurrence.  Timelines containing an empty location label are
    discarded, as before.

    Returns a list of (days, locs, labels) tuples, where `days` are ordinal
    day numbers (sorted), `locs` are integer codes and `labels` maps codes
    back to country strings.
    """
    out = []
    with open(file_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw = ast.literal_eval(line)
            except Exception:
                continue

            day_locs = defaultdict(list)
            for d, loc in raw:
                day_locs[datetime.strptime(d, "%Y-%m-%d").toordinal()].append(loc)

            pairs = []
            for day, locs in day_locs.items():
                cnt = Counter(locs)
                mx = max(cnt.values())
                for loc in reversed(locs):       # tie -> last occurrence
                    if cnt[loc] == mx:
                        pairs.append((day, loc))
                        break

            if any(l == "" for _, l in pairs):
                continue

            pairs.sort()
            labels = sorted({l for _, l in pairs})
            code = {l: i for i, l in enumerate(labels)}
            days = np.array([d for d, _ in pairs], dtype=np.int64)
            locs = np.array([code[l] for _, l in pairs], dtype=np.int64)
            out.append((days, locs, labels))

            if limit is not None and len(out) >= limit:
                break
    return out


# =============================================================================
#  KERNELS
# =============================================================================
#  All kernels are evaluated in LOG space.  This is what removes the
#  overflow/underflow that affected the submitted implementation, where
#  exp(-lambda*Delta) silently returned 0.0 for large arguments.
#
#  Every kernel satisfies f_0 == 1 and f_lambda(D)/f_lambda(D_min) -> 0 as
#  lambda -> infinity, which is what makes MostFrequent and NearestObserved
#  the two boundary points of the model (Proposition 1 of the Methods).
# =============================================================================

def log_kernel(name, D, logD, lam, shape=None):
    """log f_lambda(Delta) for a matrix of temporal distances.

    D and logD are precomputed once per timeline and reused across the whole
    lambda grid, which is the main reason this script is fast enough to add
    baselines, windows and extra kernels on top of the original workload.
    """
    if name == "PL":                       # Delta^(-lambda)
        return -lam * logD
    if name == "EXP":                      # exp(-lambda Delta)
        return -lam * D
    if name == "GAUSS":                    # exp(-lambda Delta^2 / 2)
        return -0.5 * lam * D * D
    if name == "STREXP":                   # exp(-lambda Delta^beta)
        return -lam * np.power(D, shape)
    if name == "TPL":                      # Delta^(-lambda) exp(-Delta/Delta_c)
        return -lam * logD - D / shape
    if name == "LOGNORM":                  # Delta^(-lambda) exp(-(logD)^2/2s^2)
        return -lam * logD - 0.5 * (logD / shape) ** 2
    raise ValueError(f"unknown kernel {name!r}")


# =============================================================================
#  CORE MODEL
# =============================================================================

class MemoryModel:
    """Memory-decay predictive model for one individual, one kernel.

    Fitted on `train`; evaluated on arbitrary query days.  The geometry
    (|t - s| for every query/train pair) is computed once in `set_query` and
    reused for every value of lambda.
    """

    def __init__(self, tr_days, tr_locs, n_loc, name, gamma=0, shape=None,
                 window=None):
        self.name, self.gamma, self.shape, self.window = name, gamma, shape, window
        self.tr_days = tr_days.astype(np.float64)
        self.tr_locs = tr_locs
        self.n_loc = n_loc

        # one-hot S[s, l] = 1 iff training day s is at location l
        self.S = np.zeros((len(tr_days), n_loc))
        self.S[np.arange(len(tr_days)), tr_locs] = 1.0

        # (B3) counts from the TRAINING fold only.  In the submitted code these
        # came from the full timeline, so the gamma=1 variants used held-out
        # days in their weights.
        counts = np.bincount(tr_locs, minlength=n_loc).astype(float)
        self.log_counts = np.log(np.maximum(counts, EPS)) if gamma else None

        self._q = None

    def set_query(self, q_days):
        """Precompute the distance geometry for a set of query days."""
        q = np.asarray(q_days, dtype=np.float64)
        D = np.abs(q[:, None] - self.tr_days[None, :])
        valid = D > 0                                 # exclude the day itself
        if self.window is not None:                   # (N7) window ablation
            valid &= D <= self.window
        Dsafe = np.where(valid, D, 1.0)
        self._q = (Dsafe, np.log(Dsafe), valid, len(q))

    def log_predictive(self, lam):
        """log P[x(t) = l | training data, lambda], normalised over locations."""
        D, logD, valid, nq = self._q
        lf = log_kernel(self.name, D, logD, lam, self.shape)
        lf = np.where(valid, lf, -np.inf)

        # stable per-row max shift, then a single matrix product to group the
        # contributions by location
        M = np.where(valid, lf, -np.inf).max(axis=1, keepdims=True)
        alive = np.isfinite(M[:, 0])
        Ms = np.where(alive[:, None], M, 0.0)
        E = np.exp(lf - Ms)
        E[~np.isfinite(E)] = 0.0
        W = E @ self.S                                # (n_query, n_loc)

        with np.errstate(divide="ignore"):
            lw = np.where(W > 0, np.log(np.where(W > 0, W, 1.0)) + Ms, -np.inf)
        if self.gamma:                                # preferential variant
            lw = lw + self.log_counts[None, :]
        lw[~alive, :] = -np.inf

        norm = logsumexp(lw, axis=1, keepdims=True)
        dead = ~np.isfinite(norm[:, 0])
        with np.errstate(invalid="ignore"):
            lp = lw - norm
        if dead.any():                                # no support at all
            lp[dead, :] = -np.log(self.n_loc)
        return lp

    def predictive(self, lam):
        return np.exp(self.log_predictive(lam))

    def nll(self, y, lam):
        """Negative log COMPOSITE likelihood of a fold (Eq. 6 of the paper).

        Individual log-probabilities are floored at LOGP_FLOOR.  The submitted
        script did the same thing (`log_likelihood += -100` when the emission
        probability was zero), and the floor is needed here for the same
        reason: with a finite window, or for a location with no nearby
        training day, the predictive can be exactly zero at the true location,
        which would make the whole NLL curve +inf and leave lambda unidentified.
        """
        lp = self.log_predictive(lam)
        vals = lp[np.arange(len(y)), y]
        return float(-np.maximum(vals, LOGP_FLOOR).sum())


def log_evidence(nll_grid, lam_grid):
    """log marginal composite likelihood, computed stably.  (B1)

    Replaces:
        likelihoods = np.exp(-grid_ll)
        Z = (1/(lmax-lmin)) * np.trapezoid(likelihoods, lambda_values)
        log_Z = np.log(Z) if Z > 0 else -np.inf

    which returns -inf for every user whose [-log L] exceeds 745.
    """
    dx = np.diff(lam_grid)
    q = np.zeros_like(lam_grid)
    q[:-1] += dx / 2.0
    q[1:] += dx / 2.0
    q = np.maximum(q, np.finfo(float).tiny)
    val = logsumexp(-nll_grid + np.log(q)) - np.log(lam_grid[-1] - lam_grid[0])
    return float(val) if np.isfinite(val) else np.nan

def log_evidence_loguniform(nll_grid, lam_grid):
    """log marginal composite likelihood under a log-uniform prior on the
    interior of the grid (lambda > 0). Trapezoidal rule in log(lambda)."""
    keep = lam_grid > 0                      # exclude the exact lambda_0 = 0
    x = np.log(lam_grid[keep])
    nll = nll_grid[keep]
    dx = np.diff(x)
    q = np.zeros_like(x)
    q[:-1] += dx / 2.0
    q[1:] += dx / 2.0
    q = np.maximum(q, np.finfo(float).tiny)
    val = logsumexp(-nll + np.log(q)) - np.log(x[-1] - x[0])
    return float(val) if np.isfinite(val) else np.nan


def lambda_posterior(nll_grid, lam_grid):
    """Normalised p(lambda | validation fold) on the grid.  (N6)"""
    dx = np.diff(lam_grid)
    q = np.zeros_like(lam_grid)
    q[:-1] += dx / 2.0
    q[1:] += dx / 2.0
    q = np.maximum(q, np.finfo(float).tiny)
    lp = -nll_grid + np.log(q)
    tot = logsumexp(lp)
    if not np.isfinite(tot):          # degenerate curve: fall back to uniform
        return np.full_like(lam_grid, 1.0 / len(lam_grid))
    return np.exp(lp - tot)


def posterior_summaries(post, lam_grid):
    """How well is lambda identified, and how much mass sits on the limits?

    The submitted script reported only `second_derivative`, the curvature of
    the NLL at its minimum.  That is a LOCAL width estimate and it is least
    reliable exactly where it matters: on a flat surface, or when the optimum
    sits on a boundary (where the central difference is undefined and the
    submitted code returned NaN).

    These four numbers are global, and they cost nothing because the posterior
    is already computed for the evidence integral:

      post_mean, post_sd   the location and width of p(lambda | validation).
                           post_sd separates "lambda* = 0.3 and the data say
                           so" from "lambda* = 0.3 but anything in [0.01, 5]
                           fits equally well".  These two users are otherwise
                           indistinguishable in the output.

      post_mass_lower      posterior mass on lambda = lambda_0, i.e. on the
                           MostFrequent limit.
      post_mass_upper      posterior mass on lambda = lambda_inf, i.e. on the
                           NearestObserved limit.

    The two mass terms give a GRADED version of the "saved trajectories"
    statistic of Fig. 3c, which is currently a hard binary on the argmin: a
    user whose posterior puts 49% on a boundary counts identically to one at
    0.1%.  Reporting both lets us check that the binary count is robust, which
    matters because the lambda grid has changed (B4).
    """
    m1 = float(np.sum(post * lam_grid))
    m2 = float(np.sum(post * lam_grid ** 2))
    return dict(
        post_mean=m1,
        post_sd=float(np.sqrt(max(0.0, m2 - m1 ** 2))),
        post_mass_lower=float(post[0]),
        post_mass_upper=float(post[-1]),
    )


# =============================================================================
#  BASELINES  (N2)
# =============================================================================
#  Every baseline returns a row-stochastic (n_query, n_loc) array, so that all
#  metrics are computed identically for baselines and for the proposed model.
#
#  MostFrequent and NearestObserved are NOT independent competitors: they are
#  the lambda -> 0 and lambda -> infinity limits of the memory model itself.
#  They are reported with that annotation.
# =============================================================================

def bl_most_frequent(tr_days, tr_locs, q_days, n_loc):
    """Overall most-visited location.  Equivalent to the model at lambda = 0."""
    c = np.bincount(tr_locs, minlength=n_loc).astype(float) + 1.0   # add-one
    p = c / c.sum()
    return np.repeat(p[None, :], len(q_days), axis=0)


def bl_last_observed(tr_days, tr_locs, q_days, n_loc, smooth=0.02):
    """Most recent observation strictly BEFORE the query day (persistence).

    This is the causal baseline named by R1 ("predict the last known
    location") and R4.  Note that for an imputation task the bidirectional
    version below is the fairer comparison; both are reported.
    """
    out = np.full((len(q_days), n_loc), smooth / n_loc)
    pos = np.searchsorted(tr_days, q_days, side="left") - 1
    c = np.bincount(tr_locs, minlength=n_loc).astype(float) + 1.0
    marg = c / c.sum()
    for i, j in enumerate(pos):
        if j < 0:
            out[i] = marg
        else:
            out[i, tr_locs[j]] += 1.0 - smooth
    return out / out.sum(1, keepdims=True)


def bl_nearest_observed(tr_days, tr_locs, q_days, n_loc, smooth=0.02):
    """Nearest observation in EITHER direction; ties split evenly.

    Equivalent to the model as lambda -> infinity.  This is the appropriate
    persistence baseline for imputation (as opposed to forecasting), because
    evidence on both sides of a gap is available.
    """
    out = np.full((len(q_days), n_loc), smooth / n_loc)
    D = np.abs(tr_days[None, :] - np.asarray(q_days)[:, None])
    for i in range(len(q_days)):
        m = D[i].min()
        hit = tr_locs[D[i] == m]
        for l in hit:
            out[i, l] += (1.0 - smooth) / len(hit)
    return out / out.sum(1, keepdims=True)


def bl_markov(tr_days, tr_locs, q_days, n_loc, alpha=1.0):
    """First-order Markov chain on the sequence of OBSERVATIONS.

    NOTE: this baseline ignores the size of the gap -- it applies a one-step
    transition whether the interval is one day or one year.  It is reported
    because R2.4 and R4.1 ask for it, but `bl_markov_ct` below is the fairer
    comparison on irregularly sampled data.
    """
    A = np.full((n_loc, n_loc), alpha)
    for a, b in zip(tr_locs[:-1], tr_locs[1:]):
        A[a, b] += 1.0
    A /= A.sum(1, keepdims=True)
    c = np.bincount(tr_locs, minlength=n_loc).astype(float) + 1.0
    marg = c / c.sum()

    out = np.empty((len(q_days), n_loc))
    pos = np.searchsorted(tr_days, q_days, side="left") - 1
    for i, j in enumerate(pos):
        out[i] = marg if j < 0 else A[tr_locs[j]]
    return out


def bl_markov_ct(tr_days, tr_locs, q_days, n_loc, alpha=1.0):
    """Gap-aware (continuous-time) Markov chain: P(gap) = A^(gap/mean_gap).

    Conditions on the nearest training observation on each side and combines
    the two directions by a normalised product, which is the exact posterior
    for a reversible chain.  This makes the Markov comparison fair for
    irregular sampling and prevents us from beating a strawman.
    """
    A = np.full((n_loc, n_loc), alpha)
    gaps = []
    for (da, a), (db, b) in zip(zip(tr_days[:-1], tr_locs[:-1]),
                                zip(tr_days[1:], tr_locs[1:])):
        A[a, b] += 1.0
        gaps.append(max(1, int(db - da)))
    A /= A.sum(1, keepdims=True)
    mean_gap = float(np.mean(gaps)) if gaps else 1.0

    evals, evecs = np.linalg.eig(A)
    inv = np.linalg.pinv(evecs)

    def power(t):
        t = max(t, 1e-6) / mean_gap
        P = np.real((evecs * np.power(evals.astype(complex), t)) @ inv)
        P = np.clip(P, 1e-9, None)
        return P / P.sum(1, keepdims=True)

    c = np.bincount(tr_locs, minlength=n_loc).astype(float) + 1.0
    marg = c / c.sum()

    out = np.empty((len(q_days), n_loc))
    pos = np.searchsorted(tr_days, q_days, side="left")
    cache = {}
    for i, t in enumerate(q_days):
        j, k = pos[i] - 1, pos[i]
        parts = []
        if j >= 0:
            d = int(t - tr_days[j])
            if d not in cache:
                cache[d] = power(d)
            parts.append(cache[d][tr_locs[j]])
        if k < len(tr_days):
            d = int(tr_days[k] - t)
            if d not in cache:
                cache[d] = power(d)
            parts.append(cache[d][tr_locs[k]])
        if not parts:
            out[i] = marg
        else:
            r = np.prod(parts, axis=0)
            out[i] = r / r.sum() if r.sum() > 0 else marg
    return out


BASELINES = {
    "MostFrequent":    bl_most_frequent,     # == memory model at lambda = 0
    "LastObserved":    bl_last_observed,
    "NearestObserved": bl_nearest_observed,  # == memory model at lambda -> inf
    "Markov":          bl_markov,
    "MarkovCT":        bl_markov_ct,
}


# =============================================================================
#  SPLITS
# =============================================================================

def three_way_split(days, n, scheme, rng):
    """Disjoint TRAIN / VALIDATION / TEST index sets.  (N1)

    TRAIN builds the kernel, VALIDATION selects lambda, TEST is touched once.
    No quantity reported as a result is computed on data used to fit anything.

    scheme = "random": individual observed days removed at random.  This is the
        interpolation regime of the submitted analysis (whose split was the
        deterministic `known_items[::3]`).
    scheme = "block": whole contiguous calendar blocks of BLOCK_DAYS removed,
        so held-out days are not adjacent to training days.  This is realistic
        gap filling.  (N8)
    """
    idx = np.arange(n)
    if scheme == "random":
        perm = rng.permutation(n)
        n_val = max(2, int(round(FRAC_VAL * n)))
        n_test = max(2, int(round(FRAC_TEST * n)))
        test = np.sort(perm[:n_test])
        val = np.sort(perm[n_test:n_test + n_val])
        train = np.sort(perm[n_test + n_val:])
        return train, val, test

    # block scheme -- assign whole calendar blocks to one of the three folds
    d0 = int(days[0])
    block_id = (days - d0) // BLOCK_DAYS
    uniq = np.unique(block_id)
    rng.shuffle(uniq)

    target_test = FRAC_TEST * n
    target_val = FRAC_VAL * n
    test_b, val_b, got_t, got_v = [], [], 0, 0
    for b in uniq:
        size = int((block_id == b).sum())
        if got_t < target_test:
            test_b.append(b); got_t += size
        elif got_v < target_val:
            val_b.append(b); got_v += size
    test = idx[np.isin(block_id, test_b)]
    val = idx[np.isin(block_id, val_b)]
    train = idx[~np.isin(block_id, test_b + val_b)]
    return train, val, test


def gap_geometry(tr_days, q_days):
    """Distance from each evaluated day to its nearest training day.  (N4)

    This single statistic settles the question raised by R2.2 and R4.3, namely
    whether scattered hold-out days sit "immediately next to training points".
    Because the corpus is sparse, they often do not: removing every third
    OBSERVATION from a user who posts every 40 days still leaves ~80-day
    calendar gaps.  Reporting the distribution makes the argument empirical
    rather than rhetorical.
    """
    if len(tr_days) == 0 or len(q_days) == 0:
        return np.nan, np.nan, np.nan
    D = np.abs(np.asarray(q_days)[:, None] - tr_days[None, :]).min(axis=1)
    return float(np.median(D)), float(D.mean()), float(np.percentile(D, 90))


# =============================================================================
#  METRICS  (N9)
# =============================================================================

def auroc(y, P, average):
    """One-vs-rest AUROC, robust to locations absent from the fold.

    The submitted script reported MICRO-averaged AUROC only.  For a user who
    spends 90% of their days in one country, micro-averaging is dominated by
    that country, which is precisely the concern of R2.3, R3 and R4.  Macro
    averaging weights every candidate location equally and is far more
    sensitive to the rare "away" days that actually matter.
    """
    K = P.shape[1]
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), y] = 1.0
    cols = [k for k in range(K) if 0 < Y[:, k].sum() < len(y)]
    if not cols:
        return np.nan
    try:
        if average == "micro":
            return float(roc_auc_score(Y[:, cols].ravel(), P[:, cols].ravel()))
        return float(np.mean([roc_auc_score(Y[:, k], P[:, k]) for k in cols]))
    except ValueError:
        return np.nan


def evaluate(y, P):
    """Full metric bundle for one fold."""
    P = np.clip(np.asarray(P, float), EPS, None)
    P = P / P.sum(1, keepdims=True)
    n = len(y)
    if n == 0:
        return {}
    pred = P.argmax(1)

    present = np.unique(y)
    bacc = float(np.mean([np.mean(pred[y == c] == c) for c in present]))

    Y = np.zeros_like(P)
    Y[np.arange(n), y] = 1.0

    return dict(
        T=float(np.mean(pred == y)),                 # as in the submitted paper
        auroc_micro=auroc(y, P, "micro"),            # as in the submitted paper
        auroc_macro=auroc(y, P, "macro"),            # new
        balanced_acc=bacc,                           # new
        logloss=float(-np.mean(np.log(P[np.arange(n), y]))),   # new
        brier=float(np.mean(np.sum((P - Y) ** 2, axis=1))),    # new
        n_eval=int(n),
    )


# =============================================================================
#  PER-TIMELINE COVARIATES  (N3)
# =============================================================================

def covariates(days, locs, labels):
    """Stratification variables, so that pooled results can be broken down.

    R2.3 and R4.8 both ask for this: the pooled distribution is dominated by
    near-stationary users, and a single headline AUROC hides whether
    performance degrades for mobile individuals.
    """
    n = len(days)
    span = int(days[-1] - days[0] + 1)
    counts = np.bincount(locs).astype(float)
    p_loc = counts / counts.sum()
    p_loc = p_loc[p_loc > 0]

    # entropy over the full calendar, including unobserved days as their own
    # category -- same convention as the submitted script
    n_days_total = span
    ent = -np.sum((counts / n_days_total) * np.log(counts / n_days_total))
    p_empty = (n_days_total - n) / n_days_total
    if p_empty > 0:
        ent -= p_empty * np.log(p_empty)

    return dict(
        n_obs=n,
        Nc=len(labels),                              # distinct countries (R4.8)
        span=span,
        coverage=n / max(1, span),                   # observed fraction of days
        p=float(np.mean(locs[1:] != locs[:-1])) if n > 1 else 0.0,
        modal_frac=float(counts.max() / n),          # how stationary the user is
        entropy_nats=float(ent),
        entropy_loc=float(-(p_loc * np.log(p_loc)).sum()),
        median_intertweet=float(np.median(np.diff(days))) if n > 1 else np.nan,
    )


# =============================================================================
#  ANALYSIS OF ONE TIMELINE
# =============================================================================

def analyse_timeline(uid, days, locs, labels, rng):
    """Returns a list of long-format rows for one individual."""
    rows = []
    cov = covariates(days, locs, labels)
    n = len(days)

    for scheme in SPLIT_SCHEMES:
        tr_i, va_i, te_i = three_way_split(days, n, scheme, rng)
        if len(tr_i) < 8 or len(va_i) < 2 or len(te_i) < 2:
            continue

        tr_days, tr_locs = days[tr_i], locs[tr_i]
        va_days, va_locs = days[va_i], locs[va_i]
        te_days, te_locs = days[te_i], locs[te_i]

        # (B3) candidate set from TRAINING only.  Test days at a location never
        # seen in training are impossible by construction; we count them so the
        # effect is visible rather than hidden.
        seen = np.unique(tr_locs)
        remap = -np.ones(len(labels), dtype=int)
        remap[seen] = np.arange(len(seen))
        n_loc = len(seen)
        if n_loc < MIN_LOCATIONS:
            continue

        te_mask = remap[te_locs] >= 0
        va_mask = remap[va_locs] >= 0
        n_unseen = int((~te_mask).sum())
        if te_mask.sum() < 2 or va_mask.sum() < 2:
            continue

        tr_l = remap[tr_locs]
        va_d, va_l = va_days[va_mask], remap[va_locs[va_mask]]
        te_d, te_l = te_days[te_mask], remap[te_locs[te_mask]]

        med_gap, mean_gap, p90_gap = gap_geometry(tr_days, te_d)
        base = dict(uid=uid, scheme=scheme, n_loc_train=n_loc,
                    n_unseen_test=n_unseen, gap_median=med_gap,
                    gap_mean=mean_gap, gap_p90=p90_gap, **cov)

        # ---------------- proposed model -----------------------------------
        for (name, gamma, shape) in MODELS:
            for window in WINDOWS:
                # the window ablation is only informative for the main-text
                # (gamma = 0) kernels; skip it elsewhere to save time
                if window is not None and (gamma == 1 or name not in ("PL", "EXP", "GAUSS")):
                    continue

                m = MemoryModel(tr_days, tr_l, n_loc, name, gamma, shape, window)

                # fit lambda on VALIDATION
                m.set_query(va_d)
                nll = np.array([m.nll(va_l, lam) for lam in LAMBDA_GRID])
                k_star = int(np.argmin(nll))
                lam_star = float(LAMBDA_GRID[k_star])
                logZ = log_evidence(nll, LAMBDA_GRID)           # (B1)
                logZ_lu = log_evidence_loguniform(nll, LAMBDA_GRID)
                post = lambda_posterior(nll, LAMBDA_GRID)       # (N6)
                psum = posterior_summaries(post, LAMBDA_GRID)   # (N10)

                # curvature at the optimum, as in the submitted script
                if 1 <= k_star <= len(nll) - 2:
                    h = LAMBDA_GRID[k_star + 1] - LAMBDA_GRID[k_star - 1]
                    d2 = (nll[k_star + 1] - 2 * nll[k_star] + nll[k_star - 1]) / (h / 2) ** 2
                else:
                    d2 = np.nan

                # interior/boundary status computed from the GRID ITSELF, so it
                # cannot drift out of sync with hard-coded constants
                at_lower = k_star == 0
                at_upper = k_star == len(LAMBDA_GRID) - 1

                # evaluate on TEST
                m.set_query(te_d)
                P_point = m.predictive(lam_star)
                mt = evaluate(te_l, P_point)

                rows.append({**base, "method": f"memory-{name}", "gamma": gamma,
                             "shape": shape if shape is not None else np.nan,
                             "window": -1 if window is None else window,
                             "lambda_star": lam_star, "log_evidence": logZ,
                             "min_negll": float(nll[k_star]),
                             "second_derivative": float(d2),
                             "at_lower_bound": at_lower,
                             "at_upper_bound": at_upper,
                             "interior": not (at_lower or at_upper),
                             "posterior_avg": False, **psum, **mt})

                # posterior-averaged prediction (N6), main kernels only
                if window is None and gamma == 0:
                    keep = post > 1e-4
                    if keep.any():
                        Pavg = np.zeros_like(P_point)
                        wsum = 0.0
                        for lam, w in zip(LAMBDA_GRID[keep], post[keep]):
                            Pavg += m.predictive(lam) * w
                            wsum += w
                        Pavg /= wsum
                        mt2 = evaluate(te_l, Pavg)
                        rows.append({**base, "method": f"memory-{name}",
                                     "gamma": gamma,
                                     "shape": shape if shape is not None else np.nan,
                                     "window": -1, "lambda_star": lam_star,
                                     "log_evidence": logZ,
                                     "log_evidence_loguniform": logZ_lu,
                                     "min_negll": float(nll[k_star]),
                                     "second_derivative": float(d2),
                                     "at_lower_bound": at_lower,
                                     "at_upper_bound": at_upper,
                                     "interior": not (at_lower or at_upper),
                                     "posterior_avg": True, **psum, **mt2})

        # ---------------- baselines (N2) ------------------------------------
        for bname, fn in BASELINES.items():
            try:
                P = fn(tr_days, tr_l, te_d, n_loc)
            except Exception:
                continue
            mt = evaluate(te_l, P)
            rows.append({**base, "method": bname, "gamma": np.nan,
                         "shape": np.nan, "window": -1,
                         "lambda_star": np.nan, "log_evidence": np.nan,
                         "min_negll": np.nan, "second_derivative": np.nan,
                         "at_lower_bound": False, "at_upper_bound": False,
                         "interior": False, "posterior_avg": False,
                         "post_mean": np.nan, "post_sd": np.nan,
                         "post_mass_lower": np.nan, "post_mass_upper": np.nan,
                         **mt})

    return rows


# =============================================================================
#  MAIN
# =============================================================================

def main():
    n_file = str(sys.argv[1]) if len(sys.argv) > 1 else "a0"
    delta = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    chunk = int(sys.argv[3]) if len(sys.argv) > 3 else int(n_file[1])
    cond = n_file[0]

    print(f"condition={cond}  chunk={chunk}  delta={delta}")
    print(f"lambda grid: {len(LAMBDA_GRID)} points, "
          f"[{LAMBDA_GRID[0]}, {LAMBDA_GRID[-1]}], logarithmic interior")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_name = os.path.join(
        OUTPUT_DIR, f"metrics_revision_3_condition_{cond}_chunk_{chunk}.csv")

    print(f"Output CSV created: {csv_name}")

    timeline_file = TIMELINE_TEMPLATE.format(cond=cond)
    timelines = read_timelines_from_file(timeline_file)
    print(f"File read: {timeline_file}")
    print(f"Total timelines read: {len(timelines)}")

    timelines = timelines[chunk * delta:(chunk + 1) * delta]
    print(f"Timelines to analyze in this run: {len(timelines)}")

    rng = np.random.default_rng(SEED + 1000 * ord(cond) + chunk)

    header_written = os.path.isfile(csv_name)
    buffer, BUF = [], 200

    for k, (days, locs, labels) in tqdm(enumerate(timelines), desc="Analyzing trajectories", unit="user"):

        # explicit inclusion thresholds (R4.8 asks for these to be stated)
        if len(days) < MIN_OBS or len(labels) < MIN_LOCATIONS:
            continue

        uid = chunk * delta + k
        try:
            buffer.extend(analyse_timeline(uid, days, locs, labels, rng))
        except Exception as exc:
            print(f"  [warn] uid={uid} failed: {exc}")
            continue

        if len(buffer) >= BUF:
            pd.DataFrame(buffer).to_csv(csv_name, mode="a", index=False,
                                        header=not header_written)
            header_written = True
            buffer = []

    if buffer:
        pd.DataFrame(buffer).to_csv(csv_name, mode="a", index=False,
                                    header=not header_written)

    print(f"Analysis complete -> {csv_name}")


if __name__ == "__main__":
    main()
