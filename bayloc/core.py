"""
bayloc.core -- memory-decay imputation of missing locations.

Reference implementation of the method in
  "Inferring missing locations in human mobility data using memory effects"
  (revision 1). Equations are referred to by their LaTeX labels.

What this module implements, and where it is described in the paper:
  * predictive distribution, eq:pmodel1 and eq:weights, with the kernels PL, EXP, GAUSS
    (main text), STREXP (Methods, "Additional kernel") and TPL, LOGNORM (SI);
  * fit of lambda on a validation fold by grid search over
    {0} U logspace(1e-3, 0.999*lambda_inf, 1000) U {lambda_inf}, with
    log-probabilities floored at -100 (Methods, "Fitting lambda");
  * pseudo-evidence log Z (eq:bayesian_evidence), posterior over lambda and
    the posterior-averaged predictive (eq:postavg, mass threshold 1e-4);
  * the evaluation protocol of Algorithm 1 (50/25/25 split, random or 14-day
    block hold-out, candidate set from training days only), the five
    baselines and the metrics of Methods, "Performance metrics".

The numerical core (kernels, grid, evidence, posterior, baselines, split,
metrics) is identical to the analysis script used for the paper; see
tests/test_against_pipeline.py.
"""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from datetime import date, datetime

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score

# =============================================================================
#  Configuration (values used in the paper)
# =============================================================================

LAMBDA_0 = 0.0
LAMBDA_INF = 200.0
N_LAMBDA = 1000          # interior grid points; total grid size N_LAMBDA + 2
LOGP_FLOOR = -100.0      # floor on log-probabilities inside the composite likelihood
EPS = 1e-12              # probability clipping before scoring
POST_MASS_MIN = 1e-4     # grid points kept in the posterior-averaged predictive
MIN_OBS = 30             # inclusion threshold on observed days
MIN_LOCATIONS = 2
MIN_TRAIN = 8
MIN_HELDOUT = 2
BLOCK_DAYS = 14
FRAC_VAL = 0.25
FRAC_TEST = 0.25

KERNELS = {              # name -> default shape parameter
    "PL": None, "EXP": None, "GAUSS": None,
    "STREXP": 0.5, "TPL": 365.0, "LOGNORM": 2.0,
}


class InsufficientDataError(ValueError):
    """Timeline does not meet the inclusion thresholds of Algorithm 1."""


class LambdaOptimizationError(Exception):
    """Kept for import compatibility with bayloc < 2.0. No longer raised:
    lambda_* on a boundary is a legitimate outcome (Cases 1 and 3)."""


def build_lambda_grid(lambda_inf=LAMBDA_INF, n=N_LAMBDA):
    """Exact boundaries plus a log-spaced interior."""
    interior = np.geomspace(1e-3, lambda_inf * 0.999, n)
    return np.concatenate([[LAMBDA_0], interior, [lambda_inf]])


# =============================================================================
#  Kernels (log space)
# =============================================================================

def log_kernel(name, D, logD, lam, shape=None):
    """log f_lambda(Delta). All kernels satisfy f_0 = 1."""
    if name == "PL":
        return -lam * logD
    if name == "EXP":
        return -lam * D
    if name == "GAUSS":
        return -0.5 * lam * D * D
    if name == "STREXP":
        return -lam * np.power(D, shape)
    if name == "TPL":
        return -lam * logD - D / shape
    if name == "LOGNORM":
        return -lam * logD - 0.5 * (logD / shape) ** 2
    raise ValueError(f"unknown kernel {name!r}; choose from {sorted(KERNELS)}")


class _Predictor:
    """P[x(t) = l | record, lambda] for a fixed record and fixed query days.

    The distance geometry is computed once and reused across the grid.
    """

    def __init__(self, rec_days, rec_locs, n_loc, kernel, shape=None,
                 window=None):
        self.kernel, self.shape, self.n_loc = kernel, shape, n_loc
        self.window = window
        self.rec_days = np.asarray(rec_days, dtype=np.float64)
        self.S = np.zeros((len(rec_days), n_loc))
        self.S[np.arange(len(rec_days)), rec_locs] = 1.0
        self._q = None

    def set_query(self, q_days):
        q = np.asarray(q_days, dtype=np.float64)
        D = np.abs(q[:, None] - self.rec_days[None, :])
        valid = D > 0
        if self.window is not None:
            valid &= D <= self.window
        Dsafe = np.where(valid, D, 1.0)
        self._q = (Dsafe, np.log(Dsafe), valid)
        return self

    def log_predictive(self, lam):
        D, logD, valid = self._q
        lf = np.where(valid, log_kernel(self.kernel, D, logD, lam, self.shape),
                      -np.inf)
        M = lf.max(axis=1, keepdims=True)
        alive = np.isfinite(M[:, 0])
        Ms = np.where(alive[:, None], M, 0.0)
        E = np.exp(lf - Ms)
        E[~np.isfinite(E)] = 0.0
        W = E @ self.S
        with np.errstate(divide="ignore"):
            lw = np.where(W > 0, np.log(np.where(W > 0, W, 1.0)) + Ms, -np.inf)
        lw[~alive, :] = -np.inf
        norm = logsumexp(lw, axis=1, keepdims=True)
        dead = ~np.isfinite(norm[:, 0])
        with np.errstate(invalid="ignore"):
            lp = lw - norm
        if dead.any():          # no record day in range (window only)
            lp[dead, :] = -np.log(self.n_loc)
        return lp

    def predictive(self, lam):
        return np.exp(self.log_predictive(lam))

    def nll(self, y, lam):
        """Negative log composite likelihood (eq:composite_general), floored at -100."""
        lp = self.log_predictive(lam)
        return float(-np.maximum(lp[np.arange(len(y)), y], LOGP_FLOOR).sum())


# =============================================================================
#  Evidence and posterior over lambda
# =============================================================================

def _trapezoid_logw(grid):
    dx = np.diff(grid)
    q = np.zeros_like(grid)
    q[:-1] += dx / 2.0
    q[1:] += dx / 2.0
    return np.log(np.maximum(q, np.finfo(float).tiny))


def log_evidence(nll_grid, grid):
    """log Z (eq:bayesian_evidence): uniform prior, trapezoidal rule."""
    val = logsumexp(-nll_grid + _trapezoid_logw(grid)) - np.log(grid[-1] - grid[0])
    return float(val) if np.isfinite(val) else np.nan


def lambda_posterior(nll_grid, grid):
    """Grid masses pi_k of p(lambda | validation fold)."""
    lp = -nll_grid + _trapezoid_logw(grid)
    tot = logsumexp(lp)
    if not np.isfinite(tot):
        return np.full_like(grid, 1.0 / len(grid))
    return np.exp(lp - tot)


def posterior_summaries(post, grid):
    m1 = float(np.sum(post * grid))
    m2 = float(np.sum(post * grid ** 2))
    return dict(post_mean=m1, post_sd=float(np.sqrt(max(0.0, m2 - m1 ** 2))),
                post_mass_lower=float(post[0]), post_mass_upper=float(post[-1]))


def fit_lambda(tr_days, tr_locs, va_days, va_locs, n_loc, kernel="PL",
               shape=None, window=None, grid=None):
    """Grid search of lambda on the validation fold (Algorithm 1, l. 7-15)."""
    grid = build_lambda_grid() if grid is None else grid
    pred = _Predictor(tr_days, tr_locs, n_loc, kernel, shape, window)
    pred.set_query(va_days)
    nll = np.array([pred.nll(va_locs, lam) for lam in grid])
    k = int(np.argmin(nll))
    post = lambda_posterior(nll, grid)
    case = 1 if k == 0 else (3 if k == len(grid) - 1 else 2)
    # The composite likelihood saturates in floating point once the predictive
    # at the recorded locations reaches 1: from there on it is flat, and argmin
    # returns its first point. flat_to_upper flags an interior lambda_* whose
    # objective is constant up to lambda_inf, i.e. indistinguishable from the
    # persistence limit on the validation days.
    flat = bool(case == 2 and np.all(nll[k:] == nll[k]))
    return dict(lambda_star=float(grid[k]), k_star=k, case=case,
                flat_to_upper=flat, nll_grid=nll, min_negll=float(nll[k]),
                log_evidence=log_evidence(nll, grid), posterior=post,
                **posterior_summaries(post, grid))


def predict(rec_days, rec_locs, q_days, n_loc, fit, grid, kernel="PL",
            shape=None, window=None, posterior_average=False):
    """Predictive at lambda_* or posterior-averaged (eq:postavg)."""
    pred = _Predictor(rec_days, rec_locs, n_loc, kernel, shape, window)
    pred.set_query(q_days)
    if not posterior_average:
        return pred.predictive(fit["lambda_star"])
    post = fit["posterior"]
    keep = post > POST_MASS_MIN
    P, wsum = 0.0, 0.0
    for lam, w in zip(grid[keep], post[keep]):
        P = P + pred.predictive(lam) * w
        wsum += w
    return P / wsum


# =============================================================================
#  Baselines (Methods, "Baselines")
# =============================================================================

def _smoothed_freq(tr_locs, n_loc):
    c = np.bincount(tr_locs, minlength=n_loc).astype(float) + 1.0
    return c / c.sum()


def bl_most_frequent(tr_days, tr_locs, q_days, n_loc):
    return np.repeat(_smoothed_freq(tr_locs, n_loc)[None, :], len(q_days), axis=0)


def bl_last_observed(tr_days, tr_locs, q_days, n_loc, smooth=0.02):
    out = np.full((len(q_days), n_loc), smooth / n_loc)
    pos = np.searchsorted(tr_days, q_days, side="left") - 1
    marg = _smoothed_freq(tr_locs, n_loc)
    for i, j in enumerate(pos):
        if j < 0:
            out[i] = marg
        else:
            out[i, tr_locs[j]] += 1.0 - smooth
    return out / out.sum(1, keepdims=True)


def bl_nearest_observed(tr_days, tr_locs, q_days, n_loc, smooth=0.02):
    out = np.full((len(q_days), n_loc), smooth / n_loc)
    D = np.abs(np.asarray(tr_days)[None, :] - np.asarray(q_days)[:, None])
    for i in range(len(q_days)):
        hit = tr_locs[D[i] == D[i].min()]
        for l in hit:
            out[i, l] += (1.0 - smooth) / len(hit)
    return out / out.sum(1, keepdims=True)


def bl_markov(tr_days, tr_locs, q_days, n_loc, alpha=1.0):
    A = np.full((n_loc, n_loc), alpha)
    for a, b in zip(tr_locs[:-1], tr_locs[1:]):
        A[a, b] += 1.0
    A /= A.sum(1, keepdims=True)
    marg = _smoothed_freq(tr_locs, n_loc)
    out = np.empty((len(q_days), n_loc))
    pos = np.searchsorted(tr_days, q_days, side="left") - 1
    for i, j in enumerate(pos):
        out[i] = marg if j < 0 else A[tr_locs[j]]
    return out


def bl_markov_ct(tr_days, tr_locs, q_days, n_loc, alpha=1.0):
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

    marg = _smoothed_freq(tr_locs, n_loc)
    out = np.empty((len(q_days), n_loc))
    pos = np.searchsorted(tr_days, q_days, side="left")
    cache = {}
    for i, t in enumerate(q_days):
        j, k = pos[i] - 1, pos[i]
        parts = []
        if j >= 0:
            d = int(t - tr_days[j])
            cache.setdefault(d, power(d))
            parts.append(cache[d][tr_locs[j]])
        if k < len(tr_days):
            d = int(tr_days[k] - t)
            cache.setdefault(d, power(d))
            parts.append(cache[d][tr_locs[k]])
        if not parts:
            out[i] = marg
        else:
            r = np.prod(parts, axis=0)
            out[i] = r / r.sum() if r.sum() > 0 else marg
    return out


BASELINES = {
    "MostFrequent": bl_most_frequent,       # lambda -> 0 limit
    "LastObserved": bl_last_observed,
    "NearestObserved": bl_nearest_observed, # lambda -> infinity limit
    "Markov": bl_markov,
    "MarkovCT": bl_markov_ct,
}


# =============================================================================
#  Splits and metrics
# =============================================================================

def three_way_split(days, n, scheme, rng, frac_val=FRAC_VAL,
                    frac_test=FRAC_TEST, block_days=BLOCK_DAYS):
    """Disjoint train / validation / test index sets (Methods, Data partition)."""
    idx = np.arange(n)
    if scheme == "random":
        perm = rng.permutation(n)
        n_val = max(2, int(round(frac_val * n)))
        n_test = max(2, int(round(frac_test * n)))
        test = np.sort(perm[:n_test])
        val = np.sort(perm[n_test:n_test + n_val])
        train = np.sort(perm[n_test + n_val:])
        return train, val, test
    if scheme != "block":
        raise ValueError("scheme must be 'random' or 'block'")
    block_id = (days - int(days[0])) // block_days
    uniq = np.unique(block_id)
    rng.shuffle(uniq)
    test_b, val_b, got_t, got_v = [], [], 0, 0
    for b in uniq:
        size = int((block_id == b).sum())
        if got_t < frac_test * n:
            test_b.append(b); got_t += size
        elif got_v < frac_val * n:
            val_b.append(b); got_v += size
    test = idx[np.isin(block_id, test_b)]
    val = idx[np.isin(block_id, val_b)]
    train = idx[~np.isin(block_id, test_b + val_b)]
    return train, val, test


def auroc(y, P, average):
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


def score(y, P):
    """T, micro/macro AUROC, balanced accuracy, log-loss, Brier."""
    P = np.clip(np.asarray(P, float), EPS, None)
    P = P / P.sum(1, keepdims=True)
    n = len(y)
    pred = P.argmax(1)
    Y = np.zeros_like(P)
    Y[np.arange(n), y] = 1.0
    return dict(
        T=float(np.mean(pred == y)),
        auroc_micro=auroc(y, P, "micro"),
        auroc_macro=auroc(y, P, "macro"),
        balanced_acc=float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)])),
        logloss=float(-np.mean(np.log(P[np.arange(n), y]))),
        brier=float(np.mean(np.sum((P - Y) ** 2, axis=1))),
        n_eval=int(n),
    )


# =============================================================================
#  User-facing class
# =============================================================================

def _import_rezin():
    """rezin.py is not bundled (see README); look for it in the package
    folder first and then on the Python path."""
    try:
        from . import rezin
        return rezin
    except ImportError:
        pass
    try:
        import rezin
        return rezin
    except ImportError as exc:
        raise ImportError(
            "get_residence_history() needs rezin.py from "
            "https://github.com/networkdynamics/resin (src/rezin.py). Copy it "
            "into the bayloc/ folder or anywhere on your PYTHONPATH.") from exc


def _to_date(x):
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return datetime.strptime(str(x), "%Y-%m-%d").date()


class BayLoc:
    """Memory-decay imputation for one individual.

    Parameters
    ----------
    records : list of (date, location)
        One entry per geolocated event, in the order in which they occurred.
        Dates are 'YYYY-MM-DD' strings or date objects. Several entries per
        day are allowed: each day is collapsed to its majority location, ties
        broken by the LAST occurrence (Methods, Empirical data). Pass a list,
        not a set: a set removes repeated events and has no order, so neither
        the majority nor the tie-break would be defined.
    kernel : {'PL','EXP','GAUSS','STREXP','TPL','LOGNORM'}
    shape : float, optional. Second parameter of STREXP / TPL / LOGNORM.
    lambda_inf : float. Upper end of the grid (200 in the paper).
    window : int, optional. Restrict the kernel sum to |t - s| <= window.
    min_obs : int. Inclusion threshold on observed days (30 in the paper).
    seed : int or None. Default seed for the train/validation split, the
        evaluation split and sampled imputations, so that repeated runs give
        identical results. None draws fresh entropy on every call.
    """

    def __init__(self, records, kernel="PL", shape=None, lambda_inf=LAMBDA_INF,
                 n_lambda=N_LAMBDA, window=None, min_obs=MIN_OBS, seed=0,
                 lambda_min=None, lambda_max=None):
        if lambda_min is not None or lambda_max is not None:
            warnings.warn(
                "lambda_min and lambda_max are deprecated: lambda is now selected "
                "on the grid {0} U logspace(1e-3, 0.999*lambda_inf) U {lambda_inf}. "
                "lambda_max is used as lambda_inf; lambda_min is ignored.",
                DeprecationWarning, stacklevel=2)
            if lambda_max is not None:
                lambda_inf = float(lambda_max)
        if isinstance(records, (set, frozenset)):
            warnings.warn(
                "records was passed as a set: repeated events on the same day "
                "are lost and ties are broken in arbitrary order. Pass a list "
                "in chronological order to reproduce the paper.", stacklevel=2)
            records = sorted(records, key=lambda r: (str(r[0]), str(r[1])))
        if kernel not in KERNELS:
            raise ValueError(f"unknown kernel {kernel!r}")
        self.kernel = kernel
        self.shape = KERNELS[kernel] if shape is None else shape
        self.window = window
        self.min_obs = min_obs
        self.seed = seed
        self.grid = build_lambda_grid(lambda_inf, n_lambda)

        self.daily_timeline = self._convert_input(records)
        self.consolidated_timeline = self._consolidate(records)
        self.labels = sorted(set(self.consolidated_timeline.values()))
        self._code = {l: i for i, l in enumerate(self.labels)}
        self.days = np.array([d.toordinal() for d in self.consolidated_timeline],
                             dtype=np.int64)
        self.locs = np.array([self._code[l] for l in self.consolidated_timeline.values()],
                             dtype=np.int64)

        self.fit_result = None
        self.best_lambda = None
        self.full_timeline = None
        self.guessed_timeline = None
        self.probabilities = None

    # ------------------------------------------------------------- input ----
    @staticmethod
    def _convert_input(records):
        tl = defaultdict(Counter)
        for d, loc in records:
            tl[_to_date(d)][loc] += 1
        return dict(sorted(tl.items()))

    @staticmethod
    def _consolidate(records):
        per_day = defaultdict(list)
        for d, loc in records:
            per_day[_to_date(d)].append(loc)
        out = {}
        for day, locs in per_day.items():
            cnt = Counter(locs)
            mx = max(cnt.values())
            for loc in reversed(locs):           # tie -> last occurrence
                if cnt[loc] == mx:
                    out[day] = loc
                    break
        return dict(sorted(out.items()))

    def _check(self):
        if len(self.days) < self.min_obs:
            raise InsufficientDataError(
                f"{len(self.days)} observed days < min_obs={self.min_obs}")
        if len(self.labels) < MIN_LOCATIONS:
            raise InsufficientDataError("fewer than two distinct locations")

    # --------------------------------------------------------------- fit ----
    def fit(self, scheme="random", val_frac=1 / 3, seed=None):
        """Fit lambda on a train/validation split of ALL observed days.

        For imputing genuine gaps no test fold is needed, so the observed days
        are split into training and validation days in the 2:1 proportion of
        the paper's 50/25 train/validation folds, with the same random or
        14-day block geometry. Lambda is selected on the validation days with
        the kernel built on the training days (Algorithm 1, lines 7-15).
        Returns self; results in self.fit_result.
        """
        self._check()
        seed = self.seed if seed is None else seed
        rng = np.random.default_rng(seed)
        n = len(self.days)
        tr, va, _ = three_way_split(self.days, n, scheme, rng,
                                    frac_val=val_frac, frac_test=0.0)
        # three_way_split with frac_test = 0 still reserves 2 test days in the
        # random scheme; return them to the training fold
        tr = np.sort(np.setdiff1d(np.arange(n), va))
        seen = np.unique(self.locs[tr])
        if len(seen) < MIN_LOCATIONS or len(tr) < MIN_TRAIN:
            raise InsufficientDataError("training fold too small or single-location")
        remap = -np.ones(len(self.labels), dtype=int)
        remap[seen] = np.arange(len(seen))
        vm = remap[self.locs[va]] >= 0
        if vm.sum() < MIN_HELDOUT:
            raise InsufficientDataError("validation fold too small")
        fr = fit_lambda(self.days[tr], remap[self.locs[tr]], self.days[va][vm],
                        remap[self.locs[va][vm]], len(seen), self.kernel,
                        self.shape, self.window, self.grid)
        fr.update(scheme=scheme, seed=seed, n_train=len(tr), n_val=int(vm.sum()))
        self.fit_result = fr
        self.best_lambda = fr["lambda_star"]
        return self

    # ----------------------------------------------------------- predict ----
    def predict_proba(self, query_dates, posterior_average=False):
        """P[x(t) = l | full record, lambda] for each query date (eq:pmodel1).

        Returns (array of shape (n_query, n_locations), list of labels).
        """
        if self.fit_result is None:
            self.fit()
        q = np.array([_to_date(d).toordinal() for d in query_dates], dtype=np.int64)
        P = predict(self.days, self.locs, q, len(self.labels), self.fit_result,
                    self.grid, self.kernel, self.shape, self.window,
                    posterior_average)
        return P, list(self.labels)

    def infer_missing_days(self, point="argmax", posterior_average=False,
                           seed=None):
        """Fill every unobserved day between the first and last observation.

        point='argmax' assigns the most probable location (the quantity scored
        by T; ties go to the first label in sorted order). point='sample'
        draws from the predictive with a seeded generator. The full predictive
        is stored in self.probabilities.
        """
        seed = self.seed if seed is None else seed
        if self.fit_result is None:
            self.fit(seed=seed)
        observed = set(self.days.tolist())
        missing = [d for d in range(int(self.days[0]), int(self.days[-1]) + 1)
                   if d not in observed]
        missing_dates = [date.fromordinal(d) for d in missing]
        guessed = {}
        if missing:
            P, labels = self.predict_proba(missing_dates, posterior_average)
            self.probabilities = {d: dict(zip(labels, map(float, row)))
                                  for d, row in zip(missing_dates, P)}
            if point == "argmax":
                choice = P.argmax(1)
            elif point == "sample":
                rng = np.random.default_rng(seed)
                choice = np.array([rng.choice(len(labels), p=row / row.sum())
                                   for row in P])
            else:
                raise ValueError("point must be 'argmax' or 'sample'")
            guessed = {d: labels[c] for d, c in zip(missing_dates, choice)}
        self.guessed_timeline = guessed
        full = dict(self.consolidated_timeline)
        full.update(guessed)
        self.full_timeline = dict(sorted(full.items()))
        return self.full_timeline

    # ---------------------------------------------------------- evaluate ----
    def evaluate(self, scheme="random", seed=None, baselines=True,
                 posterior_average=True):
        """Algorithm 1: 50/25/25 split, fit on validation, score on test.

        Returns a dict {method: metrics}. 'memory-<KERNEL>' uses lambda_*,
        'memory-<KERNEL>-postavg' the posterior-averaged predictive.
        """
        self._check()
        seed = self.seed if seed is None else seed
        rng = np.random.default_rng(seed)
        n = len(self.days)
        tr, va, te = three_way_split(self.days, n, scheme, rng)
        if len(tr) < MIN_TRAIN or len(va) < MIN_HELDOUT or len(te) < MIN_HELDOUT:
            raise InsufficientDataError("folds below the inclusion thresholds")
        seen = np.unique(self.locs[tr])
        n_loc = len(seen)
        if n_loc < MIN_LOCATIONS:
            raise InsufficientDataError("single location in the training fold")
        remap = -np.ones(len(self.labels), dtype=int)
        remap[seen] = np.arange(n_loc)
        vm, tm = remap[self.locs[va]] >= 0, remap[self.locs[te]] >= 0
        if vm.sum() < MIN_HELDOUT or tm.sum() < MIN_HELDOUT:
            raise InsufficientDataError("held-out folds below thresholds")
        tr_d, tr_l = self.days[tr], remap[self.locs[tr]]
        va_d, va_l = self.days[va][vm], remap[self.locs[va][vm]]
        te_d, te_l = self.days[te][tm], remap[self.locs[te][tm]]

        fr = fit_lambda(tr_d, tr_l, va_d, va_l, n_loc, self.kernel, self.shape,
                        self.window, self.grid)
        key = f"memory-{self.kernel}"
        out = {key: dict(score(te_l, predict(tr_d, tr_l, te_d, n_loc, fr, self.grid,
                                             self.kernel, self.shape, self.window)),
                         lambda_star=fr["lambda_star"], case=fr["case"],
                         log_evidence=fr["log_evidence"],
                         **{k: fr[k] for k in ("post_mean", "post_sd",
                                               "post_mass_lower", "post_mass_upper")})}
        if posterior_average:
            out[key + "-postavg"] = score(te_l, predict(
                tr_d, tr_l, te_d, n_loc, fr, self.grid, self.kernel, self.shape,
                self.window, posterior_average=True))
        if baselines:
            for name, fn in BASELINES.items():
                out[name] = score(te_l, fn(tr_d, tr_l, te_d, n_loc))
        gap = np.abs(te_d[:, None] - tr_d[None, :]).min(1)
        out["_split"] = dict(scheme=scheme, seed=seed, n_unseen_test=int((~tm).sum()),
                             gap_median=float(np.median(gap)),
                             gap_mean=float(gap.mean()),
                             gap_p90=float(np.percentile(gap, 90)))
        return out

    # ------------------------------------------------------------- extras --
    def get_residence_history(self, n_days, **infer_kwargs):
        """Residence periods lasting at least n_days, with the ResIn algorithm.

        The completed daily timeline (observed + imputed days) is compressed
        into runs of consecutive days at the same location and passed to
        rezin.rezin. ResIn returns, for each residence, the index of the RUN
        at which it starts; it is converted here to the first day of that run.
        Requires rezin.py (see README). Returns {location: [[start, end], ...]}
        with ISO dates, or None if the timeline is empty.
        """
        rezin = _import_rezin()
        if self.full_timeline is None:
            self.infer_missing_days(**infer_kwargs)
        dates = list(self.full_timeline)
        if not dates:
            return None
        runs = []
        for loc in self.full_timeline.values():
            if runs and runs[-1][0] == loc:
                runs[-1][1] += 1
            else:
                runs.append([loc, 1])
        run_start = np.concatenate([[0], np.cumsum([w for _, w in runs])])
        res = rezin.rezin([(l, w) for l, w in runs], n_days)
        out = defaultdict(list)
        for i, (loc, r) in enumerate(res):
            first = int(run_start[r])
            last = int(run_start[res[i + 1][1]]) - 1 if i + 1 < len(res) else len(dates) - 1
            out[str(loc)].append([dates[first].isoformat(), dates[last].isoformat()])
        return dict(out)

    def get_basic_stats(self):
        input_counts = Counter()
        for c in self.daily_timeline.values():
            input_counts.update(c)
        dates = list(self.daily_timeline)
        stats = {"input_data": dict(
            total_days=len(dates), unique_locations=set(input_counts),
            counts_per_location=dict(input_counts),
            start_date=min(dates) if dates else None,
            end_date=max(dates) if dates else None)}
        if self.full_timeline is None:
            self.infer_missing_days()
        inf_counts = Counter(self.full_timeline.values())
        fr = self.fit_result
        stats["inferred_timeline"] = dict(
            total_days=len(self.full_timeline),
            unique_locations=set(inf_counts),
            counts_per_location=dict(inf_counts),
            start_date=min(self.full_timeline), end_date=max(self.full_timeline),
            number_of_known_days=len(self.consolidated_timeline),
            number_of_inferred_days=len(self.guessed_timeline),
            kernel=self.kernel, optimized_lambda=self.best_lambda,
            lambda_case=fr["case"], log_evidence=fr["log_evidence"],
            posterior_mean_lambda=fr["post_mean"], posterior_sd_lambda=fr["post_sd"])
        return stats
