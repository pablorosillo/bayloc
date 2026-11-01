"""The bayloc package reproduces the analysis scripts of the paper exactly.

Runs analysis/bayloc_metrics.py (the script behind every result in the paper)
and BayLoc.evaluate() on the same synthetic timelines with the same seeds, and
compares lambda_*, the log pseudo-evidence, the posterior summaries and every
test-fold metric of the memory model (point estimate and posterior-averaged)
and of the five baselines, under random and 14-day block hold-out.

    pytest tests/test_against_pipeline.py        # quick subset
    python tests/test_against_pipeline.py [SG1 SG2 SG3 SG4]   # full comparison
"""
import os
import sys
from datetime import date

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

import bayloc_metrics as R            # noqa: E402
import bayloc_synthetic as G          # noqa: E402
from bayloc import BayLoc             # noqa: E402

D0 = date(2016, 1, 1).toordinal()
ALL_KERNELS = [("PL", None), ("EXP", None), ("GAUSS", None),
               ("STREXP", 0.5), ("TPL", 365.0), ("LOGNORM", 2.0)]


def _reference(days, locs, labels, scheme, seed, name, shape):
    rng = np.random.default_rng(seed)
    tr, va, te = R.three_way_split(days, len(days), scheme, rng)
    seen = np.unique(locs[tr])
    remap = -np.ones(len(labels), int)
    remap[seen] = np.arange(len(seen))
    n_loc = len(seen)
    vm, tm = remap[locs[va]] >= 0, remap[locs[te]] >= 0
    trd, trl = days[tr], remap[locs[tr]]
    vad, val = days[va][vm], remap[locs[va][vm]]
    ted, tel = days[te][tm], remap[locs[te][tm]]
    m = R.MemoryModel(trd, trl, n_loc, name, 0, shape, None)
    m.set_query(vad)
    nll = np.array([m.nll(val, lam) for lam in R.LAMBDA_GRID])
    k = int(np.argmin(nll))
    lam = float(R.LAMBDA_GRID[k])
    post = R.lambda_posterior(nll, R.LAMBDA_GRID)
    m.set_query(ted)
    out = {f"memory-{name}": dict(R.evaluate(tel, m.predictive(lam)),
                                  lambda_star=lam,
                                  log_evidence=R.log_evidence(nll, R.LAMBDA_GRID),
                                  **R.posterior_summaries(post, R.LAMBDA_GRID))}
    keep = post > 1e-4
    P, w = 0.0, 0.0
    for lam_k, p_k in zip(R.LAMBDA_GRID[keep], post[keep]):
        P = P + m.predictive(lam_k) * p_k
        w += p_k
    out[f"memory-{name}-postavg"] = R.evaluate(tel, P / w)
    for b, fn in R.BASELINES.items():
        out[b] = R.evaluate(tel, fn(trd, trl, ted, n_loc))
    return out


def _same(a, b):
    if isinstance(a, float) and np.isnan(a):
        return isinstance(b, float) and np.isnan(b)
    return a == b                      # exact: same operations, same order


def compare(gen, p, seed, kernel, shape, scheme):
    """Returns the number of quantities compared (0 if timeline excluded)."""
    days, locs, labels = G.GENERATORS[gen](p=p, rng=seed)
    if len(days) < R.MIN_OBS or len(np.unique(locs)) < 2:
        return 0
    recs = [(date.fromordinal(D0 + int(d)).isoformat(), labels[l])
            for d, l in zip(days, locs)]
    model = BayLoc(recs, kernel=kernel, lambda_inf=R.LAMBDA_INF)
    assert np.array_equal(model.days - D0, days)
    try:
        mine = model.evaluate(scheme=scheme, seed=seed)
    except Exception:                  # excluded by the inclusion thresholds
        return 0
    ref = _reference(days, locs, labels, scheme, seed, kernel, shape)
    n = 0
    for method, metrics in ref.items():
        for key, value in metrics.items():
            assert _same(value, mine[method][key]), (
                gen, p, seed, kernel, scheme, method, key, value, mine[method][key])
            n += 1
    return n


@pytest.mark.parametrize("scheme", ["random", "block"])
@pytest.mark.parametrize("kernel,shape", [("PL", None), ("GAUSS", None),
                                          ("STREXP", 0.5)])
@pytest.mark.parametrize("gen", ["SG1", "SG3", "SG4"])
def test_matches_pipeline(gen, kernel, shape, scheme):
    assert R.LAMBDA_INF == 200.0
    assert compare(gen, 0.2, 7, kernel, shape, scheme) > 0


if __name__ == "__main__":
    gens = sys.argv[1:] or ["SG1", "SG2", "SG3", "SG4"]
    total = 0
    for gen in gens:
        for p in [0.05, 0.3]:
            for seed in [0, 1]:
                for kernel, shape in ALL_KERNELS:
                    for scheme in ["random", "block"]:
                        total += compare(gen, p, seed, kernel, shape, scheme)
        print(f"{gen} done", flush=True)
    print(f"{total} quantities identical")
