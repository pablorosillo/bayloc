"""Unit tests for the user-facing BayLoc class."""
import sys
import types
import warnings
from datetime import date, timedelta

import numpy as np
import pytest

from bayloc import BayLoc, InsufficientDataError
from bayloc import core


def _records(pattern, start=date(2015, 1, 1), every=1):
    """pattern: list of (location, n_observed_days)."""
    out, t = [], 0
    for loc, n in pattern:
        for _ in range(n):
            out.append(((start + timedelta(t)).isoformat(), loc))
            t += every
    return out


def test_majority_and_last_occurrence_tie_break():
    recs = [("2016-01-01", "ES"), ("2016-01-01", "ES"), ("2016-01-01", "FR"),
            ("2016-01-02", "DE"), ("2016-01-02", "IT")]
    tl = BayLoc._consolidate(recs)
    assert tl[date(2016, 1, 1)] == "ES"     # majority
    assert tl[date(2016, 1, 2)] == "IT"     # tie -> last occurrence


def test_set_input_warns():
    recs = _records([("A", 20), ("B", 20)])
    with pytest.warns(UserWarning, match="set"):
        BayLoc(set(recs))


def test_legacy_lambda_arguments_warn():
    recs = _records([("A", 20), ("B", 20)])
    with pytest.warns(DeprecationWarning):
        m = BayLoc(recs, lambda_max=75)
    assert m.grid[-1] == 75


def test_inclusion_thresholds():
    with pytest.raises(InsufficientDataError):
        BayLoc(_records([("A", 10), ("B", 10)])).fit()
    with pytest.raises(InsufficientDataError):
        BayLoc(_records([("A", 50)])).fit()


def test_upper_boundary_is_case_3_not_an_error():
    # validation day at distance 100 from an A day and 101 from a B day: the
    # predictive of A increases monotonically with lambda up to lambda_inf
    fr = core.fit_lambda(np.array([100, 301]), np.array([0, 1]),
                         np.array([200]), np.array([0]), 2)
    assert fr["case"] == 3 and fr["lambda_star"] == 200.0


def test_lower_boundary_is_case_1():
    # the nearest training day is at the wrong location: lambda = 0 is optimal
    fr = core.fit_lambda(np.array([0, 1, 2, 9]), np.array([0, 0, 0, 1]),
                         np.array([10]), np.array([0]), 2)
    assert fr["case"] == 1 and fr["lambda_star"] == 0.0


def test_no_overflow_with_long_gaps_and_large_lambda():
    recs = _records([("A", 20)], every=1) + _records(
        [("B", 20)], start=date(2019, 1, 1))
    m = BayLoc(recs).fit()
    P, labels = m.predict_proba([date(2017, 6, 1)])
    assert np.all(np.isfinite(P)) and np.isclose(P.sum(), 1.0)


def test_reproducible_by_default():
    rng = np.random.default_rng(3)
    locs = np.where(rng.random(200) < 0.2, "B", "A")
    recs = [((date(2015, 1, 1) + timedelta(int(2 * i))).isoformat(), l)
            for i, l in enumerate(locs)]
    a = BayLoc(recs).infer_missing_days(point="sample")
    b = BayLoc(recs).infer_missing_days(point="sample")
    assert a == b


def test_imputation_fills_every_day_and_keeps_observed():
    recs = _records([("A", 20), ("B", 20)], every=3)
    m = BayLoc(recs)
    full = m.infer_missing_days()
    days = list(full)
    assert days[-1] - days[0] == timedelta(len(days) - 1)
    for d, loc in m.consolidated_timeline.items():
        assert full[d] == loc


def test_residence_history_maps_runs_to_dates(monkeypatch):
    # stub with ResIn's output convention: (location, index of the RUN)
    fake = types.SimpleNamespace(
        rezin=lambda H, rho: [(loc, i) for i, (loc, w) in enumerate(H) if w >= rho])
    monkeypatch.setattr(core, "_import_rezin", lambda: fake)
    recs = _records([("RU", 40), ("IE", 7), ("RU", 8), ("IE", 30)])
    m = BayLoc(recs)
    res = m.get_residence_history(7)
    assert res["RU"] == [["2015-01-01", "2015-02-09"], ["2015-02-17", "2015-02-24"]]
    assert res["IE"] == [["2015-02-10", "2015-02-16"], ["2015-02-25", "2015-03-26"]]


def test_evaluate_returns_model_and_baselines():
    rng = np.random.default_rng(0)
    locs = np.where(rng.random(150) < 0.15, "B", "A")
    recs = [((date(2015, 1, 1) + timedelta(int(2 * i))).isoformat(), l)
            for i, l in enumerate(locs)]
    out = BayLoc(recs).evaluate(scheme="block")
    for k in ["memory-PL", "memory-PL-postavg", "MostFrequent", "LastObserved",
              "NearestObserved", "Markov", "MarkovCT"]:
        assert 0.0 <= out[k]["T"] <= 1.0
