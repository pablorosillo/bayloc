"""Minimal bayloc example on a synthetic timeline.

A user lives in RU, spends two periods in IE, and posts on about one day in
eight. Run from the repository root:

    python examples/example.py
"""
from datetime import date, timedelta

import numpy as np

from bayloc import BayLoc

# ---- a synthetic record ------------------------------------------------------
rng = np.random.default_rng(42)
start = date(2015, 5, 12)
true_location = (["RU"] * 120 + ["IE"] * 60 + ["RU"] * 200 + ["IE"] * 90
                 + ["RU"] * 107)
records = []
for t, loc in enumerate(true_location):
    if rng.random() < 0.12:                         # posted that day
        n_posts = rng.integers(1, 4)
        for _ in range(n_posts):
            records.append(((start + timedelta(t)).isoformat(), loc))
    if t == 200:                    # several posts and two locations on one
        d = (start + timedelta(t)).isoformat()      # day: the majority (RU) is
        records += [(d, "RU"), (d, "FI"), (d, "RU")]  # kept for that day

# ---- fit and impute -----------------------------------------------------------
model = BayLoc(records)                             # power-law kernel, seed 0
model.fit()
fit = model.fit_result
print(f"lambda_* = {fit['lambda_star']:.3f} (case {fit['case']}), "
      f"posterior mean {fit['post_mean']:.3f} +/- {fit['post_sd']:.3f}")

full_timeline = model.infer_missing_days()          # most probable location
day = min(d for d in model.probabilities if d >= date(2015, 9, 20))  # unobserved
print(f"P[x({day})] =", {k: round(v, 3) for k, v in model.probabilities[day].items()})

# ---- accuracy of the imputation against the known truth -----------------------
truth = {start + timedelta(t): loc for t, loc in enumerate(true_location)}
imputed = model.guessed_timeline
acc = np.mean([imputed[d] == truth[d] for d in imputed])
print(f"{len(imputed)} days imputed, {acc:.1%} equal to the true location")

# ---- residence history (needs rezin.py, see README) ---------------------------
try:
    residence = model.get_residence_history(n_days=7)
    print("\nResidence history (periods of stay):")
    for place, periods in residence.items():
        print(f"{place}: {periods}")
except ImportError as exc:
    print(f"\nSkipping residence history: {exc}")

# ---- evaluation protocol of the paper (Algorithm 1) ----------------------------
print("\nTest-fold metrics, 14-day block hold-out:")
results = model.evaluate(scheme="block")
for method in ["memory-PL", "memory-PL-postavg", "NearestObserved", "MarkovCT"]:
    r = results[method]
    print(f"  {method:18s} T={r['T']:.3f}  macro-AUROC={r['auroc_macro']:.3f}  "
          f"log-loss={r['logloss']:.3f}")
