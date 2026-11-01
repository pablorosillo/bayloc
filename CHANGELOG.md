# Changelog

## 2.0.0 — revision 1 of the manuscript

This release implements the method as described in the revised manuscript
(Methods and Algorithm 1). Results in the revised paper were produced with the
analysis scripts in `analysis/`, released here unchanged except for file
paths, module names, comments, and `LAMBDA_INF = 200`, the value stated in
the paper; `tests/test_against_pipeline.py` checks that
the `bayloc` package reproduces them exactly.

### Method
- lambda is fitted on a validation fold with the kernel built on a disjoint
  training fold, instead of by leave-one-out over all observed days.
- lambda is selected by grid search over {0} ∪ 1000 log-spaced points in
  [1e-3, 0.999·200] ∪ {200}, instead of bounded Brent search on [1e-4, 75]
  with tolerance 1.0. With that tolerance the previous optimizer stopped after
  a few golden-section steps and returned a small set of values (0.61, 0.99,
  1.6, ...) regardless of the data.
- lambda_* on a boundary is returned as Case 1 (lambda = 0, most frequent
  location) or Case 3 (lambda = 200, nearest observed day). Previously the
  upper bound raised `LambdaOptimizationError`.
- Power law is Delta^(-lambda) (previously (Delta + 1e-3)^(-lambda)). Added the
  EXP, GAUSS, STREXP, TPL and LOGNORM kernels. All kernels are evaluated in
  log space, so large lambda no longer overflows.
- Log-probabilities in the composite likelihood are floored at -100
  (previously log(p + 1e-9), i.e. about -20.7).
- New: pseudo-evidence, posterior over lambda, posterior summaries and the
  posterior-averaged predictive.
- Missing days are imputed with the most probable location by default
  (previously a random draw with no seed). Seeded sampling is still available
  with `point="sample"`. The full predictive is stored in `.probabilities`.
- All random operations use `seed=0` by default, so repeated runs give
  identical results.
- `fit_result["flat_to_upper"]` flags an interior lambda_* whose composite
  likelihood is constant up to lambda_inf (the predictive has saturated at 1
  on the validation days), i.e. one that the validation days cannot
  distinguish from the persistence limit.

### Input
- Input is a list of (date, location) events in chronological order. Each day
  is collapsed to its majority location with ties broken by the last
  occurrence. Passing a set (previous API) now emits a warning: a set removes
  repeated events, so the majority rule cannot be applied, and its iteration
  order depends on `PYTHONHASHSEED`, so ties were resolved differently in
  different runs.

### Evaluation
- `compute_auroc_training_subset` (chronological split, micro-AUROC only) is
  removed. `BayLoc.evaluate()` implements Algorithm 1: 50/25/25 split, random
  or 14-day block hold-out, candidate set from training days only, the five
  baselines, and T, micro/macro AUROC, balanced accuracy, log-loss and Brier.

### Fixes in the analysis scripts used for the submitted version
- The log-evidence is computed with logsumexp. Previously exp(-NLL) underflowed
  to 0 for NLL > 745, and those timelines were dropped from the kernel
  comparison of the submitted manuscript.
- Rare-location observations are no longer duplicated in the training fold.
- The candidate location set and visit counts are built from the training fold
  only.

### Residence history
- Fixed the conversion of the ResIn output to dates. `rezin.rezin` returns the
  index of the run (block of consecutive days at one location) at which each
  residence starts; it was used as a day index, so every residence after the
  first was assigned the wrong start and end dates.

### Other
- `lambda_min` and `lambda_max` are deprecated; `lambda_max` is used as
  `lambda_inf`.
- `full_timeline` is initialised in the constructor (previously
  `get_residence_history()` and `get_basic_stats()` raised `AttributeError`
  if called before `infer_missing_days()`).
