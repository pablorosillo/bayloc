# bayloc

A Python package for inferring missing locations in sparse mobility timelines
and reconstructing residence histories. It implements the memory-decay model of

> P. Rosillo-Rodes, A. Tugores, D. Sánchez and J. J. Ramasco,
> *Inferring missing locations in human mobility data using memory effects*,
> [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX).

The location of an individual on an unobserved day *t* is predicted from the
days on which it was observed, each weighted by a decaying function of its
temporal distance to *t*:

$$
P\left[x(t) = \ell \mid \mathcal{D}, \lambda\right]
= \frac{\sum_{s \in \mathcal{T}} f_\lambda(|t-s|)\,\delta_{\ell,x(s)}}
       {\sum_{s \in \mathcal{T}} f_\lambda(|t-s|)} ,
\qquad f_\lambda(\Delta) = \Delta^{-\lambda} .
$$

The decay rate λ is fitted for each individual. λ = 0 gives the most frequent
location and λ → ∞ the location of the nearest observed day; the fitted value
interpolates between the two. Only dates and location labels are used, with no
coordinates or distances.

## Features

* Imputation of missing days with the most probable location, or by sampling,
  together with the full predictive distribution for every day.
* Kernels: power law (`PL`, default), exponential (`EXP`), Gaussian (`GAUSS`),
  stretched exponential (`STREXP`), truncated power law (`TPL`) and log-normal
  decay (`LOGNORM`).
* Fit of λ on held-out observed days by grid search, pseudo-evidence for
  comparing kernels, posterior over λ and posterior-averaged imputation.
* Residence periods with a configurable minimum duration, using the ResIn
  algorithm ([arXiv:2003.04155](https://arxiv.org/abs/2003.04155)).
* Evaluation with the protocol of the paper: training/validation/test split
  with random or 14-day block hold-out, five baselines, and the fraction of
  correct inferences, micro- and macro-averaged AUROC, balanced accuracy,
  log-loss and Brier score.
* Basic statistics of the input and completed timelines.

## Installation

```bash
git clone https://github.com/<user>/bayloc.git
cd bayloc
conda env create -f environment.yml
conda activate bayloc-env
```

or, in an existing environment, `pip install -e .`

Residence histories need `rezin.py` from the ResIn repository
([`src/rezin.py`](https://github.com/networkdynamics/resin/blob/master/src/rezin.py)),
which is not distributed with this package. Copy it into the `bayloc/` folder
or anywhere on your `PYTHONPATH`. Everything else works without it.

## Usage

The input is a list of `('YYYY-MM-DD', label)` tuples, one per event (e.g. one
per geolocated post), where `label` identifies the location (a country code in
the examples). The list does not need to be sorted and may contain several
events and several locations on the same day. Each day is assigned its majority
location, with ties broken by the location that occurs last in the list, so
pass events in the order in which they happened.

```python
from bayloc import BayLoc

records = [('2015-05-16', 'RU'), ('2015-05-21', 'RU'), ('2015-05-21', 'RU'),
           ('2015-05-21', 'FI'), ('2015-06-02', 'RU'), ...]

model = BayLoc(records)            # power-law kernel
model.fit()                        # select lambda on held-out observed days
full_timeline = model.infer_missing_days()   # {date: location}, every day
```

Pass a list, not a set: a set drops repeated events, so the majority rule
cannot be applied, and its order is arbitrary, so ties would be broken
differently from run to run. `BayLoc` warns if it receives a set.

### Fitted decay and predictive probabilities

```python
fit = model.fit_result
print(fit['lambda_star'], fit['case'], fit['post_mean'], fit['post_sd'])

day = min(model.probabilities)     # first imputed day
print(day, model.probabilities[day])
```

`case` is 1 when λ\* = 0 (the most frequent location explains the held-out
days best), 3 when λ\* is at the upper end of the grid (persistence), and 2
otherwise. `post_mean` and `post_sd` summarize the posterior over λ; a large
`post_sd` means that the data do not determine λ well, in which case the
posterior-averaged predictive is the safer choice:

```python
model.infer_missing_days(posterior_average=True)
```

Imputations are deterministic by default (`seed=0`). To sample instead of
taking the most probable location, use `infer_missing_days(point='sample')`.

### Residence history

```python
residence = model.get_residence_history(n_days=7)   # ResIn, minimum stay 7 days

print("Residence history (periods of stay):")
for place, periods in residence.items():
    print(f"{place}: {periods}")
```

### Statistics

```python
stats = model.get_basic_stats()
for k, v in stats["input_data"].items():
    print(f"{k}: {v}")
for k, v in stats["inferred_timeline"].items():
    print(f"{k}: {v}")
```

### Evaluating on your own data

`evaluate()` applies the protocol of the paper to one timeline. Observed days
are split into training (50%), validation (25%) and test (25%) days, either at
random or in whole 14-day blocks. λ is fitted on the validation days, and the
model and the baselines are scored on the test days.

```python
results = model.evaluate(scheme="block")      # or scheme="random"
for method in ["memory-PL", "memory-PL-postavg", "MostFrequent",
               "LastObserved", "NearestObserved", "Markov", "MarkovCT"]:
    print(method, results[method]["auroc_macro"], results[method]["logloss"])
```

### Other kernels

```python
BayLoc(records, kernel="EXP")
BayLoc(records, kernel="STREXP", shape=0.5)
```

`fit_result['log_evidence']` is the log pseudo-evidence of the kernel; the
difference between two kernels fitted with the same seed is the log
pseudo-Bayes factor of the paper.

## Example

`examples/example.py` generates a synthetic user who lives in RU, spends two
periods in IE and posts on about one day in eight, then runs everything above.

```bash
python examples/example.py
```

Output:

```
lambda_* = 2.046 (case 2), posterior mean 2.656 +/- 1.068
P[x(2015-09-21)] = {'IE': 0.989, 'RU': 0.011}
497 days imputed, 97.8% equal to the true location

Residence history (periods of stay):
RU: [['2015-05-16', '2015-09-08'], ['2015-11-12', '2016-05-22'], ['2016-08-20', '2016-12-04']]
IE: [['2015-09-09', '2015-11-11'], ['2016-05-23', '2016-08-19']]

Test-fold metrics, 14-day block hold-out:
  memory-PL          T=0.947  macro-AUROC=0.735  log-loss=0.558
  memory-PL-postavg  T=0.947  macro-AUROC=0.735  log-loss=0.390
  NearestObserved    T=0.947  macro-AUROC=0.750  log-loss=0.252
  MarkovCT           T=0.947  macro-AUROC=0.794  log-loss=0.252
```

Test-fold metrics of a single short timeline are noisy; the paper reports
their distribution over many timelines.

## Reproducing the paper

`analysis/` contains the scripts that produced the results of the paper: the
real-data analysis and the synthetic generators SG1–SG4. See
[`analysis/README.md`](analysis/README.md). The Twitter timelines cannot be
shared because of data protection regulations; the synthetic results can be
reproduced in full.

The test suite checks that the package gives results identical to those
scripts for every kernel and baseline, under both hold-out schemes:

```bash
pytest                                     # unit tests and a quick comparison
python tests/test_against_pipeline.py      # full comparison
```

## Assumptions

The model treats missing days as conditionally independent given the observed
days, so it returns calibrated probabilities for each day rather than a joint
reconstruction of a gap. It also assumes that whether a day is recorded does
not depend on the location occupied that day (missing at random given timing).
Performance on gaps of several weeks is lower than on isolated missing days;
see the Discussion of the paper.

## Changes

Version 2.0.0 implements the method of the revised manuscript. It changes how
λ is fitted, the default imputation and the input handling, and fixes the
dates returned by `get_residence_history()`. See [CHANGELOG.md](CHANGELOG.md).

## Contributing

Issues and pull requests are welcome.

## License

MIT License. You are free to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of this software, as long as the original
copyright notice and this permission notice appear in all copies. See the full
text of the [MIT License](https://opensource.org/licenses/MIT).

## Citation

If you use this package, please cite:

```bibtex
@article{Rosillo-Rodes2025,
  author  = {Rosillo-Rodes, Pablo and Tugores, Ant{\`o}nia and S{\'a}nchez, David and Ramasco, Jos{\'e} J.},
  title   = {Inferring missing locations in human mobility data using memory effects},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2025},
  url     = {https://arxiv.org/abs/XXXX.XXXXX},
}
```

If you use the residence history (`get_residence_history`), please also cite:

```bibtex
@article{Ruths2020,
  author  = {Derek Ruths and Caitrin Armstrong},
  title   = {The Residence History Inference Problem},
  journal = {arXiv preprint arXiv:2003.04155},
  year    = {2020},
  url     = {https://arxiv.org/abs/2003.04155},
}
```
