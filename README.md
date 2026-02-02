# bayloc

A Python package for inferring location timelines and residency history from daily location data.

## Features

* Infer missing daily locations using power-law weights following Eq.(eq) of the \[paper]\(link al arxiv).
* Compute residence periods with configurable minimum residency duration using the ResIn algorithm ([arXiv:2003.04155](https://arxiv.org/abs/2003.04155)).
* Provide basic timeline statistics.
* Compute AUROC for binary classification tasks on location data.

## Installation

```bash
conda env create -f environment.yml
conda activate bayloc-env
```

*Note:* The package `rezin` is required. Its source code is available [here](https://github.com/networkdynamics/resin).

## Usage

```python
from datetime import date
from bayloc import BayLoc
```

Input data must be a list of tuples in the format `('yyyy-mm-dd', 'CC')`, where `CC` is a label for the location (in the examples, this is the country code).

The timeline does not need to be ordered by date and may include multiple locations for the same day. For example, see 2015-05-17 in the sample below.

```python
daily_timeline = [('2015-05-12', 'RU'), ('2015-05-16', 'RU'), ('2015-05-17', 'RU'), ('2015-05-17', 'IE'), ...]
bayloc = BayLoc(daily_timeline)

full_timeline = bayloc.infer_missing_days()  # fills missing days with inferred locations

residence = bayloc.get_residence_history(n_days=7)  # residency threshold based on ResIn method

print("\nResidence history (periods of stay):")
for place, period in residence.items():
    print(f"{place}: {period}")
```

Example output:

```
Residence history (periods of stay):
RU: [['2015-05-12', '2015-06-21'], ['2015-06-29', '2015-07-01'], ...]
IE: [['2015-06-22', '2015-06-28'], ['2015-07-02', '2015-07-06'], ...]
```

You can also explore statistics of both the original and complete (original + inferred) timelines:

```python
stats = bayloc.get_basic_stats()

print("\nTimeline statistics:")

print("\nInput data stats:")
for k, v in stats["input_data"].items():
    print(f"{k}: {v}")

print("\nComplete (input + inferred) timeline stats:")
for k, v in stats["inferred_timeline"].items():
    print(f"{k}: {v}")
```

Example output:

```
Input data stats:
total_days: 44
unique_locations: {'IE', 'RU'}
counts_per_location: {'RU': 29, 'IE': 16}
start_date: 2015-05-12
end_date: 2016-12-08

Complete (input + inferred) timeline stats:
total_days: 577
unique_locations: {'IE', 'RU'}
counts_per_location: {'RU': 343, 'IE': 234}
start_date: 2015-05-12
end_date: 2016-12-08
number_of_known_days: 44
number_of_inferred_days: 533
```

## Development

* Place `rezin.py` inside the `/bayloc` folder.
* Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of this software, as long as the original copyright notice and this permission notice appear in all copies.

See the full text of the [MIT License](https://opensource.org/licenses/MIT) for details.

## Citation

If you use this library, please cite it using the following BibTeX:

```bibtex
@article{Rosillo-Rodes2025,
  author       = {Author Name},
  title        = {Title of the Work},
  journal      = {Journal or Conference},
  year         = {2025},
  url          = {URL if available},
}
```

If you use the residence history computation feature (`bayloc.get_residence_history`), please cite:

```bibtex
@article{Ruths2020,
  author       = {Derek Ruths and Caitrin Armstrong},
  title        = {The Residence History Inference Problem},
  journal      = {arXiv preprint arXiv:2003.04155},
  year         = {2020},
  url          = {https://arxiv.org/abs/2003.04155},
}
```

