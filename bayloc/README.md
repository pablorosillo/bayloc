# bayloc

A Python package for inferring location timelines and residency history from daily location data.

## Features

- Infer missing daily locations using power-law smoothing.
- Compute residence periods with configurable minimum residency duration.
- Optimize smoothing parameter (lambda) automatically.
- Provide basic timeline statistics.
- Compute AUROC for binary classification tasks on location data.

## Installation

```bash
conda env create -f environment.yml
conda activate bayloc-env
````

*Note:* The package `rezin` is required and must be installed either via pip or otherwise.

## Usage

```python
from datetime import date
from bayloc import BayLoc

# Example daily timeline
# Format must be a list of tuples. It does ot need to be ordered in time, and it may feature several different locations for the same day. For example, check 2015-05-17
daily_timeline = [('2015-05-12', 'RU'), ('2015-05-16', 'RU'), ('2015-05-17', 'RU'), ('2015-05-17', 'IE'), ('2015-05-31', 'RU'), ('2015-06-02', 'RU'), ('2015-06-10', 'RU'), ('2015-06-17', 'RU'), ('2015-06-25', 'RU'), ('2015-06-27', 'RU'), ('2015-07-04', 'RU'), ('2015-07-05', 'RU'), ('2015-07-09', 'RU'), ('2015-07-11', 'RU'), ('2015-07-25', 'RU'), ('2015-08-02', 'RU'), ('2015-08-20', 'RU'), ('2015-08-24', 'RU'), ('2015-08-28', 'RU'), ('2015-09-01', 'RU'), ('2015-09-02', 'RU'), ('2015-09-06', 'RU'), ('2015-09-16', 'RU'), ('2015-09-28', 'RU'), ('2015-09-30', 'RU'), ('2015-10-05', 'RU'), ('2015-10-08', 'RU'), ('2015-10-09', 'RU'), ('2016-09-20', 'IE'), ('2016-09-21', 'IE'), ('2016-09-22', 'IE'), ('2016-09-26', 'IE'), ('2016-09-27', 'IE'), ('2016-09-28', 'IE'), ('2016-10-01', 'IE'), ('2016-10-09', 'IE'), ('2016-10-18', 'IE'), ('2016-10-24', 'IE'), ('2016-10-27', 'IE'), ('2016-10-29', 'IE'), ('2016-11-01', 'IE'), ('2016-11-05', 'IE'), ('2016-11-09', 'IE'), ('2016-12-05', 'RU'), ('2016-12-08', 'RU')]


bayloc = BayLoc(daily_timeline)
full_timeline = bayloc.infer_missing_days()  # fills missing days with inferred locations

residence = bayloc.get_residence_history(n_days=7)  # optional, based on residency threshold following ResIn method documentation

print("\nResidence history (periods of stay):")
for place, period in residence.items():
    print(f"{place}: {period}")

# this is an output

# Residence history (periods of stay):
# RU: [['2015-05-12', '2015-06-21'], ['2015-06-29', '2015-07-01'], ['2015-07-07', '2015-07-13'], ['2015-07-21', '2015-07-23'], ['2015-07-29', '2015-08-04'], ['2015-08-10', '2015-08-14'], ['2015-08-22', '2015-08-30'], ['2015-09-03', '2015-09-07'], ['2015-09-13', '2015-09-19'], ['2015-10-23', '2016-12-08']]
# IE: [['2015-06-22', '2015-06-28'], ['2015-07-02', '2015-07-06'], ['2015-07-14', '2015-07-20'], ['2015-07-24', '2015-07-28'], ['2015-08-05', '2015-08-09'], ['2015-08-15', '2015-08-21'], ['2015-08-31', '2015-09-02'], ['2015-09-08', '2015-09-12'], ['2015-09-20', '2015-10-22']]


stats = bayloc.get_basic_stats()

print("\nTimeline statistics:")

print("\nInput data stats:")
for k, v in stats["input_data"].items():
    print(f"{k}: {v}")

print("\nComplete (input + inferred) timeline stats:")
for k, v in stats["inferred_timeline"].items():
    print(f"{k}: {v}")

# this is an output

# Basic timeline statistics:

# Input data stats:
# total_days: 44
# unique_locations: {'IE', 'RU'}
# counts_per_location: {'RU': 29, 'IE': 16}
# start_date: 2015-05-12
# end_date: 2016-12-08

# Complete (input + inferred) timeline Stats:
# total_days: 577
# unique_locations: {'IE', 'RU'}
# counts_per_location: {'RU': 343, 'IE': 234}
# start_date: 2015-05-12
# end_date: 2016-12-08
# number_of_known_days: 44
# number_of_inferred_days: 533

```

## Development

* Make sure to have `rezin.py` downloaded (añadir cita).
* Use Python 3.10+.
* Contributions welcome.

## License

MIT License


