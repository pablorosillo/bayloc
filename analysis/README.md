# Analysis scripts

These are the scripts that produced the results of the revised manuscript.
They are kept as they were run; `tests/test_against_pipeline.py` checks that
the `bayloc` package gives identical results.

| Script | Produces |
|---|---|
| `bayloc_metrics.py` | All real-data results, at country level and at NUTS3 resolution |
| `bayloc_synthetic.py` | The synthetic generators SG1–SG4 and all synthetic results |

Both scripts write long-format CSV files with one row per timeline, method and
configuration. Figures and tables are computed from these files.

## Real data

The Twitter timelines cannot be shared because of data protection regulations.
To run the script on your own data, write one timeline per line as a Python
list of `('YYYY-MM-DD', label)` tuples and set

```bash
export BAYLOC_TIMELINES="data/timelines_country_condition_{cond}.txt"
export BAYLOC_OUTPUT="results"
python analysis/bayloc_metrics.py a0 10000      # condition 'a', chunk 0
```

`{cond}` is replaced by the first character of the first argument, and the
file is processed in chunks of `delta` timelines (second argument) so that it
can be split across cluster jobs.

## Synthetic generators

Settings used in the paper (20 values of p in [0, 0.5], 500 replicates, random
hold-out, deterministic seeds):

```bash
for g in SG1 SG2 SG3 SG4; do
    python analysis/bayloc_synthetic.py $g 20 500 results_synthetic
done
```
