# ⚠️ SYNTHETIC DEVELOPMENT ARTIFACTS — NOT RESEARCH RESULTS

All files in this directory and its subdirectories were generated from
**synthetic development data** produced by `generate_development_dataset()`.

These artifacts exist for development verification purposes only. They do
NOT constitute valid research findings and must NOT be cited as experimental
results.

## To produce real results:
1. Place a real Reddit dataset at `data/raw/reddit_posts.parquet`
2. Run preprocessing: `python -m src.data.preprocessing`
3. Run the benchmark: `python -c "...see README Section 12..."`
4. Run financial analysis: `python -m src.finance.run_financial_analysis`

The pipeline will refuse to run if the data is synthetic (via `assert_real_data()`).

## Files in this directory:
- `benchmark_master.csv` — **SYNTHETIC** benchmark results (stale)
- `sentiment_metrics.csv` — **SYNTHETIC** sentiment metrics (stale)
- `experiment_results.csv` — **SYNTHETIC** experiment tracking log (stale)
- `error_analysis.csv` — **SYNTHETIC** error analysis (stale)
- `metrics/` — **SYNTHETIC** per-model metrics JSONs (stale)
- `figures/` — **SYNTHETIC** confusion matrices and plots (stale)
- `predictions/` — **SYNTHETIC** model prediction parquets (stale)

These files will be overwritten with real results when the pipeline is run
on real data.
