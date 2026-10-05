# Legacy / Stale Results Archive

All files in this directory are **LEGACY ARTIFACTS** generated from synthetic development data.

## Status: NOT FOR FINAL RESULTS

These files were generated from `generate_development_dataset()` (synthetic data).
They must NOT be consumed by the final dashboard or cited as research results.

## Archived Files

| File | Original Location | Reason |
|---|---|---|
| `LEGACY_benchmark_master.csv` | `results/benchmark_master.csv` | Synthetic data - stale |
| `LEGACY_error_analysis.csv` | `results/error_analysis.csv` | Synthetic data - stale |
| `LEGACY_experiment_results.csv` | `results/experiment_results.csv` | Synthetic data - stale |
| `LEGACY_fiqa_benchmark.csv` | `results/fiqa_benchmark.csv` | Synthetic data - stale |
| `LEGACY_sentiment_metrics.csv` | `results/sentiment_metrics.csv` | Synthetic data - stale |

## Archived On: 2026-10-05T18:43:35.077534

## How to Generate Real Results

1. Load real data: `python -m src.data.ingestion`
2. Run preprocessing: `python -m src.data.preprocessing`
3. Run teacher labeling: `python -m src.paper_pipeline --provider openai`
4. Run benchmarks: see README.md
