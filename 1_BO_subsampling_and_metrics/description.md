# Subsampling for Multi-View Kernel Methods

Experimental code to compare **subsampling strategies** for a multi-view
least-squares SVM (`Multiclass_MV_LSSVM`). For every combination of dataset,
subsampling method and subset size, the pipeline runs a nested
cross-validation with Bayesian hyperparameter optimization (Optuna) and stores
metrics, timings and the best hyperparameters as CSV files. A second script
turns those results into the figures and tables used in the paper.

> **Status:** research code. Items marked `TODO` must be completed by the
> authors (paper reference, license, dataset sources).

---

## Table of contents

- [Subsampling for Multi-View Kernel Methods](#subsampling-for-multi-view-kernel-methods)
  - [Table of contents](#table-of-contents)
  - [Overview](#overview)
  - [Repository structure](#repository-structure)
  - [Installation](#installation)
  - [Quick start](#quick-start)
  - [Experimental protocol](#experimental-protocol)
  - [Subsampling methods](#subsampling-methods)
  - [Configuration](#configuration)
  - [Outputs](#outputs)
  - [Analysis](#analysis)
  - [Reproducibility](#reproducibility)
  - [Known limitations](#known-limitations)
  - [Citation and license](#citation-and-license)

---

## Overview

```
run_experiments.py                       analyze_results.py
  datasets x methods x n_sub    ──CSV──▶   figures, tables, rankings,
  nested CV + Optuna                       statistical tests
```

Questions the experiments answer:

- How does predictive performance (accuracy, balanced accuracy, macro and
  weighted F1) evolve as the subset size `n_sub` grows?
- Which subsampling method gives the best trade-off between performance and
  cost (subsampling time + training time)?
- Which hyperparameters are selected for each method and subset size?

## Repository structure

Adjust the paths below to your actual layout.

```
.
├── get_data.py                      # get_dataset(name) -> (views, labels)
├── multi_view_learning_models/      # Multiclass_MV_LSSVM
├── subsampling_methods/             # Random / KernelHerding / Clustering
├── run_experiments.py               # Launches the experiment grid
├── analyze_results.py               # Builds figures and tables
├── results/                         # Written by run_experiments.py
└── analysis/                        # Written by analyze_results.py
```

## Installation

Developed with Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn scipy matplotlib optuna
```

`get_data.py`, `multi_view_learning_models` and `subsampling_methods` are
project modules and must be importable from the working directory.

## Quick start

1. Edit the grid at the top of `run_experiments.py`:

   ```python
   DATASETS = ["3Sources"]
   SUBSAMPLING_METHODS = ["random", "kernel_herding", "clustering"]
   N_SUBS = [50, 100, 200]
   ```

2. Run the experiments (runs are sequential and resumable):

   ```bash
   python run_experiments.py
   ```

3. Analyze the results:

   ```bash
   python analyze_results.py --results-root results --analysis-dir analysis
   ```

Tip: do a smoke test first with one dataset, one `n_sub` and a small
`N_TRIALS_PER_DIMENSION` (e.g. 1) before launching the full grid.

## Experimental protocol

Each run `(dataset, method, n_sub)` follows the same protocol.

| Step | Description |
|---|---|
| Outer CV | Stratified K-fold (`N_OUTER_FOLDS = 5`, shuffled, `SEED = 42`). Each fold gives `train_full` / `test`. |
| Inner split | Stratified hold-out of `train_full` (`VAL_SIZE = 0.25`) into `train` / `validation`. |
| Search | Optuna with the TPE sampler (seeded per fold). Each trial subsamples `train`, fits the model and returns the **validation accuracy**. |
| Refit | The best hyperparameters are used to subsample and fit on `train_full` (`REFIT_ON_TRAIN_PLUS_VAL = True`) and evaluate on the outer `test` fold. |

Test data is never used for subsampling or hyperparameter selection.

**Trials per fold:** `N_TRIALS_PER_DIMENSION * (2 * n_views + 1)`, plus
`2 * N_TRIALS_PER_DIMENSION` for kernel herding (one extra tuned
hyperparameter). Trials that raise `LinAlgError` or produce an invalid
subsample are recorded as failed and the search continues.

**Search space**

| Hyperparameter | Range | Scale |
|---|---|---|
| `gamma_reg_<v>` (per view) | 1e-3 to 1e3 | log |
| `gamma_kernel_<v>` (per view) | 1e-4 to 1e2 | log |
| `rho_ratio` | 1e-3 to 0.95 | log; `rho = rho_ratio * min(gamma_reg)` |
| `warmup_steps` (herding only) | 0.01 to 1.0 | fraction of the per-class subset size |

**Metrics (outer test fold):** accuracy, balanced accuracy, F1 macro,
F1 weighted.

**Timings (seconds, with the best hyperparameters):**

1. `subsampling_time_s`: selection of the subset.
2. `training_time_s`: model fit on the subset.

Also stored: `predict_time_s`, `optuna_time_s` (whole search) and the mean
per-trial subsampling/training time during the search.

## Subsampling methods

| Key | Class | Notes |
|---|---|---|
| `random` | `RandomSubsamplingPerView` | Uniform random selection per view. |
| `kernel_herding` | `MultiViewKernelHerding` | Uses the trial's kernel gammas; `epsilon = inf`; tunes `warmup_steps`. |
| `clustering` | `ClusteringSubsampling` | Clustering-based selection. |

**All methods are applied in a stratified way.** The method runs
independently on the rows of each class, selecting
`ceil(n_sub / n_classes)` rows per class (capped at the class size), and the
per-class subsets are merged into one subset per view. Consequently:

- Every class keeps at least one point in every view, as required by the
  model's reconstruction operator.
- The real subset size is `n_classes * ceil(n_sub / n_classes)`, which can be
  slightly larger than the nominal `n_sub`. It is stored as
  `n_sub_effective`; use it when comparing against non-subsampled baselines.

To add a method, implement a sampler with `fit(views, labels)`,
`selected_indices_` (dict per view) and `view_names_`, then register it in
`SAMPLERS` and `SAMPLER_KWARGS`.

## Configuration

All settings are constants at the top of `experiments/bo_accuracy_in_view.py`.

| Constant | Meaning |
|---|---|
| `DATASETS`, `SUBSAMPLING_METHODS`, `N_SUBS` | Experiment grid (full Cartesian product). |
| `N_OUTER_FOLDS`, `VAL_SIZE`, `SEED` | Nested CV protocol and seed. |
| `N_TRIALS_PER_DIMENSION` | Controls the number of Optuna trials. |
| `REFIT_ON_TRAIN_PLUS_VAL` | Refit on `train_full` (default) or only on `train`. |
| `RESULTS_ROOT` | Output folder. |
| `SKIP_COMPLETED` | Skip runs that already have a `summary.csv` (resume). |
| `SAVE_PREDICTIONS`, `SAVE_SELECTED_INDICES` | Save test predictions / selected rows. |

Tip: use a new `RESULTS_ROOT` when the code or the protocol changes, so that
results from different versions are not mixed.

## Outputs

`run_experiments.py` writes to `results/`:

```
results/
├── experiment_metadata.csv          # Environment, git commit, global config
├── manifest.csv                     # One row per run: status, elapsed time, error
├── all_outer_fold_metrics.csv       # All runs, one row per outer fold
├── all_best_hyperparameters.csv     # All runs, one row per outer fold
├── all_summaries.csv                # Mean/std over folds per configuration
└── <dataset>/<method>/<n_sub>/
    ├── run_config.csv               # Run configuration and dataset statistics
    ├── outer_fold_metrics.csv       # Test metrics and timings per fold
    ├── best_hyperparameters.csv     # Best hyperparameters per fold
    ├── optuna_trials_fold<k>.csv    # Full Optuna history of fold k
    ├── selected_indices.csv         # Subsampled rows per view and fold
    ├── predictions.csv              # Test predictions per fold
    └── summary.csv                  # Mean/std over folds (written last)
```

A failed run does not stop the grid: it is logged in `manifest.csv`
(`status = failed`, with the error message) and the next run starts. Relaunch
the script to retry only the missing runs.

## Analysis

`analyze_results.py` reads `all_outer_fold_metrics.csv` (and, if present,
`all_best_hyperparameters.csv` and the `run_config.csv` files) and writes to
`analysis/`:

| Output | Content |
|---|---|
| `figures/<dataset>_metrics` | Four metrics vs `n_sub` (mean ± std per method). |
| `figures/<dataset>_times` | Subsampling and training time vs `n_sub` (log scale). |
| `figures/<dataset>_tradeoff` | Accuracy vs total time. |
| `figures/average_rank_<metric>` | Average ranks with the Nemenyi critical difference. |
| `tables/table_<target>.csv/.tex` | mean ± std per (dataset, `n_sub`), best in bold (LaTeX). |
| `tables/ranking_<target>_per_config.csv` | Rank of each method in every (dataset, `n_sub`). |
| `tables/ranking_<target>_per_dataset.csv` | Average rank per dataset, overall average rank, number of wins. |
| `tables/friedman_<metric>.csv`, `wilcoxon_<metric>.csv` | Friedman test, Nemenyi CD, pairwise Wilcoxon with Holm correction. |
| `tables/best_configuration_per_dataset.csv` | Best (method, `n_sub`) per dataset and metric. |
| `tables/best_hyperparameters_summary.csv` | Median and quartiles of the selected hyperparameters. |
| `tables/dataset_table.csv` | Dataset statistics for the paper. |

Figures are saved as PNG (300 dpi) and PDF.

## Reproducibility

- Fixed seeds: outer folds, inner split, TPE sampler and subsamplers are all
  derived from `SEED`.
- `experiment_metadata.csv` stores the git commit, library versions,
  platform, hostname and the global configuration of the experiment.
- `optuna_trials_fold<k>.csv`, `selected_indices.csv` and `predictions.csv`
  allow re-analysis without re-running the experiments.
- Timings depend on the machine and on the number of threads; compare timings
  only between runs executed on the same hardware.

## Known limitations

- The statistical tests use (dataset, `n_sub`) configurations as blocks.
  Configurations from the same dataset are not independent, so p-values are
  indicative. The Nemenyi CD is computed for 3 to 10 methods.
- With 5 outer folds and a single seed, differences between methods can be
  within noise. Consider repeating with several seeds.
- There is no baseline without subsampling yet (`TODO`), so the loss of
  accuracy and the speed-up relative to the full model are not reported.
- Subset sizes are nominal; the real size is `n_sub_effective` (see
  [Subsampling methods](#subsampling-methods)).

## Citation and license

`TODO`: add the paper reference (BibTeX), dataset sources and licenses, and
the license of this repository.

```bibtex
@article{TODO,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```