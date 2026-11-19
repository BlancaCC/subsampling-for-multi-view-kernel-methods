# Multi-View Kernel Herding for Efficient Multi-View Learning

Code for the paper *Multi-View Kernel Herding for Efficient Multi-View
Learning* (B. Cano-Camarero, F. Faramarzi Lighvan, L. Houthuys).

> **Status:** the paper has been **submitted to ESANN 2027 and is under
> review**. It has not been accepted yet; this README will be updated with the
> final reference if it is published.

Kernel methods scale at least quadratically with the number of samples, and
the cost grows with the number of views. **Multi-View Kernel Herding (MKH)**
subsamples the training set separately in each view:

- kernel herding keeps each view's distribution (small MMD to the full data);
- an HSIC-type cross-view term couples the views;
- a scale-free rank-sum rule combines the two criteria.

The repository contains the sampler, the baselines, the multi-view LS-SVM
models and the scripts that run and analyze the experiments of the paper.

- **Using the method on your own data:** see [`usage/`](usage/README.md), with a
  runnable [`usage/quickstart.py`](usage/quickstart.py).
- **Reproducing the paper:** see
  [Reproducing the experiments](#reproducing-the-experiments) below.

## Repository structure

```
.
├── README.md
├── LICENSE                              # MIT
├── requirements.txt
├── usage/                               # How to use MKH (guide + runnable example)
├── Datasets/                            # .mat files (not tracked, see "Datasets")
└── 1_BO_subsampling_and_metrics/        # Run all commands from this folder
    ├── subsampling_methods/
    │   ├── MultiViewKernelHerding.py    # MKH (proposed); also SKH with warmup_steps=1.0
    │   ├── RandomSubsamplingPerView.py  # RND baseline
    │   ├── ClusteringSubsampling.py     # CLU baseline
    │   ├── FixedSizeEntropySubsampling.py  # FSE baseline (fixed-size LS-SVM)
    │   └── BaseMultiViewSampler.py      # Common fit / transform interface
    ├── multi_view_learning_models/
    │   ├── late_fusion.py               # LateFusion_LSSVM (model used in the paper)
    │   ├── Multiclass_MV_LSSVM.py       # Coupled multi-view LS-SVM
    │   ├── early_fusion.py, AW_LSSVM_partial.py   # Other models (not in the paper)
    ├── get_data/                        # Dataset loader and metadata
    ├── experiments/
    │   ├── bo_late_fusion.py            # Subsampling experiments of the paper
    │   ├── bo_full_data.py              # "Full" baseline (no subsampling)
    │   ├── bo_accuracy_in_view.py       # Shared protocol + coupled MV-LSSVM runs
    │   └── bo_early_fusion.py, bo_aw_lssvm.py     # Other models (not in the paper)
    ├── analysis_scripts/
    │   ├── bo_accuracy_in_view.py       # Tables, figures and statistical tests
    │   └── subset_properties.py         # Properties of the selected subsets
    ├── examples/mvkh.py                 # Sanity checks of the sampler (synthetic data)
    └── scripts/                         # Shell (fish) launchers of the runs
```

## Installation

Tested with Python 3.9. Dependencies: NumPy, SciPy, scikit-learn, pandas,
Optuna and Matplotlib (Matplotlib is only needed for the analysis).

```bash
git clone https://github.com/BlancaCC/subsampling-for-multi-view-kernel-methods.git
cd subsampling-for-multi-view-kernel-methods
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Check the installation (a few seconds):

```bash
python usage/quickstart.py
```

## Datasets

The experiments use nine multi-view datasets: 3Sources, YouTube, Prokaryotic,
ProteinFold, Caltech, NUS-WIDE, OutdoorScene, Cora and ACM. They are not
included in the repository. Download the `.mat` files from
<https://github.com/ChuanbinZhang/Multi-view-datasets> and place them in
`Datasets/` at the repository root:

```
Datasets/
├── 3Sources.mat    ├── Caltech.mat     ├── OutdoorScene.mat
├── ACM.mat         ├── Cora.mat        ├── Prokaryotic.mat
├── NUS-WIDE.mat    ├── ProteinFold.mat └── YouTube3views_subsampled_500.mat
```

Each file must contain a cell array `X` (one matrix per view, samples in rows)
and a label vector `y`. YouTube is used as a class-balanced subset of 500
samples with 3 views (`YouTube3views_subsampled_500.mat`).
`get_data/dataset_info.py` lists the number of views and classes of each
dataset. To add a new dataset, add an entry there and drop its `.mat` file
into `Datasets/`.

## Reproducing the experiments

All commands are run from `1_BO_subsampling_and_metrics/`:

```bash
cd 1_BO_subsampling_and_metrics
```

### 1. Subsampling experiments (Table 1, all methods)

```bash
python -m experiments.bo_late_fusion
```

This runs every combination of:

| | Values |
|---|---|
| Datasets | 3Sources, YouTube, Prokaryotic, ProteinFold, Caltech, NUS-WIDE, OutdoorScene, Cora, ACM |
| Methods | `MV_kernel_herding` (MKH), `simple_kernel_herding` (SKH), `clustering` (CLU), `random` (RND), `fixed_size_entropy` (FSE) |
| `n_sub` | 10, 25, 50, 100, 150, 200 (plus 300–600 for NUS-WIDE, Caltech, OutdoorScene, ACM and Cora; not reported in the paper) |

Results are written to `results_late_fusion/`.

### 2. Full-data baseline (row "Full" of Table 1)

```bash
python -m experiments.bo_full_data --model late_fusion
```

Results are written to `results_full_data/late_fusion/`.

### 3. Tables, figures and statistical tests

```bash
python -m analysis_scripts.bo_accuracy_in_view \
    --results-root results_late_fusion --analysis-dir analysis_late_fusion
# Optional: overlap and properties of the selected subsets
python -m analysis_scripts.subset_properties \
    --results-root results_late_fusion --analysis-dir analysis_late_fusion
```

The outputs go to `analysis_late_fusion/tables/` and
`analysis_late_fusion/figures/`:

- mean ± std of each metric per dataset and `n_sub`;
- rankings;
- Friedman, Nemenyi and Wilcoxon tests;
- subsampling and training times;
- accuracy vs. time trade-off plots.

### Protocol

These settings are fixed in `experiments/bo_accuracy_in_view.py` and
`experiments/bo_late_fusion.py`, and are shared by every script:

| Step | Setting |
|---|---|
| Preprocessing | `MinMaxScaler` per view, fitted on the training rows only. |
| Outer CV | Stratified 10-fold (`N_OUTER_FOLDS = 10`, shuffled, `SEED = 42`); the test fold is never used for subsampling or tuning. |
| Inner split | Stratified 25% hold-out of the training fold for validation (`VAL_SIZE = 0.25`). |
| Subsampling | Each sampler runs inside each class with `ceil(n_sub / C)` rows per class; the subsets are merged per view. |
| Tuning | Optuna (TPE, seeded per fold), maximizing validation accuracy. Per view: LS-SVM regularization `gamma_reg` in [1e-3, 1e3] and RBF width `gamma_kernel` in [1e-4, 1e2] (log scale). MKH also tunes `warmup_steps` in (0.01, 1.0] as a fraction of `n_sub`; MKH uses `epsilon = inf`. SKH uses `warmup_steps = 1.0` (not tuned). |
| Trials | `N_TRIALS_PER_DIMENSION * 2V = 20V` for V views, plus `MV_KH_EXTRA_TRIALS` for MKH. **The paper results use `MV_KH_EXTRA_TRIALS = 50`**; set this constant in `experiments/bo_accuracy_in_view.py` before running. |
| Refit | Best configuration refit on the whole training fold, evaluated on the test fold. |
| Metrics | Accuracy, balanced accuracy (reported in the paper), macro and weighted F1; subsampling, training and prediction times. |

### Running time and partial runs

The full grid is long (hours to days, depending on the machine). Runs are
**sequential and resumable**:

- a run whose `summary.csv` exists is skipped (`SKIP_COMPLETED = True`);
- a failed run is logged in `manifest.csv` and the grid continues.

So you can stop and relaunch the same command. For a quick smoke test, edit
the grid at the top of `experiments/bo_late_fusion.py`, for example:

```python
DATASETS = ["3Sources"]
SUBSAMPLING_METHODS = ["MV_kernel_herding", "simple_kernel_herding"]
N_SUBS = [50]
```

Each results folder records its protocol (`experiment_metadata.csv` includes
the git commit and library versions). `check_protocol` refuses to mix runs
with a different protocol in the same folder. Use a new results folder if you
change the protocol.

### Output layout

```
results_late_fusion/
├── experiment_metadata.csv          # Environment, git commit, global config
├── manifest.csv                     # One row per run: status, time, error
├── all_outer_fold_metrics.csv       # All runs, one row per outer fold
├── all_best_hyperparameters.csv
├── all_summaries.csv                # Mean/std over folds per configuration
└── <dataset>/<method>/<n_sub>/
    ├── run_config.csv, summary.csv
    ├── outer_fold_metrics.csv       # Test metrics and timings per fold
    ├── best_hyperparameters.csv
    ├── optuna_trials_fold<k>.csv    # Full Optuna history of fold k
    ├── selected_indices.csv         # Selected rows per view and fold
    └── predictions.csv              # Test predictions per fold
```

Timings depend on the hardware and the number of threads, so only compare
timings between runs on the same machine.

## Citation

The paper is under review (submitted to ESANN 2027). Until it is
published, please cite it as:

```bibtex
@unpublished{canocamarero2026multiview,
  title  = {Multi-View Kernel Herding for Efficient Multi-View Learning},
  author = {Cano-Camarero, Blanca and Faramarzi Lighvan, Farnaz and Houthuys, Lynn},
  note   = {Submitted to ESANN 2027, under review},
  year   = {2026}
}
```

## Acknowledgements

Supported by projects PID2022-139856NB-I00 and PID2025-167768NB-I00 funded by
MCIN/AEI/10.13039/501100011033/FEDER, UE, project IDEA-CM (TEC-2024/COM-89),
the ELLIS Unit Madrid, Cátedra UAM-IIC, FPI-UAM and the Flemish Government
under the Onderzoeksprogramma Artificiële Intelligentie Vlaanderen program.

## License

This code is released under the [MIT License](LICENSE). The datasets are not
covered by this license; see their original sources.
