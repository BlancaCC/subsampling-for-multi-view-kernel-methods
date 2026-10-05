"""Launches consecutive nested-CV experiments for multi-view subsampling.

The script runs the full grid DATASETS x SUBSAMPLING_METHODS x N_SUBS. For
every combination it performs a nested cross-validation (outer stratified
K-fold, inner stratified hold-out tuned with Optuna) and stores everything
as CSV files.

Run from 1_BO_subsampling_and_metrics:

    python3 -m experiments.bo_accuracy_in_view                  # with correction terms
    python3 -m experiments.bo_accuracy_in_view --no-correction  # without

Output layout (under RESULTS_ROOT, one folder per setting):

    experiment_metadata.csv          Environment and global configuration.
    manifest.csv                     One row per run: status, elapsed time.
    all_outer_fold_metrics.csv       Every run, one row per outer fold.
    all_best_hyperparameters.csv     Every run, one row per outer fold.
    all_summaries.csv                Mean/std over folds for every run.
    <dataset>/<method>/<n_sub>/
        run_config.csv               Configuration of this run.
        outer_fold_metrics.csv       Test metrics and timings per fold.
        best_hyperparameters.csv     Best hyperparameters per fold.
        optuna_trials_fold<k>.csv    Full Optuna history of each fold.
        selected_indices.csv         Selected rows per view and fold.
        predictions.csv              Test predictions per fold.
        summary.csv                  Mean/std over folds (written last).

Timings stored per fold (all in seconds):
    subsampling_time_s: subsampling with the best hyperparameters.
    training_time_s:    model fit with the best hyperparameters.
    predict_time_s:     prediction on the outer test fold.
    optuna_time_s:      whole hyperparameter search.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import math
import platform
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from get_data import get_dataset
from multi_view_learning_models import Multiclass_MV_LSSVM

# Adjust this import to wherever the sampler package lives in your project.
from subsampling_methods import (
    ClusteringSubsampling,
    FixedSizeEntropySubsampling,
    MultiViewKernelHerding,
    RandomSubsamplingPerView,
)

# ------------------------------ experiment grid -----------------------------
DATASETS: Sequence[str] = ['Cora','OutdoorScene','ACM','Caltech','Prokaryotic','ProteinFold',"NUS-WIDE", "3Sources", "YouTube"]
SUBSAMPLING_METHODS: Sequence[str] = ["random", "MV_kernel_herding", "simple_kernel_herding", "clustering", "fixed_size_entropy"]
N_SUBS: Sequence[int] = [10, 25, 50, 100, 150, 200]
# Extra subset sizes, run only on LARGE_DATASETS (> 694 samples). Smaller
# datasets run out of samples per class at these sizes.
LARGE_N_SUBS: Sequence[int] = [300, 400, 500, 600]
LARGE_DATASETS = frozenset({"NUS-WIDE", "Caltech", "OutdoorScene", "ACM", "Cora"})

# ------------------------------ protocol config -----------------------------
N_OUTER_FOLDS = 5          # Outer CV: train_full / test.
VAL_SIZE = 0.25            # Inner hold-out: train_full -> train / validation.
REFIT_ON_TRAIN_PLUS_VAL = True
SEED = 42
N_TRIALS_PER_DIMENSION = 10
# Correction terms of the MV-LSSVM extension (Multiclass_MV_LSSVM.reconstruct).
# True: removed samples' errors are reconstructed (proposed model).
# False: kept samples only, no correction terms (ablation). Set with the
# --no-correction command-line flag; each setting has its own results folder.
RECONSTRUCT = True
RESULTS_ROOTS = {
    True: Path("results_mv_lssvm_correction"),
    False: Path("results_mv_lssvm_no_correction"),
}
MODEL_NAMES = {True: "mv_lssvm_correction", False: "mv_lssvm_no_correction"}
RESULTS_ROOT = RESULTS_ROOTS[RECONSTRUCT]
MODEL_NAME = MODEL_NAMES[RECONSTRUCT]

SKIP_COMPLETED = True      # Skip runs whose summary.csv already exists.
SAVE_PREDICTIONS = True    # Save test predictions of every fold.
SAVE_SELECTED_INDICES = True  # Save the subsampled rows of every fold.

# ------------------------------ sampler registry ----------------------------
SAMPLERS: Mapping[str, type] = {
    "random": RandomSubsamplingPerView,
    "MV_kernel_herding": MultiViewKernelHerding,
    # Plain MMD herding: herding with warmup over all n_sub steps.
    "simple_kernel_herding": MultiViewKernelHerding,
    "clustering": ClusteringSubsampling,
    # Fixed-size LS-SVM prototypes (De Brabanter et al., 2010): maximum
    # quadratic Renyi entropy, per view, with the trial's kernel gammas.
    "fixed_size_entropy": FixedSizeEntropySubsampling,
}
# Fixed constructor arguments. n_sub, kernel_gammas and random_state are
# always set by this script.
SAMPLER_KWARGS: Mapping[str, Mapping[str, Any]] = {
    "random": {},
    "MV_kernel_herding": {"epsilon": np.inf},
    # warmup_steps = 1.0 (100% of n_sub): no joint HSIC phase. Fixed, not tuned.
    "simple_kernel_herding": {"epsilon": np.inf, "warmup_steps": 1.0},
    "clustering": {},
    "fixed_size_entropy": {},
}
# Herding warmup_steps, as a fraction of n_sub (0 itself is invalid).
WARMUP_RANGE = (0.01, 1.0)

METRIC_NAMES = ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted")
TIME_COLUMNS = (
    "subsampling_time_s",
    "training_time_s",
    "predict_time_s",
    "optuna_time_s",
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------------ helpers -------------------------------------
def build_runs(
    datasets: Sequence[str],
    methods: Sequence[str],
    n_subs: Sequence[int],
    large_n_subs: Sequence[int],
) -> list[tuple[str, str, int]]:
    """Lists the (dataset, method, n_sub) runs; LARGE_DATASETS also get
    the `large_n_subs` sizes."""
    return [
        (dataset, method, n_sub)
        for dataset in datasets
        for method in methods
        for n_sub in list(n_subs)
        + (list(large_n_subs) if dataset in LARGE_DATASETS else [])
    ]


def take(views: Sequence[np.ndarray], idx: np.ndarray) -> list[np.ndarray]:
    """Selects the rows `idx` from every view."""
    return [view[idx] for view in views]


def n_trials_for(method: str, n_views: int) -> int:
    """Returns the number of Optuna trials for a method and view count."""
    n_trials = N_TRIALS_PER_DIMENSION * (2 * n_views + 1)
    if method == "MV_kernel_herding":
        # Herding has one extra tuned hyperparameter (warmup_steps).
        n_trials += 2 * N_TRIALS_PER_DIMENSION
    return n_trials


def sampler_params_from(
    method: str, source: optuna.trial.Trial | Mapping[str, float]
) -> dict[str, float]:
    """Reads the tuned subsampler hyperparameters.

    Args:
        method: Subsampling method name.
        source: An Optuna trial (values are suggested) or a dict of values.

    Returns:
        Keyword arguments for the sampler. Only kernel herding has tuned
        arguments (warmup_steps).
    """
    if method != "MV_kernel_herding":
        return {}
    if isinstance(source, optuna.trial.Trial):
        warmup = source.suggest_float("warmup_steps", *WARMUP_RANGE)
    else:
        warmup = source["warmup_steps"]
    return {"warmup_steps": float(warmup)}


def params_from(
    source: optuna.trial.Trial | Mapping[str, float], n_views: int
) -> tuple[list[float], list[float], float]:
    """Reads the model hyperparameters from a trial or a dict.

    Args:
        source: An Optuna trial (values are suggested) or a dict of values.
        n_views: Number of views of the dataset.

    Returns:
        A tuple (regularization gammas, kernel gammas, rho).
    """
    if isinstance(source, optuna.trial.Trial):
        reg = [
            source.suggest_float(f"gamma_reg_{v}", 1e-3, 1e3, log=True)
            for v in range(n_views)
        ]
        ker = [
            source.suggest_float(f"gamma_kernel_{v}", 1e-4, 1e2, log=True)
            for v in range(n_views)
        ]
        rho_ratio = source.suggest_float("rho_ratio", 1e-3, 0.95, log=True)
    else:
        reg = [source[f"gamma_reg_{v}"] for v in range(n_views)]
        ker = [source[f"gamma_kernel_{v}"] for v in range(n_views)]
        rho_ratio = source["rho_ratio"]
    return reg, ker, rho_ratio * min(reg)


def subsample(
    method: str,
    views: Sequence[np.ndarray],
    labels: np.ndarray,
    n_sub: int,
    kernel_gammas: Sequence[float],
    random_state: int,
    sampler_params: Mapping[str, float] | None = None,
) -> tuple[list[np.ndarray], int]:
    """Runs the selected subsampling method in a stratified way.

    The sampler is run independently on the rows of each class, selecting
    ceil(n_sub / n_classes) rows per class (capped at the class size). The
    per-class subsets are then merged (union) into one subset per view.
    The total size is therefore n_classes * ceil(n_sub / n_classes), i.e.
    at least n_sub unless a class is smaller than its quota.

    Args:
        method: Key of SAMPLERS.
        views: Feature matrices, one per view.
        labels: Class labels.
        n_sub: Requested number of selected rows.
        kernel_gammas: Kernel gammas, one per view.
        random_state: Seed of the sampler.
        sampler_params: Tuned sampler arguments.

    Returns:
        A tuple (selected indices per view, effective n_sub). Indices are
        relative to `views`/`labels`.
    """
    classes = np.unique(labels)
    per_class = math.ceil(n_sub / len(classes))
    selected_parts: dict[str, list[np.ndarray]] = {}
    view_names: list[str] = []
    n_eff = 0
    for class_pos, cls in enumerate(classes):
        cls_idx = np.flatnonzero(labels == cls)
        m = min(per_class, len(cls_idx))
        n_eff += m
        sampler = SAMPLERS[method](
            n_sub=m,
            kernel_gammas=kernel_gammas,
            random_state=random_state + class_pos,
            **SAMPLER_KWARGS.get(method, {}),
            **(sampler_params or {}),
        )
        sampler.fit(take(views, cls_idx), labels[cls_idx])
        view_names = list(sampler.view_names_)
        for name in view_names:
            # Map class-local indices back to indices of `views`.
            local = np.asarray(sampler.selected_indices_[name], dtype=int)
            selected_parts.setdefault(name, []).append(cls_idx[local])
    # Union of the per-class subsets, one array per view.
    selected = [
        np.sort(np.concatenate(selected_parts[name])) for name in view_names
    ]
    check_class_coverage(selected, labels)
    return selected, n_eff


def build_model(
    reg: Sequence[float],
    ker: Sequence[float],
    rho: float,
    selected: Sequence[np.ndarray],
) -> Multiclass_MV_LSSVM:
    """Builds the multi-view LS-SVM."""
    return Multiclass_MV_LSSVM(
        kernel_gammas=ker, selected_indices=selected, gammas=reg, rho=rho,
        reconstruct=RECONSTRUCT,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Computes the classification metrics listed in METRIC_NAMES."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


class InvalidSubsamplingError(ValueError):
    """Raised when a subsample leaves a class without retained points."""


def check_class_coverage(
    selected: Sequence[np.ndarray], labels: np.ndarray
) -> None:
    """Checks that every view retains at least one point of every class.

    The multi-view LS-SVM needs this to reconstruct the removed points.

    Args:
        selected: Selected indices per view, relative to `labels`.
        labels: Class labels of the rows that were subsampled.

    Raises:
        InvalidSubsamplingError: If a class is missing in some view.
    """
    classes = np.unique(labels)
    for view_id, sel in enumerate(selected):
        missing = np.setdiff1d(classes, labels[np.asarray(sel)])
        if missing.size:
            raise InvalidSubsamplingError(
                f"View {view_id}: classes {missing.tolist()} have no retained "
                f"point (n_sub={len(sel)}). Use a stratified sampler or a "
                "larger n_sub."
            )


def timed(func: Callable[[], Any]) -> tuple[Any, float]:
    """Calls `func` and returns (result, elapsed seconds)."""
    start = time.perf_counter()
    result = func()
    return result, time.perf_counter() - start


def write_mapping_csv(path: Path, mapping: Mapping[str, Any]) -> None:
    """Writes a dict as a two-column (key, value) CSV."""
    pd.DataFrame(
        {"key": list(mapping), "value": [str(v) for v in mapping.values()]}
    ).to_csv(path, index=False)


def get_git_commit() -> str:
    """Returns the current git commit hash, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_experiment_metadata(root: Path) -> None:
    """Stores environment and global configuration for reproducibility."""
    root.mkdir(parents=True, exist_ok=True)
    write_mapping_csv(
        root / "experiment_metadata.csv",
        {
            "model": MODEL_NAME,
            "reconstruct": RECONSTRUCT,
            "start_time": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_commit": get_git_commit(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "optuna": optuna.__version__,
            "datasets": list(DATASETS),
            "subsampling_methods": list(SUBSAMPLING_METHODS),
            "n_subs": list(N_SUBS),
            "large_n_subs": list(LARGE_N_SUBS),
            "large_datasets": sorted(LARGE_DATASETS),
            "n_outer_folds": N_OUTER_FOLDS,
            "val_size": VAL_SIZE,
            "refit_on_train_plus_val": REFIT_ON_TRAIN_PLUS_VAL,
            "seed": SEED,
            "n_trials_per_dimension": N_TRIALS_PER_DIMENSION,
            "warmup_range": WARMUP_RANGE,
            "sampler_kwargs": dict(SAMPLER_KWARGS),
        },
    )


# ------------------------------ nested CV -----------------------------------
def run_nested_cv(
    dataset: str,
    method: str,
    n_sub: int,
    views: Sequence[np.ndarray],
    y: np.ndarray,
    out_dir: Path,
) -> None:
    """Runs one nested-CV experiment and writes its CSV files to `out_dir`.

    Args:
        dataset: Dataset name (used for identification columns only).
        method: Key of SAMPLERS.
        n_sub: Requested number of subsampled rows.
        views: Feature matrices, one per view.
        y: Class labels.
        out_dir: Output directory of this run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_views = len(views)
    n_trials = n_trials_for(method, n_views)
    run_id = {"model": MODEL_NAME, "dataset": dataset,
              "subsampling_method": method, "n_sub": n_sub}

    write_mapping_csv(
        out_dir / "run_config.csv",
        {
            **run_id,
            "reconstruct": RECONSTRUCT,
            "n_samples": len(y),
            "n_views": n_views,
            "n_features_per_view": [v.shape[1] for v in views],
            "n_classes": len(np.unique(y)),
            "samples_per_class": math.ceil(n_sub / len(np.unique(y))),
            "class_counts": np.bincount(np.unique(y, return_inverse=True)[1]),
            "n_trials": n_trials,
            "n_outer_folds": N_OUTER_FOLDS,
            "val_size": VAL_SIZE,
            "refit_on_train_plus_val": REFIT_ON_TRAIN_PLUS_VAL,
            "seed": SEED,
            "sampler_kwargs": dict(SAMPLER_KWARGS.get(method, {})),
        },
    )

    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED
    )
    metric_rows, hyper_rows, index_rows, prediction_frames = [], [], [], []

    for fold, (train_full_idx, test_idx) in enumerate(
        outer_cv.split(np.zeros(len(y)), y)
    ):
        # 1) Train / validation / test split.
        train_idx, val_idx = train_test_split(
            train_full_idx,
            test_size=VAL_SIZE,
            stratify=y[train_full_idx],
            random_state=SEED + fold,
        )
        x_train, y_train = take(views, train_idx), y[train_idx]
        x_val, y_val = take(views, val_idx), y[val_idx]

        # 2) Hyperparameter search: subsample on train, fit, score on val.
        def objective(trial: optuna.trial.Trial) -> float:
            reg, ker, rho = params_from(trial, n_views)
            sampler_params = sampler_params_from(method, trial)
            (selected, _), sub_time = timed(
                lambda: subsample(
                    method, x_train, y_train, n_sub, ker, SEED + fold,
                    sampler_params,
                )
            )
            model = build_model(reg, ker, rho, selected)
            _, train_time = timed(lambda: model.fit(x_train, y_train))
            trial.set_user_attr("subsampling_time_s", sub_time)
            trial.set_user_attr("training_time_s", train_time)
            return accuracy_score(y_val, model.predict(x_val))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED + fold),
        )
        _, optuna_time = timed(
            lambda: study.optimize(
                objective,
                n_trials=n_trials,
                catch=(np.linalg.LinAlgError, InvalidSubsamplingError),
            )
        )
        if not any(
            t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
        ):
            raise RuntimeError(
                f"All {n_trials} trials failed in fold {fold}; check n_sub "
                "and the sampler (see the warnings above)."
            )

        best = study.best_params
        reg, ker, rho = params_from(best, n_views)
        best_sampler_params = sampler_params_from(method, best)

        # 3) Final model with the best hyperparameters on the outer test.
        fit_idx = train_full_idx if REFIT_ON_TRAIN_PLUS_VAL else train_idx
        x_fit, y_fit = take(views, fit_idx), y[fit_idx]
        x_test, y_test = take(views, test_idx), y[test_idx]

        (selected, n_eff), subsampling_time = timed(
            lambda: subsample(
                method, x_fit, y_fit, n_sub, ker, SEED + fold,
                best_sampler_params,
            )
        )
        model = build_model(reg, ker, rho, selected)
        _, training_time = timed(lambda: model.fit(x_fit, y_fit))
        y_pred, predict_time = timed(lambda: model.predict(x_test))

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        metric_rows.append(
            {
                **run_id,
                "fold": fold,
                "n_sub_effective": n_eff,
                "n_fit": len(fit_idx),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "val_accuracy_best": study.best_value,
                **{
                    f"test_{k}": v
                    for k, v in compute_metrics(y_test, y_pred).items()
                },
                "subsampling_time_s": subsampling_time,
                "training_time_s": training_time,
                "predict_time_s": predict_time,
                "optuna_time_s": optuna_time,
                "optuna_mean_trial_subsampling_time_s": float(np.mean(
                    [t.user_attrs["subsampling_time_s"] for t in completed]
                )),
                "optuna_mean_trial_training_time_s": float(np.mean(
                    [t.user_attrs["training_time_s"] for t in completed]
                )),
                "n_trials_completed": len(completed),
            }
        )
        hyper_rows.append(
            {
                **run_id,
                "fold": fold,
                "best_trial_number": study.best_trial.number,
                "best_val_accuracy": study.best_value,
                **{f"best_{k}": v for k, v in best.items()},
                "best_rho": rho,
            }
        )
        if SAVE_SELECTED_INDICES:
            for view_id, sel in enumerate(selected):
                # Map to row indices of the original dataset.
                for row in fit_idx[np.asarray(sel)]:
                    index_rows.append(
                        {**run_id, "fold": fold, "view": view_id,
                         "row_index": int(row)}
                    )
        if SAVE_PREDICTIONS:
            prediction_frames.append(
                pd.DataFrame(
                    {**run_id, "fold": fold, "row_index": test_idx,
                     "y_true": y_test, "y_pred": y_pred}
                )
            )

        # Save after every outer fold so a crash loses little work.
        pd.DataFrame(metric_rows).to_csv(
            out_dir / "outer_fold_metrics.csv", index=False
        )
        pd.DataFrame(hyper_rows).to_csv(
            out_dir / "best_hyperparameters.csv", index=False
        )
        study.trials_dataframe().to_csv(
            out_dir / f"optuna_trials_fold{fold}.csv", index=False
        )
        logger.info(
            "[%s/%s/%d fold %d] val_acc=%.4f test_acc=%.4f f1_macro=%.4f "
            "subsampling=%.2fs training=%.3fs optuna=%.1fs",
            dataset, method, n_sub, fold, study.best_value,
            metric_rows[-1]["test_accuracy"], metric_rows[-1]["test_f1_macro"],
            subsampling_time, training_time, optuna_time,
        )

    if SAVE_SELECTED_INDICES:
        pd.DataFrame(index_rows).to_csv(
            out_dir / "selected_indices.csv", index=False
        )
    if SAVE_PREDICTIONS:
        pd.concat(prediction_frames).to_csv(
            out_dir / "predictions.csv", index=False
        )

    # Written last: its existence marks the run as completed.
    df = pd.DataFrame(metric_rows)
    summary_cols = [c for c in df.columns if c.startswith("test_")] + list(
        TIME_COLUMNS
    )
    df[summary_cols].agg(["mean", "std"]).T.to_csv(out_dir / "summary.csv")


# ------------------------------ aggregation ---------------------------------
def aggregate_results(root: Path) -> None:
    """Concatenates per-run CSVs into global CSVs under `root`.

    Args:
        root: Results root directory.
    """
    for name, target in (
        ("outer_fold_metrics.csv", "all_outer_fold_metrics.csv"),
        ("best_hyperparameters.csv", "all_best_hyperparameters.csv"),
    ):
        files = sorted(root.glob(f"*/*/*/{name}"))
        if files:
            pd.concat(
                [pd.read_csv(f) for f in files], ignore_index=True
            ).to_csv(root / target, index=False)

    metrics_path = root / "all_outer_fold_metrics.csv"
    if not metrics_path.exists():
        return
    df = pd.read_csv(metrics_path)
    if "training_time_s" not in df.columns:
        return  # Only legacy files (older column names) found.
    df = df.dropna(subset=["training_time_s"])  # Ignore legacy rows.
    value_cols = [c for c in df.columns if c.startswith("test_")] + list(
        TIME_COLUMNS
    )
    grouped = df.groupby(["dataset", "subsampling_method", "n_sub"])
    summary = grouped[value_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary["n_folds"] = grouped.size()
    summary.reset_index().to_csv(root / "all_summaries.csv", index=False)


def append_manifest(root: Path, row: Mapping[str, Any]) -> None:
    """Appends one row to manifest.csv (creating it if needed)."""
    path = root / "manifest.csv"
    pd.DataFrame([row]).to_csv(
        path, mode="a", header=not path.exists(), index=False
    )


# ------------------------------ main ----------------------------------------
def main() -> None:
    """Runs the whole experiment grid sequentially."""
    global RECONSTRUCT, RESULTS_ROOT, MODEL_NAME
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--no-correction", action="store_true",
        help="Train without correction terms (reconstruct=False).",
    )
    args = parser.parse_args()
    RECONSTRUCT = not args.no_correction
    RESULTS_ROOT = RESULTS_ROOTS[RECONSTRUCT]
    MODEL_NAME = MODEL_NAMES[RECONSTRUCT]

    unknown = [m for m in SUBSAMPLING_METHODS if m not in SAMPLERS]
    if unknown:
        raise KeyError(f"Unknown methods {unknown}. Options: {list(SAMPLERS)}")

    write_experiment_metadata(RESULTS_ROOT)
    runs = build_runs(DATASETS, SUBSAMPLING_METHODS, N_SUBS, LARGE_N_SUBS)
    logger.info("Starting %d %s runs.", len(runs), MODEL_NAME)

    data_cache: dict[str, tuple[list[np.ndarray], np.ndarray]] = {}
    for i, (dataset, method, n_sub) in enumerate(runs, start=1):
        out_dir = RESULTS_ROOT / dataset / method / str(n_sub)
        tag = f"[{i}/{len(runs)}] {MODEL_NAME}/{dataset}/{method}/{n_sub}"
        run_id = {"model": MODEL_NAME, "dataset": dataset,
                  "subsampling_method": method, "n_sub": n_sub}

        if SKIP_COMPLETED and (out_dir / "summary.csv").exists():
            logger.info("%s already completed, skipping.", tag)
            continue

        start = time.perf_counter()
        try:
            if dataset not in data_cache:
                views, labels = get_dataset(dataset)
                data_cache[dataset] = (list(views), np.asarray(labels))
            views, labels = data_cache[dataset]
            logger.info("%s starting.", tag)
            run_nested_cv(dataset, method, n_sub, views, labels, out_dir)
            status, error = "ok", ""
        except Exception as exc:  # Keep the grid going if one run fails.
            logger.error("%s failed:\n%s", tag, traceback.format_exc())
            status, error = "failed", f"{type(exc).__name__}: {exc}"

        append_manifest(
            RESULTS_ROOT,
            {
                **run_id,
                "status": status,
                "elapsed_s": time.perf_counter() - start,
                "finished_at": datetime.datetime.now().isoformat(
                    timespec="seconds"
                ),
                "error": error,
            },
        )
        aggregate_results(RESULTS_ROOT)

    logger.info("All runs finished. Results in %s", RESULTS_ROOT)


if __name__ == "__main__":
    main()