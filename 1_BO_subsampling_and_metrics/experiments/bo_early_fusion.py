"""Launches consecutive nested-CV experiments for the early-fusion baseline.

Same protocol as experiments/bo_accuracy_in_view.py, but the model is
EarlyFusion_LSSVM: all views are concatenated into one single-view dataset
(ViewConcatenator, fitted on the training rows of each split), the
subsampling methods are run on that single view, and one LS-SVM is trained
on the subsample. Only one regularization and one kernel gamma (and the
herding warmup) are tuned.

Run from 1_BO_subsampling_and_metrics:

    python -m experiments.bo_early_fusion

Output layout (under RESULTS_ROOT) is identical to bo_accuracy_in_view.py;
selected_indices.csv has a single view (view = 0, the concatenation).
"""

from __future__ import annotations

import datetime
import math
import platform
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from experiments.bo_accuracy_in_view import (
    SAMPLER_KWARGS,
    SAMPLERS,
    WARMUP_RANGE,
    InvalidSubsamplingError,
    aggregate_results,
    append_manifest,
    compute_metrics,
    get_git_commit,
    logger,
    sampler_params_from,
    subsample,
    take,
    timed,
    write_mapping_csv,
    FEATURE_SCALING,
    minmax_scale,
    LARGE_DATASETS,
    build_runs,
)
from get_data import get_dataset
from multi_view_learning_models import EarlyFusion_LSSVM, ViewConcatenator

MODEL_NAME = "early_fusion"

# ------------------------------ experiment grid -----------------------------
# Ordered by number of samples, small to large.
DATASETS: Sequence[str] = ["3Sources", "YouTube", "Prokaryotic", "ProteinFold", "Caltech", "NUS-WIDE", "OutdoorScene", "Cora", "ACM"]
# Single (concatenated) view: multi-view "MV_kernel_herding" has no HSIC phase
# and reduces to simple_kernel_herding, so it is not run.
SUBSAMPLING_METHODS: Sequence[str] = ["random", "simple_kernel_herding", "clustering", "fixed_size_entropy"]
N_SUBS: Sequence[int] = [10, 25, 50, 100, 150, 200]
LARGE_N_SUBS: Sequence[int] = [300, 400, 500, 600]  # LARGE_DATASETS only.

# ------------------------------ protocol config -----------------------------
N_OUTER_FOLDS = 5          # Outer CV: train_full / test.
VAL_SIZE = 0.25            # Inner hold-out: train_full -> train / validation.
REFIT_ON_TRAIN_PLUS_VAL = True
SEED = 42
N_TRIALS_PER_DIMENSION = 10
RESULTS_ROOT = Path("results_early_fusion")  # Kept apart from other models.
# ViewConcatenator scaling: "weight" (views already MinMax-scaled; weight
# view v by 1/sqrt(d_v)) or None (plain concatenation).
SCALING = "weight"

SKIP_COMPLETED = True      # Skip runs whose summary.csv already exists.
SAVE_PREDICTIONS = True    # Save test predictions of every fold.
SAVE_SELECTED_INDICES = True  # Save the subsampled rows of every fold.


# ------------------------------ helpers -------------------------------------
def n_trials_for(method: str) -> int:
    """Returns the number of Optuna trials for a method."""
    n_trials = N_TRIALS_PER_DIMENSION * 2  # gamma_reg and gamma_kernel.
    if method == "MV_kernel_herding":
        # Herding has one extra tuned hyperparameter (warmup_steps).
        n_trials += 2 * N_TRIALS_PER_DIMENSION
    return n_trials


def params_from(
    source: optuna.trial.Trial | Mapping[str, float],
) -> tuple[float, float]:
    """Reads the model hyperparameters from a trial or a dict.

    Args:
        source: An Optuna trial (values are suggested) or a dict of values.

    Returns:
        A tuple (regularization gamma, kernel gamma).
    """
    if isinstance(source, optuna.trial.Trial):
        reg = source.suggest_float("gamma_reg", 1e-3, 1e3, log=True)
        ker = source.suggest_float("gamma_kernel", 1e-4, 1e2, log=True)
    else:
        reg, ker = source["gamma_reg"], source["gamma_kernel"]
    return reg, ker


def subsample_concatenated(
    method: str,
    views: Sequence[np.ndarray],
    labels: np.ndarray,
    n_sub: int,
    ker: float,
    random_state: int,
    sampler_params: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, int]:
    """Concatenates the views and subsamples the resulting single view.

    The concatenator is fitted on `views` (the training rows), exactly as
    inside EarlyFusion_LSSVM.fit, so the sampler sees the model's input.

    Returns:
        A tuple (selected indices relative to `views`, effective n_sub).
    """
    x_concat = ViewConcatenator(SCALING).fit(views).transform(views)
    selected, n_eff = subsample(
        method, [x_concat], labels, n_sub, [ker], random_state, sampler_params
    )
    return selected[0], n_eff


def build_model(reg: float, ker: float, selected: np.ndarray) -> EarlyFusion_LSSVM:
    """Builds the early-fusion LS-SVM."""
    return EarlyFusion_LSSVM(
        kernel_gamma=ker, selected_indices=selected, gamma=reg, scaling=SCALING
    )


def write_experiment_metadata(root: Path) -> None:
    """Stores environment and global configuration for reproducibility."""
    root.mkdir(parents=True, exist_ok=True)
    write_mapping_csv(
        root / "experiment_metadata.csv",
        {
            "model": MODEL_NAME,
            "scaling": SCALING,
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
            "feature_scaling": FEATURE_SCALING,
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
        views: Feature matrices, one per view (concatenated per split).
        y: Class labels.
        out_dir: Output directory of this run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_views = len(views)
    n_trials = n_trials_for(method)
    run_id = {"model": MODEL_NAME, "dataset": dataset,
              "subsampling_method": method, "n_sub": n_sub}

    write_mapping_csv(
        out_dir / "run_config.csv",
        {
            **run_id,
            "scaling": SCALING,
            "n_samples": len(y),
            "n_views": n_views,
            "n_features_per_view": [v.shape[1] for v in views],
            "n_features_concatenated": sum(v.shape[1] for v in views),
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
        # MinMax scaling fitted on the training rows only (no leakage).
        x_train, x_val = minmax_scale(x_train, x_val)

        # 2) Hyperparameter search: subsample on train, fit, score on val.
        def objective(trial: optuna.trial.Trial) -> float:
            reg, ker = params_from(trial)
            sampler_params = sampler_params_from(method, trial)
            (selected, _), sub_time = timed(
                lambda: subsample_concatenated(
                    method, x_train, y_train, n_sub, ker, SEED + fold,
                    sampler_params,
                )
            )
            model = build_model(reg, ker, selected)
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
        reg, ker = params_from(best)
        best_sampler_params = sampler_params_from(method, best)

        # 3) Final model with the best hyperparameters on the outer test.
        fit_idx = train_full_idx if REFIT_ON_TRAIN_PLUS_VAL else train_idx
        x_fit, y_fit = take(views, fit_idx), y[fit_idx]
        x_test, y_test = take(views, test_idx), y[test_idx]
        x_fit, x_test = minmax_scale(x_fit, x_test)

        (selected, n_eff), subsampling_time = timed(
            lambda: subsample_concatenated(
                method, x_fit, y_fit, n_sub, ker, SEED + fold,
                best_sampler_params,
            )
        )
        model = build_model(reg, ker, selected)
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
            }
        )
        if SAVE_SELECTED_INDICES:
            # Map to row indices of the original dataset.
            for row in fit_idx[np.asarray(selected)]:
                index_rows.append(
                    {**run_id, "fold": fold, "view": 0, "row_index": int(row)}
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
            "[%s/%s/%s/%d fold %d] val_acc=%.4f test_acc=%.4f f1_macro=%.4f "
            "subsampling=%.2fs training=%.3fs optuna=%.1fs",
            MODEL_NAME, dataset, method, n_sub, fold, study.best_value,
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
    summary_cols = [c for c in df.columns if c.startswith("test_")] + [
        "subsampling_time_s", "training_time_s", "predict_time_s",
        "optuna_time_s",
    ]
    df[summary_cols].agg(["mean", "std"]).T.to_csv(out_dir / "summary.csv")


# ------------------------------ main ----------------------------------------
def main() -> None:
    """Runs the whole experiment grid sequentially."""
    unknown = [m for m in SUBSAMPLING_METHODS if m not in SAMPLERS]
    if unknown:
        raise KeyError(f"Unknown methods {unknown}. Options: {list(SAMPLERS)}")

    write_experiment_metadata(RESULTS_ROOT)
    runs = build_runs(DATASETS, SUBSAMPLING_METHODS, N_SUBS, LARGE_N_SUBS)
    logger.info("Starting %d early-fusion runs.", len(runs))

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
