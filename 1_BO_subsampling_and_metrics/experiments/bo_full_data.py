"""Full-data baselines: every model trained on ALL training samples (no
subsampling), with the same nested-CV protocol as the subsampling scripts.

Models (choose with --model):
    mv_lssvm      Regular MV-LSSVM: Multiclass_MV_LSSVM with every sample kept
                  in every view (P = I, so there are no correction terms).
    late_fusion   LateFusion_LSSVM, one LS-SVM per view on all samples.
    early_fusion  EarlyFusion_LSSVM on the concatenated views.
    aw_lssvm      AW_LSSVM_Partial with every sample in every view, i.e. the
                  original AW-LSSVM (adaptive weights from all other views).

Hyperparameters and their search ranges are those of the matching
subsampling script (bo_accuracy_in_view.py, bo_late_fusion.py,
bo_early_fusion.py, bo_aw_lssvm.py), so the results are directly comparable.

Run from 1_BO_subsampling_and_metrics:

    python3 -m experiments.bo_full_data --model mv_lssvm
    python3 -m experiments.bo_full_data --model late_fusion
    python3 -m experiments.bo_full_data --model early_fusion
    python3 -m experiments.bo_full_data --model aw_lssvm

Output layout (under RESULTS_ROOT/<model>), same files as the other scripts:
    <dataset>/full/all/   run_config, outer_fold_metrics, best_hyperparameters,
                          optuna_trials_fold<k>, predictions, summary
Rows have subsampling_method = "full" and n_sub = "all"; the number of
training samples is in n_sub_effective.
"""

from __future__ import annotations

import argparse
import datetime
import platform
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import experiments.bo_accuracy_in_view as mv_exp
import experiments.bo_aw_lssvm as aw_exp
import experiments.bo_early_fusion as early_exp
import experiments.bo_late_fusion as late_exp
from experiments.bo_accuracy_in_view import (
    aggregate_results,
    append_manifest,
    compute_metrics,
    get_git_commit,
    logger,
    take,
    timed,
    write_mapping_csv,
    FEATURE_SCALING,
    minmax_scale,
    check_protocol,
)
from get_data import get_dataset
from multi_view_learning_models import (
    AW_LSSVM_Partial,
    EarlyFusion_LSSVM,
    LateFusion_LSSVM,
    Multiclass_MV_LSSVM,
)

# ------------------------------ experiment grid -----------------------------
# Ordered by number of samples, small to large.
DATASETS: Sequence[str] = ["3Sources", "YouTube", "Prokaryotic", "ProteinFold", "Caltech", "NUS-WIDE", "OutdoorScene", "Cora", "ACM"]
MODELS: Sequence[str] = ["mv_lssvm", "late_fusion", "early_fusion", "aw_lssvm"]

# ------------------------------ protocol config -----------------------------
# Kept identical to the subsampling scripts.
N_OUTER_FOLDS = 10         # Outer CV: train_full / test.
VAL_SIZE = 0.25            # Inner hold-out: train_full -> train / validation.
REFIT_ON_TRAIN_PLUS_VAL = True
SEED = 42
N_TRIALS_PER_DIMENSION = 10
RESULTS_ROOT = Path("results_full_data")  # One sub-folder per model.

SKIP_COMPLETED = True      # Skip runs whose summary.csv already exists.
SAVE_PREDICTIONS = True    # Save test predictions of every fold.

METHOD = "full"            # Value of the subsampling_method column.
N_SUB = "all"              # Value of the n_sub column.


# ------------------------------ models --------------------------------------
def all_rows(n: int, n_views: int) -> list[np.ndarray]:
    """Selected indices that keep every training sample in every view."""
    return [np.arange(n)] * n_views


def mv_lssvm_spec(n_views: int) -> tuple[int, Callable, Callable]:
    """(n_trials, params_from, build_model) for the regular MV-LSSVM."""
    def params_from(source):
        return mv_exp.params_from(source, n_views)

    def build(params, n):
        reg, ker, rho = params
        # All samples kept: P = I, so reconstruct has no effect; False makes
        # it explicit that there are no correction terms.
        return Multiclass_MV_LSSVM(
            kernel_gammas=ker, selected_indices=all_rows(n, n_views),
            gammas=reg, rho=rho, reconstruct=False,
        )
    return mv_exp.n_trials_for(METHOD, n_views), params_from, build


def late_fusion_spec(n_views: int) -> tuple[int, Callable, Callable]:
    """(n_trials, params_from, build_model) for late fusion."""
    def params_from(source):
        return late_exp.params_from(source, n_views)

    def build(params, n):
        reg, ker = params
        return LateFusion_LSSVM(
            kernel_gammas=ker, selected_indices=all_rows(n, n_views), gammas=reg
        )
    return late_exp.n_trials_for(METHOD, n_views), params_from, build


def early_fusion_spec(n_views: int) -> tuple[int, Callable, Callable]:
    """(n_trials, params_from, build_model) for early fusion."""
    def build(params, n):
        reg, ker = params
        return EarlyFusion_LSSVM(
            kernel_gamma=ker, selected_indices=np.arange(n), gamma=reg,
            scaling=early_exp.SCALING,
        )
    return early_exp.n_trials_for(METHOD), early_exp.params_from, build


def aw_lssvm_spec(n_views: int) -> tuple[int, Callable, Callable]:
    """(n_trials, params_from, build_model) for AW-LSSVM on all samples."""
    def params_from(source):
        return aw_exp.params_from(source, n_views)

    def build(params, n):
        reg, ker, ro, decay = params
        return AW_LSSVM_Partial(
            kernel_gammas=ker, selected_indices=all_rows(n, n_views),
            gammas=reg, ro=ro, decay=decay,
            coupling=aw_exp.AW_COUPLING, range_steps=aw_exp.AW_RANGE_STEPS,
        )
    return aw_exp.n_trials_for(METHOD, n_views), params_from, build


MODEL_SPECS: Mapping[str, Callable[[int], tuple[int, Callable, Callable]]] = {
    "mv_lssvm": mv_lssvm_spec,
    "late_fusion": late_fusion_spec,
    "early_fusion": early_fusion_spec,
    "aw_lssvm": aw_lssvm_spec,
}
MODEL_NAMES = {m: f"{m}_full" for m in MODEL_SPECS}  # Value of the model column.


# ------------------------------ helpers -------------------------------------
def write_experiment_metadata(root: Path, model: str) -> None:
    """Stores environment and global configuration for reproducibility."""
    root.mkdir(parents=True, exist_ok=True)
    write_mapping_csv(
        root / "experiment_metadata.csv",
        {
            "model": MODEL_NAMES[model],
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
            "n_outer_folds": N_OUTER_FOLDS,
            "val_size": VAL_SIZE,
            "refit_on_train_plus_val": REFIT_ON_TRAIN_PLUS_VAL,
            "seed": SEED,
            "n_trials_per_dimension": N_TRIALS_PER_DIMENSION,
            "feature_scaling": FEATURE_SCALING,
            "early_fusion_scaling": early_exp.SCALING,
        },
    )


# ------------------------------ nested CV -----------------------------------
def run_nested_cv(
    dataset: str,
    model: str,
    views: Sequence[np.ndarray],
    y: np.ndarray,
    out_dir: Path,
) -> None:
    """Runs one nested-CV experiment and writes its CSV files to `out_dir`.

    Args:
        dataset: Dataset name (used for identification columns only).
        model: Key of MODEL_SPECS.
        views: Feature matrices, one per view.
        y: Class labels.
        out_dir: Output directory of this run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_views = len(views)
    n_trials, params_from, build_model = MODEL_SPECS[model](n_views)
    run_id = {"model": MODEL_NAMES[model], "dataset": dataset,
              "subsampling_method": METHOD, "n_sub": N_SUB}

    write_mapping_csv(
        out_dir / "run_config.csv",
        {
            **run_id,
            "n_samples": len(y),
            "n_views": n_views,
            "n_features_per_view": [v.shape[1] for v in views],
            "n_classes": len(np.unique(y)),
            "class_counts": np.bincount(np.unique(y, return_inverse=True)[1]),
            "n_trials": n_trials,
            "n_outer_folds": N_OUTER_FOLDS,
            "val_size": VAL_SIZE,
            "refit_on_train_plus_val": REFIT_ON_TRAIN_PLUS_VAL,
            "seed": SEED,
        },
    )

    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED
    )
    metric_rows, hyper_rows, prediction_frames = [], [], []

    for fold, (train_full_idx, test_idx) in enumerate(
        outer_cv.split(np.zeros(len(y)), y)
    ):
        # 1) Train / validation / test split (same seeds as the other scripts).
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

        # 2) Hyperparameter search: fit on all of train, score on val.
        def objective(trial: optuna.trial.Trial) -> float:
            clf = build_model(params_from(trial), len(train_idx))
            _, train_time = timed(lambda: clf.fit(x_train, y_train))
            trial.set_user_attr("training_time_s", train_time)
            return accuracy_score(y_val, clf.predict(x_val))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED + fold),
        )
        _, optuna_time = timed(
            lambda: study.optimize(
                objective, n_trials=n_trials, catch=(np.linalg.LinAlgError,)
            )
        )
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            raise RuntimeError(
                f"All {n_trials} trials failed in fold {fold} (see warnings)."
            )

        best = study.best_params
        best_params = params_from(best)

        # 3) Final model with the best hyperparameters on the outer test.
        fit_idx = train_full_idx if REFIT_ON_TRAIN_PLUS_VAL else train_idx
        x_fit, y_fit = take(views, fit_idx), y[fit_idx]
        x_test, y_test = take(views, test_idx), y[test_idx]
        x_fit, x_test = minmax_scale(x_fit, x_test)

        clf = build_model(best_params, len(fit_idx))
        _, training_time = timed(lambda: clf.fit(x_fit, y_fit))
        y_pred, predict_time = timed(lambda: clf.predict(x_test))

        metric_rows.append(
            {
                **run_id,
                "fold": fold,
                "n_sub_effective": len(fit_idx),
                "n_fit": len(fit_idx),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "val_accuracy_best": study.best_value,
                **{
                    f"test_{k}": v
                    for k, v in compute_metrics(y_test, y_pred).items()
                },
                "subsampling_time_s": 0.0,
                "training_time_s": training_time,
                "predict_time_s": predict_time,
                "optuna_time_s": optuna_time,
                "optuna_mean_trial_subsampling_time_s": 0.0,
                "optuna_mean_trial_training_time_s": float(np.mean(
                    [t.user_attrs["training_time_s"] for t in completed]
                )),
                "n_trials_completed": len(completed),
            }
        )
        hyper_row = {
            **run_id,
            "fold": fold,
            "best_trial_number": study.best_trial.number,
            "best_val_accuracy": study.best_value,
            **{f"best_{k}": v for k, v in best.items()},
        }
        if model == "mv_lssvm":
            hyper_row["best_rho"] = best_params[2]
        hyper_rows.append(hyper_row)
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
            "[%s/%s fold %d] val_acc=%.4f test_acc=%.4f f1_macro=%.4f "
            "training=%.3fs optuna=%.1fs",
            MODEL_NAMES[model], dataset, fold, study.best_value,
            metric_rows[-1]["test_accuracy"], metric_rows[-1]["test_f1_macro"],
            training_time, optuna_time,
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
    """Runs every dataset for the chosen model sequentially."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    args = parser.parse_args()
    model = args.model
    root = RESULTS_ROOT / model

    check_protocol(root, {"n_outer_folds": N_OUTER_FOLDS})
    write_experiment_metadata(root, model)
    logger.info("Starting %d %s runs.", len(DATASETS), MODEL_NAMES[model])

    for i, dataset in enumerate(DATASETS, start=1):
        out_dir = root / dataset / METHOD / N_SUB
        tag = f"[{i}/{len(DATASETS)}] {MODEL_NAMES[model]}/{dataset}"
        run_id = {"model": MODEL_NAMES[model], "dataset": dataset,
                  "subsampling_method": METHOD, "n_sub": N_SUB}

        if SKIP_COMPLETED and (out_dir / "summary.csv").exists():
            logger.info("%s already completed, skipping.", tag)
            continue

        start = time.perf_counter()
        try:
            views, labels = get_dataset(dataset)
            views, labels = list(views), np.asarray(labels)
            logger.info("%s starting.", tag)
            run_nested_cv(dataset, model, views, labels, out_dir)
            status, error = "ok", ""
        except Exception as exc:  # Keep the grid going if one run fails.
            logger.error("%s failed:\n%s", tag, traceback.format_exc())
            status, error = "failed", f"{type(exc).__name__}: {exc}"

        append_manifest(
            root,
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
        aggregate_results(root)

    logger.info("All runs finished. Results in %s", root)


if __name__ == "__main__":
    main()
