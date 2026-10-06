"""Analyzes the results of run_experiments.py for a paper.

INPUT PATH
    Set RESULTS_ROOT below (or pass --results-root). It must be the folder
    that contains the files written by run_experiments.py:
        all_outer_fold_metrics.csv
        all_best_hyperparameters.csv   (optional)
        <dataset>/<method>/<n_sub>/run_config.csv   (optional)

OUTPUT (under ANALYSIS_DIR)
    figures/<dataset>_metrics.{png,pdf}      Metrics vs n_sub (mean +- std).
    figures/<dataset>_times.{png,pdf}        Subsampling / training time.
    figures/<dataset>_tradeoff.{png,pdf}     Accuracy vs total time.
    figures/average_rank_<metric>.{png,pdf}  Average rank with Nemenyi CD.
    tables/table_<target>.{csv,tex}          mean +- std, best in bold.
    tables/table_<target>_numeric.csv        Numeric mean/std.
    tables/ranking_<target>_per_config.csv   Rank per (dataset, n_sub).
    tables/ranking_<target>_per_dataset.csv  Average rank per dataset.
    tables/friedman_<metric>.csv             Friedman test + Nemenyi CD.
    tables/wilcoxon_<metric>.csv             Pairwise Wilcoxon (Holm).
    tables/best_configuration_per_dataset.csv
    tables/best_hyperparameters_summary.csv
    tables/dataset_table.csv

Note on the statistical tests: the blocks are (dataset, n_sub)
configurations. Configurations of the same dataset are not independent, so
the p-values are indicative; with many datasets prefer tests per n_sub.
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

# ------------------------------ configuration -------------------------------
RESULTS_ROOT = Path("results")    # <-- PATH TO THE EXPERIMENT RESULTS
ANALYSIS_DIR = Path("analysis")   # <-- where the analysis is written

ALPHA = 0.05
FIG_DPI = 300
METRICS = ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted")

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "f1_macro": "F1 (macro)",
    "f1_weighted": "F1 (weighted)",
}
METHOD_LABELS = {
    "random": "Random",
    "MV_kernel_herding": "MV kernel herding",
    "simple_kernel_herding": "Kernel herding",
    "clustering": "Clustering",
    "fixed_size_entropy": "Fixed-size (Rényi entropy)",
}
# name -> (column, higher_is_better, decimals)
TARGETS: Mapping[str, tuple[str, bool, int]] = {
    **{m: (f"test_{m}", True, 3) for m in METRICS},
    "subsampling_time": ("subsampling_time_s", False, 3),
    "training_time": ("training_time_s", False, 3),
}
# Studentized-range critical values / sqrt(2) for alpha = 0.05 (Nemenyi).
NEMENYI_Q05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
    7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}
KEYS = ["dataset", "n_sub", "subsampling_method"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------------------ helpers -------------------------------------
def label(method: str) -> str:
    """Returns the display name of a subsampling method."""
    return METHOD_LABELS.get(method, method)


def save_figure(fig: plt.Figure, stem: Path, close: bool = True) -> None:
    """Saves a figure as PNG and PDF (stem has no extension)."""
    for ext in ("png", "pdf"):
        fig.savefig(Path(f"{stem}.{ext}"), dpi=FIG_DPI, bbox_inches="tight")
    if close:
        plt.close(fig)


def style_map(methods: Sequence[str]) -> dict[str, tuple]:
    """Maps every method to a consistent (color, marker) pair."""
    cmap = plt.get_cmap("tab10")
    markers = "osD^v<>P*X"
    return {
        m: (cmap(i % 10), markers[i % len(markers)])
        for i, m in enumerate(methods)
    }


def load_fold_metrics(root: Path) -> pd.DataFrame:
    """Loads the per-fold metrics of all runs.

    Args:
        root: Results root written by run_experiments.py.

    Returns:
        One row per (dataset, method, n_sub, fold) with a total_time_s column.

    Raises:
        FileNotFoundError: If all_outer_fold_metrics.csv does not exist.
    """
    path = root / "all_outer_fold_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run run_experiments.py first or check "
            "RESULTS_ROOT."
        )
    df = pd.read_csv(path)
    df = df.dropna(subset=["training_time_s"])  # Drop legacy-format rows.
    df["total_time_s"] = df["subsampling_time_s"] + df["training_time_s"]
    return df


# ------------------------------ tables --------------------------------------
def build_table(
    df: pd.DataFrame,
    column: str,
    higher_is_better: bool,
    decimals: int,
    methods: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Builds mean/std pivots and formatted (plain, LaTeX) tables.

    Args:
        df: Per-fold metrics.
        column: Column to summarize.
        higher_is_better: Whether larger values are better.
        decimals: Number of decimals in the formatted cells.
        methods: Column order.

    Returns:
        Tuple (mean, std, plain_table, latex_table). Rows are (dataset,
        n_sub), columns are methods; the best cell of each row is bold in
        the LaTeX table.
    """
    grouped = df.groupby(KEYS)[column]
    mean = grouped.mean().unstack("subsampling_method").reindex(columns=methods)
    std = grouped.std().unstack("subsampling_method").reindex(columns=methods)
    best = mean.max(axis=1) if higher_is_better else mean.min(axis=1)
    plain = pd.DataFrame(index=mean.index, columns=mean.columns, dtype=object)
    latex = plain.copy()
    for idx in mean.index:
        for method in mean.columns:
            value = mean.loc[idx, method]
            if pd.isna(value):
                plain.loc[idx, method] = "--"
                latex.loc[idx, method] = "--"
                continue
            spread = std.loc[idx, method]
            spread = 0.0 if pd.isna(spread) else spread
            plain.loc[idx, method] = (
                f"{value:.{decimals}f} ± {spread:.{decimals}f}"
            )
            cell = f"{value:.{decimals}f} $\\pm$ {spread:.{decimals}f}"
            if np.isclose(value, best.loc[idx]):
                cell = f"\\textbf{{{cell}}}"
            latex.loc[idx, method] = cell
    return mean, std, plain, latex


def write_tables(df: pd.DataFrame, methods: Sequence[str], out: Path) -> None:
    """Writes the mean +- std tables (CSV and LaTeX) of every target."""
    for name, (column, higher, decimals) in TARGETS.items():
        mean, std, plain, latex = build_table(
            df, column, higher, decimals, methods
        )
        names = [label(m) for m in methods]
        plain.columns = names
        latex.columns = names
        plain.to_csv(out / f"table_{name}.csv")
        numeric = pd.concat(
            {"mean": mean, "std": std}, axis=1
        )
        numeric.columns = [f"{stat}_{m}" for stat, m in numeric.columns]
        numeric.to_csv(out / f"table_{name}_numeric.csv")
        title = name.replace("_", " ")
        latex.to_latex(
            out / f"table_{name}.tex",
            escape=False,
            caption=f"{title} (mean $\\pm$ std over outer folds). "
            "Best value in bold.",
            label=f"tab:{name}",
        )


def write_best_configurations(df: pd.DataFrame, out: Path) -> None:
    """Writes the best (method, n_sub) of every dataset and metric."""
    rows = []
    for name in METRICS:
        column = TARGETS[name][0]
        grouped = (
            df.groupby(["dataset", "subsampling_method", "n_sub"])[column]
            .agg(["mean", "std"])
            .reset_index()
        )
        best = grouped.loc[grouped.groupby("dataset")["mean"].idxmax()]
        rows.append(best.assign(metric=name))
    pd.concat(rows).to_csv(
        out / "best_configuration_per_dataset.csv", index=False
    )


def write_dataset_table(root: Path, out: Path) -> None:
    """Writes dataset statistics read from the run_config.csv files."""
    rows = {}
    for path in sorted(root.glob("*/*/*/run_config.csv")):
        dataset = path.parts[-4]
        if dataset in rows:
            continue
        config = pd.read_csv(path)
        info = dict(zip(config["key"], config["value"]))
        rows[dataset] = {
            k: info.get(k)
            for k in ("n_samples", "n_views", "n_features_per_view",
                      "n_classes", "class_counts")
        }
    if rows:
        pd.DataFrame.from_dict(rows, orient="index").rename_axis(
            "dataset"
        ).to_csv(out / "dataset_table.csv")


def write_hyperparameter_summary(root: Path, out: Path) -> None:
    """Summarizes the best hyperparameters (median and quartiles)."""
    path = root / "all_best_hyperparameters.csv"
    if not path.exists():
        logger.warning("%s not found, skipping hyperparameters.", path)
        return
    hp = pd.read_csv(path)
    skip = {"best_trial_number", "best_val_accuracy"}
    cols = [
        c for c in hp.columns
        if c.startswith("best_") and c not in skip
        and pd.api.types.is_numeric_dtype(hp[c])
    ]
    grouped = hp.groupby(KEYS)[cols]
    summary = pd.concat(
        {
            "median": grouped.median(),
            "q25": grouped.quantile(0.25),
            "q75": grouped.quantile(0.75),
        },
        axis=1,
    )
    summary.columns = [f"{col}_{stat}" for stat, col in summary.columns]
    summary.to_csv(out / "best_hyperparameters_summary.csv")


# ------------------------------ rankings & tests ----------------------------
def compute_ranks(
    df: pd.DataFrame, column: str, higher_is_better: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ranks methods inside every (dataset, n_sub) configuration.

    Returns:
        Tuple (mean values, ranks); rank 1 is best, ties share the average.
    """
    mean = (
        df.groupby(KEYS)[column].mean().unstack("subsampling_method")
    )
    ranks = mean.rank(
        axis=1, ascending=not higher_is_better, method="average"
    )
    return mean, ranks


def write_rankings(df: pd.DataFrame, out: Path) -> None:
    """Writes per-config and per-dataset rankings for every target."""
    for name, (column, higher, _) in TARGETS.items():
        mean, ranks = compute_ranks(df, column, higher)
        best = mean.max(axis=1) if higher else mean.min(axis=1)
        wins = mean.eq(best, axis=0).sum()
        ranks.rename(columns=label).to_csv(
            out / f"ranking_{name}_per_config.csv"
        )
        per_dataset = ranks.groupby(level="dataset").mean()
        per_dataset.loc["Average rank (all configs)"] = ranks.mean()
        per_dataset.loc["Number of wins"] = wins
        per_dataset.rename(columns=label).to_csv(
            out / f"ranking_{name}_per_dataset.csv"
        )


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down correction of a vector of p-values."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for pos, idx in enumerate(order):
        running = max(running, (m - pos) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def statistical_tests(
    df: pd.DataFrame, metric: str, out: Path
) -> tuple[pd.Series, float | None] | None:
    """Runs Friedman, Nemenyi CD and pairwise Wilcoxon (Holm) tests.

    Args:
        df: Per-fold metrics.
        metric: Key of METRICS.
        out: Tables directory.

    Returns:
        (average ranks, Nemenyi CD) or None if there are too few blocks.
    """
    column = TARGETS[metric][0]
    mean, _ = compute_ranks(df, column, True)
    complete = mean.dropna()  # Blocks where every method has a result.
    n_blocks, k = complete.shape
    if n_blocks < 2 or k < 2:
        logger.warning("Too few blocks for tests on %s.", metric)
        return None
    avg_ranks = complete.rank(axis=1, ascending=False).mean()
    cd = None
    if k >= 3:
        chi2, p_value = stats.friedmanchisquare(
            *[complete[m] for m in complete.columns]
        )
        if k in NEMENYI_Q05:
            cd = NEMENYI_Q05[k] * np.sqrt(k * (k + 1) / (6.0 * n_blocks))
        row = {"n_blocks": n_blocks, "n_methods": k, "chi2": chi2,
               "p_value": p_value, "nemenyi_cd_0.05": cd}
        row.update({f"avg_rank_{label(m)}": r for m, r in avg_ranks.items()})
        pd.DataFrame([row]).to_csv(out / f"friedman_{metric}.csv", index=False)

    pairs = list(itertools.combinations(complete.columns, 2))
    rows = []
    for a, b in pairs:
        diff = complete[a] - complete[b]
        if np.allclose(diff, 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = stats.wilcoxon(complete[a], complete[b])
        rows.append({"method_a": label(a), "method_b": label(b),
                     "mean_diff_a_minus_b": diff.mean(),
                     "n_blocks": n_blocks, "statistic": statistic,
                     "p_value": p_value})
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_value"].to_numpy())
    result["significant"] = result["p_holm"] < ALPHA
    result.to_csv(out / f"wilcoxon_{metric}.csv", index=False)
    return avg_ranks, cd


# ------------------------------ plots ---------------------------------------
def _set_n_sub_axis(ax: plt.Axes, n_subs: Sequence[int]) -> None:
    """Uses the actual n_sub values as ticks (log scale if very spread)."""
    values = sorted(set(n_subs))
    if len(values) > 1 and values[-1] / values[0] >= 8:
        ax.set_xscale("log", base=2)
        ax.minorticks_off()
    ax.set_xticks(values)
    ax.set_xticklabels([str(v) for v in values])


def plot_metric_evolution(
    df: pd.DataFrame, dataset: str, styles: Mapping[str, tuple], out: Path
) -> None:
    """Plots every metric vs n_sub (mean +- std) for one dataset."""
    sub = df[df["dataset"] == dataset]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, metric in zip(axes.ravel(), METRICS):
        column = TARGETS[metric][0]
        for method, (color, marker) in styles.items():
            g = (
                sub[sub["subsampling_method"] == method]
                .groupby("n_sub")[column]
                .agg(["mean", "std"])
                .fillna(0.0)
            )
            if g.empty:
                continue
            ax.plot(g.index, g["mean"], color=color, marker=marker,
                    label=label(method))
            ax.fill_between(g.index, g["mean"] - g["std"],
                            g["mean"] + g["std"], color=color, alpha=0.15)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(alpha=0.3)
        _set_n_sub_axis(ax, sub["n_sub"])
    for ax in axes[-1]:
        ax.set_xlabel("n_sub")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(dataset)
    fig.tight_layout()
    save_figure(fig, out / f"{dataset}_metrics")


def plot_times(
    df: pd.DataFrame, dataset: str, styles: Mapping[str, tuple], out: Path
) -> None:
    """Plots subsampling and training time vs n_sub for one dataset."""
    sub = df[df["dataset"] == dataset]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, name in zip(axes, ("subsampling_time", "training_time")):
        column = TARGETS[name][0]
        for method, (color, marker) in styles.items():
            g = (
                sub[sub["subsampling_method"] == method]
                .groupby("n_sub")[column]
                .agg(["mean", "std"])
                .fillna(0.0)
            )
            if g.empty:
                continue
            ax.plot(g.index, g["mean"], color=color, marker=marker,
                    label=label(method))
            lower = np.maximum(g["mean"] - g["std"], g["mean"] * 0.1)
            ax.fill_between(g.index, lower, g["mean"] + g["std"],
                            color=color, alpha=0.15)
        ax.set_yscale("log")
        ax.set_title(name.replace("_", " ").capitalize())
        ax.set_xlabel("n_sub")
        ax.set_ylabel("Seconds")
        ax.grid(alpha=0.3, which="both")
        _set_n_sub_axis(ax, sub["n_sub"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(dataset)
    fig.tight_layout()
    save_figure(fig, out / f"{dataset}_times")


def plot_tradeoff(
    df: pd.DataFrame, dataset: str, styles: Mapping[str, tuple], out: Path
) -> None:
    """Plots accuracy vs total (subsampling + training) time."""
    sub = df[df["dataset"] == dataset]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for method, (color, marker) in styles.items():
        g = (
            sub[sub["subsampling_method"] == method]
            .groupby("n_sub")[["total_time_s", "test_accuracy"]]
            .mean()
        )
        if g.empty:
            continue
        ax.plot(g["total_time_s"], g["test_accuracy"], color=color,
                marker=marker, label=label(method))
        for n_sub, row in g.iterrows():
            ax.annotate(str(n_sub), (row["total_time_s"],
                                     row["test_accuracy"]),
                        textcoords="offset points", xytext=(3, 3),
                        fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Subsampling + training time (s)")
    ax.set_ylabel("Accuracy")
    ax.set_title(dataset)
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out / f"{dataset}_tradeoff", close=False)

    # Paper version: EPS, no title and no axis labels. EPS (PostScript) has
    # no transparency, so the grid gets an opaque light gray instead of alpha.
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(which="both", color="0.85", alpha=1.0)
    fig.tight_layout()
    paper_dir = out / "paper"
    paper_dir.mkdir(exist_ok=True)
    fig.savefig(paper_dir / f"{dataset}_tradeoff.eps", bbox_inches="tight")
    plt.close(fig)


def plot_average_ranks(
    avg_ranks: pd.Series, cd: float | None, metric: str, out: Path
) -> None:
    """Plots average ranks (lower is better) with the Nemenyi CD bar."""
    ranks = avg_ranks.sort_values()
    fig, ax = plt.subplots(figsize=(5.5, 0.6 * len(ranks) + 1.2))
    ax.barh([label(m) for m in ranks.index], ranks.values, color="0.6")
    ax.invert_yaxis()
    ax.set_xlabel(f"Average rank ({METRIC_LABELS[metric]}, lower is better)")
    if cd is not None:
        ax.plot([ranks.min(), ranks.min() + cd], [-0.7, -0.7], color="k",
                lw=2)
        ax.text(ranks.min() + cd / 2, -0.9, f"CD = {cd:.2f}", ha="center",
                fontsize=8)
    fig.tight_layout()
    save_figure(fig, out / f"average_rank_{metric}")


# ------------------------------ main ----------------------------------------
def main() -> None:
    """Runs the full analysis."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    args = parser.parse_args()

    tables_dir = args.analysis_dir / "tables"
    figures_dir = args.analysis_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_fold_metrics(args.results_root)
    methods = sorted(df["subsampling_method"].unique(), key=label)
    styles = style_map(methods)
    logger.info("Loaded %d rows | datasets=%s | methods=%s | n_subs=%s",
                len(df), sorted(df["dataset"].unique()), methods,
                sorted(df["n_sub"].unique()))

    write_tables(df, methods, tables_dir)
    write_rankings(df, tables_dir)
    write_best_configurations(df, tables_dir)
    write_dataset_table(args.results_root, tables_dir)
    write_hyperparameter_summary(args.results_root, tables_dir)

    for metric in METRICS:
        result = statistical_tests(df, metric, tables_dir)
        if result is not None:
            plot_average_ranks(*result, metric, figures_dir)

    for dataset in sorted(df["dataset"].unique()):
        plot_metric_evolution(df, dataset, styles, figures_dir)
        plot_times(df, dataset, styles, figures_dir)
        plot_tradeoff(df, dataset, styles, figures_dir)

    logger.info("Analysis written to %s", args.analysis_dir.resolve())


if __name__ == "__main__":
    main()