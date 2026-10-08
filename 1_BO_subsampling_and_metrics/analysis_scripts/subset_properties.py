"""Properties of the per-view subsets chosen by the subsampling methods.

Reads the selected_indices.csv files written by the multi-view experiments
(bo_accuracy_in_view.py, bo_late_fusion.py) and describes how the subsets
of the different views relate to each other, for every per-view subsampling
method (none of them forces the views to keep the same samples).

Run from 1_BO_subsampling_and_metrics:

    python3 -m analysis_scripts.subset_properties \
        --results-root results_late_fusion --analysis-dir analysis_late_fusion

OUTPUT (under <analysis-dir>/subset_properties/)
    tables/per_fold.csv          One row per (dataset, method, n_sub, fold):
        view_size            Samples per view (mean over views).
        n_fit                Training rows the subsets were drawn from.
        union_size           Samples selected in at least one view.
        union_fraction       union_size / n_fit.
        intersection_all     Samples selected in ALL views.
        classes_in_intersection / n_classes
        pairwise_intersection_{mean,min,max}   |S_u n S_v| over view pairs.
        pairwise_overlap_mean  Mean |S_u n S_v| / view size (0..1).
        pairwise_jaccard_mean  Mean |S_u n S_v| / |S_u u S_v|.
        frac_in_<k>_views    Fraction of the union selected in exactly k views.
    tables/summary.csv           per_fold averaged over folds (mean and std).
    tables/pairwise.csv          Per (dataset, method, n_sub, view pair):
                                 intersection and Jaccard, mean/std over folds.
    figures/<dataset>_overlap.{png,pdf}
        Pairwise overlap, samples in all views and union coverage vs n_sub,
        one line per method.
    figures/<dataset>_<method>_pairwise.{png,pdf}
        Heatmaps of the mean pairwise intersection, one panel per n_sub
        (diagonal = view size), for every method (or only those given with
        --heatmap-methods).
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis_scripts.bo_accuracy_in_view import (  # noqa: E402
    _set_n_sub_axis,
    label,
    save_figure,
    style_map,
)
from get_data import get_dataset  # noqa: E402

RESULTS_ROOT = Path("results_late_fusion")
ANALYSIS_DIR = Path("analysis_late_fusion")
KEYS = ["dataset", "subsampling_method", "n_sub"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------------------ loading -------------------------------------
def load_selected(root: Path) -> pd.DataFrame:
    """Concatenates every selected_indices.csv under `root` with its n_fit."""
    frames = []
    for path in sorted(root.glob("*/*/*/selected_indices.csv")):
        sel = pd.read_csv(path)
        if sel.empty or sel["view"].nunique() < 2:
            continue  # Single-view subsets (early fusion) have no pairs.
        metrics = pd.read_csv(path.with_name("outer_fold_metrics.csv"))
        frames.append(sel.merge(metrics[["fold", "n_fit"]], on="fold"))
    if not frames:
        raise FileNotFoundError(
            f"No multi-view selected_indices.csv under {root.resolve()}."
        )
    return pd.concat(frames, ignore_index=True)


# ------------------------------ properties ----------------------------------
def fold_properties(
    group: pd.DataFrame, labels: np.ndarray
) -> tuple[dict, list[dict]]:
    """Properties of one fold's per-view subsets.

    Args:
        group: selected_indices rows of one (dataset, method, n_sub, fold).
        labels: Class labels of the whole dataset (indexed by row_index).

    Returns:
        (fold-level properties, one dict per view pair).
    """
    subsets = {
        v: set(g["row_index"]) for v, g in group.groupby("view", sort=True)
    }
    views = sorted(subsets)
    n_views = len(views)
    sizes = np.array([len(subsets[v]) for v in views])
    union = set().union(*subsets.values())
    common = set.intersection(*subsets.values())

    pair_rows = []
    for u, v in itertools.combinations(views, 2):
        inter = len(subsets[u] & subsets[v])
        pair_rows.append({
            "view_u": u, "view_v": v, "intersection": inter,
            "overlap": inter / min(len(subsets[u]), len(subsets[v])),
            "jaccard": inter / len(subsets[u] | subsets[v]),
        })
    pairs = pd.DataFrame(pair_rows)

    # In how many views is each selected sample?
    counts = pd.Series(
        [sum(i in subsets[v] for v in views) for i in union]
    ).value_counts()

    n_fit = int(group["n_fit"].iloc[0])
    row = {
        "n_views": n_views,
        "view_size": float(sizes.mean()),
        "n_fit": n_fit,
        "union_size": len(union),
        "union_fraction": len(union) / n_fit,
        "intersection_all": len(common),
        "classes_in_intersection": len(np.unique(labels[list(common)])),
        "n_classes": len(np.unique(labels)),
        "pairwise_intersection_mean": pairs["intersection"].mean(),
        "pairwise_intersection_min": pairs["intersection"].min(),
        "pairwise_intersection_max": pairs["intersection"].max(),
        "pairwise_overlap_mean": pairs["overlap"].mean(),
        "pairwise_jaccard_mean": pairs["jaccard"].mean(),
        **{
            f"frac_in_{k}_views": counts.get(k, 0) / len(union)
            for k in range(1, n_views + 1)
        },
    }
    return row, pair_rows


def compute_properties(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-fold properties, per-fold pairwise intersections)."""
    fold_rows, pair_rows = [], []
    for dataset, ds_rows in selected.groupby("dataset"):
        labels = np.asarray(get_dataset(dataset)[1])
        for (method, n_sub, fold), group in ds_rows.groupby(
            ["subsampling_method", "n_sub", "fold"]
        ):
            ident = {"dataset": dataset, "subsampling_method": method,
                     "n_sub": n_sub, "fold": fold}
            row, pairs = fold_properties(group, labels)
            fold_rows.append({**ident, **row})
            pair_rows.extend({**ident, **p} for p in pairs)
    return pd.DataFrame(fold_rows), pd.DataFrame(pair_rows)


# ------------------------------ plots ---------------------------------------
def plot_overlap(summary: pd.DataFrame, dataset: str, out: Path) -> None:
    """Overlap, intersection of all views and union coverage vs n_sub."""
    sub = summary[summary["dataset"] == dataset]
    methods = sorted(sub["subsampling_method"].unique(), key=label)
    styles = style_map(methods)
    panels = [
        ("pairwise_overlap_mean", "Mean pairwise overlap\n|S_u ∩ S_v| / view size"),
        ("intersection_all", "Samples selected in all views"),
        ("union_fraction", "Union of views / training set"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
    for ax, (col, title) in zip(axes, panels):
        for method in methods:
            g = sub[sub["subsampling_method"] == method].sort_values("n_sub")
            color, marker = styles[method]
            ax.errorbar(g["n_sub"], g[f"{col}_mean"], yerr=g[f"{col}_std"],
                        color=color, marker=marker, capsize=2,
                        label=label(method))
        _set_n_sub_axis(ax, sub["n_sub"])
        ax.set_xlabel("n_sub")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(dataset)
    fig.tight_layout()
    save_figure(fig, out / f"{dataset}_overlap")


def plot_pairwise_heatmaps(
    pairwise: pd.DataFrame, summary: pd.DataFrame, dataset: str, method: str,
    out: Path,
) -> None:
    """Mean pairwise intersection heatmaps, one panel per n_sub."""
    sub = pairwise[(pairwise["dataset"] == dataset)
                   & (pairwise["subsampling_method"] == method)]
    if sub.empty:
        return
    n_subs = sorted(sub["n_sub"].unique())
    n_views = int(max(sub["view_u"].max(), sub["view_v"].max())) + 1
    n_cols = min(len(n_subs), 5)
    n_rows = int(np.ceil(len(n_subs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=(2.6 * n_cols + 0.8, 2.6 * n_rows))
    for ax, n_sub in zip(axes.flat, n_subs):
        g = sub[sub["n_sub"] == n_sub]
        view_size = summary.loc[
            (summary["dataset"] == dataset)
            & (summary["subsampling_method"] == method)
            & (summary["n_sub"] == n_sub), "view_size_mean"
        ].iloc[0]
        mat = np.full((n_views, n_views), view_size)
        for _, r in g.iterrows():
            mat[int(r["view_u"]), int(r["view_v"])] = r["intersection_mean"]
            mat[int(r["view_v"]), int(r["view_u"])] = r["intersection_mean"]
        im = ax.imshow(mat / view_size, vmin=0, vmax=1, cmap="viridis")
        if n_views <= 8:
            for i, j in itertools.product(range(n_views), repeat=2):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if mat[i, j] / view_size < 0.6
                        else "black")
        ax.set_xticks(range(n_views))
        ax.set_yticks(range(n_views))
        ax.tick_params(labelsize=7)
        ax.set_title(f"n_sub={n_sub} (|S|={view_size:.0f})", fontsize=9)
    for ax in list(axes.flat)[len(n_subs):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8,
                 label="|S_u ∩ S_v| / view size (numbers: counts)")
    fig.suptitle(f"{dataset}: {label(method)}, mean pairwise intersection")
    save_figure(fig, out / f"{dataset}_{method}_pairwise")


# ------------------------------ main ----------------------------------------
def main() -> None:
    """Computes and writes the subset properties."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--heatmap-methods", nargs="+", default=None,
                        help="Methods that get pairwise heatmaps "
                             "(default: all methods found).")
    args = parser.parse_args()

    out = args.analysis_dir / "subset_properties"
    tables_dir, figures_dir = out / "tables", out / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    selected = load_selected(args.results_root)
    per_fold, pairs = compute_properties(selected)
    logger.info("Loaded %d (dataset, method, n_sub, fold) subsets | methods=%s",
                len(per_fold), sorted(per_fold["subsampling_method"].unique()))

    value_cols = [c for c in per_fold.columns if c not in KEYS + ["fold"]]
    summary = per_fold.groupby(KEYS)[value_cols].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    pairwise = (
        pairs.groupby(KEYS + ["view_u", "view_v"])[["intersection", "jaccard"]]
        .agg(["mean", "std"])
    )
    pairwise.columns = ["_".join(c) for c in pairwise.columns]
    pairwise = pairwise.reset_index()

    per_fold.to_csv(tables_dir / "per_fold.csv", index=False)
    summary.to_csv(tables_dir / "summary.csv", index=False)
    pairwise.to_csv(tables_dir / "pairwise.csv", index=False)

    heatmap_methods = args.heatmap_methods or sorted(
        summary["subsampling_method"].unique(), key=label
    )
    for dataset in sorted(summary["dataset"].unique()):
        plot_overlap(summary, dataset, figures_dir)
        for method in heatmap_methods:
            plot_pairwise_heatmaps(pairwise, summary, dataset, method,
                                   figures_dir)

    logger.info("Subset properties written to %s", out.resolve())


if __name__ == "__main__":
    main()
