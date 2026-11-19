"""Quick start: Multi-View Kernel Herding (MKH) with a late-fusion LS-SVM.

Runs in a few seconds on synthetic data and shows the three ways of using
the method:

    1. Plain use of the sampler on a multi-view dataset.
    2. Class-stratified use, as in the paper (one sampler run per class).
    3. Training the late-fusion LS-SVM on the selected subsets and comparing
       with random subsampling and with training on all the data.

Run it from the repository root:

    python usage/quickstart.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# The library code lives in 1_BO_subsampling_and_metrics/.
CODE_DIR = Path(__file__).resolve().parents[1] / "1_BO_subsampling_and_metrics"
sys.path.insert(0, str(CODE_DIR))

from multi_view_learning_models import LateFusion_LSSVM  
from subsampling_methods import (  
    MultiViewKernelHerding,
    RandomSubsamplingPerView,
)

SEED = 0
N_SUB = 60  # Instances selected per view (spread evenly over the classes).

# ---------------------------------------------------------------------------
# 0. A synthetic 3-view dataset: three disjoint blocks of features.
# ---------------------------------------------------------------------------
X, y = make_classification(
    n_samples=1500, n_features=30, n_informative=15, n_redundant=5,
    n_classes=3, n_clusters_per_class=2, class_sep=1.2, random_state=SEED,
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED
)
blocks = [slice(0, 10), slice(10, 20), slice(20, 30)]
views_tr, views_te = [], []
for sl in blocks:
    scaler = MinMaxScaler().fit(X_tr[:, sl])  # Same scaling as the paper.
    views_tr.append(scaler.transform(X_tr[:, sl]))
    views_te.append(scaler.transform(X_te[:, sl]))

# RBF kernel width per view. In the experiments these are tuned with Optuna;
# here we use a simple scale-aware default.
kernel_gammas = [1.0 / v.shape[1] for v in views_tr]
reg_gammas = [1.0] * len(views_tr)  # LS-SVM regularization per view.

# ---------------------------------------------------------------------------
# 1. Plain use: one call selects N_SUB rows in every view.
# ---------------------------------------------------------------------------
sampler = MultiViewKernelHerding(
    n_sub=N_SUB,
    kernel_gammas=kernel_gammas,  # Scalar, list, array or {view_name: gamma}.
    warmup_steps=0.5,             # Fraction of n_sub (or an int) of pure herding.
    epsilon=np.inf,               # MMD tolerance; inf = paper setting.
    random_state=SEED,
)
sampler.fit(views_tr, y_tr)
print("1. Plain use")
for name, idx in sampler.selected_indices_.items():
    print(f"   {name}: {len(idx)} rows, first five {idx[:5].tolist()}")
v0, v1 = sampler.view_names_[:2]
print(f"   Final cross-view statistic HSIC_{v0},{v1} = "
      f"{sampler.traces_[v0][v1] / N_SUB**2:.2e}\n")


# ---------------------------------------------------------------------------
# 2. Class-stratified use (paper protocol): run the sampler inside each class
#    with ceil(n_sub / C) rows per class and merge the subsets of each view.
# ---------------------------------------------------------------------------
def stratified_subsample(sampler_cls, views, labels, n_sub, seed, **kwargs):
    """Returns one array of selected row indices per view."""
    classes = np.unique(labels)
    per_class = math.ceil(n_sub / len(classes))
    parts = [[] for _ in views]
    for k, c in enumerate(classes):
        rows = np.flatnonzero(labels == c)
        s = sampler_cls(n_sub=min(per_class, len(rows)),
                        random_state=seed + k, **kwargs)
        s.fit([v[rows] for v in views], labels[rows])
        for v, name in enumerate(s.view_names_):
            parts[v].append(rows[s.selected_indices_[name]])
    return [np.sort(np.concatenate(p)) for p in parts]


def evaluate(selected):
    """Trains the late-fusion LS-SVM on the subsets; returns test bal. acc."""
    model = LateFusion_LSSVM(kernel_gammas=kernel_gammas,
                             selected_indices=selected, gammas=reg_gammas)
    model.fit(views_tr, y_tr)
    return balanced_accuracy_score(y_te, model.predict(views_te))


samplers = {
    "MKH (ours)": (MultiViewKernelHerding,
                   dict(kernel_gammas=kernel_gammas, warmup_steps=0.5,
                        epsilon=np.inf)),
    # Single-view kernel herding = MKH with warm-up over all steps.
    "SKH": (MultiViewKernelHerding,
            dict(kernel_gammas=kernel_gammas, warmup_steps=1.0,
                 epsilon=np.inf)),
    "Random": (RandomSubsamplingPerView, {}),
}

# ---------------------------------------------------------------------------
# 3. Train the late-fusion LS-SVM on each subset and compare.
# ---------------------------------------------------------------------------
print(f"2-3. Class-stratified subsampling (n_sub = {N_SUB} per view) "
      f"+ late-fusion LS-SVM, {len(y_tr)} training rows")
for label, (cls, kw) in samplers.items():
    t0 = time.perf_counter()
    selected = stratified_subsample(cls, views_tr, y_tr, N_SUB, SEED, **kw)
    t_sub = time.perf_counter() - t0
    print(f"   {label:11s} balanced accuracy {evaluate(selected):.3f}"
          f"   (subsampling {t_sub:.3f} s)")

t0 = time.perf_counter()
full = [np.arange(len(y_tr))] * len(views_tr)
acc_full = evaluate(full)
print(f"   {'Full data':11s} balanced accuracy {acc_full:.3f}"
      f"   (training {time.perf_counter() - t0:.3f} s)")
