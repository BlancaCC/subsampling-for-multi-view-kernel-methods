# How to use Multi-View Kernel Herding

This folder explains how to use **Multi-View Kernel Herding (MKH)** on your own
multi-view data. [`quickstart.py`](quickstart.py) is a runnable example that
takes a few seconds:

```bash
# from the repository root
python usage/quickstart.py
```

## What the method does

You have `n` instances observed in `V` paired views (one feature matrix per
view, same rows in every view). MKH selects `n_sub` rows **in each view**, and
the selected rows may differ from view to view:

1. **Within each view**, it runs kernel herding on the centered RBF kernel, so
   that the selected subset keeps the view's distribution (small MMD to the
   full data).
2. **Across views**, after a warm-up of pure herding, each view scores its
   candidates by how much they would increase an HSIC-type dependence
   statistic between the views that have already picked at this step. It then
   selects the candidate with the best sum of ranks of the MMD score and the
   cross-view score, among the candidates within a tolerance `epsilon` of the
   best MMD score.

The subsets are then used to train any per-view kernel model. In the paper
this is a late-fusion LS-SVM: one LS-SVM per view, with scores averaged over
the views.

## Minimal example

```python
import numpy as np
from subsampling_methods import MultiViewKernelHerding

X_views = [X_view_0, X_view_1, X_view_2]   # or {"text": ..., "image": ...}

sampler = MultiViewKernelHerding(
    n_sub=100,                 # rows selected in every view
    kernel_gammas=[0.1, 0.05, 0.01],
    warmup_steps=0.5,          # first 50% of the steps: pure herding
    epsilon=np.inf,            # MMD tolerance (setting used in the paper)
    random_state=42,
)
sampler.fit(X_views, y)

idx = sampler.selected_indices_         # {"view_0": array, "view_1": ..., ...}
subsets = sampler.transform(X_views)    # {"view_0": {"X", "y", "indices"}, ...}
```

The code lives in `1_BO_subsampling_and_metrics/`. Run your script from that
folder, or add it to `sys.path` as `quickstart.py` does.

## Parameters

| Parameter | Meaning | Values |
|---|---|---|
| `n_sub` | Rows selected in every view. | `1 <= n_sub <= n` |
| `kernel_gammas` | RBF width `gamma` of each view, `k(x, x') = exp(-gamma ||x - x'||^2)`. Use the same widths as the downstream model. | Scalar (all views), list/array (one per view) or dict `{view_name: gamma}` |
| `warmup_steps` | Initial steps of pure MMD herding, including the random first pick. | Int in `[1, n_sub]`, float in `(0, 1]` (fraction of `n_sub`) or `None` (= 1). Default `0.25` |
| `epsilon` | MMD tolerance: only candidates with MMD score within `epsilon` of the best are considered. | `np.inf` (paper; pure rank-sum), `0` (pure herding), or anything in between |
| `random_state` | Seed for the first pick and the random view order. | Int or `None` |

Special cases:

- `warmup_steps=1.0` or `epsilon=0` gives **single-view kernel herding** (SKH)
  run independently in each view. This is the SKH baseline of the paper.
- With a single view, MKH reduces to kernel herding.

## Fitted attributes

| Attribute | Content |
|---|---|
| `selected_indices_` | Dict `{view_name: array of n_sub row indices}`, in selection order. |
| `selected_X_`, `selected_y_` | Selected features and labels of each view. |
| `view_names_` | View names (`view_0`, `view_1`, ... when a list is given). |
| `traces_` | Dict of dicts with the cross-view traces `T_vu`. `traces_[v][u] / n_sub**2` is the HSIC-type statistic of the selected pairs. |

## Class-stratified selection (paper protocol)

In the paper, every sampler is run **inside each class**, with
`ceil(n_sub / C)` rows per class (`C` classes), and the per-class subsets of
each view are merged. This keeps every class represented in every view. See
`stratified_subsample` in [`quickstart.py`](quickstart.py) and `subsample` in
`1_BO_subsampling_and_metrics/experiments/bo_accuracy_in_view.py`.

## Training a model on the subsets

```python
from multi_view_learning_models import LateFusion_LSSVM

selected = [sampler.selected_indices_[name] for name in sampler.view_names_]
model = LateFusion_LSSVM(kernel_gammas=[0.1, 0.05, 0.01],
                         selected_indices=selected,
                         gammas=[1.0, 1.0, 1.0])   # LS-SVM regularization
model.fit(X_views, y)          # full training views + indices of the subsets
y_pred = model.predict(X_views_test)
```

Each per-view LS-SVM costs `O(n_sub^3)` instead of `O(n^3)`.

## Practical tips

- **Scale the features** first. The experiments use `MinMaxScaler` fitted
  on the training rows of each view.
- **Tune `kernel_gammas` together with the model.** The sampler and the
  model should use the same kernel. In the experiments, both are tuned with
  Optuna on a validation split, and `warmup_steps` is tuned in `(0.01, 1.0]`.
- **Cost.** Centering the Gram matrices takes `O(V n^2)` time and memory, so
  the method is meant for datasets whose `n x n` kernel fits in memory. Each
  selection step costs `O(n)` per view plus the cross-view score.
- **Other samplers.** `RandomSubsamplingPerView`, `ClusteringSubsampling` and
  `FixedSizeEntropySubsampling` share the same interface (`fit`,
  `selected_indices_`, `transform`), so they can be swapped in directly.
