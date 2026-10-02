"""
Late-fusion baseline: one independent LS-SVM per view, each trained on that
view's own subsample, combined at the decision level by uniform soft voting.

Training (per view v, one-vs-rest for m classes):
    standard LS-SVM on (X_v[S_v], y[S_v]) with its own kernel width and gamma_v.
    Equivalent to Multiclass_MV_LSSVM with V=1, rho=0, reconstruct=False.

Fusion (uniform soft voting over the views available for each sample):
    F(x) = 1/|A(x)| * sum_{v in A(x)} f_v(x),    y_hat = argmax_l F_l(x)
    A(x) = views available for sample x (given by view_mask; default all).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .Multiclass_MV_LSSVM import Multiclass_MV_LSSVM


class LateFusion_LSSVM(BaseEstimator, ClassifierMixin):
    """
    Parameters
    ----------
    kernel_gammas : list of RBF gammas, one per view
    selected_indices : list of index arrays (into the training set), one per view
    gammas : list of length V, LS-SVM regularisation constant gamma^[v] per view
    """

    def __init__(self, kernel_gammas, selected_indices, gammas):
        self.kernel_gammas = kernel_gammas
        self.selected_indices = selected_indices
        self.gammas = gammas

    def fit(self, X_list, y):
        y = np.asarray(y)
        V = len(X_list)
        if not (len(self.kernel_gammas) == len(self.selected_indices)
                == len(self.gammas) == V):
            raise ValueError("kernel_gammas, selected_indices and gammas need "
                             "one entry per view")
        self.classes_ = np.unique(y)
        # Each view model receives the full y so all share the same classes_.
        self.models_ = [
            Multiclass_MV_LSSVM([self.kernel_gammas[v]],
                                [self.selected_indices[v]],
                                gammas=[self.gammas[v]], rho=0.0,
                                reconstruct=False).fit([X_list[v]], y)
            for v in range(V)
        ]
        return self

    def view_decision_functions(self, X_list):
        """Per-view scores, shape (V, n_samples, n_classes)."""
        return np.stack([self.models_[v].decision_function([X_list[v]])
                         for v in range(len(self.models_))])

    def decision_function(self, X_list, view_mask=None):
        """Uniform soft vote over available views.

        view_mask : None (all views available) or bool array (n_samples, V);
            view_mask[i, v] = True if view v is available for sample i.
            Features of unavailable views are ignored (may be any placeholder).
        """
        S = self.view_decision_functions(X_list)          # (V, n, m)
        V, n, _ = S.shape
        mask = (np.ones((n, V), bool) if view_mask is None
                else np.asarray(view_mask, bool))
        if mask.shape != (n, V):
            raise ValueError(f"view_mask must have shape ({n}, {V})")
        counts = mask.sum(axis=1)
        if np.any(counts == 0):
            raise ValueError("some samples have no available view")
        W = mask.T[:, :, None].astype(float)              # (V, n, 1)
        return (W * S).sum(axis=0) / counts[:, None]

    def predict(self, X_list, view_mask=None):
        return self.classes_[np.argmax(self.decision_function(X_list, view_mask),
                                       axis=1)]


if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from .Multiclass_MV_LSSVM import stratified_random_indices

    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y,
                                          random_state=0)
    V = 8
    to_views = lambda A: [A[:, 8 * v:8 * (v + 1)] for v in range(V)]
    Xtr_l, Xte_l = to_views(Xtr), to_views(Xte)

    rng = np.random.default_rng(0)
    sel = [stratified_random_indices(ytr, 0.1, rng) for _ in range(V)]
    clf = LateFusion_LSSVM([0.5] * V, sel, gammas=[1.0] * V).fit(Xtr_l, ytr)

    print(f"Test error, all views:      {1 - np.mean(clf.predict(Xte_l) == yte):.4f}")
    mask = rng.random((len(yte), V)) < 0.5
    mask[np.arange(len(yte)), rng.integers(0, V, len(yte))] = True
    print(f"Test error, ~50% views/sample: "
          f"{1 - np.mean(clf.predict(Xte_l, mask) == yte):.4f}")
