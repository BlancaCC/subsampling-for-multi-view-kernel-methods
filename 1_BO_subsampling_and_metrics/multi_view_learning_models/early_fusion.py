"""
Early-fusion baseline: all views are concatenated at the input into one
single-view dataset, which is subsampled and fed to one standard LS-SVM.

Fusion (ViewConcatenator, fitted on the training rows only):
    scaling="view": every feature is z-scored, then view v is divided by
        sqrt(d_v) so each view contributes equally to the squared distances
        of the single RBF kernel, whatever its dimension or feature scale.
    scaling="weight": view v is only divided by sqrt(d_v) (for inputs that
        are already scaled, e.g. MinMax), so each view counts equally.
    scaling=None:   raw concatenation.

Training (one-vs-rest for m classes):
    standard LS-SVM on (X_concat[S], y[S]) with one kernel width and one gamma.
    Equivalent to Multiclass_MV_LSSVM with V=1, rho=0, reconstruct=False.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from .Multiclass_MV_LSSVM import Multiclass_MV_LSSVM


class ViewConcatenator(BaseEstimator, TransformerMixin):
    """Concatenates a list of views into one feature matrix.

    Parameters
    ----------
    scaling : "view" (z-score features, then weight view v by 1/sqrt(d_v)),
        "weight" (only the 1/sqrt(d_v) weighting) or None (raw).
    """

    def __init__(self, scaling="view"):
        self.scaling = scaling

    def fit(self, X_list, y=None):
        if self.scaling not in ("view", "weight", None):
            raise ValueError("scaling must be 'view', 'weight' or None")
        X_list = [np.asarray(X, float) for X in X_list]
        self.n_views_ = len(X_list)
        if self.scaling == "view":
            self.mean_ = [X.mean(axis=0) for X in X_list]
            # Constant features keep std 1 so they become 0, not NaN.
            self.std_ = [np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
                         for X in X_list]
        return self

    def transform(self, X_list):
        if len(X_list) != self.n_views_:
            raise ValueError(f"expected {self.n_views_} views, got {len(X_list)}")
        X_list = [np.asarray(X, float) for X in X_list]
        if self.scaling == "view":
            X_list = [(X - m) / s / np.sqrt(X.shape[1])
                      for X, m, s in zip(X_list, self.mean_, self.std_)]
        elif self.scaling == "weight":
            X_list = [X / np.sqrt(X.shape[1]) for X in X_list]
        return np.hstack(X_list)


class EarlyFusion_LSSVM(BaseEstimator, ClassifierMixin):
    """
    Parameters
    ----------
    kernel_gamma : RBF gamma of the single kernel on the concatenated features
    selected_indices : index array (into the training set) of the subsample
    gamma : LS-SVM regularisation constant (larger = less regularised)
    scaling : passed to ViewConcatenator ("view", "weight" or None)
    """

    def __init__(self, kernel_gamma, selected_indices, gamma, scaling="view"):
        self.kernel_gamma = kernel_gamma
        self.selected_indices = selected_indices
        self.gamma = gamma
        self.scaling = scaling

    def fit(self, X_list, y):
        # Scaling statistics use all training rows (cheap, O(N d)); only the
        # subsample enters the kernel system.
        self.concatenator_ = ViewConcatenator(self.scaling).fit(X_list)
        X = self.concatenator_.transform(X_list)
        self.model_ = Multiclass_MV_LSSVM(
            [self.kernel_gamma], [self.selected_indices], gammas=[self.gamma],
            rho=0.0, reconstruct=False,
        ).fit([X], y)
        self.classes_ = self.model_.classes_
        return self

    def decision_function(self, X_list):
        return self.model_.decision_function([self.concatenator_.transform(X_list)])

    def predict(self, X_list):
        return self.classes_[np.argmax(self.decision_function(X_list), axis=1)]


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
    sel = stratified_random_indices(ytr, 0.1, rng)
    for scaling in ("view", None):
        clf = EarlyFusion_LSSVM(0.5 if scaling else 0.05, sel, gamma=10.0,
                                scaling=scaling).fit(Xtr_l, ytr)
        print(f"Test error, scaling={scaling}: "
              f"{1 - np.mean(clf.predict(Xte_l) == yte):.4f}")
