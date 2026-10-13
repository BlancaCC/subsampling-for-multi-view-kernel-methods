"""
Adaptive-Weighted Multi-View LS-SVM (AW-LSSVM) with per-view, independently
subsampled views (partial / non-aligned MV data).

Each view v is trained as a one-vs-rest LS-SVM on its own kept subset
S_v = selected_indices[v] (same convention as Multiclass_MV_LSSVM /
LateFusion_LSSVM), but instead of a fixed, closed-form coupling operator
(the P/rho reconstruction of Multiclass_MV_LSSVM) the per-sample
regularisation is adapted over T = range_steps rounds from the *other*
views' mistakes, following "Adaptive Weighted LS-SVM for Multi-View
Classification" (Eqs. 1-7, Algorithm 1), restricted to each pair of views'
common support, S_v ^ S_u (the samples both v and u kept).

Per view v, class l (one-vs-rest), at round t:
    min_{w,b,e}  1/2 w'w + 1/2 gamma_v sum_{k in S_v} e_k^2
    s.t.  w'phi(x_k) + b = y_k - e_k,   k in S_v
Dual (as in Multiclass_MV_LSSVM / MultiClassLSSVM), but weighted:
    [0 y'; y  Omega + D_v^t][b; alpha] = [0; 1],
    D_v^t = diag_{k in S_v}( 1 / (gamma_v + ro_v * u_k^t) )
where u_k^t >= 0 is sample k's adaptive weight (u_k^0 = 0, i.e. round 0 is
plain LS-SVM for every view).

Adaptive weights (uniform-average variant, Eq. 2, restricted to the views
that actually kept each sample):
    u_k^t += decay**t * (1 / |O_v(k)|) * sum_{u in O_v(k)} e_u(k)
where O_v(k) = the OTHER views that also kept k and e_u(k) is view u's
squared error on k -- a per-sample divisor, since not every view keeps
every sample.
    - coupling="hard"  : only samples view v itself misclassifies feed the
      regulariser (squared error of correctly-classified samples is
      zeroed first); coupling="all" uses every sample's squared error.
A sample kept by only one view gets no coupling term at all (no other view
shares it) and trains exactly like plain LS-SVM for that sample -- nothing
breaks, there's just no compensation signal available for it.

Final prediction is a uniform soft vote (Eq. 7) over whichever views are
available for each test sample -- the same `view_mask` convention as
`LateFusion_LSSVM.decision_function` -- except the per-view scores come
from these adaptively-coupled models instead of independently-trained ones.

Relation to the other two MV techniques in this folder
--------------------------------------------------------
- vs. Multiclass_MV_LSSVM: same `selected_indices` per-view subsampling
  convention, but the cross-view coupling is an adaptive, iteratively
  re-weighted *sample regulariser* instead of a fixed closed-form error-
  reconstruction operator (no P matrices, no single joint dual solve).
- vs. LateFusion_LSSVM: same final soft-vote fusion (and the same
  `view_mask` test-time convention), but views are trained with cross-
  view-informed adaptive weights instead of fully independently.
  `range_steps=1` recovers LateFusion_LSSVM exactly (round 0 is always
  unweighted, so no coupling ever gets applied).
"""
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics.pairwise import rbf_kernel


class AW_LSSVM_Partial(BaseEstimator, ClassifierMixin):
    """
    Parameters
    ----------
    kernel_gammas : list of RBF gammas, one per view
    selected_indices : list of index arrays (into the training set), one per
        view -- same convention as Multiclass_MV_LSSVM / LateFusion_LSSVM.
    gammas : list of length V, LS-SVM regularisation constant gamma^[v] per
        view (larger = less regularised). Distinct from kernel_gammas.
    ro : None, a scalar, or a list of length V -- coupling-term
        regularisation weight per view (see module docstring).
        None -> 10.0 for every view.
    decay : float in (0, 1] -- geometric decay `decay**t` applied to the
        round-t coupling increment.
    coupling : "hard" (only samples a view itself misclassifies feed the
        regulariser) or anything else, e.g. "all" (every sample's squared
        error feeds it).
    range_steps : number of adaptive-weighting rounds T. `range_steps=1`
        reduces to LateFusion_LSSVM (no coupling ever applied).
    """

    def __init__(self, kernel_gammas, selected_indices, gammas, ro=None,
                 decay=0.7, coupling="hard", range_steps=10):
        self.kernel_gammas = kernel_gammas
        self.selected_indices = selected_indices
        self.gammas = gammas
        self.ro = ro
        self.decay = decay
        self.coupling = coupling
        self.range_steps = range_steps

    # ------------------------------------------------------------------
    # adaptive-weighting internals (restricted to the views that kept each sample)
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_errors(y_cls, y_pred_cls, scores, coupling):
        """Per-sample, per-class squared error used to build next round's
        regulariser. `y_cls`/`y_pred_cls` are integer class indices
        (0..m-1); `scores` is the (n, m) decision matrix they came from.
        """
        n, m = scores.shape
        label_matrix = -np.ones((n, m))
        label_matrix[np.arange(n), y_cls] = 1.0
        errors_orig = 1.0 - label_matrix * scores
        squared = errors_orig ** 2
        if coupling == "hard":
            misclassified = (y_pred_cls != y_cls)[:, None]
            errors = np.where(misclassified, squared, 0.0)
        else:
            errors = squared
        return errors

    def _update_weights_partial(self, weights_global, errors_global, presence_mask, t):
        """For each view c and (global) sample k, only the other views that
        actually kept k contribute to c's regulariser at k: the mean of
        their squared errors on k (per-sample divisor |O_c(k)|).
        """
        V, N, m = errors_global.shape
        for c in range(V):
            other_mask = presence_mask.copy()
            other_mask[c, :] = False  # exclude the view being updated
            counts = other_mask.sum(axis=0).astype(float)
            safe_counts = np.where(counts == 0, 1.0, counts)
            for cls in range(m):
                contrib = np.sum(errors_global[:, :, cls] * other_mask, axis=0)
                weights_global[c, :, cls] += (self.decay ** t) * (contrib / safe_counts)
        return weights_global

    # ------------------------------------------------------------------
    # per-view weighted one-vs-rest LS-SVM dual solve
    # ------------------------------------------------------------------
    @staticmethod
    def _fit_view(Kv, y_bin_all, weights_v, gamma_v, ro_v):
        """One-vs-rest LS-SVM dual solve for a single view, every class at
        once. `Kv` is the (n_v, n_v) train-train RBF Gram matrix (fixed over
        the rounds, so computed once in fit); `y_bin_all` is (m, n_v) of +-1
        labels (one row per class); `weights_v` is (n_v, m) adaptive weights
        (all-0 => plain LS-SVM). Returns (alpha_y, b): alpha_y shape
        (m, n_v), b shape (m,).
        """
        n_v = Kv.shape[0]
        m = y_bin_all.shape[0]
        alpha_y = np.zeros((m, n_v))
        b = np.zeros(m)
        for l in range(m):
            y_bin = y_bin_all[l]
            D = np.diag(1.0 / (gamma_v + ro_v * weights_v[:, l]))
            Omega = Kv * np.outer(y_bin, y_bin)
            A = np.zeros((n_v + 1, n_v + 1))
            A[0, 1:] = y_bin
            A[1:, 0] = y_bin
            A[1:, 1:] = Omega + D
            A += 1e-9 * np.eye(n_v + 1)
            rhs = np.zeros(n_v + 1)
            rhs[1:] = 1.0
            sol = np.linalg.solve(A, rhs)
            b[l] = sol[0]
            alpha_y[l] = sol[1:] * y_bin
        return alpha_y, b

    # ------------------------------------------------------------------
    def fit(self, X_list, y):
        y = np.asarray(y)
        V = len(X_list)
        N = len(y)
        if not (len(self.kernel_gammas) == len(self.selected_indices)
                == len(self.gammas) == V):
            raise ValueError("kernel_gammas, selected_indices and gammas need "
                              "one entry per view")
        self.classes_ = np.unique(y)
        m = len(self.classes_)
        y_cls = np.searchsorted(self.classes_, y)

        gammas = np.asarray(self.gammas, dtype=float)
        if gammas.shape != (V,) or np.any(gammas <= 0):
            raise ValueError("gammas must be a list of V positive values, "
                              "one regularisation constant per view")

        ro = self.ro
        if ro is None:
            ro = 10.0
        ro = np.full(V, float(ro)) if np.isscalar(ro) else np.asarray(ro, dtype=float)
        if ro.shape != (V,):
            raise ValueError("ro must be a scalar or a list of length V")

        idx = [np.asarray(self.selected_indices[v], dtype=int) for v in range(V)]
        for v in range(V):
            if len(np.unique(idx[v])) != len(idx[v]):
                raise ValueError(f"selected_indices[{v}] contains duplicates")
        self.idx_ = idx
        self.n_views_ = V

        presence_mask = np.zeros((V, N), dtype=bool)
        for v in range(V):
            presence_mask[v, idx[v]] = True
        coverage = presence_mask.sum(axis=0)
        covered = coverage > 0
        if not np.all(covered):
            n_uncovered = int((~covered).sum())
            warnings.warn(
                f"{n_uncovered} training sample(s) are not kept by any view "
                "(independent per-view subsampling can do this); they are simply "
                "unused by every view and excluded from the training-committee "
                "diagnostic (sc_train_), not an error."
            )
        self.presence_mask_ = presence_mask
        self.coverage_ = coverage
        safe_coverage = np.where(covered, coverage, 1)  # avoid 0/0 below; result unused where ~covered

        self.sub_X_ = [np.asarray(X_list[v])[idx[v]] for v in range(V)]
        y_cls_sub = [y_cls[idx[v]] for v in range(V)]
        y_bin_sub = [np.stack([np.where(y_cls_sub[v] == l, 1.0, -1.0) for l in range(m)])
                     for v in range(V)]

        # Gram matrices depend only on the kept samples: compute once.
        K_sub = [rbf_kernel(self.sub_X_[v], gamma=self.kernel_gammas[v])
                 for v in range(V)]

        weights_global = np.zeros((V, N, m))

        self.sc_train_ = []

        t = 0
        for t in range(self.range_steps):
            errors_global = np.zeros((V, N, m))
            y_sum_global = np.zeros((N, m))
            alpha_y_this_t, b_this_t = [], []

            for v in range(V):
                weights_v = weights_global[v][idx[v], :]  # (n_v, m); all-0 at t=0
                Kv = K_sub[v]
                alpha_y_v, b_v = self._fit_view(
                    Kv, y_bin_sub[v], weights_v, gammas[v], ro[v],
                )
                alpha_y_this_t.append(alpha_y_v)
                b_this_t.append(b_v)

                F_v = (Kv @ alpha_y_v.T) + b_v[None, :]  # (n_v, m)
                y_sum_global[idx[v], :] += F_v

                y_pred_local = np.argmax(F_v, axis=1)
                errors_global[v, idx[v], :] = self._sample_errors(
                    y_cls_sub[v], y_pred_local, F_v, self.coupling
                )

            self.alpha_y_, self.b_ = alpha_y_this_t, b_this_t

            y_pred_com = np.argmax(y_sum_global / safe_coverage[:, None], axis=1)
            self.y_pred_com_ = self.classes_[y_pred_com]
            self.sc_train_.append(balanced_accuracy_score(y_cls[covered], y_pred_com[covered]))

            if np.sum(errors_global) == 0:
                break

            weights_global = self._update_weights_partial(
                weights_global, errors_global, presence_mask, t
            )

        self.weights_ = weights_global
        self.t_ = t
        return self

    # ------------------------------------------------------------------
    def view_decision_functions(self, X_list):
        """Per-view scores from the final adaptively-weighted models,
        shape (V, n_samples, n_classes)."""
        V = self.n_views_
        m = len(self.classes_)
        n = np.asarray(X_list[0]).shape[0]
        S = np.zeros((V, n, m))
        for v in range(V):
            Kt = rbf_kernel(np.asarray(X_list[v]), self.sub_X_[v], gamma=self.kernel_gammas[v])
            S[v] = Kt @ self.alpha_y_[v].T + self.b_[v][None, :]
        return S

    def decision_function(self, X_list, view_mask=None):
        """Uniform soft vote over available views (same convention as
        `LateFusion_LSSVM.decision_function`).

        view_mask : None (all views available) or bool array (n_samples, V);
            view_mask[i, v] = True if view v is available for test sample i.
            Features of unavailable views are ignored (may be any placeholder).
        """
        S = self.view_decision_functions(X_list)           # (V, n, m)
        V, n, _ = S.shape
        mask = (np.ones((n, V), bool) if view_mask is None
                else np.asarray(view_mask, bool))
        if mask.shape != (n, V):
            raise ValueError(f"view_mask must have shape ({n}, {V})")
        counts = mask.sum(axis=1)
        if np.any(counts == 0):
            raise ValueError("some samples have no available view")
        W = mask.T[:, :, None].astype(float)                # (V, n, 1)
        return (W * S).sum(axis=0) / counts[:, None]

    def predict(self, X_list, view_mask=None):
        return self.classes_[np.argmax(self.decision_function(X_list, view_mask), axis=1)]


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

    # Each view independently keeps ~50% of each class (a stand-in for a
    # per-view subsampling technique, e.g. MMD herding) -- views overlap
    # only partially, exactly the setting this class is for.
    rng = np.random.default_rng(0)
    sel = [stratified_random_indices(ytr, 0.5, rng) for _ in range(V)]

    clf = AW_LSSVM_Partial([0.5] * V, sel, gammas=[1.0] * V, ro=[5.0] * V,
                           decay=0.7, coupling="hard",
                           range_steps=8).fit(Xtr_l, ytr)

    print(f"rounds actually run: {clf.t_ + 1} / {clf.range_steps}")
    print(f"train balanced acc per round: {[round(s, 3) for s in clf.sc_train_]}")
    print(f"Test error, all views:      {1 - np.mean(clf.predict(Xte_l) == yte):.4f}")

    mask = rng.random((len(yte), V)) < 0.5
    mask[np.arange(len(yte)), rng.integers(0, V, len(yte))] = True
    print(f"Test error, ~50% views/sample: "
          f"{1 - np.mean(clf.predict(Xte_l, mask) == yte):.4f}")