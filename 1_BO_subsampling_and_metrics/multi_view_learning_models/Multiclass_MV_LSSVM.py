"""
Multi-View LS-SVM with per-view, class-stratified subsampling (MMD-mean error
reconstruction).  One-vs-rest for m = n_classes binary subproblems.

Model (per subproblem l, implicit below):
    e_full^[v] = P^[v] e_S^[v]
      - retained row k          -> unit row on k
      - removed row k of class c -> (1/Ns^c) * indicator of retained pts of class c
        (only if reconstruct=True; otherwise removed rows are zero)
    min  1/2 sum_v w_v'w_v
       + 1/2 sum_v gamma_v ||P_v e_v||^2
       + rho sum_{v<u} (P_v e_v)'(P_u e_u)
    s.t. Z_v' w_v + y_S b_v = 1 - e_v

KKT:  C_v e_v + rho sum_{u!=v} M_vu e_u = alpha_v,   C_v = gamma_v P_v'P_v,
      M_vu = P_v'P_u  ->  e = D alpha,  D = blockmatrix^{-1}
Dual: [0 Y'; Y Omega + D][b; alpha] = [0; 1]

Convention: coupling is summed over unordered pairs v<u, so the gradient is
rho * M_vu (equivalent to the ordered-pair sum with rho/2).

Ablation switch
---------------
reconstruct=True  : proposed model (MV-LSSVM extention with correction terms, removed errors = class-wise kept mean).
reconstruct=False : ablation 1 (kept samples only, no correction terms).
    Rows of removed samples in P are zero, so P_v'P_v = I and
    M_vu = P_v'P_u = overlap indicator (1 where the same sample is kept in
    both views). The model is:
        min  1/2 sum_v w_v'w_v
           + 1/2 sum_v gamma_v e_v'e_v                      (own kept samples)
           + rho sum_{v<u} sum_{k in S_v cap S_u} e_v[k] e_u[k]  (shared samples)
    i.e. regular MV-LSSVM generalised to view subsets that do not line up.
 
Special cases of reconstruct=False:
    - Views keep DIFFERENT samples (the subsampling case): each view's gamma
      term uses its own kept samples; rho couples only samples kept in both
      views of a pair.
    - Views keep the SAME samples: original MV-LSSVM trained on that subset.
    - No shared samples between two views: their coupling vanishes; with no
      overlap at all the model is V independent LS-SVMs.
If every view keeps ALL samples, both settings reduce to the original
MV-LSSVM on the full data.
"""
import warnings
import numpy as np
from scipy.linalg import block_diag
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel


def build_reconstruction_operator(idx, y_cls, N, reconstruct=True):
    """P in R^{N x Ns}: full-sample errors = P @ e_S (stratified per class).

    reconstruct=False -> rows of removed samples stay zero (ablation 1).
    """
    idx = np.asarray(idx, dtype=int)
    if len(np.unique(idx)) != len(idx):
        raise ValueError("selected indices contain duplicates")
    Ns = len(idx)
    P = np.zeros((N, Ns))
    P[idx, np.arange(Ns)] = 1.0
    if not reconstruct:
        return P
    removed = np.setdiff1d(np.arange(N), idx)
    y_S = y_cls[idx]
    for c in np.unique(y_cls):
        pos_S = np.where(y_S == c)[0]
        rem_c = removed[y_cls[removed] == c]
        if len(rem_c) == 0:
            continue
        if len(pos_S) == 0:
            raise ValueError(f"class {c} has removed points but no retained point "
                             f"in this view (subsampling must be stratified)")
        P[np.ix_(rem_c, pos_S)] = 1.0 / len(pos_S)
    return P


class Multiclass_MV_LSSVM(BaseEstimator, ClassifierMixin):
    """
    Parameters
    ----------
    kernel_gammas : list of RBF gammas, one per view
    selected_indices : list of index arrays (into the training set), one per view
    gammas : list of length V, LS-SVM regularisation constant gamma^[v] per view
        (larger = less regularised). Distinct from kernel_gammas (RBF widths).
    rho : coupling parameter (>= 0). Must keep the block matrix PSD.
    view_weights : None (uniform 1/V) or list beta_v
    reconstruct : True = proposed model with class-wise mean-error reconstruction;
        False = ablation 1 (kept samples only, no correction terms).
    """

    def __init__(self, kernel_gammas, selected_indices, gammas, rho=0.0,
                 view_weights=None, reconstruct=True):
        self.kernel_gammas = kernel_gammas
        self.selected_indices = selected_indices
        self.gammas = gammas
        self.rho = rho
        self.view_weights = view_weights
        self.reconstruct = reconstruct

    # ------------------------------------------------------------------
    def _build_D(self, P_list, gam):
        V = len(P_list)
        blocks = [[None] * V for _ in range(V)]
        for v in range(V):
            for u in range(V):
                PtP = P_list[v].T @ P_list[u]
                blocks[v][u] = gam[v] * PtP if v == u else self.rho * PtP
        B = np.block(blocks)
        B = 0.5 * (B + B.T)
        lam_min = np.linalg.eigvalsh(B)[0]
        if lam_min <= 1e-10:
            warnings.warn(f"Coupled error matrix not positive definite "
                          f"(min eig={lam_min:.2e}); reduce rho or raise gamma.")
        return np.linalg.inv(B)

    def fit(self, X_list, y):
        y = np.asarray(y)
        V = len(X_list)
        N = len(y)
        self.classes_ = np.unique(y)
        y_cls = np.searchsorted(self.classes_, y)
        m = len(self.classes_)
        gam = np.asarray(self.gammas, float)
        if gam.shape != (V,) or np.any(gam <= 0):
            raise ValueError("gammas must be a list of V positive values, "
                             "one regularisation constant per view")
        if len(self.kernel_gammas) != V or len(self.selected_indices) != V:
            raise ValueError("kernel_gammas and selected_indices need one entry per view")

        idx = [np.asarray(self.selected_indices[v], dtype=int) for v in range(V)]
        P_list = [build_reconstruction_operator(idx[v], y_cls, N, self.reconstruct)
                  for v in range(V)]
        D = self._build_D(P_list, gam)

        Ns = [len(i) for i in idx]
        Ntot = sum(Ns)
        offs = np.cumsum([0] + Ns)

        self.sub_X_ = [np.asarray(X_list[v])[idx[v]] for v in range(V)]
        K_blocks = [rbf_kernel(self.sub_X_[v], gamma=self.kernel_gammas[v])
                    for v in range(V)]
        K_M = block_diag(*K_blocks)

        self.alpha_y_ = []   # per class: list per view of alpha*y
        self.b_ = np.zeros((m, V))
        for l in range(m):
            y_l = [np.where(y_cls[idx[v]] == l, 1.0, -1.0) for v in range(V)]
            yM = np.concatenate(y_l)
            Y_M = block_diag(*[yl.reshape(-1, 1) for yl in y_l])   # (Ntot, V)
            Omega = K_M * np.outer(yM, yM)
            A = np.zeros((V + Ntot, V + Ntot))
            A[:V, V:] = Y_M.T
            A[V:, :V] = Y_M
            A[V:, V:] = Omega + D
            rhs = np.concatenate([np.zeros(V), np.ones(Ntot)])
            sol = np.linalg.solve(A, rhs)
            self.b_[l] = sol[:V]
            alpha = sol[V:]
            self.alpha_y_.append([alpha[offs[v]:offs[v + 1]] * y_l[v]
                                  for v in range(V)])
        return self

    def decision_function(self, X_list):
        V = len(X_list)
        beta = (np.full(V, 1.0 / V) if self.view_weights is None
                else np.asarray(self.view_weights, float))
        m = len(self.classes_)
        F = np.zeros((np.asarray(X_list[0]).shape[0], m))
        for v in range(V):
            Kt = rbf_kernel(np.asarray(X_list[v]), self.sub_X_[v],
                            gamma=self.kernel_gammas[v])
            for l in range(m):
                F[:, l] += beta[v] * (Kt @ self.alpha_y_[l][v] + self.b_[l, v])
        return F

    def predict(self, X_list):
        return self.classes_[np.argmax(self.decision_function(X_list), axis=1)]


def stratified_random_indices(y, frac, rng):
    """Helper: stratified random subsample (stand-in for MMD herding)."""
    out = []
    for c in np.unique(y):
        ic = np.where(y == c)[0]
        out.append(rng.choice(ic, max(1, int(round(frac * len(ic)))), replace=False))
    return np.sort(np.concatenate(out))


# ----------------------------------------------------------------------
# Ablation 1: proposed (reconstruct=True) vs kept-samples-only (False)
# ----------------------------------------------------------------------
def run_ablation(X_train_list, y_train, X_test_list, y_test, selected_indices,
                 kernel_gammas, params_by_setting):
    """Fit both settings on the SAME subsets and return test errors.

    params_by_setting : dict {True: {"gammas": [...], "rho": r},
                              False: {"gammas": [...], "rho": r}}
        Hyperparameters tuned separately for each setting.
    """
    results = {}
    for reconstruct in (True, False):
        p = params_by_setting[reconstruct]
        model = Multiclass_MV_LSSVM(kernel_gammas, selected_indices,
                                    gammas=p["gammas"], rho=p["rho"],
                                    reconstruct=reconstruct)
        model.fit(X_train_list, y_train)
        results[reconstruct] = 1.0 - np.mean(model.predict(X_test_list) == y_test)
    return results


if __name__ == "__main__":
    # Minimal demo on Digits split into 8 row-views (random stratified subsets
    # as a placeholder for the herding output; fixed hyperparameters).
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y,
                                          random_state=0)
    V = 8
    to_views = lambda A: [A[:, 8 * v:8 * (v + 1)] for v in range(V)]
    Xtr_l, Xte_l = to_views(Xtr), to_views(Xte)

    rng = np.random.default_rng(0)
    sel = [stratified_random_indices(ytr, 0.1, rng) for _ in range(V)]
    kg = [0.5] * V
    params = {True: {"gammas": [1.0] * V, "rho": 0.05},
              False: {"gammas": [1.0] * V, "rho": 0.05}}

    res = run_ablation(Xtr_l, ytr, Xte_l, yte, sel, kg, params)
    print(f"Test error, proposed (reconstruct=True):  {res[True]:.4f}")
    print(f"Test error, ablation (reconstruct=False): {res[False]:.4f}")