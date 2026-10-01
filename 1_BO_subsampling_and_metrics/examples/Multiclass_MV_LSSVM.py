from sklearn.datasets import make_classification
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from multi_view_learning_models import Multiclass_MV_LSSVM, build_reconstruction_operator, stratified_random_indices



# ----------------------------------------------------------------------
# Sanity tests
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    rng = np.random.default_rng(0)
    N = 120
    X, y = make_classification(N, n_features=8, n_informative=6, n_classes=3,
                               n_clusters_per_class=1, random_state=0)
    Xl = [X[:, :4], X[:, 4:]]
    kg = [0.1, 0.1]

    # 1) P operator reproduces the paper's single-class formulas
    yy = np.zeros(N, int)
    idx = np.sort(rng.choice(N, 30, replace=False))
    P = build_reconstruction_operator(idx, yy, N)
    Ns, Nr = len(idx), N - len(idx)
    C_paper = np.eye(Ns) + Nr / Ns**2 * np.ones((Ns, Ns))
    assert np.allclose(P.T @ P, C_paper)
    print("OK  P'P == I + Nr/Ns^2 11'")

    # 2) rho=0, Nr=0 -> independent per-view LS-SVMs
    full = [np.arange(N)] * 2
    gs = [2.0, 8.0]   # different regularisation per view
    clf = Multiclass_MV_LSSVM(kg, full, gammas=gs, rho=0.0).fit(Xl, y)
    for v in range(2):
        K = rbf_kernel(Xl[v], gamma=kg[v])
        for l in range(3):
            yl = np.where(y == l, 1.0, -1.0)
            A = np.zeros((N + 1, N + 1))
            A[0, 1:] = yl; A[1:, 0] = yl
            A[1:, 1:] = K * np.outer(yl, yl) + np.eye(N) / gs[v]
            s = np.linalg.solve(A, np.r_[0, np.ones(N)])
            assert np.isclose(s[0], clf.b_[l, v])
            assert np.allclose(s[1:] * yl, clf.alpha_y_[l][v])
    print("OK  rho=0, Nr=0 reduces to independent per-view LS-SVMs")

    # 3) constraint y'alpha = 0 and e = D alpha consistency on subsampled fit
    sel = [stratified_random_indices(y, 0.5, rng) for _ in range(2)]
    clf = Multiclass_MV_LSSVM(kg, sel, gammas=[2.0, 8.0], rho=0.5).fit(Xl, y)
    for l in range(3):
        for v in range(2):
            yl = np.where(y[sel[v]] == l, 1.0, -1.0)
            assert abs(yl @ (clf.alpha_y_[l][v] * yl)) < 1e-8
    print("OK  y_S' alpha = 0 on subsampled, coupled fit")

    # 4) accuracy
    perm = rng.permutation(N); tr, te = perm[:80], perm[80:]
    sel = [stratified_random_indices(y[tr], 0.5, rng) for _ in range(2)]
    clf = Multiclass_MV_LSSVM(kg, sel, gammas=[2.0, 8.0], rho=0.3).fit(
        [x[tr] for x in Xl], y[tr])
    print("test acc:", (clf.predict([x[te] for x in Xl]) == y[te]).mean())