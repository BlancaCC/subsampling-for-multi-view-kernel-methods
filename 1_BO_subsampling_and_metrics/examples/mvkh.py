"""Example: multi-view MMD-HSIC herding on a synthetic 3-view problem.

Adjust the import below to wherever the sampler lives in your package.
"""
import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from subsampling_methods import MultiViewKernelHerding

SEED = 0
N_SUB = 100

# ----------------------------------------------------------------------
# 1. Synthetic data, split into 3 views (disjoint feature blocks).
# ----------------------------------------------------------------------
X, y = make_classification(
    n_samples=1500, n_features=30, n_informative=15, n_redundant=5,
    n_classes=3, n_clusters_per_class=2, class_sep=1.2, random_state=SEED,
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED
)

blocks = {"view_0": slice(0, 10), "view_1": slice(10, 20), "view_2": slice(20, 30)}
views_tr, views_te = {}, {}
for name, sl in blocks.items():
    scaler = StandardScaler().fit(X_tr[:, sl])
    views_tr[name] = scaler.transform(X_tr[:, sl])
    views_te[name] = scaler.transform(X_te[:, sl])

# Gamma per view: 1 / (n_features_in_view) is a simple, scale-aware choice.
gammas = {name: 1.0 / Xv.shape[1] for name, Xv in views_tr.items()}


# ----------------------------------------------------------------------
# 2. Helpers: MMD of a subset, late-fusion evaluation.
# ----------------------------------------------------------------------
def mmd(Xv, idx, gamma):
    """Empirical MMD between the subset Xv[idx] and the full view Xv."""
    K = rbf_kernel(Xv, gamma=gamma)
    a = K.mean()
    b = K[np.ix_(idx, idx)].mean()
    c = K[:, idx].mean()
    return np.sqrt(max(a + b - 2 * c, 0.0))


def late_fusion_accuracy(selected):
    """Train one classifier per view on its own subset, average probabilities."""
    probs = []
    for name, idx in selected.items():
        clf = LogisticRegression(max_iter=1000)
        clf.fit(views_tr[name][idx], y_tr[idx])
        probs.append(clf.predict_proba(views_te[name]))
    return accuracy_score(y_te, np.mean(probs, axis=0).argmax(axis=1))


def hsic_paired(Xv, Xu, iv, iu, gv, gu):
    """T_vu(t)/t^2 with dataset-centered kernels and position pairing (Lemma 1)."""
    def cgram(X, g):
        K = rbf_kernel(X, gamma=g)
        return K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()
    Kv, Ku = cgram(Xv, gv), cgram(Xu, gu)
    T = np.sum(Kv[np.ix_(iv, iv)] * Ku[np.ix_(iu, iu)])
    return T / len(iv) ** 2


# ----------------------------------------------------------------------
# 3. Run the sampler (and baselines).
# ----------------------------------------------------------------------
configs = {
    "MMD only   (eps=0)":   dict(epsilon=0.0),
    "rank-sum   (eps=inf)": dict(epsilon=np.inf),
}

results = {}
for label, kw in configs.items():
    sampler = MultiViewKernelHerding(
        n_sub=N_SUB, kernel_gammas=gammas, warmup_steps=0.25,
        random_state=SEED, **kw,
    )
    sampler.fit_transform(views_tr, y_tr)
    results[label] = sampler

rng = np.random.default_rng(SEED)
random_sel = {n: rng.choice(len(y_tr), N_SUB, replace=False) for n in views_tr}

# ----------------------------------------------------------------------
# 4. Report.
# ----------------------------------------------------------------------
names = list(views_tr)
pairs = [(names[a], names[b]) for a in range(len(names)) for b in range(a + 1, len(names))]

def report(label, sel):
    mmds = [mmd(views_tr[n], sel[n], gammas[n]) for n in names]
    hs = [hsic_paired(views_tr[u], views_tr[v], sel[u], sel[v], gammas[u], gammas[v])
          for u, v in pairs]
    print(f"{label:22s} | mean MMD {np.mean(mmds):.4f} | mean paired HSIC {np.mean(hs):.5f}"
          f" | late-fusion acc {late_fusion_accuracy(sel):.3f}")

print(f"n_train={len(y_tr)}, n_sub={N_SUB}, views={names}\n")
report("random", random_sel)
for label, s in results.items():
    report(label, s.selected_indices_)

# Sanity checks
for label, s in results.items():
    for n, idx in s.selected_indices_.items():
        assert len(idx) == N_SUB and len(set(idx)) == N_SUB, "indices must be distinct"
    u, v = pairs[0]
    direct = hsic_paired(views_tr[u], views_tr[v],
                         s.selected_indices_[u], s.selected_indices_[v],
                         gammas[u], gammas[v])
    incremental = s.traces_[u][v] / N_SUB ** 2
    assert np.isclose(direct, incremental), (direct, incremental)
print("\nChecks passed: distinct indices; incremental trace == Lemma 1 double sum.")
print("First 10 picks per view:", {n: s.tolist()[:10] for n, s in
                                   results['rank-sum   (eps=inf)'].selected_indices_.items()})