import numpy as np

from .BaseMultiViewSampler import BaseMultiViewSubsampler
from .MultiViewKernelHerding import MultiViewKernelHerding


class FixedSizeEntropySubsampling(BaseMultiViewSubsampler):
    """Fixed-size LS-SVM prototype selection, drawn independently per view.

    Implements the prototype-vector selection of De Brabanter et al.,
    "Optimized fixed-size kernel models for large data sets", CSDA 2010:
    the selected set S (|S| = n_sub) maximizes the quadratic Renyi entropy

        H_R(S) = -log( (1 / n_sub^2) * sum_{i,j in S} k(x_i, x_j) ),

    i.e. minimizes the sum of the RBF Gram matrix of S, which spreads the
    prototypes over the input space. Active selection: start from a random
    S; at every iteration swap a random member p for a random non-member q
    and keep the swap only if the entropy increases.

    The entropy change of a swap is computed incrementally in O(n_sub * d)
    (the "optimized" part), from the row sums r of the Gram matrix of S:

        sum' = sum - 2 r_p + k_pp + 2 (r_q - k_qp) + k_qq,

    instead of recomputing the O(n_sub^2 * d) sum.

    Args:
        n_sub: Number of samples to select from every view.
        kernel_gammas: RBF gamma per view (scalar, list/array or dict, as
            in MultiViewKernelHerding). Used for the entropy kernel when
            entropy_gamma="model".
        warmup_steps: Accepted only so the constructor is interchangeable
            with MultiViewKernelHerding; not used.
        epsilon: Same as warmup_steps; not used.
        random_state: Seeds the initial set and the swap proposals.
        n_iter: Number of swap proposals per view. None uses 10 * n_samples
            (every sample is proposed about 10 times on average).
        entropy_gamma: "model" uses kernel_gammas; "silverman" uses
            Silverman's rule of thumb (as in the paper) per view,
            gamma = 1 / (2 h^2), h = sigma * (4 / ((d + 2) n))^(1 / (d + 4)),
            sigma = mean feature standard deviation. For high-dimensional
            views the Silverman bandwidth makes off-diagonal kernel values
            vanish, so "model" is the default.

    Attributes:
        entropy_: After fitting, dict view name -> final Renyi entropy.
        n_accepted_: After fitting, dict view name -> accepted swaps.
    """

    def __init__(
        self,
        n_sub,
        kernel_gammas=None,
        warmup_steps=None,
        epsilon=None,
        random_state=None,
        n_iter=None,
        entropy_gamma="model",
    ):
        super().__init__(random_state=random_state)

        if not isinstance(n_sub, (int, np.integer)):
            raise TypeError("n_sub must be an integer.")

        if n_sub <= 0:
            raise ValueError("n_sub must be greater than zero.")

        if n_iter is not None and n_iter < 0:
            raise ValueError("n_iter must be None or non-negative.")

        if entropy_gamma not in ("model", "silverman"):
            raise ValueError("entropy_gamma must be 'model' or 'silverman'.")

        self.n_sub = int(n_sub)
        self.kernel_gammas = kernel_gammas
        self.warmup_steps = warmup_steps
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.entropy_gamma = entropy_gamma

    def _select_indices(self, views, y):
        """Selects `n_sub` maximum-entropy prototypes per view.

        Returns:
            dict[str, np.ndarray]: view name -> selected indices, as the
                base class expects.
        """
        n_samples = len(y)

        if len(views) == 0:
            raise ValueError("At least one feature view is required.")

        if self.n_sub > n_samples:
            raise ValueError(
                f"n_sub={self.n_sub} cannot be greater than the "
                f"number of training samples ({n_samples})."
            )

        if self.entropy_gamma == "model":
            if self.kernel_gammas is None:
                raise ValueError(
                    "kernel_gammas is required when entropy_gamma='model'."
                )
            # Same scalar/list/dict handling as MultiViewKernelHerding.
            gammas = MultiViewKernelHerding._resolve_kernel_gammas(self, views)
        else:
            gammas = {
                name: self._silverman_gamma(X) for name, X in views.items()
            }

        n_iter = 10 * n_samples if self.n_iter is None else int(self.n_iter)

        self.entropy_ = {}
        self.n_accepted_ = {}
        selected = {}
        for view_name, X in views.items():
            idx, entropy, n_accepted = self._select_view(
                np.asarray(X, dtype=float), gammas[view_name], n_iter
            )
            selected[view_name] = idx
            self.entropy_[view_name] = entropy
            self.n_accepted_[view_name] = n_accepted
        return selected

    def _select_view(self, X, gamma, n_iter):
        """Active selection of one view. Returns (indices, entropy, accepts)."""
        n_samples, m = X.shape[0], self.n_sub
        sq_norms = np.einsum("ij,ij->i", X, X)

        def kernel_row(i, members):
            """k(x_i, x_j) for j in members (RBF, k_ii = 1)."""
            d2 = sq_norms[i] + sq_norms[members] - 2.0 * X[members] @ X[i]
            return np.exp(-gamma * np.maximum(d2, 0.0))

        perm = self.rng_.permutation(n_samples)
        members, outside = perm[:m].copy(), perm[m:].copy()

        # Gram matrix of S and its row sums (k_ii = 1 on the diagonal).
        d2 = (sq_norms[members][:, None] + sq_norms[members][None, :]
              - 2.0 * X[members] @ X[members].T)
        gram = np.exp(-gamma * np.maximum(d2, 0.0))
        row_sums = gram.sum(axis=1)
        total = row_sums.sum()

        n_accepted = 0
        if len(outside) > 0:
            for _ in range(n_iter):
                pos = self.rng_.integers(m)
                out_pos = self.rng_.integers(len(outside))
                q = outside[out_pos]

                k_q = kernel_row(q, members)          # k(q, S), includes k_qp.
                r_q = k_q.sum()
                new_total = (total - 2.0 * row_sums[pos] + 1.0
                             + 2.0 * (r_q - k_q[pos]) + 1.0)

                # Higher entropy <=> smaller Gram sum.
                if new_total < total:
                    # Row sums of the other members: drop p, add q.
                    row_sums += k_q - gram[:, pos]
                    k_q[pos] = 1.0                    # k_qq on the diagonal.
                    gram[pos, :] = k_q
                    gram[:, pos] = k_q
                    row_sums[pos] = k_q.sum()
                    outside[out_pos] = members[pos]
                    members[pos] = q
                    total = new_total
                    n_accepted += 1

        entropy = -np.log(total / m**2)
        return np.sort(members), float(entropy), n_accepted

    @staticmethod
    def _silverman_gamma(X):
        """RBF gamma from Silverman's rule of thumb for one view."""
        n, d = X.shape
        sigma = float(np.mean(np.std(X, axis=0)))
        if sigma <= 0:
            raise ValueError("Constant view: Silverman bandwidth is zero.")
        h = sigma * (4.0 / ((d + 2) * n)) ** (1.0 / (d + 4))
        return 1.0 / (2.0 * h**2)
