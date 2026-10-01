import numpy as np

from scipy.stats import rankdata
from sklearn.metrics.pairwise import rbf_kernel

from .BaseMultiViewSampler import BaseMultiViewSubsampler


class MultiViewKernelHerding(BaseMultiViewSubsampler):
    """Joint multi-view MMD-HSIC herding sampler.

    This sampler jointly selects representative samples from multiple
    feature views using a combination of:

    - MMD-based representativeness within each view.
    - HSIC-based dependence between the (position-paired) selected samples
      of different views.

    For each view, an RBF Gram matrix is computed and centered. The centered
    kernel is used to maintain an MMD accumulator vector. During the
    warm-up phase, samples are selected purely according to the MMD
    criterion.

    After warm-up, at every step the views are processed in a uniformly
    random order. The first view selects by pure MMD. Every following view
    selects, among its MMD-tolerance set

        G = {i : m_i <= min_i' m_i' + epsilon},

    the candidate minimizing rank(m_i) + rank(h_i), where

        m_i = a_{v,i}                          (MMD score)
        h_i = sum_{u in P} Delta_{vu}(i, j_u)  (dependence score)

    and P is the set of views already processed at this step, with anchors
    j_u = pi_u(t). Ties share the minimum rank; remaining ties in the rank
    sum are broken by the smaller MMD score.

    The sampler maintains the selected samples as ordered sequences. The
    k-th selections of two views are paired by position. Different views
    may therefore select different global sample indices.

    Args:
        n_sub: Number of samples to select from every view.
        kernel_gammas: RBF kernel gamma parameters. Can be:

            - A scalar shared by all views.
            - A list or tuple with one gamma per view.
            - A NumPy array with one gamma per view.
            - A dictionary mapping view names to gamma values.

        warmup_steps: Number of initial selections (including the random
            first one) performed using pure MMD herding. Can be an integer
            with 1 <= warmup_steps <= n_sub, a float in (0, 1] interpreted
            as a fraction of n_sub, or None (equivalent to 1: only the
            random first selection precedes joint optimization).
        epsilon: MMD tolerance in [0, inf]. ``np.inf`` gives the plain
            rank-sum rule, ``0`` gives pure MMD herding.
        random_state: Controls random view ordering and initial sample
            selection.

    Attributes:
        traces_: After fitting, dict of dicts with the final pairwise
            cross-view traces T_{vu}(n_sub). HSIC_{vu} = T_{vu} / n_sub**2.

    Example:
        sampler = MultiViewKernelHerding(
            n_sub=100,
            kernel_gammas={
                "view_0": 0.1,
                "view_1": 0.05,
                "view_2": 0.01,
            },
            warmup_steps=10,
            epsilon=np.inf,
            random_state=42,
        )

        result = sampler.fit_transform(X_views, y)

        indices_view_0 = sampler.selected_indices_["view_0"]
        indices_view_1 = sampler.selected_indices_["view_1"]
    """

    def __init__(
        self,
        n_sub,
        kernel_gammas,
        warmup_steps=0.25,
        epsilon=np.inf,
        random_state=None,
    ):
        """Initializes the joint multiview herding sampler.

        Args:
            n_sub: Number of samples to select from every view.
            kernel_gammas: RBF kernel gamma parameters.
            warmup_steps: Number of selections performed using pure MMD
                herding before joint MMD-HSIC optimization.
            epsilon: MMD tolerance in [0, inf].
            random_state: Controls reproducibility.

        Raises:
            TypeError: If `n_sub` or `warmup_steps` has an invalid type.
            ValueError: If `n_sub`, `warmup_steps` or `epsilon` is invalid.
        """
        super().__init__(random_state=random_state)

        if not isinstance(n_sub, (int, np.integer)):
            raise TypeError("n_sub must be an integer.")

        if n_sub <= 0:
            raise ValueError("n_sub must be greater than zero.")

        if warmup_steps is None:
            warmup_steps = 1
        elif isinstance(warmup_steps, (float, np.floating)):
            if not 0.0 < warmup_steps <= 1.0:
                raise ValueError(
                    "A float warmup_steps must be in the interval (0, 1]."
                )
            warmup_steps = int(np.floor(warmup_steps * n_sub))
        elif isinstance(warmup_steps, (int, np.integer)):
            if warmup_steps <= 0:
                raise ValueError(
                    "warmup_steps must be greater than zero."
                )
        else:
            raise TypeError(
                "warmup_steps must be None, an integer, or a float in (0, 1]."
            )

        # A fractional warm-up always results in at least one iteration
        # (the random first selection).
        warmup_steps = int(max(1, warmup_steps))

        if warmup_steps > n_sub:
            raise ValueError(
                "warmup_steps cannot be greater than n_sub."
            )

        if epsilon is None or np.isnan(epsilon) or epsilon < 0:
            raise ValueError(
                "epsilon must be in [0, inf]."
            )

        self.n_sub = int(n_sub)
        self.kernel_gammas = kernel_gammas
        self.warmup_steps = warmup_steps
        self.epsilon = float(epsilon)

    def _select_indices(self, views, y):
        """Selects samples jointly across all feature views.

        Args:
            views: Dictionary mapping view names to feature matrices.
            y: One-dimensional array containing the training labels.

        Returns:
            dict[str, np.ndarray]: Selected global sample indices for
                every feature view, as ordered sequences.

        Raises:
            ValueError: If `n_sub` exceeds the number of training samples
                or no view is given.
        """
        n_samples = len(y)
        view_names = list(views.keys())
        n_views = len(view_names)

        if n_views == 0:
            raise ValueError(
                "At least one feature view is required."
            )

        if self.n_sub > n_samples:
            raise ValueError(
                f"n_sub={self.n_sub} cannot be greater than the "
                f"number of training samples ({n_samples})."
            )

        gammas = self._resolve_kernel_gammas(views)

        # ------------------------------------------------------------------
        # Precompute centered Gram matrices (and their diagonals).
        # ------------------------------------------------------------------
        centered_kernels = {
            view_name: self._compute_centered_gram_matrix(
                X_view,
                gamma=gammas[view_name],
            )
            for view_name, X_view in views.items()
        }

        diagonals = {
            view_name: np.diag(K).copy()
            for view_name, K in centered_kernels.items()
        }

        # ------------------------------------------------------------------
        # State per view:
        #   accumulators[v] : a_v^{(t)} = sum_k K~_v[:, pi_v(k)]
        #   selected[v]     : ordered sequence pi_v
        #   candidate_pools : U_v(t)
        # State per pair:
        #   traces[v][u]    : T_{vu}(t)  (monitoring only; constant in the
        #                     candidate, so not needed for selection)
        # ------------------------------------------------------------------
        accumulators = {
            view_name: np.zeros(n_samples, dtype=float)
            for view_name in view_names
        }

        selected = {view_name: [] for view_name in view_names}

        candidate_pools = {
            view_name: np.arange(n_samples, dtype=int)
            for view_name in view_names
        }

        traces = {
            view_u: {
                view_v: 0.0
                for view_v in view_names
                if view_u != view_v
            }
            for view_u in view_names
        }

        # ------------------------------------------------------------------
        # Step t = 1: independent random first selection in every view.
        # ------------------------------------------------------------------
        for view_name in view_names:
            first_idx = int(self.rng_.choice(n_samples))

            self._update_sampling_state(
                view_name=view_name,
                selected_index=first_idx,
                selected=selected,
                candidate_pools=candidate_pools,
                accumulators=accumulators,
                centered_kernels=centered_kernels,
            )

        # T_{vu}(1) = K~_v[i,i] * K~_u[j,j]
        for u_idx in range(n_views):
            for v_idx in range(u_idx + 1, n_views):
                view_u = view_names[u_idx]
                view_v = view_names[v_idx]

                i = selected[view_u][0]
                j = selected[view_v][0]

                value = diagonals[view_u][i] * diagonals[view_v][j]

                traces[view_u][view_v] = value
                traces[view_v][view_u] = value

        # ------------------------------------------------------------------
        # Warm-up: pure MMD herding, t = 2, ..., warmup_steps.
        # ------------------------------------------------------------------
        for t in range(2, self.warmup_steps + 1):
            for view_name in view_names:
                idx = self._select_mmd_sample(
                    accumulators=accumulators[view_name],
                    candidate_pool=candidate_pools[view_name],
                )

                self._update_sampling_state(
                    view_name=view_name,
                    selected_index=idx,
                    selected=selected,
                    candidate_pools=candidate_pools,
                    accumulators=accumulators,
                    centered_kernels=centered_kernels,
                )

            self._update_traces(
                selected=selected,
                centered_kernels=centered_kernels,
                diagonals=diagonals,
                traces=traces,
                t=t,
                view_names=view_names,
            )

        # ------------------------------------------------------------------
        # Joint rank-sum phase, t = warmup_steps + 1, ..., n_sub.
        # ------------------------------------------------------------------
        for t in range(self.warmup_steps + 1, self.n_sub + 1):
            order = [
                view_names[k]
                for k in self.rng_.permutation(n_views)
            ]

            # processed = [(view, anchor_index j_u = pi_u(t)), ...]
            processed = []

            for position, current_view in enumerate(order):
                if position == 0:
                    # First view: pure MMD.
                    idx = self._select_mmd_sample(
                        accumulators=accumulators[current_view],
                        candidate_pool=candidate_pools[current_view],
                    )
                else:
                    idx = self._select_joint_sample(
                        current_view=current_view,
                        processed=processed,
                        selected=selected,
                        accumulators=accumulators,
                        candidate_pools=candidate_pools,
                        centered_kernels=centered_kernels,
                        diagonals=diagonals,
                        t=t,
                    )

                self._update_sampling_state(
                    view_name=current_view,
                    selected_index=idx,
                    selected=selected,
                    candidate_pools=candidate_pools,
                    accumulators=accumulators,
                    centered_kernels=centered_kernels,
                )

                processed.append((current_view, idx))

            self._update_traces(
                selected=selected,
                centered_kernels=centered_kernels,
                diagonals=diagonals,
                traces=traces,
                t=t,
                view_names=view_names,
            )

        self.traces_ = traces

        return {
            view_name: np.asarray(selected[view_name], dtype=int)
            for view_name in view_names
        }

    def _select_mmd_sample(
        self,
        accumulators,
        candidate_pool,
    ):
        """Selects a candidate using the MMD criterion.

        The centered-kernel formulation selects i* = argmin_i a_i over the
        currently available candidates.

        Args:
            accumulators: MMD accumulator vector of the view.
            candidate_pool: Currently available global indices.

        Returns:
            int: Selected global sample index.
        """
        if len(candidate_pool) == 0:
            raise RuntimeError("The candidate pool is empty.")

        candidate_scores = accumulators[candidate_pool]

        return int(candidate_pool[int(np.argmin(candidate_scores))])

    def _select_joint_sample(
        self,
        current_view,
        processed,
        selected,
        accumulators,
        candidate_pools,
        centered_kernels,
        diagonals,
        t,
    ):
        """Selects a sample using the MMD-HSIC rank-sum criterion.

        For every candidate i in the MMD-tolerance set G of the current
        view, with processed views P and anchors j_u = pi_u(t):

            m_i = a_{v,i}^{(t-1)}
            h_i = sum_{u in P} [ 2 * sum_{k<t} K~_v[i, pi_v(k)]
                                           * K~_u[j_u, pi_u(k)]
                                 + K~_v[i,i] * K~_u[j_u,j_u] ]

        and the selected index minimizes rank(m_i) + rank(h_i) over G,
        with ties sharing the minimum rank and remaining ties broken by
        the smaller m_i.

        Args:
            current_view: View currently being optimized.
            processed: List of (view, anchor_index) already processed at
                step t.
            selected: Selected index sequences for all views.
            accumulators: MMD accumulator vectors.
            candidate_pools: Candidate pools for all views.
            centered_kernels: Centered Gram matrices for all views.
            diagonals: Diagonals of the centered Gram matrices.
            t: Current step (1-based); the new selection is pi_v(t).

        Returns:
            int: Selected global sample index.
        """
        pool = candidate_pools[current_view]

        if len(pool) == 0:
            raise RuntimeError(
                f"Candidate pool for view '{current_view}' is empty."
            )

        K_v = centered_kernels[current_view]
        pi_v_prev = np.asarray(
            selected[current_view][: t - 1],
            dtype=int,
        )

        # MMD scores.
        m = accumulators[current_view][pool]

        # Dependence scores: sum over processed views of Delta_{vu}.
        K_v_pool = K_v[np.ix_(pool, pi_v_prev)]  # (|pool|, t-1)
        diag_v_pool = diagonals[current_view][pool]

        h = np.zeros(len(pool), dtype=float)

        for other_view, anchor in processed:
            K_u = centered_kernels[other_view]
            pi_u_prev = np.asarray(
                selected[other_view][: t - 1],
                dtype=int,
            )

            k_u = K_u[anchor, pi_u_prev]  # (t-1,)

            h += 2.0 * (K_v_pool @ k_u)
            h += diag_v_pool * diagonals[other_view][anchor]

        # MMD-tolerance set G.
        in_g = m <= m.min() + self.epsilon
        g_positions = np.flatnonzero(in_g)

        m_g = m[g_positions]
        h_g = h[g_positions]

        rank_sum = self._rank_ascending(m_g) + self._rank_ascending(h_g)

        # Minimize rank sum; ties broken by smaller m (then by position).
        best = int(np.lexsort((m_g, rank_sum))[0])

        return int(pool[g_positions[best]])

    def _update_sampling_state(
        self,
        view_name,
        selected_index,
        selected,
        candidate_pools,
        accumulators,
        centered_kernels,
    ):
        """Updates a view after selecting one sample.

        The selected sample is appended to the sequence, removed from the
        candidate pool, and its centered-kernel column is added to the MMD
        accumulator: a_v <- a_v + K~_v[:, i*].

        Args:
            view_name: Name of the view being updated.
            selected_index: Global index selected for the view.
            selected: Selected index sequences.
            candidate_pools: Candidate pools for all views.
            accumulators: MMD accumulators for all views.
            centered_kernels: Centered Gram matrices for all views.
        """
        selected_index = int(selected_index)

        selected[view_name].append(selected_index)

        pool = candidate_pools[view_name]
        candidate_pools[view_name] = pool[pool != selected_index]

        accumulators[view_name] += (
            centered_kernels[view_name][:, selected_index]
        )

    def _update_traces(
        self,
        selected,
        centered_kernels,
        diagonals,
        traces,
        t,
        view_names,
    ):
        """Updates pairwise cross-view traces after step t.

        With i = pi_v(t) and j = pi_u(t):

            T_{vu}(t) = T_{vu}(t-1) + Delta_{vu}(i, j; t)
            Delta = 2 * sum_{k<t} K~_v[i, pi_v(k)] * K~_u[j, pi_u(k)]
                    + K~_v[i,i] * K~_u[j,j]

        This must be called after all views have selected their t-th
        sample, so that every sequence has length t.

        Args:
            selected: Selected index sequences for every view.
            centered_kernels: Centered Gram matrices.
            diagonals: Diagonals of the centered Gram matrices.
            traces: Pairwise trace accumulators (updated in place).
            t: Current step (1-based).
            view_names: Ordered list of view names.
        """
        if t <= 1:
            return

        n_views = len(view_names)

        for v_idx in range(n_views):
            for u_idx in range(v_idx + 1, n_views):
                view_v = view_names[v_idx]
                view_u = view_names[u_idx]

                i = selected[view_v][t - 1]
                j = selected[view_u][t - 1]

                pi_v_prev = np.asarray(
                    selected[view_v][: t - 1],
                    dtype=int,
                )
                pi_u_prev = np.asarray(
                    selected[view_u][: t - 1],
                    dtype=int,
                )

                k_v = centered_kernels[view_v][i, pi_v_prev]
                k_u = centered_kernels[view_u][j, pi_u_prev]

                delta = (
                    2.0 * float(k_v @ k_u)
                    + diagonals[view_v][i] * diagonals[view_u][j]
                )

                traces[view_v][view_u] += delta
                traces[view_u][view_v] = traces[view_v][view_u]

    @staticmethod
    def _rank_ascending(values):
        """Computes ascending ranks with ties sharing the minimum rank.

        The smallest value receives rank zero. Equal values receive the
        same (minimum) rank, i.e. rk(x_i) = |{i' : x_i' < x_i}|.

        Args:
            values: One-dimensional array of values to rank.

        Returns:
            np.ndarray: Integer ascending ranks.
        """
        return (
            rankdata(values, method="min").astype(int) - 1
        )

    def _resolve_kernel_gammas(self, views):
        """Resolves one RBF gamma value for every feature view.

        Args:
            views: Dictionary mapping view names to feature matrices.

        Returns:
            dict[str, float]: Mapping from view names to gamma values.

        Raises:
            ValueError: If a view is missing a gamma value or the number
                of gamma values does not match the number of views.
        """
        view_names = list(views.keys())

        if isinstance(self.kernel_gammas, dict):
            missing = [
                name
                for name in view_names
                if name not in self.kernel_gammas
            ]

            if missing:
                raise ValueError(
                    "Missing kernel gamma values for views: "
                    f"{missing}"
                )

            gammas = {
                name: float(self.kernel_gammas[name])
                for name in view_names
            }

        elif isinstance(self.kernel_gammas, (list, tuple, np.ndarray)):
            if len(self.kernel_gammas) != len(view_names):
                raise ValueError(
                    "The number of kernel gamma values must "
                    "match the number of views."
                )

            gammas = {
                name: float(gamma)
                for name, gamma in zip(view_names, self.kernel_gammas)
            }

        else:
            gammas = {
                name: float(self.kernel_gammas)
                for name in view_names
            }

        for view_name, gamma in gammas.items():
            if not np.isfinite(gamma):
                raise ValueError(
                    f"RBF gamma for '{view_name}' must be finite."
                )

            if gamma <= 0:
                raise ValueError(
                    f"RBF gamma for '{view_name}' must be positive, "
                    f"got {gamma}."
                )

        return gammas

    @staticmethod
    def _compute_centered_gram_matrix(
        X,
        gamma,
    ):
        """Computes a centered RBF Gram matrix.

        K~ = H K H with H = I - (1/n) 11^T, evaluated element-wise as
        K - row_mean - column_mean + grand_mean to avoid building H.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            gamma: RBF kernel gamma parameter.

        Returns:
            np.ndarray: Centered RBF Gram matrix.
        """
        K = rbf_kernel(X, gamma=gamma)

        row_mean = np.mean(K, axis=1, keepdims=True)
        column_mean = np.mean(K, axis=0, keepdims=True)
        grand_mean = np.mean(K)

        return K - row_mean - column_mean + grand_mean