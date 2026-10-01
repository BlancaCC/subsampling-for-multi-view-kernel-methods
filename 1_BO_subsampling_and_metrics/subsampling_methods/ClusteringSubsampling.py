import numpy as np

from .BaseMultiViewSampler import BaseMultiViewSubsampler
from sklearn.cluster import KMeans

# # Cluster the samples into n_sub groups.
#         kmeans = KMeans(
#             n_clusters=n_sub,
#             n_init=3,
#             random_state=42,
#         ).fit(X_v)

#         closest_idx = []

#         # Find the nearest unique training sample to each centroid.
#         for c in kmeans.cluster_centers_:
#             dists = np.linalg.norm(X_v - c, axis=1)

#             for idx in np.argsort(dists):
#                 if idx not in closest_idx:
#                     closest_idx.append(idx)
#                     break

#         selected.append(np.array(closest_idx, dtype=int))
class ClusteringSubsampling(BaseMultiViewSubsampler):
    """Clustering-based subsampling, drawn independently for each view.

    `kernel_gammas`, `warmup_steps` and `epsilon` are accepted only so the
    constructor is interchangeable with MultiViewKernelHerding. They are
    stored (so scikit-learn's get_params/clone work) but not used.
    """

    def __init__(
        self,
        n_sub,
        kernel_gammas=None,
        warmup_steps=None,
        epsilon=None,
        random_state=None,
    ):
        super().__init__(random_state=random_state)

        if not isinstance(n_sub, (int, np.integer)):
            raise TypeError("n_sub must be an integer.")

        if n_sub <= 0:
            raise ValueError("n_sub must be greater than zero.")

        self.n_sub = int(n_sub)
        self.kernel_gammas = kernel_gammas
        self.warmup_steps = warmup_steps
        self.epsilon = epsilon

    def _select_indices(self, views, y):
        """Draws `n_sub` distinct global indices per view.

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
        subsampled_views = {}
        for view_name, view_features in views.items():
            # Cluster the samples into n_sub groups.
            kmeans = KMeans(
                n_clusters=self.n_sub,
                n_init=3,
                random_state=self.random_state,
            ).fit(view_features)

            closest_idx = []

            # Find the nearest unique training sample to each centroid.
            for c in kmeans.cluster_centers_:
                dists = np.linalg.norm(view_features - c, axis=1)

                for idx in np.argsort(dists):
                    if idx not in closest_idx:
                        closest_idx.append(idx)
                        break

            subsampled_views[view_name] = np.array(closest_idx, dtype=int)
        # self.rng_ is seeded from random_state in fit(), so this is
        # reproducible (the global numpy `choice` was not).
        return subsampled_views