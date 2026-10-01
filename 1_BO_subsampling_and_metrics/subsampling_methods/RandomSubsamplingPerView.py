import numpy as np

from .BaseMultiViewSampler import BaseMultiViewSubsampler


class RandomSubsamplingPerView(BaseMultiViewSubsampler):
    """Random subsampling, drawn independently for each view.

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

        # self.rng_ is seeded from random_state in fit(), so this is
        # reproducible (the global numpy `choice` was not).
        return {
            view_name: self.rng_.choice(
                n_samples, size=self.n_sub, replace=False
            )
            for view_name in views
        }