from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class BaseMultiViewSubsampler(
    BaseEstimator,
    TransformerMixin,
    ABC,
):
    """Base class for scikit-learn-compatible multi-view subsampling.

    This class provides the common interface and bookkeeping required for
    subsampling methods operating on multiple feature views.

    A multi-view dataset is represented as either:

        1. A list or tuple of feature matrices::

            X_views = [X_view_1, X_view_2, ..., X_view_V]

        2. A dictionary mapping view names to feature matrices::

            X_views = {
                "view_1": X_view_1,
                "view_2": X_view_2,
                ...
            }

    All views must contain the same number of samples and therefore share
    the same sample-level indexing.

    Subclasses only need to implement the `_select_indices` method.
    The parent class handles validation, fitting, transformation, and
    storage of the selected samples and labels.

    Attributes:
        n_views_: Number of feature views.
        view_names_: Names associated with the feature views.
        n_samples_in_: Number of training samples.
        n_features_in_: Number of features in each view.
        selected_indices_: Dictionary containing the selected global
            sample indices for each view.
        selected_X_: Dictionary containing the selected samples for
            each view.
        selected_y_: Dictionary containing the class labels associated
            with the selected samples for each view.
    """

    def __init__(self, random_state=None):
        """Initializes the multi-view subsampler.

        Args:
            random_state: Controls the randomness of the subsampling
                procedure. Can be an integer, None, or any value accepted
                by NumPy's random number generator.
        """
        self.random_state = random_state

    def fit(self, X_views, y):
        """Fits the subsampler to a multi-view training dataset.

        The actual selection strategy is delegated to `_select_indices`,
        which must be implemented by subclasses.

        Args:
            X_views: Multi-view feature matrices. Can be a list, tuple,
                or dictionary of arrays. Every view must contain the same
                number of samples.
            y: One-dimensional array-like object containing the class label
                for every sample.

        Returns:
            BaseMultiViewSubsampler: The fitted subsampler.
        """
        # Normalize the input representation so that internally all
        # operations use a dictionary of view_name -> feature matrix.
        self._validate_input(X_views, y)

        # Store the training data metadata required by scikit-learn and
        # useful for later transformations.
        self.n_samples_in_ = len(y)
        self.n_views_ = len(self.views_)
        self.classes_ = np.unique(self.y_)

        self.n_features_in_ = {
            view_name: X.shape[1]
            for view_name, X in self.views_.items()
        }

        # Create a reproducible random number generator for subclasses.
        self.rng_ = np.random.default_rng(self.random_state)

        # Delegate the actual subsampling algorithm to the subclass.
        selected_indices = self._select_indices(
            self.views_,
            self.y_,
        )

        # Validate the result returned by the subclass.
        self._validate_selected_indices(selected_indices)

        # Store selected indices as NumPy arrays.
        self.selected_indices_ = {
            view_name: np.asarray(
                selected_indices[view_name],
                dtype=int,
            )
            for view_name in self.view_names_
        }

        # Materialize the selected samples and their corresponding labels.
        self.selected_X_ = {
            view_name: self.views_[view_name][
                self.selected_indices_[view_name]
            ]
            for view_name in self.view_names_
        }

        self.selected_y_ = {
            view_name: self.y_[
                self.selected_indices_[view_name]
            ]
            for view_name in self.view_names_
        }

        return self

    def transform(self, X_views):
        """Returns the selected samples from a multi-view dataset.

        For a fitted subsampler, the stored sample indices refer to the
        original training samples. Therefore, `transform` is intended
        primarily to retrieve the selected training samples from the same
        dataset used during `fit`.

        Args:
            X_views: Multi-view feature matrices having the same sample
                ordering as the data supplied to `fit`.

        Returns:
            dict: A dictionary containing the selected data for every view.
                Each entry has the form::

                    {
                        "X": selected_samples,
                        "y": selected_labels,
                        "indices": selected_global_indices,
                    }

        Raises:
            ValueError: If the number of samples differs from the data
                used during fitting.
        """
        check_is_fitted(
            self,
            [
                "selected_indices_",
                "n_samples_in_",
                "view_names_",
            ],
        )

        views = self._normalize_views(X_views)

        if len(views) != self.n_views_:
            raise ValueError(
                "The number of views supplied to transform() does not "
                "match the number of views used during fit()."
            )

        for view_name in self.view_names_:
            if view_name not in views:
                raise ValueError(
                    f"Missing view '{view_name}' in transform()."
                )

            if views[view_name].shape[0] != self.n_samples_in_:
                raise ValueError(
                    f"View '{view_name}' contains "
                    f"{views[view_name].shape[0]} samples, but the "
                    f"fitted data contains {self.n_samples_in_} samples."
                )

        # Retrieve the same global training samples selected during fit.
        selected = {}

        for view_name in self.view_names_:
            indices = self.selected_indices_[view_name]

            selected[view_name] = {
                "X": views[view_name][indices],
                "y": self.y_[indices],
                "indices": indices.copy(),
            }

        return selected

    def fit_transform(self, X_views, y=None, **fit_params):
        """Fits the sampler and returns the selected training samples.

        This method follows the scikit-learn TransformerMixin convention.

        Args:
            X_views: Multi-view feature matrices.
            y: Training class labels.
            **fit_params: Additional parameters forwarded to `fit()`.

        Returns:
            dict: Selected samples, labels, and global sample indices
                for every view.
        """
        if y is None:
            raise ValueError(
                "y must be provided for supervised multi-view "
                "subsampling."
            )

        self.fit(X_views, y, **fit_params)

        return {
            view_name: {
                "X": self.selected_X_[view_name],
                "y": self.selected_y_[view_name],
                "indices": self.selected_indices_[view_name].copy(),
            }
            for view_name in self.view_names_
        }

    @abstractmethod
    def _select_indices(self, views, y):
        """Select sample indices for every view.

        This method must be implemented by subclasses.

        The returned indices must be expressed in terms of the original
        training dataset. For example, if sample 125 is selected in view
        "image", the returned value must contain `125`, not the local
        position of that sample inside a class-specific subset.

        Args:
            views: Dictionary mapping view names to feature matrices.
            y: Training class labels.

        Returns:
            dict[str, np.ndarray]: Selected global sample indices for
                every view.

        Example:
            {
                "view_1": np.array([0, 4, 12, 20]),
                "view_2": np.array([1, 4, 8, 20]),
            }
        """
        raise NotImplementedError

    def _validate_input(self, X_views, y):
        """Validates and stores the training multi-view dataset.

        Args:
            X_views: Multi-view feature matrices.
            y: Training labels.

        Raises:
            ValueError: If the views or labels have incompatible shapes.
        """
        views = self._normalize_views(X_views)

        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError(
                "y must be a one-dimensional array of class labels."
            )

        if len(views) == 0:
            raise ValueError(
                "At least one feature view must be provided."
            )

        n_samples = len(y)

        for view_name, X in views.items():
            if X.ndim != 2:
                raise ValueError(
                    f"View '{view_name}' must be a 2-dimensional "
                    f"array, got shape {X.shape}."
                )

            if X.shape[0] != n_samples:
                raise ValueError(
                    f"View '{view_name}' contains {X.shape[0]} samples, "
                    f"but y contains {n_samples} labels."
                )

            if X.shape[1] == 0:
                raise ValueError(
                    f"View '{view_name}' contains no features."
                )

        # Store normalized training data.
        self.views_ = views
        self.y_ = y

    def _normalize_views(self, X_views):
        """Converts supported multi-view input formats to a dictionary.

        Args:
            X_views: List, tuple, or dictionary of feature matrices.

        Returns:
            dict[str, np.ndarray]: Normalized mapping from view names
                to feature matrices.

        Raises:
            TypeError: If the input is not a supported representation.
        """
        if isinstance(X_views, dict):
            views = {
                str(name): np.asarray(X)
                for name, X in X_views.items()
            }

        elif isinstance(X_views, (list, tuple)):
            views = {
                f"view_{i}": np.asarray(X)
                for i, X in enumerate(X_views)
            }

        else:
            raise TypeError(
                "X_views must be a dictionary, list, or tuple of "
                "feature matrices."
            )

        self.view_names_ = list(views.keys())

        return views

    def _validate_selected_indices(self, selected_indices):
        """Validates indices returned by a subsampling strategy.

        Args:
            selected_indices: Dictionary mapping each view name to its
                selected global sample indices.

        Raises:
            ValueError: If indices are invalid or a view is missing.
        """
        if not isinstance(selected_indices, dict):
            raise ValueError(
                "_select_indices() must return a dictionary mapping "
                "view names to arrays of sample indices."
            )

        expected_views = set(self.view_names_)
        returned_views = set(selected_indices.keys())

        if expected_views != returned_views:
            raise ValueError(
                "The views returned by _select_indices() do not match "
                "the input views. "
                f"Expected {sorted(expected_views)}, "
                f"got {sorted(returned_views)}."
            )

        for view_name in self.view_names_:
            indices = np.asarray(
                selected_indices[view_name],
                dtype=int,
            )

            if indices.ndim != 1:
                raise ValueError(
                    f"Selected indices for '{view_name}' must be "
                    "one-dimensional."
                )

            if len(indices) == 0:
                raise ValueError(
                    f"No samples were selected for view '{view_name}'."
                )

            if np.any(indices < 0) or np.any(
                indices >= self.n_samples_in_
            ):
                raise ValueError(
                    f"Selected indices for '{view_name}' contain "
                    "values outside [0, {self.n_samples_in_ - 1}]."
                )

            if len(np.unique(indices)) != len(indices):
                raise ValueError(
                    f"Selected indices for '{view_name}' contain "
                    "duplicate samples."
                )