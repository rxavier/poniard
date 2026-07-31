from __future__ import annotations

from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    RandomizedSearchCV,
)


class TuningMixin:
    """Mixin for hyperparameter tuning methods."""

    def tune_estimator(
        self,
        estimator_name: str,
        X,
        y,
        grid: dict | None = None,
        mode: str = "grid",
        tuned_estimator_name: str | None = None,
        **kwargs,
    ):
        """Hyperparameter tuning for a single estimator.

        Poniard ships no default hyperparameter grids: `grid` must be supplied
        explicitly. The grid is passed through to the search class, so keys must
        follow sklearn's pipeline convention and be prefixed with the estimator
        name (e.g. ``{"LogisticRegression__C": [...]}``).

        Parameters
        ----------
        estimator_name :
            Estimator to tune.
        X :
            Features.
        y :
            Target.
        grid :
            Hyperparameter grid. Required. Keys must be prefixed with the
            estimator name (``<estimator_name>__<param>``).
        mode :
            Type of search. Either "grid", "halving" or "random". Default "grid".
        tuned_estimator_name :
            Estimator name when adding to `pipelines`. Default None.
        kwargs :
            Passed to the search class constructor.

        Returns
        -------
        PoniardBaseEstimator
            Self.
        """
        estimator = clone(self.pipelines[estimator_name])
        if not grid:
            raise ValueError(
                "`grid` must be provided: poniard ships no default hyperparameter grids."
            )
        self._pass_instance_attrs(estimator)

        scoring = self._first_scorer(sklearn_scorer=True)
        if mode == "random":
            search = RandomizedSearchCV(
                estimator,
                grid,
                scoring=scoring,
                cv=self.cv,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                **kwargs,
            )
        elif mode == "halving":
            search = HalvingGridSearchCV(
                estimator,
                grid,
                scoring=scoring,
                cv=self.cv,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                **kwargs,
            )
        else:
            search = GridSearchCV(
                estimator,
                grid,
                scoring=scoring,
                cv=self.cv,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                **kwargs,
            )
        search.fit(X, y)
        tuned_estimator_name = tuned_estimator_name or f"{estimator_name}_tuned"
        self.add_estimators(
            estimators={
                tuned_estimator_name: clone(search.best_estimator_._final_estimator)
            }
        )
        return self
