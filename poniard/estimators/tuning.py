from __future__ import annotations

from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    RandomizedSearchCV,
)

from ..utils.hyperparameters import get_grid


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
    ) -> GridSearchCV | RandomizedSearchCV:
        """Hyperparameter tuning for a single estimator.

        Parameters
        ----------
        estimator_name :
            Estimator to tune.
        X :
            Features.
        y :
            Target.
        grid :
            Hyperparameter grid. Default None, which uses the grids available for default
            estimators.
        mode :
            Type of search. Eitherr "grid", "halving" or "random". Default "grid".
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
            try:
                grid = get_grid(estimator_name)
                grid = {f"{estimator_name}__{k}": v for k, v in grid.items()}
            except KeyError:
                raise NotImplementedError(
                    f"Estimator {estimator_name} has no predefined hyperparameter grid, so it has to be supplied."
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
