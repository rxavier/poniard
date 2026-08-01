from __future__ import annotations

__all__ = ["PoniardRegressor"]

from collections.abc import Callable, Sequence

from sklearn.base import RegressorMixin, TransformerMixin, clone
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection._split import BaseCrossValidator, BaseShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

from .core import PoniardBaseEstimator


class PoniardRegressor(PoniardBaseEstimator):
    """Cross validate multiple regressors, rank them, fine tune them and ensemble them.

    PoniardRegressor takes a list/dict of scikit-learn estimators and compares their performance
    on a list/dict of scikit-learn metrics using a predefined scikit-learn cross-validation
    strategy.

    Parameters
    ----------
    estimators :
        Estimators to evaluate.
    metrics :
        Metrics to compute for each estimator. This is more restrictive than sklearn's scoring
        parameter, as it does not allow callable scorers. Single strings are cast to lists
        automatically.
    preprocess : bool, optional
        If True, impute missing values, standard scale numeric data and one-hot or ordinal
        encode categorical data.
    custom_preprocessor :
        Preprocessor used instead of the default preprocessing pipeline. It must be able to be
        included directly in a scikit-learn Pipeline.
    cv :
        Cross validation strategy. Either an integer, a scikit-learn cross validation object,
        or an iterable.
    verbose :
        Verbosity level. Propagated to every scikit-learn function and estimator.
    random_state :
        RNG. Propagated to every scikit-learn function and estimator. The default None sets
        random_state to 0 so that cross_validate results are comparable.
    n_jobs :
        Controls parallel processing. -1 uses all cores. Propagated to every scikit-learn
        function.
    """

    def __init__(
        self,
        estimators: dict[str, RegressorMixin] | Sequence[RegressorMixin] | None = None,
        metrics: str | dict[str, Callable] | Sequence[str] | None = None,
        preprocess: bool = True,
        custom_preprocessor: Pipeline | TransformerMixin | None = None,
        cv: int | BaseCrossValidator | BaseShuffleSplit | Sequence = None,
        verbose: bool = False,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ):
        super().__init__(
            estimators=estimators,
            metrics=metrics,
            preprocess=preprocess,
            custom_preprocessor=custom_preprocessor,
            cv=cv,
            verbose=verbose,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    @property
    def poniard_task(self) -> str:
        """Return the task name."""
        return "regression"

    @property
    def _default_estimators(self) -> list[RegressorMixin]:
        return [
            LinearRegression(),
            ElasticNet(random_state=self.random_state),
            LinearSVR(verbose=self.verbose, random_state=self.random_state, max_iter=5000),
            KNeighborsRegressor(),
            DecisionTreeRegressor(random_state=self.random_state),
            RandomForestRegressor(
                random_state=self.random_state, verbose=self.verbose, n_jobs=self.n_jobs
            ),
            HistGradientBoostingRegressor(random_state=self.random_state, verbose=self.verbose),
        ]

    def _build_metrics(self) -> dict[str, Callable] | list[str]:
        return [
            "neg_mean_squared_error",
            "neg_mean_absolute_percentage_error",
            "neg_median_absolute_error",
            "r2",
        ]

    def _build_cv(self) -> BaseCrossValidator:
        cv = self.cv or 5
        if isinstance(cv, int):
            return KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        else:
            if isinstance(cv, (BaseCrossValidator, BaseShuffleSplit)):
                cv = clone(cv, safe=False)
            self._pass_instance_attrs(cv)
            return cv
