from typing import List, Optional, Union, Callable, Dict, Any, Sequence

from sklearn.base import RegressorMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit, KFold

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from xgboost import XGBRegressor
from sklearn.dummy import DummyRegressor

from poniard.estimators.core import PoniardBaseEstimator
from poniard.plot.plot_factory import PoniardPlotFactory


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
    scaler :
        Numeric scaler method. Either "standard", "minmax", "robust" or scikit-learn Transformer.
    numeric_imputer :
        Imputation method. Either "simple", "iterative" or scikit-learn Transformer.
    custom_preprocessor :
        Preprocessor used instead of the default preprocessing pipeline. It must be able to be
        included directly in a scikit-learn Pipeline.
    numeric_threshold :
        Number features with unique values above a certain threshold will be treated as numeric. If
        float, the threshold is `numeric_threshold * samples`.
    cardinality_threshold :
        Non-number features with cardinality above a certain threshold will be treated as
        ordinal encoded instead of one-hot encoded. If float, the threshold is
        `cardinality_threshold * samples`.
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
        function and estimator.
    plugins :
        Plugin instances that run in set moments of setup, fit and plotting.
    plot_options :
        :class:poniard.plot.plot_factory.PoniardPlotFactory instance specifying Plotly format
        options or None, which sets the default factory.
    cache_transformations :
        Whether to cache transformations and set the `memory` parameter for Pipelines. This can
        speed up slow transformations as they are not recalculated for each estimator.

    Attributes
    ----------
    estimators_ :
        Estimators used for scoring.
    preprocessor_ :
        Pipeline that preprocesses the data.
    metrics_ :
        Metrics used for scoring estimators during fit and hyperparameter optimization.
    cv_ :
        Cross validation strategy.
    """

    def __init__(
        self,
        estimators: Optional[
            Union[Dict[str, RegressorMixin], Sequence[RegressorMixin]]
        ] = None,
        metrics: Optional[Union[str, Dict[str, Callable], Sequence[str]]] = None,
        preprocess: bool = True,
        scaler: Optional[Union[str, TransformerMixin]] = None,
        numeric_imputer: Optional[Union[str, TransformerMixin]] = None,
        custom_preprocessor: Union[None, Pipeline, TransformerMixin] = None,
        numeric_threshold: Union[int, float] = 0.1,
        cardinality_threshold: Union[int, float] = 20,
        cv: Union[int, BaseCrossValidator, BaseShuffleSplit, Sequence] = None,
        verbose: int = 0,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        plugins: Optional[Sequence[Any]] = None,
        plot_options: Optional[PoniardPlotFactory] = None,
        cache_transformations: bool = False,
    ):
        super().__init__(
            estimators=estimators,
            metrics=metrics,
            preprocess=preprocess,
            scaler=scaler,
            numeric_imputer=numeric_imputer,
            custom_preprocessor=custom_preprocessor,
            numeric_threshold=numeric_threshold,
            cardinality_threshold=cardinality_threshold,
            cv=cv,
            verbose=verbose,
            random_state=random_state,
            n_jobs=n_jobs,
            plugins=plugins,
            plot_options=plot_options,
            cache_transformations=cache_transformations,
        )

    @property
    def _base_estimators(self) -> List[RegressorMixin]:
        return [
            LinearRegression(),
            ElasticNet(random_state=self.random_state),
            LinearSVR(
                verbose=self.verbose, random_state=self.random_state, max_iter=5000
            ),
            KNeighborsRegressor(),
            DecisionTreeRegressor(random_state=self.random_state),
            RandomForestRegressor(
                random_state=self.random_state, verbose=self.verbose, n_jobs=self.n_jobs
            ),
            HistGradientBoostingRegressor(
                random_state=self.random_state, verbose=self.verbose
            ),
            XGBRegressor(random_state=self.random_state),
            DummyRegressor(strategy="mean"),
        ]

    def _build_metrics(self) -> Union[Dict[str, Callable], List[str], Callable]:
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
            self._pass_instance_attrs(cv)
            return cv
