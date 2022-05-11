from typing import List, Optional, Union, Callable, Dict, Any, Sequence

import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection._split import BaseCrossValidator, BaseShuffleSplit, KFold

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

from poniard.core import PoniardBaseEstimator


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
        Metrics to compute for each estimator.
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
        Features with unique values above a certain threshold will be treated as numeric. If
        float, the threshold is `numeric_threshold * samples`.
    cardinality_threshold :
        Non-numeric features with cardinality above a certain threshold will be treated as
        ordinal encoded instead of one-hot encoded. If float, the threshold is
        `cardinality_threshold * samples`.
    cv :
        Cross validation strategy. Either an integer, a scikit-learn cross validation object,
        or an iterable.
    verbose :
        Verbosity level. Propagated to every scikit-learn function and estiamtor.
    random_state :
        RNG. Propagated to every scikit-learn function and estiamtor.
    n_jobs :
        Controls parallel processing. -1 uses all cores. Propagated to every scikit-learn
        function and estimator.

    Attributes
    ----------
    estimators_ :
        Estimators used for scoring.
    preprocessor_ :
        Pipeline that preprocesses the data.
    metrics_ :
        Metrics used for scoring estimators during fit and hyperparameter optimization.
    """

    def __init__(
        self,
        estimators: Optional[
            Union[Dict[str, RegressorMixin], List[RegressorMixin]]
        ] = None,
        metrics: Optional[Union[Dict[str, Callable], List[str], Callable]] = None,
        preprocess: bool = True,
        scaler: Optional[Union[str, TransformerMixin]] = None,
        numeric_imputer: Optional[Union[str, TransformerMixin]] = None,
        custom_preprocessor: Union[None, Pipeline, TransformerMixin] = None,
        numeric_threshold: Union[int, float] = 0.2,
        cardinality_threshold: Union[int, float] = 50,
        cv: Union[int, BaseCrossValidator, BaseShuffleSplit, Sequence] = None,
        verbose: int = 0,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
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
            cv=cv or KFold(n_splits=5, shuffle=True, random_state=random_state),
            verbose=verbose,
            random_state=random_state,
            n_jobs=n_jobs,
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

    def _build_metrics(
        self, y: Union[pd.DataFrame, np.ndarray]
    ) -> Union[Dict[str, Callable], List[str], Callable]:
        return [
            "neg_mean_squared_error",
            "neg_mean_absolute_percentage_error",
            "neg_median_absolute_error",
            "r2",
        ]
