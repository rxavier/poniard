from typing import List, Optional, Union, Iterable, Callable, Dict

import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.model_selection._split import BaseCrossValidator, BaseShuffleSplit, KFold

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
)
from xgboost import XGBRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    make_scorer,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from first_stab.core import MultiEstimatorBase


class MultiRegressor(MultiEstimatorBase):
    def __init__(
        self,
        estimators: Optional[
            Union[Dict[str, RegressorMixin], List[RegressorMixin]]
        ] = None,
        metrics: Optional[Union[Dict[str, Callable], List[str], Callable]] = None,
        preprocess: bool = True,
        scaler: Optional[str] = None,
        imputer: Optional[str] = None,
        numeric_threshold: Union[int, float] = 0.2,
        cardinality_threshold: Union[int, float] = 50,
        cv: Union[int, BaseCrossValidator, BaseShuffleSplit, Iterable] = None,
        verbose: int = 0,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = -1,
    ):
        super().__init__(
            estimators=estimators,
            metrics=metrics,
            preprocess=preprocess,
            scaler=scaler,
            imputer=imputer,
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
            RandomForestRegressor(random_state=self.random_state, verbose=self.verbose,
                                  n_jobs=self.n_jobs),
            AdaBoostRegressor(random_state=self.random_state),
            HistGradientBoostingRegressor(
                random_state=self.random_state, verbose=self.verbose
            ),
            XGBRegressor(random_state=self.random_state),
            DummyRegressor(),
        ]

    def _build_metrics(self, y: Union[pd.DataFrame, np.ndarray]) -> None:
        self.metrics_ = {
            "rmse": make_scorer(
                mean_squared_error, squared=False, greater_is_better=False
            ),
            "mape": make_scorer(
                mean_absolute_percentage_error, greater_is_better=False
            ),
            "median_absolute_error": make_scorer(
                median_absolute_error, greater_is_better=False
            ),
            "r2": make_scorer(r2_score, greater_is_better=True),
        }
        return
