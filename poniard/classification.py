from typing import List, Optional, Union, Iterable, Callable, Dict

import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection._split import (
    BaseCrossValidator,
    BaseShuffleSplit,
    StratifiedKFold,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier

from poniard.core import PoniardBaseEstimator


class PoniardClassifier(PoniardBaseEstimator):
    def __init__(
        self,
        estimators: Optional[
            Union[Dict[str, ClassifierMixin], List[ClassifierMixin]]
        ] = None,
        metrics: Optional[Union[Dict[str, Callable], List[str], Callable]] = None,
        preprocess: bool = True,
        scaler: Optional[str] = None,
        imputer: Optional[str] = None,
        custom_preprocessor: Union[None, Pipeline, TransformerMixin] = None,
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
            custom_preprocessor=custom_preprocessor,
            cardinality_threshold=cardinality_threshold,
            cv=cv
            or StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            verbose=verbose,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    @property
    def _base_estimators(self) -> List[ClassifierMixin]:
        return [
            LogisticRegression(
                random_state=self.random_state, verbose=self.verbose, max_iter=5000
            ),
            GaussianNB(),
            LinearSVC(
                random_state=self.random_state, verbose=self.verbose, max_iter=5000
            ),
            KNeighborsClassifier(),
            DecisionTreeClassifier(random_state=self.random_state),
            RandomForestClassifier(
                random_state=self.random_state, verbose=self.verbose, n_jobs=self.n_jobs
            ),
            HistGradientBoostingClassifier(
                random_state=self.random_state, verbose=self.verbose
            ),
            XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric="mlogloss",
            ),
            DummyClassifier(strategy="prior"),
        ]

    def _build_metrics(self, y: Union[pd.DataFrame, np.ndarray]) -> None:
        if y.ndim > 1 or len(np.unique(y)) > 2:
            self.metrics_ = [
                #"roc_auc_score",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
            ]
        else:
            self.metrics_ = [
                "roc_auc",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
        return
