from typing import List, Optional, Union, Iterable, Callable, Dict

import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.model_selection._split import BaseCrossValidator, BaseShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from first_stab.core import MultiEstimatorBase


class MultiClassifier(MultiEstimatorBase):
    def __init__(
        self,
        estimators: Optional[
            Union[Dict[str, ClassifierMixin], List[ClassifierMixin]]
        ] = None,
        metrics: Optional[Union[Dict[str, Callable], List[str], Callable]] = None,
        preprocess: bool = True,
        scaler: Optional[str] = None,
        imputer: Optional[str] = None,
        numeric_threshold: Union[int, float] = 0.2,
        cardinality_threshold: Union[int, float] = 50,
        cv: Union[int, BaseCrossValidator, BaseShuffleSplit, Iterable] = 5,
        verbose: int = 0,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            estimators=estimators,
            metrics=metrics,
            preprocess=preprocess,
            scaler=scaler,
            imputer=imputer,
            numeric_threshold=numeric_threshold,
            cardinality_threshold=cardinality_threshold,
            cv=cv,
            verbose=verbose,
            random_state=random_state,
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
                random_state=self.random_state, verbose=self.verbose
            ),
            AdaBoostClassifier(random_state=self.random_state),
            HistGradientBoostingClassifier(
                random_state=self.random_state, verbose=self.verbose
            ),
            XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric="mlogloss",
            ),
            DummyClassifier(),
        ]

    def _build_metrics(self, y: Union[pd.DataFrame, np.ndarray]) -> None:
        if y.ndim > 1 or len(np.unique(y)) > 2:
            self.metrics_ = {
                # "roc_auc": make_scorer(roc_auc_score, average="macro"),
                "accuracy": make_scorer(accuracy_score),
                "precision": make_scorer(precision_score, average="macro"),
                "recall": make_scorer(recall_score, average="macro"),
                "f1": make_scorer(f1_score, average="macro"),
            }
        else:
            self.metrics_ = {
                "roc_auc": make_scorer(roc_auc_score),
                "accuracy": make_scorer(accuracy_score),
                "precision": make_scorer(precision_score),
                "recall": make_scorer(recall_score),
                "f1": make_scorer(f1_score),
            }
        return
