from typing import List, Optional, Union, Iterable, Callable, Dict

import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.model_selection._split import BaseCrossValidator, BaseShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from first_stab.core import MultiEstimatorBase


class MultiClassifier(MultiEstimatorBase):
    def __init__(
        self,
        estimators: Optional[List[ClassifierMixin]] = None,
        metrics: Optional[Union[Dict[str, Callable], List[str], Callable]] = None,
        preprocess: bool = True,
        cv: Union[int, BaseCrossValidator, BaseShuffleSplit, Iterable] = 5,
        verbose: int = 0,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            estimators=estimators,
            metrics=metrics,
            preprocess=preprocess,
            cv=cv,
            verbose=verbose,
            random_state=random_state,
        )

    @property
    def _base_estimators(self) -> List[ClassifierMixin]:
        return [
            LogisticRegression(random_state=self.random_state, verbose=self.verbose),
            GaussianNB(),
            SVC(random_state=self.random_state, verbose=self.verbose),
            KNeighborsClassifier(),
            DecisionTreeClassifier(random_state=self.random_state),
            RandomForestClassifier(
                random_state=self.random_state, verbose=self.verbose
            ),
            AdaBoostClassifier(random_state=self.random_state),
            HistGradientBoostingClassifier(
                random_state=self.random_state, verbose=self.verbose
            ),
            DummyClassifier(),
        ]

    def _build_metrics(self, y: Union[pd.DataFrame, np.ndarray]) -> None:
        if y.ndim > 1 or len(np.unique(y)) > 2:
            self.metrics_ = {
                "accuracy": make_scorer(accuracy_score),
                "precision": make_scorer(precision_score, average="micro"),
                "recall": make_scorer(recall_score, average="micro"),
                "f1": make_scorer(f1_score, average="micro"),
            }
        else:
            self.metrics_ = {
                "accuracy": make_scorer(accuracy_score),
                "precision": make_scorer(precision_score),
                "recall": make_scorer(recall_score),
                "f1": make_scorer(f1_score),
            }
        return
