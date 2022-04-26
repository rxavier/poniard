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
    """Cross validate multiple classifiers, rank them, fine tune them and ensemble them.

    PoniardClassifier takes a list/dict of scikit-learn estimators and compares their performance
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
            Union[Dict[str, ClassifierMixin], List[ClassifierMixin]]
        ] = None,
        metrics: Optional[Union[Dict[str, Callable], List[str], Callable]] = None,
        preprocess: bool = True,
        scaler: Optional[Union[str, TransformerMixin]] = None,
        numeric_imputer: Optional[Union[str, TransformerMixin]] = None,
        custom_preprocessor: Union[None, Pipeline, TransformerMixin] = None,
        numeric_threshold: Union[int, float] = 0.2,
        cardinality_threshold: Union[int, float] = 50,
        cv: Union[int, BaseCrossValidator, BaseShuffleSplit, Iterable] = None,
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

    def _build_metrics(
        self, y: Union[pd.DataFrame, np.ndarray]
    ) -> Union[Dict[str, Callable], List[str], Callable]:
        if y.ndim > 1 or len(np.unique(y)) > 2:
            return [
                # "roc_auc_score",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
            ]
        else:
            return [
                "roc_auc",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
