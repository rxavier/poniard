from __future__ import annotations

__all__ = ['PoniardClassifier']

from collections.abc import Callable
from typing import Sequence

from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection._split import BaseCrossValidator, BaseShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .core import PoniardBaseEstimator


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
        estimators: dict[str, ClassifierMixin] | Sequence[ClassifierMixin] | None = None,
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
    def _default_estimators(self) -> list[ClassifierMixin]:
        return [
            LogisticRegression(
                random_state=self.random_state, verbose=self.verbose, max_iter=5000
            ),
            GaussianNB(),
            KNeighborsClassifier(),
            DecisionTreeClassifier(random_state=self.random_state),
            RandomForestClassifier(
                random_state=self.random_state, verbose=self.verbose, n_jobs=self.n_jobs
            ),
            HistGradientBoostingClassifier(
                random_state=self.random_state, verbose=self.verbose
            ),
        ]

    def _build_metrics(self) -> dict[str, Callable] | list[str]:
        if self.target_info["type_"] == "multilabel-indicator":
            return [
                "roc_auc",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
            ]
        elif self.target_info["type_"] == "multiclass":
            return [
                "roc_auc_ovr",
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

    def _build_cv(self) -> BaseCrossValidator:
        cv = self.cv or 5
        if isinstance(cv, int):
            if self.target_info["type_"] in ("binary", "multiclass"):
                return StratifiedKFold(
                    n_splits=cv, shuffle=True, random_state=self.random_state
                )
            else:
                return KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        else:
            self._pass_instance_attrs(cv)
            return cv
