# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/estimators.classification.ipynb.

# %% auto 0
__all__ = ['PoniardClassifier']

# %% ../../nbs/estimators.classification.ipynb 3
from typing import List, Optional, Union, Callable, Dict, Any, Sequence

import numpy as np
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection._split import BaseCrossValidator, BaseShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from xgboost import XGBClassifier

from .core import PoniardBaseEstimator
from ..plot.plot_factory import PoniardPlotFactory

# %% ../../nbs/estimators.classification.ipynb 4
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
    plugins :
        Plugin instances that run in set moments of setup, fit and plotting.
    plot_options :
        :class:poniard.plot.plot_factory.PoniardPlotFactory instance specifying Plotly format
        options or None, which sets the default factory.
    """

    def __init__(
        self,
        estimators: Optional[
            Union[Dict[str, ClassifierMixin], Sequence[ClassifierMixin]]
        ] = None,
        metrics: Optional[Union[str, Dict[str, Callable], Sequence[str]]] = None,
        preprocess: bool = True,
        custom_preprocessor: Union[None, Pipeline, TransformerMixin] = None,
        cv: Union[int, BaseCrossValidator, BaseShuffleSplit, Sequence] = None,
        verbose: int = 0,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        plugins: Optional[Sequence[Any]] = None,
        plot_options: Optional[PoniardPlotFactory] = None,
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
            plugins=plugins,
            plot_options=plot_options,
        )

    @property
    def _default_estimators(self) -> List[ClassifierMixin]:
        return [
            LogisticRegression(
                random_state=self.random_state, verbose=self.verbose, max_iter=5000
            ),
            GaussianNB(),
            SVC(
                kernel="linear",
                probability=True,
                random_state=self.random_state,
                verbose=self.verbose,
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
            ),
        ]

    def _build_metrics(self) -> Union[Dict[str, Callable], List[str], Callable]:
        y = self.y
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
            if (self.y is not None) and (
                self.target_info["type_"] in ("binary", "multiclass")
            ):
                return StratifiedKFold(
                    n_splits=cv, shuffle=True, random_state=self.random_state
                )
            else:
                return KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        else:
            self._pass_instance_attrs(cv)
            return cv
