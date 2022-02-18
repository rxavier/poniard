import warnings
import inspect
from typing import List, Optional, Union, Iterable, Callable, Dict, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.base import ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection._split import BaseCrossValidator, BaseShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import (
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
)
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer
from sklearn.exceptions import UndefinedMetricWarning


class MultiEstimatorBase(object):
    def __init__(
        self,
        estimators: Optional[
            Union[
                List[ClassifierMixin],
                Dict[str, ClassifierMixin],
                List[RegressorMixin],
                Dict[str, RegressorMixin],
            ]
        ] = None,
        metrics: Optional[Union[Dict[str, Callable], List[str], Callable]] = None,
        preprocess: bool = True,
        cv: Union[int, BaseCrossValidator, BaseShuffleSplit, Iterable] = 5,
        verbose: int = 0,
        random_state: Optional[int] = None,
    ):
        self.metrics = metrics
        self.preprocess = preprocess
        self.cv = cv
        self.verbose = verbose
        self.random_state = random_state
        self.estimators = estimators

    def _build_initial_estimators(self) -> None:
        try:
            # If the estimators_ dict exists, don't build it again
            self.estimators_
            return
        except AttributeError:
            pass
        if isinstance(self.estimators, dict):
            self.estimators_ = self.estimators
        elif self.estimators:
            if any([inspect.isclass(v) for v in self.estimators]):
                raise ValueError("Pass an instance of an estimator, not the class.")
            self.estimators_ = {
                estimator.__class__.__name__: estimator for estimator in self.estimators
            }
        else:
            self.estimators_ = {
                estimator.__class__.__name__: estimator
                for estimator in self._base_estimators
            }
        return

    @property
    def _base_estimators(self) -> List[ClassifierMixin]:
        return [
            DummyRegressor(),
            DummyClassifier(),
        ]

    def _build_preprocessor(self, X: Union[pd.DataFrame, np.ndarray, List]) -> None:
        if isinstance(X, pd.DataFrame):
            categorical_columns = make_column_selector(dtype_exclude=np.number)
            numeric_columns = make_column_selector(dtype_include=np.number)
            numeric_preprocessor = make_pipeline(
                SimpleImputer(strategy="mean"), StandardScaler()
            )
            categorical_preprocessor = make_pipeline(
                SimpleImputer(strategy="most_frequent"), OneHotEncoder()
            )
            preprocessor = make_column_transformer(
                (numeric_preprocessor, numeric_columns),
                (categorical_preprocessor, categorical_columns),
            )
        else:
            if X.dtype.kind in "biufc":
                preprocessor = make_pipeline(
                    SimpleImputer(strategy="mean"), StandardScaler()
                )
            else:
                preprocessor = make_pipeline(
                    SimpleImputer(strategy="most_frequent"), OneHotEncoder()
                )
        self.preprocessor_ = preprocessor
        return

    def _build_metrics(self, y: Union[pd.DataFrame, np.ndarray]) -> None:
        self.metrics_ = {
            "accuracy": make_scorer(accuracy_score, greater_is_better=True),
        }
        return

    def _run_experiments(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray],
    ) -> None:
        self._build_initial_estimators()
        results = {}
        pbar = tqdm(self.estimators_.items())
        for i, (name, estimator) in enumerate(pbar):
            pbar.set_description(f"{name}")
            if self.preprocess:
                final_estimator = make_pipeline(self.preprocessor_, estimator)
            else:
                final_estimator = estimator
            if name == "DummyClassifier":
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                    result = cross_validate(
                        final_estimator,
                        X,
                        y,
                        scoring=self.metrics_,
                        cv=self.cv,
                        return_train_score=True,
                        return_estimator=True,
                        verbose=self.verbose,
                    )
            else:
                result = cross_validate(
                    final_estimator,
                    X,
                    y,
                    scoring=self.metrics_,
                    cv=self.cv,
                    return_train_score=True,
                    return_estimator=True,
                    verbose=self.verbose,
                )
            results.update({name: result})
            if i == len(pbar) - 1:
                pbar.set_description("Completed")
        self._experiment_results = results
        return

    def _process_experiment_results(self) -> None:
        results = pd.DataFrame(self._experiment_results).T.drop(["estimator"], axis=1)
        means = results.apply(lambda x: np.mean(x.values.tolist(), axis=1))
        stds = results.apply(lambda x: np.std(x.values.tolist(), axis=1))
        means = means[list(means.columns[2:]) + ["fit_time", "score_time"]]
        stds = stds[list(stds.columns[2:]) + ["fit_time", "score_time"]]
        self._means = means.sort_values(means.columns[0], ascending=False)
        self._stds = stds.reindex(self._means.index)
        return

    def run_experiments(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        y: Union[pd.DataFrame, np.ndarray, List],
    ) -> None:
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            X = np.array(X)
        if not isinstance(y, (pd.DataFrame, np.ndarray)):
            y = np.array(y)

        if not self.metrics:
            self._build_metrics(y)
        else:
            self.metrics_ = self.metrics

        if self.preprocess:
            self._build_preprocessor(X)

        self._run_experiments(X, y)
        self._process_experiment_results()
        return

    def add_estimators(
        self, new_estimators: Union[Dict[str, ClassifierMixin], List[ClassifierMixin]]
    ) -> None:
        if not isinstance(new_estimators, dict):
            new_estimators = {
                estimator.__class__.__name__: estimator for estimator in new_estimators
            }
        if any([inspect.isclass(v) for v in new_estimators.values()]):
            raise ValueError("Pass an instance of an estimator, not the class.")
        self._build_initial_estimators()
        self.estimators_.update(new_estimators)
        return

    def show_results(
        self,
        std: bool = False,
        wrt_dummy: bool = False,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        means = self._means
        stds = self._stds
        if wrt_dummy:
            dummy_means = means.loc[means.index.str.contains("Dummy")]
            dummy_stds = stds.loc[stds.index.str.contains("Dummy")]
            means = means / dummy_means.squeeze()
            stds = stds / dummy_stds.squeeze()
        if std is True:
            return means, stds
        else:
            return means

    def get_estimator(
        self, name: str, with_pipeline: bool = True, fitted: bool = False
    ) -> ClassifierMixin:
        model = self._experiment_results[name]["estimator"][0]
        if not with_pipeline:
            model = model._final_estimator
        if fitted:
            return model
        else:
            return clone(model)

    def get_preprocessor(self):
        return self.preprocessor_

    def build_ensemble(
        self,
        method: str = "stacking",
        estimators: Optional[List[str]] = None,
        top_n: int = 3,
        **kwargs,
    ) -> Union[ClassifierMixin, RegressorMixin]:
        if method not in ["voting", "stacking"]:
            raise ValueError("Method must be either voting or stacking.")
        if estimators:
            models = [
                (name, self._experiment_results[name]["estimator"][0]._final_estimator)
                for name in estimators
            ]
        else:
            models = [
                (name, self._experiment_results[name]["estimator"][0]._final_estimator)
                for name in self._means.index[:top_n]
            ]
        if method == "voting":
            if self.__class__.__name__ == "MultiClassifier":
                ensemble = VotingClassifier(
                    estimators=models, verbose=self.verbose, **kwargs
                )
            else:
                ensemble = VotingRegressor(
                    estimators=models, verbose=self.verbose, **kwargs
                )
        else:
            if self.__class__.__name__ == "MultiClassifier":
                ensemble = StackingClassifier(
                    estimators=models, verbose=self.verbose, cv=self.cv, **kwargs
                )
            else:
                ensemble = StackingRegressor(
                    estimators=models, verbose=self.verbose, cv=self.cv, **kwargs
                )
        return ensemble
