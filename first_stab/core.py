import warnings
import inspect
import itertools
from typing import List, Optional, Union, Iterable, Callable, Dict, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.base import ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection._split import BaseCrossValidator, BaseShuffleSplit
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.ensemble import (
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.exceptions import UndefinedMetricWarning

from first_stab.utils import cramers_v


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
        scaler: Optional[str] = None,
        imputer: Optional[str] = None,
        numeric_threshold: Union[int, float] = 0.2,
        cardinality_threshold: Union[int, float] = 50,
        cv: Union[int, BaseCrossValidator, BaseShuffleSplit, Iterable] = None,
        verbose: int = 0,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = -1,
    ):
        self.metrics = metrics
        self.preprocess = preprocess
        self.scaler = scaler or "standard"
        self.imputer = imputer or "simple"
        self.numeric_threshold = numeric_threshold
        self.cardinality_threshold = cardinality_threshold
        self.cv = cv or StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        self.verbose = verbose
        self.random_state = random_state
        self.estimators = estimators
        self.n_jobs = n_jobs

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

    def _classify_features(self, X: Union[pd.DataFrame, np.ndarray]):
        numeric = []
        categorical_high = []
        categorical_low = []
        if isinstance(self.cardinality_threshold, int):
            card_threshold = self.cardinality_threshold
        else:
            card_threshold = int(self.cardinality_threshold * X.shape[0])
        if isinstance(self.numeric_threshold, int):
            num_threshold = self.numeric_threshold
        else:
            num_threshold = int(self.numeric_threshold * X.shape[0])
        if isinstance(X, pd.DataFrame):
            numeric.extend(X.select_dtypes(include="float").columns)
            ints = X.select_dtypes(include="int").columns
            if len(ints) > 0:
                warnings.warn(
                    "Integer columns found. If they are not categorical, consider casting to float so no assumptions have to be made about their cardinality.",
                    UserWarning,
                    stacklevel=2,
                )
            for column in ints:
                if X[column].nunique() > num_threshold:
                    numeric.append(column)
                elif X[column].nunique() > card_threshold:
                    categorical_high.append(column)
                else:
                    categorical_low.append(column)
            strings = X.select_dtypes(exclude="number").columns
            for column in strings:
                if X[column].nunique() > card_threshold:
                    categorical_high.append(column)
                else:
                    categorical_low.append(column)
        else:
            if np.issubdtype(X.dtype, float):
                numeric.extend(range(X.shape[1]))
            elif np.issubdtype(X.dtype, int):
                warnings.warn(
                    "Integer columns found. If they are not categorical, consider casting to float so no assumptions have to be made about their cardinality.",
                    UserWarning,
                    stacklevel=2,
                )
                for i in range(X.shape[1]):
                    if np.unique(X[:, i]).shape[0] > num_threshold:
                        numeric.append(i)
                    elif np.unique(X[:, i]).shape[0] > card_threshold:
                        categorical_high.append(i)
                    else:
                        categorical_low.append(i)
            else:
                for i in range(X.shape[1]):
                    if np.unique(X[:, i]).shape[0] > card_threshold:
                        categorical_high.append(i)
                    else:
                        categorical_low.append(i)
        self._assumed_types = {
            "numeric": numeric,
            "categorical_high": categorical_high,
            "categorical_low": categorical_low,
        }
        return numeric, categorical_high, categorical_low

    def _build_preprocessor(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        numeric, categorical_high, categorical_low = self._classify_features(X=X)
        if self.scaler == "standard":
            scaler = StandardScaler()
        else:
            scaler = RobustScaler()
        cat_imputer = SimpleImputer(strategy="most_frequent", verbose=self.verbose)
        if self.imputer == "simple":
            num_imputer = SimpleImputer(strategy="mean", verbose=self.verbose)
        else:
            num_imputer = IterativeImputer(random_state=self.random_state)
        numeric_preprocessor = make_pipeline(num_imputer, scaler)
        cat_low_preprocessor = make_pipeline(
            cat_imputer, OneHotEncoder(drop="first", handle_unknown="ignore")
        )
        cat_high_preprocessor = make_pipeline(
            cat_imputer,
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=99999),
        )
        if isinstance(X, pd.DataFrame):
            preprocessor = make_column_transformer(
                (numeric_preprocessor, numeric),
                (cat_low_preprocessor, categorical_low),
                (cat_high_preprocessor, categorical_high),
                n_jobs=self.n_jobs,
            )
        else:
            if np.issubdtype(X.dtype, float):
                preprocessor = numeric_preprocessor
            elif np.issubdtype(X.dtype, int):
                preprocessor = make_column_transformer(
                    (numeric_preprocessor, numeric),
                    (cat_low_preprocessor, categorical_low),
                    (cat_high_preprocessor, categorical_high),
                    n_jobs=self.n_jobs,
                )
            else:
                preprocessor = make_column_transformer(
                    (cat_low_preprocessor, categorical_low),
                    (cat_high_preprocessor, categorical_high),
                    n_jobs=self.n_jobs,
                )
        self.preprocessor_ = preprocessor
        return

    def _build_metrics(self, y: Union[pd.DataFrame, np.ndarray]) -> None:
        self.metrics_ = {
            "accuracy": make_scorer(accuracy_score),
        }
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

    def _setup_experiments(
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

        self._build_initial_estimators()
        return X, y

    def score_estimators(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        y: Union[pd.DataFrame, np.ndarray, List],
    ) -> None:
        X, y = self._setup_experiments(X, y)

        results = {}
        pbar = tqdm(self.estimators_.items())
        for i, (name, estimator) in enumerate(pbar):
            pbar.set_description(f"{name}")
            if self.preprocess:
                final_estimator = make_pipeline(self.preprocessor_, estimator)
            else:
                final_estimator = estimator
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                warnings.filterwarnings(
                    "ignore", message=".*will be encoded as all zeros"
                )
                result = cross_validate(
                    final_estimator,
                    X,
                    y,
                    scoring=self.metrics_,
                    cv=self.cv,
                    return_train_score=True,
                    return_estimator=True,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs,
                )
            results.update({name: result})
            if i == len(pbar) - 1:
                pbar.set_description("Completed")
        self._experiment_results = results

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
        self, name: str, include_preprocessor: bool = True, fitted: bool = False
    ) -> ClassifierMixin:
        model = self._experiment_results[name]["estimator"][0]
        if not include_preprocessor:
            model = model._final_estimator
        if fitted:
            return model
        else:
            return clone(model)

    # TODO: Add a method to get the best estimator and its report

    # TODO: Add a method to pop an estimator or several

    def build_ensemble(
        self,
        method: str = "stacking",
        estimators: Optional[List[str]] = None,
        top_n: int = 3,
        include_preprocessor: bool = True,
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
        # TODO: Allow adding to estimator dict and cross validating
        if include_preprocessor:
            return make_pipeline(self.preprocessor_, ensemble)
        else:
            return ensemble

    def get_predictions_similarity(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        y: Union[pd.DataFrame, np.ndarray, List],
    ) -> pd.DataFrame:
        X, y = self._setup_experiments(X, y)

        results = {}
        pbar = tqdm(self.estimators_.items())
        for i, (name, estimator) in enumerate(pbar):
            pbar.set_description(f"{name}")
            if self.preprocess:
                final_estimator = make_pipeline(self.preprocessor_, estimator)
            else:
                final_estimator = estimator
            result = cross_val_predict(
                final_estimator,
                X,
                y,
                cv=self.cv,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
            )
            results.update({name: result})
            if i == len(pbar) - 1:
                pbar.set_description("Completed")
        results = pd.DataFrame(results)
        if self.__class__.__name__ == "MultiClassifier":
            estimator_names = [
                x
                for x in self.estimators_
                if x not in ["DummyClassifier", "DummyRegressor"]
            ]
            table = pd.DataFrame(
                data=np.nan, index=estimator_names, columns=estimator_names
            )
            for row, col in itertools.combinations_with_replacement(
                table.index[::-1], 2
            ):
                cramer = cramers_v(results[row], results[col])
                if row == col:
                    table.loc[row, col] = 1
                else:
                    table.loc[row, col] = cramer
                    table.loc[col, row] = cramer
        else:
            table = results.corr()
        return table
