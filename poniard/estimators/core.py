from __future__ import annotations
from multiprocessing.sharedctypes import Value
import warnings
import itertools
import inspect
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Callable, Dict, Tuple, Any, Sequence, Iterable

import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin, clone
from sklearn.model_selection._split import BaseCrossValidator, BaseShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import (
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import (
    cross_validate,
    cross_val_predict,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.impute import SimpleImputer
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils.multiclass import type_of_target

from poniard.preprocessing import DatetimeEncoder, TargetEncoder
from poniard.utils import cramers_v
from poniard.utils import GRID
from poniard.plot import PoniardPlotFactory


class PoniardBaseEstimator(ABC):
    """Base estimator that sets up all the functionality for the classifier and regressor.

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
    scaler :
        Numeric scaler method. Either "standard", "minmax", "robust" or scikit-learn Transformer.
    high_cardinality_encoder :
        Encoder for categorical features with high cardinality. Either "target" or "ordinal",
        or scikit-learn Transformer.
    numeric_imputer :
        Imputation method. Either "simple", "iterative" or scikit-learn Transformer.
    custom_preprocessor :
        Preprocessor used instead of the default preprocessing pipeline. It must be able to be
        included directly in a scikit-learn Pipeline.
    numeric_threshold :
        Number features with unique values above a certain threshold will be treated as numeric. If
        float, the threshold is `numeric_threshold * samples`.
    cardinality_threshold :
        Non-number features with cardinality above a certain threshold will be treated as
        ordinal encoded instead of one-hot encoded. If float, the threshold is
        `cardinality_threshold * samples`.
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
    cache_transformations :
        Whether to cache transformations and set the `memory` parameter for Pipelines. This can speed up slow transformations as they are not recalculated for each estimator.

    Attributes
    ----------
    estimators_ :
        Estimators used for scoring.
    preprocessor_ :
        Pipeline that preprocesses the data.
    metrics_ :
        Metrics used for scoring estimators during fit and hyperparameter optimization.
    cv_ :
        Cross validation strategy.
    """

    def __init__(
        self,
        estimators: Optional[
            Union[
                Sequence[ClassifierMixin],
                Dict[str, ClassifierMixin],
                Sequence[RegressorMixin],
                Dict[str, RegressorMixin],
            ]
        ] = None,
        metrics: Optional[Union[str, Dict[str, Callable], Sequence[str]]] = None,
        preprocess: bool = True,
        scaler: Optional[Union[str, TransformerMixin]] = None,
        high_cardinality_encoder: Optional[Union[str, TransformerMixin]] = None,
        numeric_imputer: Optional[Union[str, TransformerMixin]] = None,
        custom_preprocessor: Union[None, Pipeline, TransformerMixin] = None,
        numeric_threshold: Union[int, float] = 0.1,
        cardinality_threshold: Union[int, float] = 20,
        cv: Union[int, BaseCrossValidator, BaseShuffleSplit, Sequence] = None,
        verbose: int = 0,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        plugins: Optional[Sequence[Any]] = None,
        plot_options: Optional[PoniardPlotFactory] = None,
        cache_transformations: bool = False,
    ):
        # TODO: Ugly check that metrics conforms to expected types. Should improve.
        if metrics and (
            (
                isinstance(metrics, Sequence)
                and not all(isinstance(m, str) for m in metrics)
            )
            or (
                isinstance(metrics, Dict)
                and not all(isinstance(m, str) for m in metrics.keys())
                and not all(isinstance(m, Callable) for m in metrics.values())
            )
        ):
            raise ValueError(
                "metrics can only be a string, a sequence of strings, a dict with "
                "strings as keys and callables as values, or None."
            )
        self.metrics = metrics
        self.preprocess = preprocess
        self.scaler = scaler or "standard"
        self.high_cardinality_encoder = high_cardinality_encoder or "target"
        self.numeric_imputer = numeric_imputer or "simple"
        self.numeric_threshold = numeric_threshold
        self.custom_preprocessor = custom_preprocessor
        self.cardinality_threshold = cardinality_threshold
        self.cv = cv
        self.verbose = verbose
        self.random_state = random_state or 0
        self.estimators = estimators
        self.n_jobs = n_jobs
        self.plugins = (
            plugins if isinstance(plugins, Sequence) or plugins is None else [plugins]
        )
        self.plot_options = plot_options or PoniardPlotFactory()

        self._fitted_estimator_ids = []
        self._build_initial_estimators()
        if self.plugins:
            [setattr(plugin, "_poniard", self) for plugin in self.plugins]
        self.plot = self.plot_options
        self.plot._poniard = self

        if cache_transformations:
            self._memory = joblib.Memory("transformation_cache", verbose=self.verbose)
        else:
            self._memory = None

    def fit(self) -> PoniardBaseEstimator:
        """This is the main Poniard method. It uses scikit-learn's `cross_validate` function to
        score all :attr:`metrics_` for every :attr:`preprocessor_` | :attr:`estimators_`, using
        :attr:`cv` for cross validation.

        After running :meth:`fit`, both :attr:`X` and :attr:`y` will be held as attributes.

        Parameters
        ----------
        X :
            Features.
        y :
            Target.

        Returns
        -------
        PoniardBaseEstimator
            Self.
        """
        if not hasattr(self, "cv_"):
            raise ValueError("`setup` must be called before `fit`.")
        self._run_plugin_methods("on_fit_start")

        results = {}
        filtered_estimators = {
            name: estimator
            for name, estimator in self.estimators_.items()
            if id(estimator) not in self._fitted_estimator_ids
        }
        pbar = tqdm(filtered_estimators.items())
        for i, (name, estimator) in enumerate(pbar):
            pbar.set_description(f"{name}")
            if self.preprocess:
                final_estimator = Pipeline(
                    [("preprocessor", self.preprocessor_), (name, estimator)],
                    memory=self._memory,
                )
            else:
                final_estimator = Pipeline([(name, estimator)])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                warnings.filterwarnings(
                    "ignore", message=".*will be encoded as all zeros"
                )
                result = cross_validate(
                    final_estimator,
                    self.X,
                    self.y,
                    scoring=self.metrics_,
                    cv=self.cv_,
                    return_train_score=True,
                    return_estimator=True,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs,
                )
            results.update({name: result})
            self._fitted_estimator_ids.append(id(estimator))
            if i == len(pbar) - 1:
                pbar.set_description("Completed")
        if hasattr(self, "_experiment_results"):
            self._experiment_results.update(results)
        else:
            self._experiment_results = results

        self._process_results()
        self._process_long_results()
        self._run_plugin_methods("on_fit_end")
        return self

    def _predict(
        self, method: str, estimator_names: Optional[Sequence[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Helper method for predicting targets or target probabilities with cross validation.
        Accepts predict, predict_proba, predict_log_proba or decision_function."""
        if not hasattr(self, "cv_"):
            raise ValueError("`setup` must be called before `predict`.")
        X, y = self.X, self.y
        if not estimator_names:
            estimator_names = [estimator for estimator in self.estimators_.keys()]
        results = {}
        pbar = tqdm(estimator_names)
        for i, name in enumerate(pbar):
            pbar.set_description(f"{name}")
            estimator = self.estimators_[name]
            if self.preprocess:
                final_estimator = Pipeline(
                    [("preprocessor", self.preprocessor_), (name, estimator)],
                    memory=self._memory,
                )
            else:
                final_estimator = estimator
            try:
                result = cross_val_predict(
                    final_estimator,
                    X,
                    y,
                    cv=self.cv_,
                    method=method,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs,
                )
            except AttributeError:
                warnings.warn(
                    f"{name} does not support `{method}` method. Filling with nan.",
                    stacklevel=2,
                )
                result = np.empty(self.y.shape)
                result[:] = np.nan
            results.update({name: result})

            if not hasattr(self, "_experiment_results"):
                self._experiment_results = {}
                self._experiment_results.update({name: {method: result}})
            elif name not in self._experiment_results:
                self._experiment_results.update({name: {method: result}})
            else:
                self._experiment_results[name][method] = result

            if i == len(pbar) - 1:
                pbar.set_description("Completed")
        return results

    def predict(
        self, estimator_names: Optional[Sequence[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Get cross validated target predictions where each sample belongs to a single test set.

        Parameters
        ----------
        estimator_names :
            Estimators to include. If None, predict all estimators.

        Returns
        -------
        Dict
            Dict where keys are estimator names and values are numpy arrays of predictions.
        """
        return self._predict(method="predict", estimator_names=estimator_names)

    def predict_proba(
        self, estimator_names: Optional[Sequence[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Get cross validated target probability predictions where each sample belongs to a
        single test set.

        Returns
        -------
        Dict
            Dict where keys are estimator names and values are numpy arrays of prediction
            probabilities.
        """
        return self._predict(method="predict_proba", estimator_names=estimator_names)

    def decision_function(
        self, estimator_names: Optional[Sequence[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Get cross validated decision function predictions where each sample belongs to a
        single test set.

        Parameters
        ----------
        estimator_names :
            Estimators to include. If None, predict all estimators.

        Returns
        -------
        Dict
            Dict where keys are estimator names and values are numpy arrays of prediction
            probabilities.
        """
        return self._predict(
            method="decision_function", estimator_names=estimator_names
        )

    def predict_all(
        self, estimator_names: Optional[Sequence[str]] = None
    ) -> Tuple[Dict[str, np.ndarray]]:
        """Get cross validated target predictions, probabilities and decision functions
        where each sample belongs to all test sets.

        Parameters
        ----------
        estimator_names :
            Estimators to include. If None, predict all estimators.

        Returns
        -------
        Dict
            Dict where keys are estimator names and values are numpy arrays of prediction
            probabilities.
        """
        return (
            self._predict(method="predict", estimator_names=estimator_names),
            self._predict(method="predict_proba", estimator_names=estimator_names),
            self._predict(method="decision_function", estimator_names=estimator_names),
        )

    @property
    @abstractmethod
    def _base_estimators(self) -> List[ClassifierMixin]:
        return [
            DummyRegressor(),
            DummyClassifier(),
        ]

    def _build_initial_estimators(
        self,
    ) -> Dict[str, Union[ClassifierMixin, RegressorMixin]]:
        """Build :attr:`estimators_` dict where keys are the estimator class names.

        Adds dummy estimators if not included during construction. Does nothing if
        :attr:`estimators_` exists.

        """
        if hasattr(self, "estimators_"):
            return

        if isinstance(self.estimators, dict):
            initial_estimators = self.estimators.copy()
        elif self.estimators:
            initial_estimators = {
                estimator.__class__.__name__: estimator for estimator in self.estimators
            }
        else:
            initial_estimators = {
                estimator.__class__.__name__: estimator
                for estimator in self._base_estimators
            }
        if (
            self._check_estimator_type() == "classifier"
            and "DummyClassifier" not in initial_estimators.keys()
        ):
            initial_estimators.update(
                {"DummyClassifier": DummyClassifier(strategy="prior")}
            )
        elif (
            self._check_estimator_type() == "regressor"
            and "DummyRegressor" not in initial_estimators.keys()
        ):
            initial_estimators.update(
                {"DummyRegressor": DummyRegressor(strategy="mean")}
            )

        for estimator in initial_estimators.values():
            self._pass_instance_attrs(estimator)
        self.estimators_ = initial_estimators
        return

    def setup(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        y: Union[pd.DataFrame, np.ndarray, List],
    ) -> PoniardBaseEstimator:
        """Orchestrator.

        Converts inputs to arrays if necessary, sets :attr:`metrics_`,
        :attr:`preprocessor_`, attr:`cv_` and :attr:`estimators_`.

        Parameters
        ----------
        X :
            Features.
        y :
            Target

        """
        self._run_plugin_methods("on_setup_start")
        if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)):
            X = np.array(X)
        if not isinstance(y, (pd.DataFrame, pd.Series, np.ndarray)):
            y = np.array(y)
        self.X = X
        self.y = y

        if self.metrics:
            self.metrics_ = (
                self.metrics if not isinstance(self.metrics, str) else [self.metrics]
            )
        else:
            self.metrics_ = self._build_metrics()
        print(f"Main metric: {self._first_scorer(sklearn_scorer=False)}")

        if self.preprocess:
            if self.custom_preprocessor:
                self.preprocessor_ = self.custom_preprocessor
            else:
                self.preprocessor_ = self._build_preprocessor()

        self.cv_ = self._build_cv()

        self._run_plugin_methods("on_setup_end")
        return self

    def _infer_dtypes(self) -> Tuple[List[str], List[str], List[str]]:
        """Infer feature types (numeric, low-cardinality categorical or high-cardinality
        categorical).

        Returns
        -------
        List[str], List[str], List[str]
            Three lists with column names or indices.
        """
        X = self.X
        numeric = []
        categorical_high = []
        categorical_low = []
        datetime = []
        if isinstance(self.cardinality_threshold, int):
            self.cardinality_threshold_ = self.cardinality_threshold
        else:
            self.cardinality_threshold_ = int(self.cardinality_threshold * X.shape[0])
        if isinstance(self.numeric_threshold, int):
            self.numeric_threshold_ = self.numeric_threshold
        else:
            self.numeric_threshold_ = int(self.numeric_threshold * X.shape[0])
        print(
            "Minimum unique values to consider a number feature numeric:",
            self.numeric_threshold_,
        )
        print(
            "Minimum unique values to consider a non-number feature high cardinality:",
            self.cardinality_threshold_,
            end="\n\n",
        )
        if isinstance(X, pd.DataFrame):
            datetime = X.select_dtypes(
                include=["datetime64[ns]", "datetimetz"]
            ).columns.tolist()
            numbers = X.select_dtypes(include="number").columns
            for column in numbers:
                if X[column].nunique() > self.numeric_threshold_:
                    numeric.append(column)
                elif X[column].nunique() > self.cardinality_threshold_:
                    categorical_high.append(column)
                else:
                    categorical_low.append(column)
            strings = X.select_dtypes(exclude=["number", "datetime"]).columns
            for column in strings:
                if X[column].nunique() > self.cardinality_threshold_:
                    categorical_high.append(column)
                else:
                    categorical_low.append(column)
        else:
            if np.issubdtype(X.dtype, np.datetime64):
                datetime.extend(range(X.shape[1]))
            if np.issubdtype(X.dtype, np.number):
                for i in range(X.shape[1]):
                    if np.unique(X[:, i]).shape[0] > self.numeric_threshold_:
                        numeric.append(i)
                    elif np.unique(X[:, i]).shape[0] > self.cardinality_threshold_:
                        categorical_high.append(i)
                    else:
                        categorical_low.append(i)
            else:
                for i in range(X.shape[1]):
                    if np.unique(X[:, i]).shape[0] > self.cardinality_threshold_:
                        categorical_high.append(i)
                    else:
                        categorical_low.append(i)
        self._inferred_dtypes = {
            "numeric": numeric,
            "categorical_high": categorical_high,
            "categorical_low": categorical_low,
            "datetime": datetime,
        }
        print(
            "Inferred feature types:",
            pd.DataFrame.from_dict(self._inferred_dtypes, orient="index").T.fillna(""),
            sep="\n",
        )
        return numeric, categorical_high, categorical_low, datetime

    def reassign_types(
        self,
        numeric: Optional[List[Union[str, int]]] = None,
        categorical_high: Optional[List[Union[str, int]]] = None,
        categorical_low: Optional[List[Union[str, int]]] = None,
        datetime: Optional[List[Union[str, int]]] = None,
    ) -> PoniardBaseEstimator:
        """Reassign feature types.

        Parameters
        ----------
        numeric :
            List of column names or indices. Default None.
        categorical_high :
            List of column names or indices. Default None.
        categorical_low :
            List of column names or indices. Default None.
        datetime :
            List of column names or indices. Default None.

        Returns
        -------
        PoniardBaseEstimator
            self.
        """
        assigned_types = {
            "numeric": numeric or [],
            "categorical_high": categorical_high or [],
            "categorical_low": categorical_low or [],
            "datetime": datetime or [],
        }
        self._inferred_dtypes = assigned_types
        print(
            "Assigned feature types:",
            pd.DataFrame.from_dict(self._inferred_dtypes, orient="index").T.fillna(""),
            sep="\n",
        )
        # Don't build the preprocessor if no preprocessing should be done or a
        # custom preprocessor was set.
        if not self.preprocess or self.custom_preprocessor is not None:
            return self
        self.preprocessor_ = self._build_preprocessor(assigned_types=assigned_types)
        # TODO: Clearing the `_fitted_estimator_ids` attr is a hacky way of ensuring that doing
        # [fit -> reassign_types -> fit] actually fits models. Ideally, build the
        # preprocessor + estimator pipeline during setup and save those IDs when calling fit.
        self._fitted_estimator_ids = []
        self._run_plugin_methods("on_setup_end")
        return self

    def _build_preprocessor(
        self, assigned_types: Optional[Dict[str, List[Union[str, int]]]] = None
    ) -> Pipeline:
        """Build default preprocessor.

        The preprocessor imputes missing values, scales numeric features and encodes categorical
        features according to inferred types.

        """
        X = self.X
        if hasattr(self, "preprocessor_") and not assigned_types:
            return self.preprocessor_
        if assigned_types:
            numeric = assigned_types["numeric"]
            categorical_high = assigned_types["categorical_high"]
            categorical_low = assigned_types["categorical_low"]
            datetime = assigned_types["datetime"]
        else:
            numeric, categorical_high, categorical_low, datetime = self._infer_dtypes()

        if isinstance(self.scaler, TransformerMixin):
            scaler = self.scaler
        elif self.scaler == "standard":
            scaler = StandardScaler()
        elif self.scaler == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()

        target_is_multilabel = type_of_target(self.y) in [
            "multilabel-indicator",
            "multiclass-multioutput",
            "continuous-multioutput",
        ]
        if isinstance(self.high_cardinality_encoder, TransformerMixin):
            high_cardinality_encoder = self.high_cardinality_encoder
        elif self.high_cardinality_encoder == "target":
            if target_is_multilabel:
                warnings.warn(
                    "TargetEncoder is not supported for multilabel or multioutput targets. "
                    "Switching to OrdinalEncoder.",
                    stacklevel=2,
                )
                high_cardinality_encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=99999
                )
            else:
                if self._check_estimator_type() == "classifier":
                    task = "classification"
                else:
                    task = "regression"
                high_cardinality_encoder = TargetEncoder(
                    task=task, handle_unknown="ignore"
                )
        else:
            high_cardinality_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=99999
            )

        cat_date_imputer = SimpleImputer(strategy="most_frequent")

        if isinstance(self.numeric_imputer, TransformerMixin):
            num_imputer = self.numeric_imputer
        elif self.numeric_imputer == "iterative":
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer

            num_imputer = IterativeImputer(random_state=self.random_state)
        else:
            num_imputer = SimpleImputer(strategy="mean")

        numeric_preprocessor = Pipeline(
            [("numeric_imputer", num_imputer), ("scaler", scaler)]
        )
        cat_low_preprocessor = Pipeline(
            [
                ("categorical_imputer", cat_date_imputer),
                (
                    "one-hot_encoder",
                    OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
                ),
            ]
        )
        cat_high_preprocessor = Pipeline(
            [
                ("categorical_imputer", cat_date_imputer),
                (
                    "high_cardinality_encoder",
                    high_cardinality_encoder,
                ),
            ],
        )
        datetime_preprocessor = Pipeline(
            [
                (
                    "datetime_encoder",
                    DatetimeEncoder(),
                ),
                ("datetime_imputer", cat_date_imputer),
            ],
        )
        if isinstance(X, pd.DataFrame):
            type_preprocessor = ColumnTransformer(
                [
                    ("numeric_preprocessor", numeric_preprocessor, numeric),
                    (
                        "categorical_low_preprocessor",
                        cat_low_preprocessor,
                        categorical_low,
                    ),
                    (
                        "categorical_high_preprocessor",
                        cat_high_preprocessor,
                        categorical_high,
                    ),
                    ("datetime_preprocessor", datetime_preprocessor, datetime),
                ],
                n_jobs=self.n_jobs,
            )
        else:
            if np.issubdtype(X.dtype, np.datetime64):
                type_preprocessor = datetime_preprocessor
            elif np.issubdtype(X.dtype, np.number):
                type_preprocessor = ColumnTransformer(
                    [
                        ("numeric_preprocessor", numeric_preprocessor, numeric),
                        (
                            "categorical_low_preprocessor",
                            cat_low_preprocessor,
                            categorical_low,
                        ),
                        (
                            "categorical_high_preprocessor",
                            cat_high_preprocessor,
                            categorical_high,
                        ),
                    ],
                    n_jobs=self.n_jobs,
                )
            else:
                type_preprocessor = ColumnTransformer(
                    [
                        (
                            "categorical_low_preprocessor",
                            cat_low_preprocessor,
                            categorical_low,
                        ),
                        (
                            "categorical_high_preprocessor",
                            cat_high_preprocessor,
                            categorical_high,
                        ),
                    ],
                    n_jobs=self.n_jobs,
                )
        # Some transformers might not be applied to any features, so we remove them.
        non_empty_transformers = [
            x for x in type_preprocessor.transformers if x[2] != []
        ]
        type_preprocessor.transformers = non_empty_transformers
        # If type_preprocessor has a single transformer, use the transformer directly.
        # This transformer generally is a Pipeline.
        if len(type_preprocessor.transformers) == 1:
            type_preprocessor = type_preprocessor.transformers[0][1]
        preprocessor = Pipeline(
            [
                ("type_preprocessor", type_preprocessor),
                ("remove_invariant", VarianceThreshold()),
            ],
            memory=self._memory,
        )
        return preprocessor

    def add_preprocessing_step(
        self,
        step: Union[
            Union[Pipeline, TransformerMixin, ColumnTransformer],
            Tuple[str, Union[Pipeline, TransformerMixin, ColumnTransformer]],
        ],
        position: Union[str, int] = "end",
    ) -> Pipeline:
        """Add a preprocessing step to :attr:`preprocessor_`.

        Parameters
        ----------
        step :
            A tuple of (str, transformer) or a scikit-learn transformer. Note that
            the transformer can also be a Pipeline or ColumnTransformer.
        position :
            Either an integer denoting before which step in the existing preprocessing pipeline
            the new step should be added, or 'start' or 'end'.

        Returns
        -------
        PoniardBaseEstimator
            self
        """
        if not isinstance(position, int) and position not in ["start", "end"]:
            raise ValueError("`position` can only be int, 'start' or 'end'.")
        existing_preprocessor = self.preprocessor_
        if not isinstance(step, Tuple):
            step = (f"step_{step.__class__.__name__.lower()}", step)
        if isinstance(position, str) and isinstance(existing_preprocessor, Pipeline):
            if position == "start":
                position = 0
            elif position == "end":
                position = len(existing_preprocessor.steps)
        if isinstance(existing_preprocessor, Pipeline):
            existing_preprocessor.steps.insert(position, step)
        else:
            if isinstance(position, int):
                raise ValueError(
                    "If the existing preprocessor is not a Pipeline, only 'start' and "
                    "'end' are accepted as `position`."
                )
            if position == "start":
                self.preprocessor_ = Pipeline(
                    [step, ("initial_preprocessor", self.preprocessor_)],
                    memory=self._memory,
                )
            else:
                self.preprocessor_ = Pipeline(
                    [("initial_preprocessor", self.preprocessor_), step],
                    memory=self._memory,
                )
        # TODO: Clearing the `_fitted_estimator_ids` attr is a hacky way of ensuring that doing
        # [fit -> add_preprocessing_step -> fit] actually fits models. Ideally, build the
        # preprocessor + estimator pipeline during setup and save those IDs when calling fit.
        self._fitted_estimator_ids = []
        self._run_plugin_methods("on_setup_end")
        return self

    @abstractmethod
    def _build_metrics(self) -> Union[Dict[str, Callable], List[str]]:
        """Build metrics."""
        return ["accuracy"]

    def show_results(
        self,
        std: bool = False,
        wrt_dummy: bool = False,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """Return dataframe containing scoring results. By default returns the mean score and fit
        and score times. Optionally returns standard deviations as well.

        Parameters
        ----------
        std :
            Whether to return standard deviation of the scores. Default False.
        wrt_dummy :
            Whether to compute each score/time with respect to the dummy estimator results. Default
            False.

        Returns
        -------
        Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]
            Results
        """
        means = self._means
        stds = self._stds
        if wrt_dummy:
            dummy_means = means.loc[means.index.str.contains("Dummy")]
            dummy_stds = stds.loc[stds.index.str.contains("Dummy")]
            means = means / dummy_means.squeeze()
            stds = stds / dummy_stds.squeeze()
        if std:
            return means, stds
        else:
            return means

    @abstractmethod
    def _build_cv(self):
        return self.cv

    def add_estimators(
        self, estimators: Union[Dict[str, ClassifierMixin], Sequence[ClassifierMixin]]
    ) -> PoniardBaseEstimator:
        """Include new estimator. This is the recommended way of adding an estimator (as opposed
        to modifying :attr:`estimators_` directly), since it also injects random state, n_jobs
        and verbosity.

        Parameters
        ----------
        estimators :
            Estimators to add.

        Returns
        -------
        PoniardBaseEstimator
            Self.

        """
        if not isinstance(estimators, (Sequence, dict)):
            estimators = [estimators]
        if not isinstance(estimators, dict):
            new_estimators = {
                estimator.__class__.__name__: estimator for estimator in estimators
            }
        else:
            new_estimators = estimators
        for new_estimator in new_estimators.values():
            self._pass_instance_attrs(new_estimator)
        self.estimators_.update(new_estimators)
        return self

    def remove_estimators(
        self, estimator_names: Sequence[str], drop_results: bool = True
    ) -> PoniardBaseEstimator:
        """Remove estimators. This is the recommended way of removing an estimator (as opposed
        to modifying :attr:`estimators_` directly), since it also removes the associated rows from
        the results tables.

        Parameters
        ----------
        estimator_names :
            Estimators to remove.
        drop_results :
            Whether to remove the results associated with the estimators. Default True.

        Returns
        -------
        PoniardBaseEstimator
            Self.
        """
        pruned_estimators = {
            k: v for k, v in self.estimators_.items() if k not in estimator_names
        }
        if len(pruned_estimators) == 0:
            raise ValueError("Cannot remove all estimators.")
        self.estimators_ = pruned_estimators
        if drop_results and hasattr(self, "_means"):
            self._means = self._means.loc[~self._means.index.isin(estimator_names)]
            self._stds = self._stds.loc[~self._stds.index.isin(estimator_names)]
            self._experiment_results = {
                k: v
                for k, v in self._experiment_results.items()
                if k not in estimator_names
            }
        self._run_plugin_methods("on_remove_estimators")
        return self

    def get_estimator(
        self,
        estimator_name: str,
        include_preprocessor: bool = True,
        retrain: bool = False,
    ) -> Union[Pipeline, ClassifierMixin, RegressorMixin]:
        """Obtain an estimator in :attr:`estimators_` by name. This is useful for extracting default
        estimators or hyperparmeter-optimized estimators (after using :meth:`tune_estimator`).

        Parameters
        ----------
        estimator_name :
            Estimator name.
        include_preprocessor :
            Whether to return a pipeline with a preprocessor or just the estimator. Default True.
        retrain :
            Whether to retrain with full data. Default False.

        Returns
        -------
        ClassifierMixin
            Estimator.
        """
        model = self._experiment_results[estimator_name]["estimator"][0]
        if not include_preprocessor:
            model = model._final_estimator
        model = clone(model)
        if retrain:
            model.fit(self.X, self.y)
        self._run_plugin_methods(
            "on_get_estimator", estimator=model, name=estimator_name
        )
        return model

    def build_ensemble(
        self,
        method: str = "stacking",
        estimator_names: Optional[Sequence[str]] = None,
        top_n: Optional[int] = 3,
        sort_by: Optional[str] = None,
        ensemble_name: Optional[str] = None,
        **kwargs,
    ) -> PoniardBaseEstimator:
        """Combine estimators into an ensemble.

        By default, orders estimators according to the first metric.

        Parameters
        ----------
        method :
            Ensemble method. Either "stacking" or "voring". Default "stacking".
        estimator_names :
            Names of estimators to include. Default None, which uses `top_n`
        top_n :
            How many of the best estimators to include.
        sort_by :
            Which metric to consider for ordering results. Default None, which uses the first metric.
        ensemble_name :
            Ensemble name when adding to :attr:`estimators_`. Default None.

        Returns
        -------
        PoniardBaseEstimator
            Self.

        Raises
        ------
        ValueError
            If `method` is not "stacking" or "voting".
        """
        if method not in ["voting", "stacking"]:
            raise ValueError("Method must be either voting or stacking.")
        if estimator_names:
            models = [
                (name, self._experiment_results[name]["estimator"][0]._final_estimator)
                for name in estimator_names
            ]
        else:
            if sort_by:
                sorter = sort_by
            else:
                sorter = self._means.columns[0]
            models = [
                (name, self._experiment_results[name]["estimator"][0]._final_estimator)
                for name in self._means.sort_values(sorter, ascending=False).index[
                    :top_n
                ]
            ]
        if method == "voting":
            if self._check_estimator_type() == "classifier":
                ensemble = VotingClassifier(
                    estimators=models, verbose=self.verbose, **kwargs
                )
            else:
                ensemble = VotingRegressor(
                    estimators=models, verbose=self.verbose, **kwargs
                )
        else:
            if self._check_estimator_type() == "classifier":
                ensemble = StackingClassifier(
                    estimators=models, verbose=self.verbose, cv=self.cv_, **kwargs
                )
            else:
                ensemble = StackingRegressor(
                    estimators=models, verbose=self.verbose, cv=self.cv_, **kwargs
                )
        ensemble_name = ensemble_name or ensemble.__class__.__name__
        self.add_estimators(estimators={ensemble_name: ensemble})
        return self

    def get_predictions_similarity(
        self,
        on_errors: bool = True,
    ) -> pd.DataFrame:
        """Compute correlation/association between cross validated predictions for each estimator.

        This can be useful for ensembling.

        Parameters
        ----------
        on_errors :
            Whether to compute similarity on prediction errors instead of predictions. Default
            True.

        Returns
        -------
        pd.DataFrame
            Similarity.
        """
        if self.y.ndim > 1:
            raise ValueError("y must be a 1-dimensional array.")
        raw_results = self.predict()
        results = raw_results.copy()
        for name, result in raw_results.items():
            if on_errors:
                if self._check_estimator_type() == "regressor":
                    results[name] = self.y - result
                else:
                    results[name] = np.where(result == self.y, 1, 0)
        results = pd.DataFrame(results)
        if self._check_estimator_type() == "classifier":
            estimator_names = [x for x in results.columns if x != "DummyClassifier"]
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
            table = results.drop("DummyRegressor", axis=1).corr()
        return table

    def tune_estimator(
        self,
        estimator_name: str,
        grid: Optional[Dict] = None,
        mode: str = "grid",
        tuned_estimator_name: Optional[str] = None,
    ) -> Union[GridSearchCV, RandomizedSearchCV]:
        """Hyperparameter tuning for a single estimator.

        Parameters
        ----------
        estimator_name :
            Estimator to tune.
        grid :
            Hyperparameter grid. Default None, which uses the grids available for default
            estimators.
        mode :
            Type of search. Eithe "grid", "halving" or "random". Default "grid".
        tuned_estimator_name :
            Estimator name when adding to :attr:`estimators_`. Default None.

        Returns
        -------
        PoniardBaseEstimator
            Self.

        Raises
        ------
        KeyError
            If no grid is defined and the estimator is not a default one.
        """
        X, y = self.X, self.y
        estimator = clone(self._experiment_results[estimator_name]["estimator"][0])
        if not grid:
            try:
                grid = GRID[estimator_name]
                grid = {f"{estimator_name}__{k}": v for k, v in grid.items()}
            except KeyError:
                raise KeyError(
                    f"Estimator {estimator_name} has no predefined hyperparameter grid, so it has to be supplied."
                )
        self._pass_instance_attrs(estimator)

        scoring = self._first_scorer(sklearn_scorer=True)
        if mode == "random":
            search = RandomizedSearchCV(
                estimator,
                grid,
                scoring=scoring,
                cv=self.cv_,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        elif mode == "halving":
            from sklearn.experimental import enable_halving_search_cv
            from sklearn.model_selection import HalvingGridSearchCV

            search = HalvingGridSearchCV(
                estimator,
                grid,
                scoring=scoring,
                cv=self.cv_,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        else:
            search = GridSearchCV(
                estimator,
                grid,
                scoring=scoring,
                cv=self.cv_,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
            )
        search.fit(X, y)
        tuned_estimator_name = tuned_estimator_name or f"{estimator_name}_tuned"
        self.add_estimators(
            estimators={
                tuned_estimator_name: clone(search.best_estimator_._final_estimator)
            }
        )
        return self

    def _process_results(self) -> None:
        """Compute mean and standard deviations of  experiment results."""
        # TODO: This processes every result, even those that were processed
        # in previous runs (before add_estimators). Should be made more efficient
        results = pd.DataFrame(self._experiment_results).T
        results = results.loc[
            :,
            [
                x
                for x in results.columns
                if x
                not in ["estimator", "predict", "predict_proba", "decision_function"]
            ],
        ]
        means = results.apply(lambda x: np.mean(x.values.tolist(), axis=1))
        stds = results.apply(lambda x: np.std(x.values.tolist(), axis=1))
        means = means[list(means.columns[2:]) + ["fit_time", "score_time"]]
        stds = stds[list(stds.columns[2:]) + ["fit_time", "score_time"]]
        self._means = means.sort_values(means.columns[0], ascending=False)
        self._stds = stds.reindex(self._means.index)
        return

    def _process_long_results(self) -> None:
        """Prepare experiment results for plotting."""
        base = pd.DataFrame(self._experiment_results).T.drop(["estimator"], axis=1)
        melted = (
            base.rename_axis("Model")
            .reset_index()
            .melt(id_vars="Model", var_name="Metric", value_name="Score")
            .explode("Score")
        )
        melted["Type"] = "Fold"
        means = melted.groupby(["Model", "Metric"])["Score"].mean().reset_index()
        means["Type"] = "Mean"
        melted = pd.concat([melted, means])
        melted["Model"] = melted["Model"].str.replace(
            "Classifier|Regressor", "", regex=True
        )

        self._long_results = melted
        return

    def _first_scorer(self, sklearn_scorer: bool) -> Union[str, Callable]:
        """Helper method to get the first scoring function or name."""
        if isinstance(self.metrics_, Sequence):
            return self.metrics_[0]
        elif isinstance(self.metrics_, dict):
            if sklearn_scorer:
                return list(self.metrics_.values())[0]
            else:
                return list(self.metrics_.keys())[0]
        else:
            raise ValueError(
                "self.metrics_ can only be a sequence of str or dict of str: callable."
            )

    def _train_test_split_from_cv(self):
        """Split data in a 80/20 fashion following the cross-validation strategy defined in the constructor."""
        if isinstance(self.cv_, (int, Iterable)):
            cv_params_for_split = {}
        else:
            cv_params_for_split = {
                k: v
                for k, v in vars(self.cv_).items()
                if k in ["shuffle", "random_state"]
            }
            stratify = self.y if "Stratified" in self.cv_.__class__.__name__ else None
            cv_params_for_split.update({"stratify": stratify})
        return train_test_split(self.X, self.y, test_size=0.2, **cv_params_for_split)

    def _check_estimator_type(self) -> Optional[str]:
        """Utility to check whether self is a Poniard regressor or classifier.

        Returns
        -------
        Optional[str]
            "classifier", "regressor" or None
        """
        from poniard import PoniardRegressor, PoniardClassifier

        if isinstance(self, PoniardRegressor):
            return "regressor"
        elif isinstance(self, PoniardClassifier):
            return "classifier"
        else:
            return None

    def _pass_instance_attrs(self, obj: Union[ClassifierMixin, RegressorMixin]):
        """Helper method to propagate instance attributes to objects."""
        for attr, value in zip(
            ["random_state", "verbose", "verbosity"],
            [self.random_state, self.verbose, self.verbose],
        ):
            if hasattr(obj, attr):
                setattr(obj, attr, value)
        return

    def _run_plugin_methods(self, method: str, **kwargs):
        """Helper method to run plugin methods by name."""
        if not self.plugins:
            return
        for plugin in self.plugins:
            fetched_method = getattr(plugin, method, None)
            if callable(fetched_method):
                accepted_kwargs = inspect.getargs(fetched_method.__code__).args
                kwargs = {k: v for k, v in kwargs.items() if k in accepted_kwargs}
                fetched_method(**kwargs)
        return

    def __repr__(self):
        return f"""{self.__class__.__name__}(estimators={self.estimators}, metrics={self.metrics},
    preprocess={self.preprocess}, scaler={self.scaler}, numeric_imputer={self.numeric_imputer},
    custom_preprocessor={self.custom_preprocessor}, numeric_threshold={self.numeric_threshold},
    cardinality_threshold={self.cardinality_threshold}, cv={self.cv}, verbose={self.verbose},
    random_state={self.random_state}, n_jobs={self.n_jobs}, plugins={self.plugins},
    plot_options={str(self.plot_options)})
            """

    def __add__(
        self, estimators: Union[Dict[str, ClassifierMixin], Sequence[ClassifierMixin]]
    ) -> PoniardBaseEstimator:
        """Add estimators to a Poniard Estimator.

        Parameters
        ----------
        estimators :
            List or dict of estimators to add.

        Returns
        -------
        PoniardBaseEstimator
            Self.
        """
        return self.add_estimators(estimators)

    def __sub__(self, estimator_names: Sequence[str]) -> PoniardBaseEstimator:
        """Remove an estimator and its results.

        Parameters
        ----------
        estimator :
            List of estimators names.

        Returns
        -------
        PoniardBaseEstimator
            Self.
        """
        return self.remove_estimators(estimator_names, drop_results=True)

    def __getitem__(self, estimators: Union[str, Sequence[str]]) -> pd.DataFrame:
        """Get results by indexing with estimator names.

        Parameters
        ----------
        estimators :
            Estimator name(s) as string or list of strings.

        Returns
        -------
        pd.DataFrame
            Filtered results.
        """
        return self.show_results().loc[estimators, :]
