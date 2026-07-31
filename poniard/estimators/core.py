from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import (
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.model_selection._split import BaseCrossValidator, BaseShuffleSplit
from sklearn.pipeline import Pipeline
from tqdm import tqdm

try:
    import polars as pl
except ImportError:
    pl = None

try:
    from IPython.display import HTML, display

    _has_ipython = True
except ImportError:
    _has_ipython = False

from ..preprocessing import PoniardPreprocessor
from ..utils.estimate import element_to_list_maybe, get_target_info
from ..utils.utils import get_kwargs, non_default_repr
from .ensemble import EnsembleMixin
from .results import ResultsMixin
from .tuning import TuningMixin

__all__ = ['PoniardBaseEstimator']

class PoniardBaseEstimator(ResultsMixin, EnsembleMixin, TuningMixin, ABC):
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
        estimators: (
            Sequence[ClassifierMixin]
            | dict[str, ClassifierMixin]
            | Sequence[RegressorMixin]
            | dict[str, RegressorMixin]
        ) = None,
        metrics: str | dict[str, Callable] | Sequence[str] | None = None,
        preprocess: bool = True,
        custom_preprocessor: Pipeline | TransformerMixin | PoniardPreprocessor | None = None,
        cv: int | BaseCrossValidator | BaseShuffleSplit | Sequence = None,
        verbose: bool = False,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ):

        self._init_params = get_kwargs()
        if metrics and (
            (
                isinstance(metrics, Sequence)
                and not all(isinstance(m, str) for m in metrics)
            )
            or (
                isinstance(metrics, dict)
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
        self.custom_preprocessor = custom_preprocessor
        self.cv = cv
        self.verbose = verbose
        self.random_state = random_state or 0
        self.estimators = element_to_list_maybe(estimators)
        self.n_jobs = n_jobs

        self._memory = None
        self._tqdm_leave = os.getenv("PONIARD_TQDM_LEAVE", "False") == "True"

        self._added_estimators = {}
        self._removed_estimators = []

    @property
    def poniard_task(self) -> str | None:
        """Check whether self is a Poniard regressor or classifier.

        Returns
        -------
        str | None
            "regression", "classification" or None
        """
        from poniard import PoniardClassifier, PoniardRegressor

        if isinstance(self, PoniardRegressor):
            return "regression"
        elif isinstance(self, PoniardClassifier):
            return "classification"
        else:
            return None

    def setup(
        self,
        X: pd.DataFrame | np.ndarray | list,
        y: pd.DataFrame | np.ndarray | list,
        show_info: bool = True,
    ) -> PoniardBaseEstimator:
        """Configure the estimator: infer types, build preprocessing, build pipelines.

        Call this before `fit()` to set up the pipeline. You can modify
        the preprocessor or estimator list between `setup()` and `fit()`.

        Parameters
        ----------
        X :
            Features.
        y :
            Target.
        show_info :
            Whether to print information about the target, metrics and type inference.

        Returns
        -------
        PoniardBaseEstimator
            Self.
        """
        self._configure(X, y, show_info)
        return self

    def _configure(self, X, y, show_info):
        """Infer types, build preprocessing, pipelines and CV.

        Shared by `setup` and `fit`. Re-running it with the same inputs
        re-configures consistently, which keeps `setup(X, y)` → adjust →
        `fit(X, y)` valid even after `reassign_types` / `add_preprocessing_step`.

        Returns
        -------
        tuple
            The converted `X` and `y`, which `fit` uses for cross-validation.
        """
        if pl is not None and isinstance(X, (pl.DataFrame, pl.Series)):
            X = X.to_pandas()
        if pl is not None and isinstance(y, (pl.DataFrame, pl.Series)):
            y = y.to_pandas()
        if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)):
            X = np.array(X)
        if not isinstance(y, (pd.DataFrame, pd.Series, np.ndarray)):
            y = np.array(y)
        self.show_info = show_info
        self.target_info = get_target_info(y, self.poniard_task)
        if self.target_info["type_"] == "multiclass-multioutput":
            raise NotImplementedError(
                "multiclass-multioutput targets are not supported as "
                "no sklearn metrics support them."
            )
        if self.metrics:
            self.metrics = element_to_list_maybe(self.metrics)
        else:
            self.metrics = self._build_metrics()

        if self.preprocess:
            if self.custom_preprocessor and not isinstance(
                self.custom_preprocessor, PoniardPreprocessor
            ):
                self.preprocessor = self.custom_preprocessor
            else:
                self.preprocessor = self._build_preprocessor(X, y)
            self._pass_instance_attrs(self.preprocessor)
            self._ensure_pandas_output(self.preprocessor)

        if self.show_info:
            self._print_setup_info()

        self.pipelines = self._build_pipelines()
        self.cv = self._build_cv()

        return X, y

    def _print_setup_info(self):
        type_ = self.target_info["type_"]
        shape = self.target_info["shape"]
        nunique = self.target_info["nunique"]
        main_metric = self._first_scorer(sklearn_scorer=False)
        if hasattr(self, "_poniard_preprocessor"):
            num_thresh = self._poniard_preprocessor.numeric_threshold
            cat_thresh = self._poniard_preprocessor.cardinality_threshold
        if _has_ipython:
            display(
                HTML(
                    f"""
                         <h2>Setup info</h2>
                         <h3>Target</h3>
                             <p><b>Type:</b> {type_}</p>
                             <p><b>Shape:</b> {shape}</p>
                             <p><b>Unique values:</b> {nunique}</p>
                             <h3>Metrics</h3>
                             <b>Main metric:</b> {main_metric}
                             """
                )
            )
            if hasattr(self, "_poniard_preprocessor"):
                display(
                    HTML(
                        f""" <h3>Feature type inference</h3>
                                <p><b>Minimum unique values to consider a number-like feature numeric:</b> {num_thresh}</p>
                                <p><b>Minimum unique values to consider a categorical feature high cardinality:</b> {cat_thresh}</p>
                                <p><b>Inferred feature types:</b></p>
                                {self._poniard_preprocessor.inferred_types_df.to_html()}"""
                    )
                )
        else:
            print("Target info", "-----------", sep="\n")
            print(
                f"Type: {type_}",
                f"Shape: {shape}",
                f"Unique values: {nunique}",
                sep="\n",
                end="\n\n",
            )

            print(
                "Main metric",
                "-----------",
                main_metric,
                sep="\n",
                end="\n\n",
            )
            if hasattr(self, "_poniard_preprocessor"):
                print(
                    "Thresholds",
                    "----------",
                    f"Minimum unique values to consider a feature numeric: {num_thresh}",
                    f"Minimum unique values to consider a categorical high cardinality: {cat_thresh}",
                    sep="\n",
                    end="\n\n",
                )
                print("Inferred feature types", "----------------------", sep="\n")
                print(self._poniard_preprocessor.inferred_types_df)

    def _build_preprocessor(self, X, y) -> Pipeline:
        """Build default preprocessor using `PoniardPreprocessor`.

        The preprocessor imputes missing values, scales numeric features and encodes categorical
        features according to inferred types.

        """
        if self.custom_preprocessor:
            self._poniard_preprocessor = self.custom_preprocessor
        else:
            self._poniard_preprocessor = PoniardPreprocessor()
        self._memory = self._poniard_preprocessor._memory
        self._poniard_preprocessor.build(
            X=X, y=y, task=self.poniard_task, target_info=self.target_info
        )
        self.feature_types = self._poniard_preprocessor.feature_types
        return self._poniard_preprocessor.preprocessor

    @property
    @abstractmethod
    def _default_estimators(self) -> list[ClassifierMixin]:
        return []

    def _make_pipeline(self, name: str, estimator) -> Pipeline:
        """Create a Pipeline for an estimator, optionally including the preprocessor.

        The wrapping pipeline is configured to output pandas DataFrames on
        ``transform``. This propagates to any preprocessor step (including a
        user-supplied custom one) so downstream code keeps the pandas contract
        without relying on a global sklearn config.
        """
        if self.preprocess:
            pipe = Pipeline(
                [("preprocessor", self.preprocessor), (name, estimator)],
                memory=self._memory,
            )
        else:
            pipe = Pipeline([(name, estimator)])
        if self.preprocess:
            pipe.set_output(transform="pandas")
        return pipe

    @staticmethod
    def _ensure_pandas_output(estimator) -> None:
        """Configure a preprocessor (Pipeline or ColumnTransformer) to output
        pandas DataFrames on ``transform``.

        Called for user-supplied custom preprocessors that are not
        ``PoniardPreprocessor`` instances, so the pandas in/out contract is
        honored without relying on a global sklearn config.
        """
        set_output = getattr(estimator, "set_output", None)
        if callable(set_output):
            try:
                estimator.set_output(transform="pandas")
            except (TypeError, ValueError):
                pass

    @staticmethod
    def _generate_estimator_name(
        estimator, existing_names: set[str], all_estimators: list
    ) -> str:
        """Generate a name for an estimator.

        Default is the class name. If it clashes, append _2, _3, etc.
        """
        class_name = estimator.__class__.__name__
        if class_name not in existing_names:
            return class_name
        i = 2
        while f"{class_name}_{i}" in existing_names:
            i += 1
        return f"{class_name}_{i}"

    def _build_pipelines(
        self,
    ) -> dict[str, ClassifierMixin | RegressorMixin]:
        """Build `pipelines` dict where keys are estimator names.

        Names can be:
        - A dict: keys are names, values are estimators
        - A list of tuples: (name, estimator)
        - A list of estimators: class name if unique, short prefix if duplicates

        Adds dummy estimators if not included during construction.
        """
        if isinstance(self.estimators, dict):
            estimators = self.estimators.copy()
        elif self.estimators:
            estimators = {}
            for item in self.estimators:
                if isinstance(item, tuple):
                    name, estimator = item
                else:
                    name = self._generate_estimator_name(
                        item, set(estimators.keys()), self.estimators
                    )
                    estimator = item
                estimators[name] = estimator
        else:
            estimators = {}
            for estimator in self._default_estimators:
                name = self._generate_estimator_name(
                    estimator, set(estimators.keys()), self._default_estimators
                )
                estimators[name] = estimator
        estimators.update(self._added_estimators)
        for name in self._removed_estimators:
            estimators.pop(name, None)
        estimators = self._add_dummy_estimators(estimators)

        for estimator in estimators.values():
            self._pass_instance_attrs(estimator)

        pipelines = {
            name: self._make_pipeline(name, estimator)
            for name, estimator in estimators.items()
        }
        self._fitted_pipeline_names = set()
        return pipelines

    def _add_dummy_estimators(self, estimators: dict):
        existing_names = set(estimators.keys())
        if "DummyClassifier" in existing_names or "DummyRegressor" in existing_names:
            return estimators
        if self.poniard_task == "classification":
            dummy = DummyClassifier(strategy="prior")
            name = self._generate_estimator_name(
                dummy, existing_names, list(estimators.values())
            )
            estimators[name] = dummy
        elif self.poniard_task == "regression":
            dummy = DummyRegressor(strategy="mean")
            name = self._generate_estimator_name(
                dummy, existing_names, list(estimators.values())
            )
            estimators[name] = dummy
        return estimators

    @abstractmethod
    def _build_metrics(self) -> dict[str, Callable] | list[str]:
        """Build metrics."""
        return ["accuracy"]

    @abstractmethod
    def _build_cv(self):
        return self.cv

    def fit(self, X, y, show_info: bool = True) -> PoniardBaseEstimator:
        """Fit the estimator: build preprocessing pipeline, cross-validate all estimators.

        This is the main method. It infers feature types, builds preprocessing,
        and cross-validates all estimators, collecting results.

        Parameters
        ----------
        X :
            Features.
        y :
            Target.
        show_info :
            Whether to print information about the target, metrics and type inference.

        Returns
        -------
        PoniardBaseEstimator
            Self.
        """
        X, y = self._configure(X, y, show_info)

        # Cross-validate
        results = {}
        filtered_pipelines = {
            name: pipeline
            for name, pipeline in self.pipelines.items()
            if name not in self._fitted_pipeline_names
        }
        pbar = tqdm(filtered_pipelines.items(), leave=self._tqdm_leave)
        for i, (name, pipeline) in enumerate(pbar):
            pbar.set_description(f"{name}")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                warnings.filterwarnings(
                    "ignore", message=".*will be encoded as all zeros"
                )
                result = cross_validate(
                    pipeline,
                    X,
                    y,
                    scoring=self.metrics,
                    cv=self.cv,
                    return_train_score=True,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs,
                )
            results.update({name: result})
            self._fitted_pipeline_names.add(name)
            if i == len(pbar) - 1:
                pbar.set_description("Completed")
        if hasattr(self, "_experiment_results"):
            self._experiment_results.update(results)
        else:
            self._experiment_results = results

        self._process_results()
        self._process_long_results()
        return self

    def _predict(
        self, method: str, X, y, estimator_names: Sequence[str] | None = None
    ) -> dict[str, np.ndarray]:
        """Helper method for predicting targets or target probabilities with cross validation.
        Accepts predict, predict_proba, predict_log_proba or decision_function."""
        if not hasattr(self, "cv"):
            raise ValueError("`setup` must be called before `predict`.")
        estimator_names = element_to_list_maybe(estimator_names)
        if not estimator_names:
            estimator_names = [estimator for estimator in self.pipelines.keys()]
        results = {}
        pbar = tqdm(estimator_names, leave=self._tqdm_leave)
        for i, name in enumerate(pbar):
            pbar.set_description(f"{name}")
            pipeline = self.pipelines[name]
            try:
                result = cross_val_predict(
                    pipeline,
                    X,
                    y,
                    cv=self.cv,
                    method=method,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs,
                )
            except AttributeError:
                warnings.warn(
                    f"{name} does not support `{method}` method. Filling with nan.",
                    stacklevel=2,
                )
                result = np.empty(y.shape)
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
        self, X, y, estimator_names: Sequence[str] | None = None
    ) -> dict[str, np.ndarray]:
        """Get cross validated target predictions where each sample belongs to a single test set.

        Parameters
        ----------
        X :
            Features.
        y :
            Target.
        estimator_names :
            Estimators to include. If None, predict all estimators.

        Returns
        -------
        dict
            Dict where keys are estimator names and values are numpy arrays of predictions.
        """
        return self._predict(method="predict", X=X, y=y, estimator_names=estimator_names)

    def predict_proba(
        self, X, y, estimator_names: Sequence[str] | None = None
    ) -> dict[str, np.ndarray]:
        """Get cross validated target probability predictions where each sample belongs to a
        single test set.

        Parameters
        ----------
        X :
            Features.
        y :
            Target.
        estimator_names :
            Estimators to include. If None, predict all estimators.

        Returns
        -------
        dict
            Dict where keys are estimator names and values are numpy arrays of prediction
            probabilities.
        """
        return self._predict(method="predict_proba", X=X, y=y, estimator_names=estimator_names)

    def decision_function(
        self, X, y, estimator_names: Sequence[str] | None = None
    ) -> dict[str, np.ndarray]:
        """Get cross validated decision function predictions where each sample belongs to a
        single test set.

        Parameters
        ----------
        X :
            Features.
        y :
            Target.
        estimator_names :
            Estimators to include. If None, predict all estimators.

        Returns
        -------
        dict
            Dict where keys are estimator names and values are numpy arrays of decision functions.
        """
        return self._predict(
            method="decision_function", X=X, y=y, estimator_names=estimator_names
        )

    def predict_all(
        self, X, y, estimator_names: Sequence[str] | None = None
    ) -> tuple[dict[str, np.ndarray], ...]:
        """Get cross validated target predictions, probabilities and decision functions
        where each sample belongs to a test set.

        Parameters
        ----------
        X :
            Features.
        y :
            Target.
        estimator_names :
            Estimators to include. If None, predict all estimators.

        Returns
        -------
        tuple[dict]
            Tuple of dicts where keys are estimator names and values are numpy arrays of
            predictions.
        """
        return (
            self._predict(method="predict", X=X, y=y, estimator_names=estimator_names),
            self._predict(method="predict_proba", X=X, y=y, estimator_names=estimator_names),
            self._predict(method="decision_function", X=X, y=y, estimator_names=estimator_names),
        )

    def reassign_types(
        self,
        numeric: list[str | int] | None = None,
        categorical_high: list[str | int] | None = None,
        categorical_low: list[str | int] | None = None,
        datetime: list[str | int] | None = None,
        keep_remainder: bool = True,
    ) -> PoniardBaseEstimator:
        """Reassign feature types. By default, leaves ommitted features as they were.

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
        keep_remainder :
            Whether to keep features not specified in the method parameters
            as is or drop them

        Returns
        -------
        PoniardBaseEstimator
            self.
        """
        numeric = numeric or []
        categorical_high = categorical_high or []
        categorical_low = categorical_low or []
        datetime = datetime or []
        if keep_remainder:
            assigned_types = self.feature_types.copy()
            swapped = numeric + categorical_high + categorical_low + datetime
            for k in self.feature_types.keys():
                assigned_types[k] = [x for x in assigned_types[k] if x not in swapped]
            for k, new in zip(
                assigned_types.keys(),
                [numeric, categorical_high, categorical_low, datetime],
            ):
                assigned_types[k] = assigned_types[k] + new
        else:
            assigned_types = {
                "numeric": numeric or [],
                "categorical_high": categorical_high or [],
                "categorical_low": categorical_low or [],
                "datetime": datetime or [],
            }
        if self.show_info:
            assigned_types_df = pd.DataFrame.from_dict(
                assigned_types, orient="index"
            ).T.fillna("")

            if _has_ipython:
                display(
                    HTML(
                        f"""<p><b>Assigned feature types:</b></p>
                            {assigned_types_df.to_html()}"""
                    )
                )
            else:
                print("Assigned feature types", "----------------------", sep="\n")
                print(assigned_types_df)

        self.feature_types = assigned_types
        self._poniard_preprocessor.feature_types = assigned_types
        self._poniard_preprocessor.build()
        self.preprocessor = self._poniard_preprocessor.preprocessor
        self.pipelines = self._build_pipelines()
        return self

    def add_preprocessing_step(
        self,
        step: (
            Pipeline | TransformerMixin | ColumnTransformer
            | tuple[str, Pipeline | TransformerMixin | ColumnTransformer]
        ),
        position: str | int = "end",
    ) -> Pipeline:
        """Add a preprocessing step.

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
        PoniardPreprocessor
            self
        """
        if not isinstance(position, int) and position not in ["start", "end"]:
            raise ValueError("`position` can only be int, 'start' or 'end'.")
        existing_preprocessor = self.preprocessor
        if not isinstance(step, tuple):
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
                self.preprocessor = Pipeline(
                    [step, ("initial_preprocessor", self.preprocessor)],
                    memory=self._memory,
                )
            else:
                self.preprocessor = Pipeline(
                    [("initial_preprocessor", self.preprocessor), step],
                    memory=self._memory,
                )
        self.pipelines = self._build_pipelines()
        return self

    def add_estimators(
        self, estimators: dict[str, ClassifierMixin] | Sequence[ClassifierMixin]
    ) -> PoniardBaseEstimator:
        """Include new estimator. This is the recommended way of adding an estimator (as opposed
        to modifying `pipelines` directly), since it also injects random state, n_jobs
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
        estimators = element_to_list_maybe(estimators)
        if isinstance(estimators, dict):
            new_estimators = estimators
        else:
            new_estimators = {}
            existing_names = set(self.pipelines.keys())
            all_estimators = list(estimators)
            for item in estimators:
                if isinstance(item, tuple):
                    name, estimator = item
                else:
                    name = self._generate_estimator_name(
                        item, existing_names, all_estimators
                    )
                    estimator = item
                    existing_names.add(name)
                new_estimators[name] = estimator
        for new_estimator in new_estimators.values():
            self._pass_instance_attrs(new_estimator)
        self._added_estimators.update(new_estimators)
        self.pipelines.update({
            name: self._make_pipeline(name, estimator)
            for name, estimator in new_estimators.items()
        })
        return self

    def remove_estimators(
        self, estimator_names: Sequence[str], drop_results: bool = True
    ) -> PoniardBaseEstimator:
        """Remove estimators. This is the recommended way of removing an estimator (as opposed
        to modifying `pipelines` directly), since it also removes the associated rows from
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
        estimator_names = element_to_list_maybe(estimator_names)
        self._removed_estimators.extend(estimator_names)
        pruned_estimators = {
            k: v for k, v in self.pipelines.items() if k not in estimator_names
        }
        if len(pruned_estimators) == 0:
            raise ValueError("Cannot remove all estimators.")
        self.pipelines = pruned_estimators
        if drop_results and hasattr(self, "_means"):
            self._means = self._means.loc[~self._means.index.isin(estimator_names)]
            self._stds = self._stds.loc[~self._stds.index.isin(estimator_names)]
            self._experiment_results = {
                k: v
                for k, v in self._experiment_results.items()
                if k not in estimator_names
            }
            self._process_long_results()
        return self

    def get_estimator(
        self,
        estimator_name: str,
        include_preprocessor: bool = True,
        X: pd.DataFrame | np.ndarray | list | None = None,
        y: pd.DataFrame | np.ndarray | list | None = None,
        retrain: bool = False,
    ) -> Pipeline | ClassifierMixin | RegressorMixin:
        """Export an estimator as a plain scikit-learn object you own.

        This is the supported way to leave Poniard: the returned object is a
        plain `sklearn.pipeline.Pipeline` (or a bare estimator when
        ``include_preprocessor=False``) with no poniard references, so you can
        save it, deploy it, or continue working on it without Poniard installed.
        Use it to extract default estimators or hyperparameter-optimized
        estimators (after using `PoniardBaseEstimator.tune_estimator`).

        Parameters
        ----------
        estimator_name :
            Estimator name.
        include_preprocessor :
            Whether to return a pipeline with a preprocessor or just the
            estimator. Default True.
        X :
            Features. Required if retrain is True.
        y :
            Target. Required if retrain is True.
        retrain :
            Whether to retrain the clone with full data. Pass X and y to get a
            fitted pipeline ready to predict. Default False returns an
            unfitted clone.

        Returns
        -------
        sklearn.pipeline.Pipeline | ClassifierMixin | RegressorMixin
            A plain scikit-learn pipeline or estimator with no poniard references.
        """
        model = self.pipelines[estimator_name]
        if not include_preprocessor:
            model = model._final_estimator
        model = clone(model)
        if retrain:
            if X is None or y is None:
                raise ValueError("X and y must be provided when retrain=True.")
            model.fit(X, y)
        return model

    def save(self, path: str | os.PathLike) -> None:
        """Save the fitted estimator to disk with joblib.

        Use `PoniardClassifier.load` / `PoniardRegressor.load` to restore it.
        A fitted estimator round-trips `fit` → `save` → `load` → `get_results`
        without losing results.

        Parameters
        ----------
        path :
            Where to write the estimator.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> PoniardBaseEstimator:
        """Load an estimator saved with `save`.

        Parameters
        ----------
        path :
            Location of the saved estimator.

        Returns
        -------
        PoniardBaseEstimator
            The restored estimator.
        """
        return joblib.load(path)

    def _train_test_split_from_cv(self, X, y):
        """Split data in a 80/20 fashion following the cross-validation strategy defined in the constructor."""
        if isinstance(self.cv, (int, Iterable)):
            cv_params_for_split = {}
        else:
            cv_params_for_split = {
                k: v
                for k, v in vars(self.cv).items()
                if k in ["shuffle", "random_state"]
            }
            stratify = y if "Stratified" in self.cv.__class__.__name__ else None
            cv_params_for_split.update({"stratify": stratify})
        return train_test_split(X, y, test_size=0.2, **cv_params_for_split)

    def _pass_instance_attrs(self, obj: ClassifierMixin | RegressorMixin):
        """Helper method to propagate instance attributes to objects."""
        for attr, value in [
            ("random_state", self.random_state),
            ("verbose", self.verbose),
        ]:
            if hasattr(obj, attr):
                setattr(obj, attr, value)

    def _get_or_compute_prediction(self, X, y, estimator_name: str, method: str):
        """Get predictions (either predict, predict_proba or decision_function) for a given
        estimator or compute if not available."""
        try:
            return self._experiment_results[estimator_name][method]
        except KeyError:
            self._predict(method=method, X=X, y=y, estimator_names=[estimator_name])
            return self._experiment_results[estimator_name][method]

    def __repr__(self):
        return non_default_repr(self)

    def __add__(
        self,
        estimators: dict[str, ClassifierMixin | RegressorMixin]
        | Sequence[ClassifierMixin | RegressorMixin],
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
        estimators = element_to_list_maybe(estimators)
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
        estimator_names = element_to_list_maybe(estimator_names)
        return self.remove_estimators(estimator_names, drop_results=True)

    def __getitem__(
        self, estimator_name: str
    ) -> Pipeline | ClassifierMixin | RegressorMixin:
        """Get an estimator by indexing with its name

        Parameters
        ----------
        estimator_name :
            Estimator name as string.

        Returns
        -------
        Pipeline | ClassifierMixin | RegressorMixin
            Built estimator.
        """
        return self.get_estimator(estimator_name)
