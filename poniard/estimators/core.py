from __future__ import annotations
import warnings
import itertools
import inspect
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Callable, Dict, Tuple, Any, Sequence, Iterable

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin, clone
from sklearn.model_selection import (
    BaseCrossValidator,
    BaseShuffleSplit,
    train_test_split,
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
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
        function.
    plugins :
        Plugin instances that run in set moments of setup, fit and plotting.
    plot_options :
        :class:poniard.plot.plot_factory.PoniardPlotFactory instance specifying Plotly format
        options or None, which sets the default factory.

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
            Union[
                List[ClassifierMixin],
                Dict[str, ClassifierMixin],
                List[RegressorMixin],
                Dict[str, RegressorMixin],
            ]
        ] = None,
        metrics: Optional[Union[str, Dict[str, Callable], List[str]]] = None,
        preprocess: bool = True,
        scaler: Optional[Union[str, TransformerMixin]] = None,
        numeric_imputer: Optional[Union[str, TransformerMixin]] = None,
        custom_preprocessor: Union[None, Pipeline, TransformerMixin] = None,
        numeric_threshold: Union[int, float] = 0.1,
        cardinality_threshold: Union[int, float] = 50,
        cv: Union[int, BaseCrossValidator, BaseShuffleSplit, Sequence] = None,
        verbose: int = 0,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        plugins: Optional[List[Any]] = None,
        plot_options: Optional[PoniardPlotFactory] = None,
    ):
        # TODO: Ugly check that metrics conforms to expected types. Should improve.
        if metrics and (
            (
                isinstance(metrics, (List, Tuple))
                and not all(isinstance(m, str) for m in metrics)
            )
            or (
                isinstance(metrics, Dict)
                and not all(isinstance(m, str) for m in metrics.values())
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
        self.numeric_imputer = numeric_imputer or "simple"
        self.numeric_threshold = numeric_threshold
        self.custom_preprocessor = custom_preprocessor
        self.cardinality_threshold = cardinality_threshold
        self.cv = cv
        self.verbose = verbose
        self.random_state = random_state
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

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        y: Union[pd.DataFrame, np.ndarray, List],
    ) -> PoniardBaseEstimator:
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
        self._run_plugin_methods("on_fit_start")
        self._setup_experiments(X, y)

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
                    [("preprocessor", self.preprocessor_), (name, estimator)]
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
                    cv=self.cv,
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

    def fit_new(self) -> PoniardBaseEstimator:
        """Helper method for fitting new estimators. Doesn't require features or target as those
        are registered when :meth:`fit` is called for the first time."""
        self.fit(self.X, self.y)
        return self

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

    def _setup_experiments(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        y: Union[pd.DataFrame, np.ndarray, List],
    ) -> None:
        """Orchestrator.

        Converts inputs to arrays if necessary, sets :attr:`metrics_`,
        :attr:`preprocessor_` and :attr:`estimators_`.

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
        self._run_plugin_methods("on_setup_end")
        return

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
        if isinstance(self.cardinality_threshold, int):
            self.cardinality_threshold_ = self.cardinality_threshold
        else:
            self.cardinality_threshold_ = int(self.cardinality_threshold * X.shape[0])
        if isinstance(self.numeric_threshold, int):
            self.numeric_threshold_ = self.numeric_threshold
        else:
            self.numeric_threshold_ = int(self.numeric_threshold * X.shape[0])
        print(
            "Minimum unique values to consider an integer feature numeric:",
            self.numeric_threshold_,
        )
        print(
            "Minimum unique values to consider a non-float feature high cardinality:",
            self.cardinality_threshold_,
            end="\n\n",
        )
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
                if X[column].nunique() > self.numeric_threshold_:
                    numeric.append(column)
                elif X[column].nunique() > self.cardinality_threshold_:
                    categorical_high.append(column)
                else:
                    categorical_low.append(column)
            strings = X.select_dtypes(exclude="number").columns
            for column in strings:
                if X[column].nunique() > self.cardinality_threshold_:
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
        }
        print(
            "Inferred feature types:",
            pd.DataFrame.from_dict(self._inferred_dtypes, orient="index").T,
            sep="\n",
        )
        return numeric, categorical_high, categorical_low

    def _build_preprocessor(self) -> Pipeline:
        """Build default preprocessor.

        The preprocessor imputes missing values, scales numeric features and encodes categorical
        features according to inferred types.

        """
        X = self.X
        if hasattr(self, "preprocessor_"):
            return self.preprocessor_
        numeric, categorical_high, categorical_low = self._infer_dtypes()

        if isinstance(self.scaler, TransformerMixin):
            scaler = self.scaler
        elif self.scaler == "standard":
            scaler = StandardScaler()
        elif self.scaler == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()

        cat_imputer = SimpleImputer(strategy="most_frequent")

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
                ("categorical_imputer", cat_imputer),
                (
                    "one-hot_encoder",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                ),
            ]
        )

        cat_high_preprocessor = Pipeline(
            [
                ("categorical_imputer", cat_imputer),
                (
                    "ordinal_encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=99999
                    ),
                ),
            ],
        )
        if isinstance(X, pd.DataFrame):
            preprocessor = ColumnTransformer(
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
            if np.issubdtype(X.dtype, float):
                preprocessor = numeric_preprocessor
            elif np.issubdtype(X.dtype, int):
                preprocessor = ColumnTransformer(
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
                preprocessor = ColumnTransformer(
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
        return preprocessor

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

    def add_estimators(
        self, estimators: Union[Dict[str, ClassifierMixin], List[ClassifierMixin]]
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
        self, names: List[str], drop_results: bool = True
    ) -> PoniardBaseEstimator:
        """Remove estimators. This is the recommended way of removing an estimator (as opposed
        to modifying :attr:`estimators_` directly), since it also removes the associated rows from
        the results tables.

        Parameters
        ----------
        names :
            Estimators to remove.
        drop_results :
            Whether to remove the results associated with the estimators. Default True.

        Returns
        -------
        PoniardBaseEstimator
            Self.
        """
        self.estimators_ = {k: v for k, v in self.estimators_.items() if k not in names}
        if drop_results:
            self._means = self._means.loc[~self._means.index.isin(names)]
            self._stds = self._stds.loc[~self._stds.index.isin(names)]
            self._experiment_results = {
                k: v for k, v in self._experiment_results.items() if k not in names
            }
        self._run_plugin_methods("on_remove_estimators")
        return self

    def get_estimator(
        self, name: str, include_preprocessor: bool = True, retrain: bool = False
    ) -> Union[Pipeline, ClassifierMixin, RegressorMixin]:
        """Obtain an estimator in :attr:`estimators_` by name. This is useful for extracting default
        estimators or hyperparmeter-optimized estimators (after using :meth:`tune_estimator`).

        Parameters
        ----------
        name :
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
        model = self._experiment_results[name]["estimator"][0]
        if not include_preprocessor:
            model = model._final_estimator
        model = clone(model)
        if retrain:
            model.fit(self.X, self.y)
        self._run_plugin_methods("on_get_estimator", estimator=model, name=name)
        return model

    def build_ensemble(
        self,
        method: str = "stacking",
        estimator_names: Optional[List[str]] = None,
        top_n: int = 3,
        sort_by: Optional[str] = None,
        include_preprocessor: bool = True,
        name: Optional[str] = None,
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
        include_preprocessor :
            Whether to include preprocessing. Default True.
        name :
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
                    estimators=models, verbose=self.verbose, cv=self.cv, **kwargs
                )
            else:
                ensemble = StackingRegressor(
                    estimators=models, verbose=self.verbose, cv=self.cv, **kwargs
                )
        name = name or ensemble.__class__.__name__
        self.add_estimators(estimators={name: ensemble})
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
        X, y = self.X, self.y
        results = {}
        pbar = tqdm(self.estimators_.items())
        for i, (name, estimator) in enumerate(pbar):
            pbar.set_description(f"{name}")
            if self.preprocess:
                final_estimator = Pipeline(
                    [("preprocessor", self.preprocessor_), ("estimator", estimator)]
                )
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
            if on_errors:
                if self._check_estimator_type() == "regressor":
                    result = y - result
                else:
                    result = np.where(result == y, 1, 0)
            results.update({name: result})
            if i == len(pbar) - 1:
                pbar.set_description("Completed")
        results = pd.DataFrame(results)
        if self._check_estimator_type() == "classifier":
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

    def tune_estimator(
        self,
        estimator_name: str,
        grid: Optional[Dict] = None,
        mode: str = "grid",
        name: Optional[str] = None,
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
        name :
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
                cv=self.cv,
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
                cv=self.cv,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        else:
            search = GridSearchCV(
                estimator,
                grid,
                scoring=scoring,
                cv=self.cv,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
            )
        search.fit(X, y)
        name = name or f"{estimator_name}_tuned"
        self.add_estimators(
            estimators={name: clone(search.best_estimator_._final_estimator)}
        )
        return self

    def get_permutation_importances(
        self,
        estimator_name: str,
        n_repeats: int = 10,
        std: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Compute permutation importances mean and standard deviations for a single estimator.

        Kwargs are passed to the sklearn permutation importance function.

        Parameters
        ----------
        estimator_name :
            Estimator to tune.
        n_repeats :
            How many times to repeat random permutations of a single feature. Default 10.
        std : bool, optional
            Whether to return standard deviations. Default True.

        Returns
        -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]
            Permutation importances mean, optionally also standard deviations.
        """
        estimator = clone(self._experiment_results[estimator_name]["estimator"][0])
        scoring = self._first_scorer(sklearn_scorer=True)

        X_train, X_test, y_train, y_test = self._train_test_split_from_cv()
        estimator.fit(X_train, y_train)
        result = permutation_importance(
            estimator,
            X_test,
            y_test,
            scoring=scoring,
            random_state=self.random_state,
            n_repeats=n_repeats,
            n_jobs=self.n_jobs,
            **kwargs,
        )
        self._experiment_results[estimator_name]["permutation_importances"] = result
        if isinstance(self.X, pd.DataFrame):
            new_idx = self.X.columns
        else:
            new_idx = range(self.X.shape[1])
        means = pd.DataFrame(result["importances_mean"])
        means.columns = [f"Permutation importances mean ({n_repeats} repeats)"]
        means.index = new_idx
        if std:
            stds = pd.DataFrame(result["importances_std"])
            stds.columns = [f"Permutation importances std. ({n_repeats} repeats)"]
            stds.index = new_idx
            return means, stds
        else:
            return means

    def _process_results(self) -> None:
        """Compute mean and standard deviations of  experiment results."""
        # TODO: This processes every result, even those that were processed
        # in previous runs (before add_estimators). Should be made more efficient
        results = pd.DataFrame(self._experiment_results).T.drop(["estimator"], axis=1)
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
        if isinstance(self.metrics_, (List, Tuple)):
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
        if isinstance(self.cv, (int, Iterable)):
            cv_params_for_split = {}
        else:
            cv_params_for_split = {
                k: v
                for k, v in vars(self.cv).items()
                if k in ["shuffle", "random_state"]
            }
            stratify = self.y if "Stratified" in self.cv.__class__.__name__ else None
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

    def _pass_instance_attrs(self, estimator: Union[ClassifierMixin, RegressorMixin]):
        """Helper method to propagate instance attributes to estimators."""
        for attr, value in zip(
            ["random_state", "verbose", "verbosity"],
            [self.random_state, self.verbose, self.verbose],
        ):
            if attr in estimator.__dict__:
                setattr(estimator, attr, value)
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

    def __add__(
        self, estimators: Union[Dict[str, ClassifierMixin], List[ClassifierMixin]]
    ) -> PoniardBaseEstimator:
        return self.add_estimators(estimators)

    def __sub__(self, estimator: List[str]) -> PoniardBaseEstimator:
        return self.remove_estimators(estimator, drop_results=True)
