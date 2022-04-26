import warnings
import inspect
import itertools
from typing import List, Optional, Union, Iterable, Callable, Dict, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin, clone
from sklearn.model_selection._split import BaseCrossValidator, BaseShuffleSplit
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
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
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
from poniard.hyperparameters import GRID


class PoniardBaseEstimator(object):
    """Base estimator that sets up all the functionality for the classifier and regressor.

    Parameters
    ----------
    estimators :
        Estimators to evaluate.
    metrics :
        Metrics to compute for each estimator.
    preprocess : bool, optional
        If True, impute missing values, standard scale numeric data and one-hot or ordinal encode categorical data.
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
            Union[
                List[ClassifierMixin],
                Dict[str, ClassifierMixin],
                List[RegressorMixin],
                Dict[str, RegressorMixin],
            ]
        ] = None,
        metrics: Optional[Union[Dict[str, Callable], List[str], Callable]] = None,
        preprocess: bool = True,
        scaler: Optional[Union[str, TransformerMixin]] = None,
        numeric_imputer: Optional[Union[str, TransformerMixin]] = None,
        custom_preprocessor: Union[None, Pipeline, TransformerMixin] = None,
        numeric_threshold: Union[int, float] = 0.1,
        cardinality_threshold: Union[int, float] = 50,
        cv: Union[int, BaseCrossValidator, BaseShuffleSplit, Iterable] = None,
        verbose: int = 0,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
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

        self.processed_estimator_ids = []

    def _build_initial_estimators(self) -> None:
        """Build :attr:`estimators_` dict where keys are the estimator class names.

        Adds dummy estimators if not included during construction. Does nothing if
        :attr:`estimators_` exists.

        Raises
        ------
        ValueError
            If a class instead of a class instance is passed.
        """
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
        if (
            self.__class__.__name__ == "PoniardClassifier"
            and "DummyClassifier" not in self.estimators_.keys()
        ):
            self.estimators_.update(
                {"DummyClassifier": DummyClassifier(strategy="prior")}
            )
        elif (
            self.__class__.__name__ == "PoniardRegressor"
            and "DummyRegressor" not in self.estimators_.keys()
        ):
            self.estimators_.update({"DummyRegressor": DummyRegressor(strategy="mean")})

        for estimator in self.estimators_.values():
            self._pass_instance_attrs(estimator)
        return

    @property
    def _base_estimators(self) -> List[ClassifierMixin]:
        return [
            DummyRegressor(),
            DummyClassifier(),
        ]

    def _infer_dtypes(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Infer feature types (numeric, low-cardinality categorical or high-cardinality
        categorical).

        Parameters
        ----------
        X :
            Input features.

        Returns
        -------
        List[str], List[str], List[str]
            Three lists with column names or indices.
        """
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
        self._inferred_dtypes = {
            "numeric": numeric,
            "categorical_high": categorical_high,
            "categorical_low": categorical_low,
        }
        return numeric, categorical_high, categorical_low

    def _build_preprocessor(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        """Build default preprocessor and assign to :attr:`preprocessor_`.

        The preprocessor imputes missing values, scales numeric features and encodes categorical
        features according to inferred types.

        Parameters
        ----------
        X :
            Input features.
        """
        try:
            self.preprocessor_
            return
        except AttributeError:
            pass
        numeric, categorical_high, categorical_low = self._infer_dtypes(X=X)

        if isinstance(self.scaler, TransformerMixin):
            scaler = self.scaler
        elif self.scaler == "standard":
            scaler = StandardScaler()
        elif self.scaler == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()

        cat_imputer = SimpleImputer(strategy="most_frequent", verbose=self.verbose)

        if isinstance(self.numeric_imputer, TransformerMixin):
            num_imputer = self.numeric_imputer
        elif self.numeric_imputer == "iterative":
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer

            num_imputer = IterativeImputer(random_state=self.random_state)
        else:
            num_imputer = SimpleImputer(strategy="mean", verbose=self.verbose)

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
        """Build metrics and assign to :attr:`metrics_`.

        Parameters
        ----------
        y :
            Target. Used to determine the task (regression, binary classification or multiclass
            classification).
        """
        self.metrics_ = ["accuracy"]
        return

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

    def _setup_experiments(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        y: Union[pd.DataFrame, np.ndarray, List],
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray]]:
        """Orhcestrator.

        Converts inputs to arrays if necessary, sets :attr:`metrics_`,
        :attr:`preprocessor_` and :attr:`estimators_`.

        Parameters
        ----------
        X :
            Features.
        y :
            Target

        Returns
        -------
        Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray]]
            X, y as numpy arrays or pandas dataframes.
        """
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            X = np.array(X)
        if not isinstance(y, (pd.DataFrame, np.ndarray)):
            y = np.array(y)

        if not self.metrics:
            self._build_metrics(y)
        else:
            self.metrics_ = self.metrics

        if self.preprocess:
            if self.custom_preprocessor:
                self.preprocessor_ = self.custom_preprocessor
            else:
                self._build_preprocessor(X)

        self._build_initial_estimators()
        return X, y

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        y: Union[pd.DataFrame, np.ndarray, List],
    ) -> None:
        """This is the main Poniard method. It uses scikit-learn's `cross_validate` function to
        score all :attr:`metrics_` for every :attr:`preprocessor_` | :attr:`estimators_`, using
        :attr:`cv` for cross validation.


        Parameters
        ----------
        X :
            Features.
        y :
            Target.
        """
        X, y = self._setup_experiments(X, y)

        results = {}
        filtered_estimators = {
            name: estimator
            for name, estimator in self.estimators_.items()
            if id(estimator) not in self.processed_estimator_ids
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
            self.processed_estimator_ids.append(id(estimator))
            if i == len(pbar) - 1:
                pbar.set_description("Completed")
        try:
            self._experiment_results.update(results)
        except AttributeError:
            self._experiment_results = results

        self._process_results()
        return

    def add_estimators(
        self, new_estimators: Union[Dict[str, ClassifierMixin], List[ClassifierMixin]]
    ) -> None:
        """Include new estimator. This is the recommended way of adding an estimator (as opposed
        to modifying :attr:`estimators_` directly), since it also injects random state, n_jobs
        and verbosity.

        Parameters
        ----------
        new_estimators :
            Estimators to add.

        Raises
        ------
        ValueError
            If a class and not an instance class is passed.
        """
        if not isinstance(new_estimators, dict):
            new_estimators = {
                estimator.__class__.__name__: estimator for estimator in new_estimators
            }
        for estimator in new_estimators.values():
            self._pass_instance_attrs(estimator)
        if any([inspect.isclass(v) for v in new_estimators.values()]):
            raise ValueError("Pass an instance of an estimator, not the class.")
        self._build_initial_estimators()
        self.estimators_.update(new_estimators)
        return

    def remove_estimators(self, names: List[str], drop_results: bool = True) -> None:
        """Remove estimators. This is the recommended way of removing an estimator (as opposed
        to modifying :attr:`estimators_` directly), since it also removes the associated rows from
        the results tables.

        Parameters
        ----------
        names :
            Estimators to remove.
        drop_results :
            Whether to remove the results associated with the estimators. Default True.
        """
        self.estimators_ = {k: v for k, v in self.estimators_.items() if k not in names}
        if drop_results:
            self._means = self._means.loc[~self._means.index.isin(names)]
            self._stds = self._stds.loc[~self._stds.index.isin(names)]
            self._experiment_results = {
                k: v for k, v in self._experiment_results.items() if k not in names
            }
        return

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
        if std is True:
            return means, stds
        else:
            return means

    def get_estimator(
        self, name: str, include_preprocessor: bool = True, fitted: bool = False
    ) -> ClassifierMixin:
        """Obtain an estimator in :attr:`estimators_` by name. This is useful for extracting default
        estimators or hyperparmeter-optimized estimators (after using :meth:`tune_estimator`).

        Parameters
        ----------
        name :
            Estimator name.
        include_preprocessor :
            Whether to return a pipeline with a preprocessor or just the estimator. Default True.
        fitted :
            Whether to return a fitted estimator/pipeline or a clone. Default False.

        Returns
        -------
        ClassifierMixin
            Estimator.
        """
        model = self._experiment_results[name]["estimator"][0]
        if not include_preprocessor:
            model = model._final_estimator
        if fitted:
            return model
        else:
            return clone(model)

    def build_ensemble(
        self,
        method: str = "stacking",
        estimator_names: Optional[List[str]] = None,
        top_n: int = 3,
        sort_by: Optional[str] = None,
        include_preprocessor: bool = True,
        add_to_estimators: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> Union[ClassifierMixin, RegressorMixin]:
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
        add_to_estimators :
            Whether to include in :attr:`estimators_`. Default False.
        name :
            Ensemble name when adding to :attr:`estimators_`. Default None.

        Returns
        -------
        Union[ClassifierMixin, RegressorMixin]
           scikit-learn ensemble

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
            if self.__class__.__name__ == "PoniardClassifier":
                ensemble = VotingClassifier(
                    estimators=models, verbose=self.verbose, **kwargs
                )
            else:
                ensemble = VotingRegressor(
                    estimators=models, verbose=self.verbose, **kwargs
                )
        else:
            if self.__class__.__name__ == "PoniardClassifier":
                ensemble = StackingClassifier(
                    estimators=models, verbose=self.verbose, cv=self.cv, **kwargs
                )
            else:
                ensemble = StackingRegressor(
                    estimators=models, verbose=self.verbose, cv=self.cv, **kwargs
                )
        if add_to_estimators:
            name = name or ensemble.__class__.__name__
            self.add_estimators(new_estimators={name: ensemble})
        if include_preprocessor:
            return make_pipeline(self.preprocessor_, ensemble)
        else:
            return ensemble

    def get_predictions_similarity(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        y: Union[pd.DataFrame, np.ndarray, List],
    ) -> pd.DataFrame:
        """Compute correlation/association between cross validated predictions for each estimator.

        This can be useful for ensembling.

        Parameters
        ----------
        X :
            Features.
        y :
            Target.

        Returns
        -------
        pd.DataFrame
            Similarity.
        """
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
        if self.__class__.__name__ == "PoniardClassifier":
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
        X: Union[pd.DataFrame, np.ndarray, List],
        y: Union[pd.DataFrame, np.ndarray, List],
        include_preprocessor: bool = True,
        grid: Optional[Dict] = None,
        mode: str = "grid",
        add_to_estimators: bool = False,
        name: Optional[str] = None,
    ) -> Union[GridSearchCV, RandomizedSearchCV]:
        """Hyperparameter tuning for a single estimator.

        Parameters
        ----------
        estimator_name :
            Estimator to tune.
        X :
            Features.
        y :
            Target.
        include_preprocessor :
            Whether to include :attr:`preprocessor_`. Default True.
        grid :
            Hyperparameter grid. Default None, which uses the grids available for default
            estimators.
        mode :
            Type of search. Eithe "grid", "halving" or "random". Default "grid".
        add_to_estimators :
            Whether to include in :attr:`estimators_`. Default False.
        name :
            Estimator name when adding to :attr:`estimators_`. Default None.

        Returns
        -------
        Union[GridSearchCV, RandomizedSearchCV]
            scikit-learn search object.

        Raises
        ------
        KeyError
            If no grid is defined and the estimator is not a default one.
        """
        X, y = self._setup_experiments(X, y)
        estimator = self.estimators_[estimator_name]
        if not grid:
            try:
                grid = GRID[estimator_name]
                grid = {f"{estimator_name}__{k}": v for k, v in grid.items()}
            except KeyError:
                raise KeyError(
                    f"Estimator {estimator_name} has no predefined hyperparameter grid, so it has to be supplied."
                )
        self._pass_instance_attrs(estimator)

        if include_preprocessor:
            estimator = Pipeline(
                [("preprocessor", self.preprocessor_), (estimator_name, estimator)]
            )
        else:
            estimator = Pipeline([(estimator_name, estimator)])

        if isinstance(self.metrics_, dict):
            scoring = list(self.metrics_.values())[0]
        elif isinstance(self.metrics_, (list, tuple, set)):
            scoring = self.metrics_[0]
        else:
            scoring = self.metrics_
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
        if add_to_estimators:
            name = name or f"{estimator_name}_tuned"
            self.add_estimators(
                new_estimators={name: clone(search.best_estimator_._final_estimator)}
            )
        return search

    def _pass_instance_attrs(self, estimator: Union[ClassifierMixin, RegressorMixin]):
        for attr, value in zip(
            ["random_state", "verbose", "verbosity", "n_jobs"],
            [self.random_state, self.verbose, self.verbose, self.n_jobs],
        ):
            if attr in estimator.__dict__:
                setattr(estimator, attr, value)
