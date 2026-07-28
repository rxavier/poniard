from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

try:
    import polars as pl
except ImportError:
    pl = None

from ..utils.estimate import get_target_info
from ..utils.utils import get_kwargs, non_default_repr
from .datetime import DatetimeEncoder

if TYPE_CHECKING:
    from poniard.estimators.core import PoniardBaseEstimator

__all__ = ['PoniardPreprocessor']

class PoniardPreprocessor:
    """Base preprocessor that builds an easily modifiable pipeline based
    on feature data types.

    Parameters
    ----------
    scaler :
        Numeric scaler method. Either "standard", "minmax", "robust" or scikit-learn Transformer.
    high_cardinality_encoder :
        Encoder for categorical features with high cardinality. Either "target" or "ordinal",
        or scikit-learn Transformer.
    numeric_imputer :
        Imputation method. Either "simple", "iterative" or scikit-learn Transformer.
    numeric_threshold :
        Number features with unique values above a certain threshold will be treated as numeric. If
        float, the threshold is `numeric_threshold * samples`.
    cardinality_threshold :
        Non-number features with cardinality above a certain threshold will be treated as
        ordinal encoded instead of one-hot encoded. If float, the threshold is
        `cardinality_threshold * samples`.
    cache_transformations :
        Whether to cache transformations and set the `memory` parameter for Pipelines. This can
        speed up slow transformations as they are not recalculated for each estimator.
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
        task: str | None = None,
        scaler: str | TransformerMixin | None = None,
        high_cardinality_encoder: str | TransformerMixin | None = None,
        numeric_imputer: str | TransformerMixin | None = None,
        custom_preprocessor: Pipeline | TransformerMixin | None = None,
        numeric_threshold: int | float = 0.1,
        cardinality_threshold: int | float = 20,
        verbose: bool = False,
        random_state: int | None = None,
        n_jobs: int | None = None,
        cache_transformations: bool = False,
    ):
        self._init_params = get_kwargs()
        self.task = task
        self.scaler = scaler or "standard"
        self.high_cardinality_encoder = high_cardinality_encoder or "target"
        self.numeric_imputer = numeric_imputer or "simple"
        self.numeric_threshold = numeric_threshold
        self.cardinality_threshold = cardinality_threshold
        self.verbose = verbose
        self.random_state = random_state or 0
        self.n_jobs = n_jobs
        if cache_transformations:
            self._memory = joblib.Memory("transformation_cache", verbose=self.verbose)
        else:
            self._memory = None

        self._poniard: PoniardBaseEstimator | None = None

    def build(
        self,
        X: pd.DataFrame | np.ndarray | list | None = None,
        y: pd.DataFrame | np.ndarray | list | None = None,
    ) -> PoniardPreprocessor:
        """Builds the preprocessor according to the input data.

        Gets the data from the main `PoniardBaseEstimator` (if available) or processes the input data,
        calls the type inference method, sets up the transformers and builds the pipeline.

        Parameters
        ----------
        X :
            Features
        y :
            Target.
        """
        if not self.task and not self._poniard:
            raise ValueError(
                "A task must be defined on initialization if not used within a Poniard estimator."
            )

        self._setup_data(X=X, y=y)
        X = self.X

        try:
            numeric = self.feature_types["numeric"]
            categorical_high = self.feature_types["categorical_high"]
            categorical_low = self.feature_types["categorical_low"]
            datetime = self.feature_types["datetime"]
        except AttributeError:
            numeric, categorical_high, categorical_low, datetime = self._infer_dtypes()

        self.task = self.task or self._poniard.poniard_task
        try:
            self.target_info = self._poniard.target_info
        except AttributeError:
            self.target_info = get_target_info(self.y, self.task)
        (
            numeric_preprocessor,
            cat_low_preprocessor,
            cat_high_preprocessor,
            datetime_preprocessor,
        ) = self._setup_transformers()

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
        non_empty_transformers = [
            x for x in type_preprocessor.transformers if x[2] != []
        ]
        type_preprocessor.transformers = non_empty_transformers
        if len(type_preprocessor.transformers) == 1:
            type_preprocessor = type_preprocessor.transformers[0][1]
        preprocessor = Pipeline(
            [
                ("type_preprocessor", type_preprocessor),
                ("remove_invariant", VarianceThreshold()),
            ],
            memory=self._memory,
        )
        self.preprocessor = preprocessor
        return self

    def _setup_data(
        self,
        X: pd.DataFrame | np.ndarray | list | None = None,
        y: pd.DataFrame | np.ndarray | list | None = None,
    ) -> PoniardPreprocessor:
        if (X is None or y is None) and self._poniard is None:
            raise NotImplementedError(
                "Both X and y need to be passed if not using the "
                "preprocessor within a Poniard estimator."
            )
        elif self._poniard is not None:
            if X is not None or y is not None:
                warnings.warn(
                    "Input data will be ignored since the preprocessor is working "
                    "within a Poniard estimator",
                    stacklevel=2,
                )
            self.X = self._poniard.X
            self.y = self._poniard.y
        else:
            if pl is not None and isinstance(X, (pl.DataFrame, pl.Series)):
                X = X.to_pandas()
            if pl is not None and isinstance(y, (pl.DataFrame, pl.Series)):
                y = y.to_pandas()
            if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)):
                X = np.array(X)
            if not isinstance(y, (pd.DataFrame, pd.Series, np.ndarray)):
                y = np.array(y)
            self.X = X
            self.y = y
        return self

    def _setup_transformers(self):
        if isinstance(self.scaler, TransformerMixin):
            scaler = self.scaler
        elif self.scaler == "standard":
            scaler = StandardScaler()
        elif self.scaler == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()

        target_is_multilabel = self.target_info["type_"] in [
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
                high_cardinality_encoder = TargetEncoder(cv=3)
        else:
            high_cardinality_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=99999
            )

        cat_date_imputer = SimpleImputer(strategy="most_frequent")

        if isinstance(self.numeric_imputer, TransformerMixin):
            num_imputer = self.numeric_imputer
        elif self.numeric_imputer == "iterative":
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
                    OneHotEncoder(
                        drop="if_binary", handle_unknown="ignore", sparse_output=False
                    ),
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
        return (
            numeric_preprocessor,
            cat_low_preprocessor,
            cat_high_preprocessor,
            datetime_preprocessor,
        )

    def _infer_dtypes(self) -> tuple[list, list, list, list]:
        """Infer feature types (numeric, low-cardinality categorical,
        high-cardinality categorical, datetime).

        Returns
        -------
        tuple[list, list, list, list]
            Four lists with column names or indices.
        """
        X = self.X
        numeric: list = []
        categorical_high: list = []
        categorical_low: list = []
        datetime_cols: list = []

        if not isinstance(self.cardinality_threshold, int):
            self.cardinality_threshold = int(self.cardinality_threshold * X.shape[0])
        if not isinstance(self.numeric_threshold, int):
            self.numeric_threshold = int(self.numeric_threshold * X.shape[0])

        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                dtype = X[col].dtype
                nunique = X[col].nunique()

                if pd.api.types.is_datetime64_any_dtype(dtype):
                    datetime_cols.append(col)
                elif pd.api.types.is_numeric_dtype(dtype):
                    if nunique > self.numeric_threshold:
                        numeric.append(col)
                    elif nunique > self.cardinality_threshold:
                        categorical_high.append(col)
                    else:
                        categorical_low.append(col)
                else:
                    # strings, objects, categorical, boolean
                    if nunique > self.cardinality_threshold:
                        categorical_high.append(col)
                    else:
                        categorical_low.append(col)
        else:
            for i in range(X.shape[1]):
                col = X[:, i]
                nunique = len(np.unique(col))
                if np.issubdtype(col.dtype, np.datetime64):
                    datetime_cols.append(i)
                elif np.issubdtype(col.dtype, np.number):
                    if nunique > self.numeric_threshold:
                        numeric.append(i)
                    elif nunique > self.cardinality_threshold:
                        categorical_high.append(i)
                    else:
                        categorical_low.append(i)
                else:
                    if nunique > self.cardinality_threshold:
                        categorical_high.append(i)
                    else:
                        categorical_low.append(i)

        self.feature_types = {
            "numeric": numeric,
            "categorical_high": categorical_high,
            "categorical_low": categorical_low,
            "datetime": datetime_cols,
        }
        self.inferred_types_df = pd.DataFrame.from_dict(
            self.feature_types, orient="index"
        ).T.fillna("")
        self._run_plugin_method_maybe("on_infer_types")
        return numeric, categorical_high, categorical_low, datetime_cols

    def _run_plugin_method_maybe(self, method: str, **kwargs):
        if self._poniard is not None:
            self._poniard._run_plugin_method(method, **kwargs)
        return

    def __repr__(self):
        return non_default_repr(self)
