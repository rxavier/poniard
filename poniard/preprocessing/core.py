from __future__ import annotations

import os
import tempfile
import warnings
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)
from sklearn.utils.validation import _check_feature_names_in

from ..utils.estimate import Task, coerce_input, get_target_info
from ..utils.utils import non_default_repr
from .datetime import DatetimeEncoder

__all__ = ["PoniardPreprocessor", "infer_feature_types"]


class _ToCategorical(BaseEstimator, TransformerMixin):
    """Cast the output of a pandas-returning encoder to ``category`` dtype.

    Used by the ``"native"`` profile so HistGradientBoosting's
    ``categorical_features="from_dtype"`` recognizes the ordinal-encoded
    columns as categorical. Missing values (NaN) are preserved as missing.
    """

    def fit(self, X, y=None) -> _ToCategorical:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.astype({col: "category" for col in X.columns})
        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            input_features = _check_feature_names_in(self, input_features)
        return np.asarray(input_features, dtype=object)


def infer_feature_types(
    X: pd.DataFrame | np.ndarray,
    numeric_threshold: int | float,
    cardinality_threshold: int | float,
) -> dict[str, list]:
    """Infer feature types from the data.

    Features are classified as numeric, low-cardinality categorical,
    high-cardinality categorical or datetime. Float thresholds are
    interpreted as a fraction of the number of rows.

    Parameters
    ----------
    X :
        Features (pandas DataFrame or numpy array).
    numeric_threshold :
        Number of unique values above which a number-like feature is treated
        as numeric. If float, `numeric_threshold * n_samples`.
    cardinality_threshold :
        Non-number features with more unique values than this are treated as
        high-cardinality (ordinal/target encoded). If float, it is
        `cardinality_threshold * n_samples`.

    Returns
    -------
    dict[str, list]
        Keys ``numeric``, ``categorical_high``, ``categorical_low``,
        ``datetime``; values are column names (DataFrame input) or indices
        (array input).
    """
    numeric: list = []
    categorical_high: list = []
    categorical_low: list = []
    datetime_cols: list = []

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    cardinality_threshold = (
        cardinality_threshold
        if isinstance(cardinality_threshold, int)
        else int(cardinality_threshold * X.shape[0])
    )
    numeric_threshold = (
        numeric_threshold
        if isinstance(numeric_threshold, int)
        else int(numeric_threshold * X.shape[0])
    )

    for col in X.columns:
        dtype = X[col].dtype
        nunique = X[col].nunique()

        if pd.api.types.is_datetime64_any_dtype(dtype):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            if nunique > numeric_threshold:
                numeric.append(col)
            elif nunique > cardinality_threshold:
                categorical_high.append(col)
            else:
                categorical_low.append(col)
        else:
            # strings, objects, categorical, boolean
            if nunique > cardinality_threshold:
                categorical_high.append(col)
            else:
                categorical_low.append(col)

    return {
        "numeric": numeric,
        "categorical_high": categorical_high,
        "categorical_low": categorical_low,
        "datetime": datetime_cols,
    }


class PoniardPreprocessor:
    """Base preprocessor that builds an easily modifiable pipeline based
    on feature data types.

    Parameters
    ----------
    profile :
        Which transformation profile to use. ``"default"`` imputes, scales and one-hot/
        target-encodes for generic estimators. ``"native"`` leaves numeric and datetime
        features untouched and ordinal-encodes categoricals as pandas ``category`` dtype,
        for estimators that handle missing values and categoricals natively (e.g.
        HistGradientBoosting). No scaling, no imputation, no variance threshold.
    scaler :
        Numeric scaler method. Either "standard", "minmax", "robust" or scikit-learn Transformer.
    high_cardinality_encoder :
        Encoder for categorical features with high cardinality. Either "target" or "ordinal",
        or scikit-learn Transformer.
    numeric_imputer :
        Imputation method for numeric features. Either "mean", "median" (default), "iterative"
        or scikit-learn Transformer. Numeric imputation also emits a missingness indicator
        column per input feature.
    categorical_imputer :
        Imputer for categorical features. Either "most_frequent" or "constant" (which fills
        with the string ``"missing"`` so one-hot encoding surfaces missingness), or a
        scikit-learn Transformer.
    cyclical_datetime :
        Whether to encode periodic datetime levels (hour, month, weekday, ...) as sin/cos
        pairs instead of plain integers.
    ohe_min_frequency :
        Minimum frequency for categories kept as separate one-hot columns. Categories rarer
        than this collapse into sklearn's infrequent bucket. An int is an absolute count; a
        float is a fraction of samples. ``None`` keeps every observed category.
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
    cache_dir :
        Directory used to cache transformations when ``cache_transformations`` is True. If None
        (the default), a temporary directory is created and cleaned up when the preprocessor is
        garbage collected. If a path is provided, the user is responsible for its contents.
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
        task: Task | None = None,
        profile: Literal["default", "native"] = "default",
        scaler: Literal["standard", "minmax", "robust"] | TransformerMixin | None = None,
        high_cardinality_encoder: (Literal["target", "ordinal"] | TransformerMixin | None) = None,
        numeric_imputer: Literal["iterative", "mean", "median"] | TransformerMixin | None = None,
        categorical_imputer: Literal["most_frequent", "constant"] | TransformerMixin | None = None,
        cyclical_datetime: bool = False,
        ohe_min_frequency: int | float | None = 5,
        numeric_threshold: int | float = 0.1,
        cardinality_threshold: int | float = 20,
        verbose: bool = False,
        random_state: int | None = None,
        n_jobs: int | None = None,
        cache_transformations: bool = False,
        cache_dir: str | os.PathLike | None = None,
    ):
        self._init_params = {
            "task": task,
            "profile": profile,
            "scaler": scaler,
            "high_cardinality_encoder": high_cardinality_encoder,
            "numeric_imputer": numeric_imputer,
            "categorical_imputer": categorical_imputer,
            "cyclical_datetime": cyclical_datetime,
            "ohe_min_frequency": ohe_min_frequency,
            "numeric_threshold": numeric_threshold,
            "cardinality_threshold": cardinality_threshold,
            "verbose": verbose,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "cache_transformations": cache_transformations,
            "cache_dir": cache_dir,
        }
        self.task = task
        self.profile = profile
        self.scaler = scaler or "standard"
        self.high_cardinality_encoder = high_cardinality_encoder or "target"
        self.numeric_imputer = numeric_imputer or "median"
        self.categorical_imputer = categorical_imputer or "most_frequent"
        self.cyclical_datetime = cyclical_datetime
        self.ohe_min_frequency = ohe_min_frequency
        self.numeric_threshold = numeric_threshold
        self.cardinality_threshold = cardinality_threshold
        self.verbose = verbose
        self.random_state = random_state or 0
        self.n_jobs = n_jobs
        self._cache_tempdir = None
        if cache_transformations:
            if cache_dir is None:
                self._cache_tempdir = tempfile.TemporaryDirectory(prefix="poniard_cache_")
                cache_dir = self._cache_tempdir.name
            self._memory = joblib.Memory(str(cache_dir), verbose=self.verbose)
        else:
            self._memory = None

    def build(
        self,
        X: pd.DataFrame | np.ndarray | list | None = None,
        y: pd.DataFrame | np.ndarray | list | None = None,
        task: Task | None = None,
        target_info: dict | None = None,
        feature_types: dict | None = None,
    ) -> PoniardPreprocessor:
        """Builds the preprocessor according to the input data.

        Processes the input data, calls the type inference method, sets up the transformers
        and builds the pipeline.

        Parameters
        ----------
        X :
            Features
        y :
            Target.
        task :
            Task type ("classification" or "regression"). Overrides the task set at init.
        target_info :
            Target info dict. If None, computed from y and task.
        feature_types :
            Explicit feature type assignment. If given, type inference is skipped and
            ``inferred_types_df`` is refreshed from this mapping.
        """
        if task:
            self.task = task
        if not self.task:
            raise ValueError("A task must be defined on initialization or passed to build().")

        self._setup_data(X=X, y=y)
        X = self.X

        if feature_types is not None:
            self.feature_types = feature_types
            self.inferred_types_df = self._feature_types_df(feature_types)
            numeric = feature_types["numeric"]
            categorical_high = feature_types["categorical_high"]
            categorical_low = feature_types["categorical_low"]
            datetime = feature_types["datetime"]
        else:
            try:
                numeric = self.feature_types["numeric"]
                categorical_high = self.feature_types["categorical_high"]
                categorical_low = self.feature_types["categorical_low"]
                datetime = self.feature_types["datetime"]
            except AttributeError:
                numeric, categorical_high, categorical_low, datetime = self._infer_dtypes()

        if target_info:
            self.target_info = target_info
        else:
            self.target_info = get_target_info(self.y, self.task)
        (
            numeric_preprocessor,
            cat_low_preprocessor,
            cat_high_preprocessor,
            datetime_preprocessor,
        ) = self._setup_transformers()

        if isinstance(X, pd.DataFrame):
            transformers = [
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
            ]
        else:
            transformers = [
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
            ]
            if np.issubdtype(X.dtype, np.datetime64):
                transformers = [("datetime_preprocessor", datetime_preprocessor, datetime)]
        type_preprocessor = ColumnTransformer(
            [t for t in transformers if t[2] != []],
            n_jobs=self.n_jobs,
            verbose_feature_names_out=False,
        )
        type_preprocessor.set_output(transform="pandas")
        preprocessor_steps = [("type_preprocessor", type_preprocessor)]
        if self.profile != "native":
            preprocessor_steps.append(("remove_invariant", VarianceThreshold()))
        preprocessor = Pipeline(
            preprocessor_steps,
            memory=self._memory,
        )
        preprocessor.set_output(transform="pandas")
        self.preprocessor = preprocessor
        return self

    def _setup_data(
        self,
        X: pd.DataFrame | np.ndarray | list | None = None,
        y: pd.DataFrame | np.ndarray | list | None = None,
    ) -> PoniardPreprocessor:
        if X is not None and y is not None:
            self.X = coerce_input(X)
            self.y = coerce_input(y)
        elif not hasattr(self, "X") or not hasattr(self, "y"):
            raise ValueError(
                "X and y must be passed to build() (or set by a previous build() call)."
            )
        return self

    def _setup_transformers(self):
        if self.profile == "native":
            return self._setup_native_transformers()

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
                    handle_unknown="use_encoded_value", unknown_value=np.nan
                )
            else:
                high_cardinality_encoder = TargetEncoder(cv=3)
        else:
            high_cardinality_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=np.nan
            )

        if isinstance(self.categorical_imputer, TransformerMixin):
            cat_imputer = self.categorical_imputer
        elif self.categorical_imputer == "constant":
            cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
        else:
            cat_imputer = SimpleImputer(strategy="most_frequent")

        if isinstance(self.numeric_imputer, TransformerMixin):
            num_imputer = self.numeric_imputer
        elif self.numeric_imputer == "iterative":
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer

            num_imputer = IterativeImputer(random_state=self.random_state)
        else:
            num_imputer = SimpleImputer(strategy=self.numeric_imputer, add_indicator=True)

        numeric_preprocessor = Pipeline([("numeric_imputer", num_imputer), ("scaler", scaler)])
        cat_low_preprocessor = Pipeline(
            [
                ("categorical_imputer", cat_imputer),
                (
                    "one-hot_encoder",
                    OneHotEncoder(
                        drop="if_binary",
                        handle_unknown="ignore",
                        sparse_output=False,
                        min_frequency=self.ohe_min_frequency,
                    ),
                ),
            ]
        )
        cat_high_preprocessor = Pipeline(
            [
                ("categorical_imputer", cat_imputer),
                (
                    "high_cardinality_encoder",
                    high_cardinality_encoder,
                ),
            ],
        )
        datetime_preprocessor = Pipeline(
            [
                ("datetime_encoder", DatetimeEncoder(cyclical=self.cyclical_datetime)),
                ("datetime_imputer", SimpleImputer(strategy="median")),
                ("scaler", scaler),
            ],
        )
        return (
            numeric_preprocessor,
            cat_low_preprocessor,
            cat_high_preprocessor,
            datetime_preprocessor,
        )

    def _setup_native_transformers(self):
        """Transformers for the ``"native"`` profile.

        Numeric and datetime features pass through untouched (tree models learn
        NaN split directions themselves), and categoricals are ordinal-encoded
        to pandas ``category`` dtype so estimators with
        ``categorical_features="from_dtype"`` split on them directly.
        """
        categorical_preprocessor = Pipeline(
            [
                (
                    "ordinal_encoder",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
                ),
                ("to_categorical", _ToCategorical()),
            ]
        )
        return (
            "passthrough",
            categorical_preprocessor,
            categorical_preprocessor,
            DatetimeEncoder(cyclical=self.cyclical_datetime),
        )

    def _infer_dtypes(self) -> tuple[list, list, list, list]:
        """Infer feature types for ``self.X`` and store them.

        Returns
        -------
        tuple[list, list, list, list]
            Four lists with column names or indices.
        """
        self.feature_types = infer_feature_types(
            self.X, self.numeric_threshold, self.cardinality_threshold
        )
        self.inferred_types_df = self._feature_types_df(self.feature_types)
        return (
            self.feature_types["numeric"],
            self.feature_types["categorical_high"],
            self.feature_types["categorical_low"],
            self.feature_types["datetime"],
        )

    @staticmethod
    def _feature_types_df(feature_types: dict) -> pd.DataFrame:
        return pd.DataFrame.from_dict(feature_types, orient="index").T.fillna("")

    def __repr__(self):
        return non_default_repr(self)
