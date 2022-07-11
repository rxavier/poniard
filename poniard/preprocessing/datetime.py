from __future__ import annotations
from typing import Sequence, Optional, Union, List
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DateLevel(Enum):
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    WEEKDAY = "weekday"
    HOUR = "hour"
    MINUTE = "minute"
    SECOND = "second"
    MICROSECOND = "microsecond"
    NANOSECOND = "nanosecond"


class DatetimeEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self, levels: Optional[Sequence[DateLevel]] = None, fmt: Optional[str] = None
    ):
        self.levels = levels
        self.fmt = fmt

    def _more_tags(self):
        return {
            "X_types": ["2darray", "string"],
            "preserves_dtype": [],
        }

    def fit(self, X: Union[pd.DataFrame, np.ndarray, List], y=None) -> DatetimeEncoder:
        if isinstance(X, pd.DataFrame):
            if X.dtypes.nunique() > 1 and not all(
                pd.api.types.is_datetime64_any_dtype(dt) for dt in X.dtypes
            ):
                raise ValueError(
                    "If data contains more than one type, they all have to be datetime64 (any)."
                )
            elif X.dtypes[0] in (object, str):
                X = X.apply(pd.to_datetime, format=self.fmt)
            self.colnames_ = X.columns
        X = self._validate_data(X=X, y=None, force_all_finite="allow-nan")

        self.valid_features_ = {}
        if self.levels:
            levels = self.levels
        else:
            levels = list(DateLevel)
        for col in range(X.shape[1]):
            valid_single_feature = []
            for level in levels:
                dates = pd.DatetimeIndex(X[:, col])
                if dates.tz:
                    dates = dates.tz_convert(None)
                encoded = getattr(dates, level.value)
                if encoded.nunique() > 1:
                    valid_single_feature.append(level)
            self.valid_features_.update({col: valid_single_feature})

        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = sum(
            [len(features) for features in self.valid_features_.values()]
        )
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if X.dtypes.nunique() > 1 and not all(
                pd.api.types.is_datetime64_any_dtype(dt) for dt in X.dtypes
            ):
                raise ValueError(
                    "If data contains more than one type, they all have to be datetime64 (any)."
                )
            elif X.dtypes[0] in (object, str):
                X = X.apply(pd.to_datetime, format=self.fmt)
        X = self._validate_data(X=X, y=None, force_all_finite="allow-nan")

        all_encoded = []
        for col, levels in self.valid_features_.items():
            for level in levels:
                dates = pd.DatetimeIndex(X[:, col])
                if dates.tz:
                    dates = dates.tz_convert(None)
                encoded = getattr(dates, level.value)
                all_encoded.append(encoded)
        output = np.stack(all_encoded, axis=1)
        return output

    def get_feature_names_out(self, input_features=None) -> List[str]:
        feature_names = []
        colnames_ = getattr(self, "colnames_", None)
        for i in self.valid_features_.keys():
            prefix = str(i) if colnames_ is None else colnames_[i]
            for feature in self.valid_features_[i]:
                feature_names.append(f"{prefix}_{feature.value}")
        return feature_names

    def get_feature_names(self, input_features=None) -> List[str]:
        return self.get_feature_names_out()


# TODO: TEST CV
# TODO: RUN SKLEARN CHECKS
