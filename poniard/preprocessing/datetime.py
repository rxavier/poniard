from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data

__all__ = ['DateLevel', 'DatetimeEncoder']


class DateLevel(Enum):
    """An enum representing different date levels."""

    YEAR = "year"
    QUARTER = "quarter"
    MONTH = "month"
    DAY = "day"
    HOUR = "hour"
    MINUTE = "minute"
    SECOND = "second"
    MICROSECOND = "microsecond"
    NANOSECOND = "nanosecond"
    WEEKDAY = "weekday"
    DAYOFYEAR = "dayofyear"
    DAYSINMONTH = "daysinmonth"


class DatetimeEncoder(BaseEstimator, TransformerMixin):
    """An encoder for datetime columns that outputs integer features.

    `levels` is a list of `DateLevel` that define which date features to extract, i.e,
    [`DateLevel.HOUR`, `DateLevel.MINUTE`] will extract hours and minutes. If left to the
    default `None`, all available features will be extracted initially, but zero variance
    features will be dropped (for example, because the dates don't have seconds).

    Parameters
    ----------
    levels :
        Date features to extract.
    fmt :
        Date format for string conversion if inputs are not datetime-like objects.
        Follows standard Pandas/stdlib formatting, e.g. '%Y-%m-%d %H:%M:%S'.
    """

    def __init__(
        self, levels: Sequence[DateLevel] | None = None, fmt: str | None = None
    ):
        self.levels = levels
        self.fmt = fmt

    def fit(self, X: pd.DataFrame | np.ndarray | list, y=None) -> DatetimeEncoder:
        """Fit the DatetimeEncoder.

        Parameters
        ----------
        X :
            Datetime-like features.
        y :
            Unused.

        Returns
        -------
        DatetimeEncoder
            Fitted `DatetimeEncoder`.
        """
        if isinstance(X, pd.DataFrame):
            if X.dtypes.nunique() > 1 and not all(
                pd.api.types.is_datetime64_any_dtype(dt) for dt in X.dtypes
            ):
                raise ValueError(
                    "If data contains more than one type, "
                    "they all have to be datetime64 (any)."
                )
            elif X.dtypes.iloc[0] in (object, str):
                X = X.apply(pd.to_datetime, format=self.fmt)
            input_names = list(X.columns)
        else:
            input_names = [str(i) for i in range(X.shape[1])]
        X = validate_data(self, X=X, y=None, ensure_all_finite="allow-nan")

        self.valid_features_ = {}
        levels = self.levels if self.levels else list(DateLevel)
        for col in range(X.shape[1]):
            valid_single_feature = []
            for level in levels:
                dates = pd.DatetimeIndex(X[:, col])
                if dates.tz:
                    dates = dates.tz_convert(None)
                encoded = getattr(dates, level.value)
                if encoded.nunique() > 1:
                    valid_single_feature.append(level)
            self.valid_features_[col] = valid_single_feature

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = input_names
        self.n_features_out_ = sum(
            len(features) for features in self.valid_features_.values()
        )
        return self

    def transform(self, X: pd.DataFrame | np.ndarray | list) -> np.ndarray:
        """Apply transformation. Will ignore zero variance features seen during fit.

        Parameters
        ----------
        X :
            The data to encode.

        Returns
        -------
        np.ndarray
            Transformed input.
        """
        if isinstance(X, pd.DataFrame):
            if X.dtypes.nunique() > 1 and not all(
                pd.api.types.is_datetime64_any_dtype(dt) for dt in X.dtypes
            ):
                raise ValueError(
                    "If data contains more than one type, "
                    "they all have to be datetime64 (any)."
                )
            elif X.dtypes.iloc[0] in (object, str):
                X = X.apply(pd.to_datetime, format=self.fmt)
        X = validate_data(self, X=X, y=None, ensure_all_finite="allow-nan", reset=False)

        all_encoded = []
        for col, levels in self.valid_features_.items():
            for level in levels:
                dates = pd.DatetimeIndex(X[:, col])
                if dates.tz:
                    dates = dates.tz_convert(None)
                encoded = getattr(dates, level.value)
                all_encoded.append(encoded)
        return np.column_stack(all_encoded)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get feature names for output."""
        feature_names = []
        input_names = getattr(self, "feature_names_in_", None)
        for col, levels in self.valid_features_.items():
            prefix = str(col) if input_names is None else input_names[col]
            for level in levels:
                feature_names.append(f"{prefix}_{level.value}")
        return feature_names
