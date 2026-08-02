from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data

__all__ = ["DatetimeEncoder"]


def _is_string_typed(X: pd.DataFrame) -> bool:
    """Whether the first column is string-like (object, str, or pandas ``string``)."""
    return pd.api.types.is_string_dtype(X.dtypes.iloc[0]) and not isinstance(
        X.dtypes.iloc[0], pd.CategoricalDtype
    )


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
    cyclical :
        Whether to emit sin/cos pairs for periodic levels (hour, minute, second, month,
        quarter, weekday, dayofyear) instead of plain integer values, so wrap-around
        (hour 23 to 0) is visible to models.
    """

    _PERIODS: dict[DateLevel, int] = {
        DateLevel.HOUR: 24,
        DateLevel.MINUTE: 60,
        DateLevel.SECOND: 60,
        DateLevel.QUARTER: 4,
        DateLevel.MONTH: 12,
        DateLevel.WEEKDAY: 7,
        DateLevel.DAYOFYEAR: 366,
        DateLevel.MICROSECOND: 1_000_000,
        DateLevel.NANOSECOND: 1_000_000_000,
    }

    def __init__(
        self,
        levels: Sequence[DateLevel] | None = None,
        fmt: str | None = None,
        cyclical: bool = False,
    ):
        self.levels = levels
        self.fmt = fmt
        self.cyclical = cyclical

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
                    "If data contains more than one type, they all have to be datetime64 (any)."
                )
            elif _is_string_typed(X):
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
            sum(2 if self.cyclical and level in self._PERIODS else 1 for level in features)
            for features in self.valid_features_.values()
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
                    "If data contains more than one type, they all have to be datetime64 (any)."
                )
            elif _is_string_typed(X):
                X = X.apply(pd.to_datetime, format=self.fmt)
        X = validate_data(self, X=X, y=None, ensure_all_finite="allow-nan", reset=False)

        all_encoded = []
        for col, levels in self.valid_features_.items():
            for level in levels:
                dates = pd.DatetimeIndex(X[:, col])
                if dates.tz:
                    dates = dates.tz_convert(None)
                encoded = getattr(dates, level.value)
                if self.cyclical and level in self._PERIODS:
                    angle = 2 * np.pi * encoded / self._PERIODS[level]
                    all_encoded.append(np.sin(angle))
                    all_encoded.append(np.cos(angle))
                else:
                    all_encoded.append(encoded)
        return np.column_stack(all_encoded)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Get feature names for output."""
        feature_names = []
        input_names = getattr(self, "feature_names_in_", None)
        for col, levels in self.valid_features_.items():
            prefix = str(col) if input_names is None else input_names[col]
            for level in levels:
                if self.cyclical and level in self._PERIODS:
                    feature_names.append(f"{prefix}_{level.value}_sin")
                    feature_names.append(f"{prefix}_{level.value}_cos")
                else:
                    feature_names.append(f"{prefix}_{level.value}")
        return np.asarray(feature_names)
