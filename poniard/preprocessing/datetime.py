from __future__ import annotations
from typing import Sequence, Optional, Union, List
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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
    """An encoder for datetime columns that outputs integer features

    `levels` is a list of :class:`DateLevel` that define which date features to extract, i.e,
    [`DateLevel.HOUR`, `DateLevel.MINUTE`] will extract hours and minutes. If left to the
    default `None`, all available features will be extracted initially, but zero variance
    features will be dropped (for example, because the dates don't have seconds).

    `fmt` is the format of the datetime string, used to parse from strings to date if inputs
    are note datetime-like objects. For example, '%Y-%m-%d %H:%M:%S'.

    Parameters
    ----------
    levels :
        Date features to extract.
    fmt :
        Date format for string conversion. Follows standard Pandas/stdlib formatting.
    """

    def __init__(
        self, levels: Optional[Sequence[DateLevel]] = None, fmt: Optional[str] = None
    ):

        self.levels = levels
        self.fmt = fmt

    def _more_tags(self):
        return {
            "X_types": ["2darray", "string"],
            "preserves_dtype": [],
            "allow_nan": True,
        }

    def fit(self, X: Union[pd.DataFrame, np.ndarray, List], y=None) -> DatetimeEncoder:
        """Fit the DatetimeEncoder.

        While this transformer is generally stateless, during meth:`fit` it checks whether any of
        the extracted features have zero variance (only one unique value) and sets those levels to be
        ignored, even during meth:`transform`.

        Parameters
        ----------
        X :
            Datetime-like features..
        y :
            Unused.

        Returns
        -------
        DatetimeEncoder
            Fitted DatetimeEncoder.
        """
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
        """Apply transformation. Will ignore zero variance features seen during meth:`fit`.

        Parameters
        ----------
        X :
            The data to encode.

        Returns
        -------
        X :
            Transformed input.
        """
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
        """Get feature names for output."""
        feature_names = []
        colnames_ = getattr(self, "colnames_", None)
        for i in self.valid_features_.keys():
            prefix = str(i) if colnames_ is None else colnames_[i]
            for feature in self.valid_features_[i]:
                feature_names.append(f"{prefix}_{feature.value}")
        return feature_names

    def get_feature_names(self, input_features=None) -> List[str]:
        return self.get_feature_names_out()
