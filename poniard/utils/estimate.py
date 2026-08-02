from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target

try:
    import polars as pl
except ImportError:
    pl = None

Task = Literal["regression", "classification"]
"""The two supported tasks."""


def coerce_input(X):
    """Convert polars input to pandas and coerce other sequences to numpy.

    Shared by the estimator and preprocessor configuration paths so both
    accept the same input types (polars, pandas, numpy, lists).
    """
    if pl is not None and isinstance(X, (pl.DataFrame, pl.Series)):
        X = X.to_pandas()
    if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)):
        X = np.array(X)
    return X


def get_target_info(y: pd.DataFrame | pd.Series | np.ndarray, task: Task) -> dict:
    """Return a dict containing basic information about the target array."""
    y = np.array(y)
    type_of_target_ = type_of_target(y)
    # sklearn's type_of_target incorrectly assumes that int-like float arrays are always
    # multiclass. This doesn't make sense in general, and for example, the diabetes
    # dataset is 'multiclass' according to this function when it should be 'continuous'.
    if type_of_target_ == "multiclass" and task == "regression":
        type_of_target_ = "continuous"
    return dict(type_=type_of_target_, ndim=y.ndim, shape=y.shape, nunique=np.unique(y).size)


def element_to_list_maybe(obj):
    if (isinstance(obj, (Sequence, dict)) and not isinstance(obj, str)) or obj is None:
        return obj
    else:
        return [obj]
