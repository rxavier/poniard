from __future__ import annotations

# This file is based on part of the Dython package (https://github.com/shakedzy/dython)
# Because I only needed Cramér's V, adding a whole dependency
# did not make sense.
import warnings
from typing import Literal

import numpy as np
import pandas as pd
import scipy.stats as ss


def cramers_v(
    x: pd.Series | np.ndarray | list,
    y: pd.Series | np.ndarray | list,
    bias_correction: bool = True,
    nan_strategy: Literal["replace", "drop"] = "replace",
    nan_replace_value: int | float = 0,
):
    """
    Calculates Cramer's V statistic for categorical-categorical association.

    This is a symmetric coefficient: V(x,y) = V(y,x)
    Original function taken from: https://stackoverflow.com/a/46498792/5863503
    Wikipedia: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V

    Parameters
    ----------
    x :
        A sequence of categorical measurements
    y :
        A sequence of categorical measurements
    bias_correction :
        Use bias correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328.
    nan_strategy :
        How to handle missing values: can be either 'drop' to remove samples
        with missing values, or 'replace' to replace all missing values with
        the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value :
        The value used to replace missing values with. Only applicable when
        nan_strategy is set to 'replace'.

    Returns
    -------
    float in the range of [0,1]
    """
    if nan_strategy == "replace":
        x, y = replace_nan_with_value(x, y, nan_replace_value)
    elif nan_strategy == "drop":
        x, y = remove_incomplete_samples(x, y)
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    if bias_correction:
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        if min((kcorr - 1), (rcorr - 1)) == 0:
            warnings.warn(
                "Unable to calculate Cramer's V using bias correction. Consider using bias_correction=False",
                RuntimeWarning,
            )
            return np.nan
        else:
            return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    else:
        return np.sqrt(phi2 / min(k - 1, r - 1))


def replace_nan_with_value(x, y, value):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_arr[np.isnan(x_arr)] = value
    y_arr[np.isnan(y_arr)] = value
    return x_arr, y_arr


def remove_incomplete_samples(x, y):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    if isinstance(x, list):
        return x_arr[mask].tolist(), y_arr[mask].tolist()
    return x_arr[mask], y_arr[mask]
