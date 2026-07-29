from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target


def get_target_info(y: pd.DataFrame | pd.Series | np.ndarray, task: str) -> dict:
    """Return a dict containing basic information about the target array."""
    y = np.array(y)
    type_of_target_ = type_of_target(y)
    # sklearn's type_of_target incorrectly assumes that int-like float arrays are always
    # multiclass. This doesn't make sense in general, and for example, the diabetes
    # dataset is 'multiclass' according to this function when it should be 'continuous'.
    if type_of_target_ == "multiclass" and task == "regression":
        type_of_target_ = "continuous"
    return dict(
        type_=type_of_target_, ndim=y.ndim, shape=y.shape, nunique=np.unique(y).size
    )


def element_to_list_maybe(obj):
    if (isinstance(obj, (Sequence, dict)) and not isinstance(obj, str)) or obj is None:
        return obj
    else:
        return [obj]
