import itertools
from typing import Sequence

import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer


def date_processor(X: np.array):
    output = []
    for col in range(X.shape[1]):
        series = pd.Series(X[:, col])
        output.append(
            pd.DataFrame(
                {
                    f"{col}_year": series.dt.year,
                    f"{col}_month": series.dt.month,
                    f"{col}_day": series.dt.day,
                    f"{col}_weekday": series.dt.weekday,
                }
            )
        )
    output = pd.concat(output, axis=1)
    return output


def date_feature_names(ft: FunctionTransformer, input_features: Sequence[str]):
    input_features = list(input_features)
    suffixes = ["_year", "_month", "_day", "_weekday"]
    product = itertools.product(input_features, suffixes)
    return [f"{feature[0]}{feature[1]}" for feature in product]


# USAGE
# data = pd.DataFrame({"A": list(range(100)), "B": pd.date_range(start="2020-01-01", freq="D", periods=100)})
# pre = ColumnTransformer([("date", FunctionTransformer(date_processor, feature_names_out=date_feature_names, validate=True), ["B"])], remainder="passthrough")
# pre.fit_transform(data)
