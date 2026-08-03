"""Per-estimator preprocessors: the default profile and the native HGB profile.

This example uses data with missing values, low- and high-cardinality
categoricals, and a datetime column, then shows:

1. The default preprocessor: median imputation with missingness indicators,
   one-hot encoding (with min_frequency), and scaling.
2. Routing HistGradientBoosting to the "native" profile via preprocessor_map,
   which leaves numeric/datetime untouched and ordinal-encodes categoricals to
   pandas category dtype so HGB handles them natively.
3. Registering a custom template and assigning it at runtime.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from poniard import PoniardClassifier
from poniard.preprocessing import PoniardPreprocessor

rng = np.random.RandomState(42)
n = 120
X = pd.DataFrame(
    {
        "num": rng.normal(size=n),
        "num_missing": np.where(rng.rand(n) < 0.2, np.nan, rng.normal(size=n)),
        "cat_low": rng.choice(["a", "b", "c"], size=n),
        "cat_high": [f"c_{i}" for i in range(n)],
        "dt": pd.date_range("2020-01-01", periods=n, freq="D"),
    }
)
y = make_classification(n_samples=n, n_features=5, random_state=42)[1]

# 1. Default preprocessor on all estimators.
clf = PoniardClassifier(
    estimators=[LogisticRegression(max_iter=1000), HistGradientBoostingClassifier()],
    cv=3,
    random_state=0,
)
clf.setup(X, y, show_info=False)
print("Registered preprocessors:", list(clf.preprocessors))
print("Default profile steps:", [s for s, _ in clf.preprocessors["default"].steps])
print()

# 2. Route HGB to the "native" profile. It keeps NaNs and categoricals raw
#    (as category dtype) instead of imputing/one-hot encoding them away.
clf = PoniardClassifier(
    estimators=[
        HistGradientBoostingClassifier(),
        LogisticRegression(max_iter=1000),
    ],
    cv=3,
    random_state=0,
    preprocessor_map={"HistGradientBoostingClassifier": "native"},
)
clf.fit(X, y, show_info=False)
print("Preprocessor map:", clf.preprocessor_map)
hgb_est = clf.pipelines["HistGradientBoostingClassifier"].named_steps[
    "HistGradientBoostingClassifier"
]
print(
    "HGB categorical_features (auto-set):",
    hgb_est.categorical_features,
)
print(clf.get_results()[["test_roc_auc", "test_neg_log_loss"]].round(3))
print()

# 3. Register a custom template and assign it at runtime. A registered template
#    must handle the full feature set, so build one from a configured
#    PoniardPreprocessor (here: min-max scaling instead of the standard scaler).
clf = PoniardClassifier(
    estimators=[HistGradientBoostingClassifier(), LogisticRegression(max_iter=1000)],
    cv=3,
    random_state=0,
    preprocessor_map={"HistGradientBoostingClassifier": "native"},
)
clf.setup(X, y, show_info=False)
minmax_pp = PoniardPreprocessor(scaler="minmax")
minmax_pp.build(X, y, task="classification")
clf.add_preprocessor("minmax", minmax_pp.preprocessor)
clf.set_preprocessor("LogisticRegression", "minmax")
print("Map after set_preprocessor:", clf.preprocessor_map)
clf.fit(X, y, show_info=False)
assert not clf.get_results().isna().any().any()
print("Mixed-preprocessor experiment scored cleanly.")
