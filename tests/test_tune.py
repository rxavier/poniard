import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from poniard import PoniardClassifier


@pytest.mark.parametrize(
    "grid,mode",
    [
        (None, "grid"),
        ({"LogisticRegression__C": np.linspace(0.1, 1, num=4)}, "halving"),
        ({"LogisticRegression__penalty": ["l2", None]}, "random"),
    ],
)
def test_tune(grid, mode):
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1])
    x = pd.DataFrame(np.random.normal(size=(len(y), 5)))
    clf = PoniardClassifier(
        estimators=[LogisticRegression()],
        random_state=True,
    )
    clf.setup(x, y)
    clf.fit(x, y)
    clf.tune_estimator("LogisticRegression", x, y, grid, mode)
    clf.fit(x, y)
    assert clf.get_results().shape[0] == 3
    assert clf.get_results().isna().sum().sum() == 0
