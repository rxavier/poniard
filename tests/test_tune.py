import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from poniard import PoniardClassifier


@pytest.mark.parametrize(
    "grid,mode",
    [
        (None, "grid"),
        ({"LogisticRegression__C": np.linspace(0.1, 1, num=4)}, "halving"),
        ({"LogisticRegression__penalty": ["l1", "none"]}, "random"),
    ],
)
def test_tune(grid, mode):
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1])
    x = pd.DataFrame(np.random.normal(size=(len(y), 5)))
    reg = PoniardClassifier(
        estimators=[LogisticRegression()],
        random_state=True,
    )
    reg.fit(x, y)
    reg.tune_estimator("LogisticRegression", grid, mode)
    reg.fit_new()
    assert reg.show_results().shape[0] == 3
    assert reg.show_results().isna().sum().sum() == 0
