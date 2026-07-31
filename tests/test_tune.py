import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from poniard import PoniardClassifier


def _xy():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1])
    x = pd.DataFrame(np.random.normal(size=(len(y), 5)))
    return x, y


@pytest.mark.parametrize(
    "grid,mode",
    [
        ({"LogisticRegression__C": np.linspace(0.1, 1, num=4)}, "grid"),
        ({"LogisticRegression__C": np.linspace(0.1, 1, num=4)}, "halving"),
        ({"LogisticRegression__penalty": ["l2", None]}, "random"),
    ],
)
def test_tune(grid, mode):
    x, y = _xy()
    clf = PoniardClassifier(
        estimators=[LogisticRegression()],
        random_state=0,
    )
    clf.setup(x, y)
    clf.fit(x, y)
    clf.tune_estimator("LogisticRegression", x, y, grid, mode)
    clf.fit(x, y)
    assert clf.get_results().shape[0] == 3
    assert clf.get_results().isna().sum().sum() == 0
    assert "LogisticRegression_tuned" in clf.pipelines


def test_tune_bare_param_names_are_prefixed():
    x, y = _xy()
    clf = PoniardClassifier(estimators=[LogisticRegression()], random_state=0)
    clf.setup(x, y)
    clf.fit(x, y)
    clf.tune_estimator("LogisticRegression", x, y, grid={"C": [0.1, 1.0]})
    results = clf.get_tuning_results("LogisticRegression_tuned")
    assert "LogisticRegression__C" in results["grid"]
    assert "C" not in results["grid"]
    assert any(k.endswith("__C") or k == "LogisticRegression__C" for k in results["best_params_"])


def test_tune_mixed_grid_keys():
    x, y = _xy()
    clf = PoniardClassifier(estimators=[LogisticRegression()], random_state=0)
    clf.setup(x, y)
    clf.fit(x, y)
    clf.tune_estimator(
        "LogisticRegression",
        x,
        y,
        grid={"C": [0.5, 1.0], "LogisticRegression__max_iter": [100, 200]},
    )
    grid = clf.get_tuning_results("LogisticRegression_tuned")["grid"]
    assert grid == {
        "LogisticRegression__C": [0.5, 1.0],
        "LogisticRegression__max_iter": [100, 200],
    }


def test_tune_grid_required():
    x, y = _xy()
    clf = PoniardClassifier(estimators=[LogisticRegression()], random_state=0)
    clf.setup(x, y)
    clf.fit(x, y)
    with pytest.raises(ValueError, match="grid"):
        clf.tune_estimator("LogisticRegression", x, y)
    with pytest.raises(ValueError, match="grid"):
        clf.tune_estimator("LogisticRegression", x, y, grid={})


def test_tune_unknown_estimator():
    x, y = _xy()
    clf = PoniardClassifier(estimators=[LogisticRegression()], random_state=0)
    clf.setup(x, y)
    clf.fit(x, y)
    with pytest.raises(KeyError, match="Unknown estimator"):
        clf.tune_estimator("Nope", x, y, grid={"C": [1.0]})


def test_tune_rejects_name_collision():
    x, y = _xy()
    clf = PoniardClassifier(estimators=[LogisticRegression()], random_state=0)
    clf.setup(x, y)
    clf.fit(x, y)
    clf.tune_estimator("LogisticRegression", x, y, grid={"C": [0.1, 1.0]})
    with pytest.raises(ValueError, match="already exists"):
        clf.tune_estimator("LogisticRegression", x, y, grid={"C": [0.2, 2.0]})


def test_tune_custom_name_and_get_results():
    x, y = _xy()
    clf = PoniardClassifier(estimators=[LogisticRegression()], random_state=0)
    clf.setup(x, y)
    clf.fit(x, y)
    clf.tune_estimator(
        "LogisticRegression",
        x,
        y,
        grid={"C": [0.1, 1.0]},
        tuned_estimator_name="lr_search",
    )
    out = clf.get_tuning_results("lr_search")
    assert out["baseline"] == "LogisticRegression"
    assert out["mode"] == "grid"
    assert isinstance(out["best_params_"], dict)
    assert isinstance(out["best_score_"], float)
    assert isinstance(out["search"], GridSearchCV)
    assert "lr_search" in clf.pipelines
    # single tune → bare dict without name key
    assert clf.get_tuning_results()["baseline"] == "LogisticRegression"


def test_get_tuning_results_requires_tune():
    x, y = _xy()
    clf = PoniardClassifier(estimators=[LogisticRegression()], random_state=0)
    clf.setup(x, y)
    clf.fit(x, y)
    with pytest.raises(ValueError, match="No tuning results"):
        clf.get_tuning_results()
