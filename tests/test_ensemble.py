import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import LinearSVR

from poniard import PoniardRegressor, PoniardClassifier


@pytest.mark.parametrize(
    "method,estimator_names,top_n,sort_by",
    [
        ("stacking", ["LinearRegression", "DecisionTreeRegressor"], None, None),
        ("voting", None, 2, "test_r2"),
        ("voting", None, 2, None),
    ],
)
def test_ensemble(method, estimator_names, top_n, sort_by):
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])
    x = pd.DataFrame(np.random.normal(size=(len(y), 5)))
    reg = PoniardRegressor(
        estimators=[DecisionTreeRegressor(), LinearRegression(), LinearSVR()],
        random_state=True,
    )
    reg.setup(x, y)
    reg.fit()
    reg.build_ensemble(
        method=method, estimator_names=estimator_names, top_n=top_n, sort_by=sort_by
    )
    reg.fit()
    results = reg.get_results()
    ensemble_class_name = method.capitalize() + "Regressor"
    ensemble = reg.get_estimator(ensemble_class_name)
    ensemble_estimators = [x[0] for x in ensemble[-1].estimators]
    assert results.shape[0] == 5
    assert "StackingRegressor" in results.index or "VotingRegressor" in results.index
    if estimator_names:
        assert all(estimator in ensemble_estimators for estimator in estimator_names)
    elif sort_by:
        sorted = results.sort_values(sort_by, ascending=False)
        sorted.drop([ensemble_class_name], inplace=True)
        assert all(x in ensemble_estimators for x in sorted.index[:top_n])
    else:
        sorted = results.sort_values(results.columns[0], ascending=False)
        sorted.drop([ensemble_class_name], inplace=True)
        assert all(x in ensemble_estimators for x in sorted.index[:top_n])


@pytest.mark.parametrize("reg_or_clf,on_errors", [("reg", True), ("clf", False)])
def test_predictions_similarity(reg_or_clf, on_errors):
    if reg_or_clf == "reg":
        est = PoniardRegressor(estimators=[LinearRegression(), DecisionTreeRegressor()])
        y = np.random.normal(size=10)
    else:
        est = PoniardClassifier(
            estimators=[LogisticRegression(), DecisionTreeClassifier()]
        )
        y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])
    x = pd.DataFrame(np.random.normal(size=(len(y), 5)))
    est.setup(x, y)
    est.fit()
    result = est.get_predictions_similarity(on_errors=on_errors)
    assert result.shape == (2, 2)
    assert result.iloc[1, 0] == result.iloc[0, 1]
