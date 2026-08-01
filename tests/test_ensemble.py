import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from poniard import PoniardClassifier, PoniardRegressor


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
    reg.fit(x, y)
    reg.build_ensemble(
        method=method, estimator_names=estimator_names, top_n=top_n, sort_by=sort_by
    )
    reg.fit(x, y)
    results = reg.get_results()
    ensemble_class_name = method.capitalize() + "Regressor"
    ensemble = reg.get_estimator(ensemble_class_name)
    ensemble_estimators = [x[0] for x in ensemble[-1].estimators]
    assert results.shape[0] == 5
    assert "StackingRegressor" in results.index or "VotingRegressor" in results.index
    dummy = set(reg._dummy_names())
    if estimator_names:
        assert all(estimator in ensemble_estimators for estimator in estimator_names)
    else:
        sorter = sort_by or results.columns[0]
        sorted_names = [
            n for n in results.sort_values(sorter, ascending=False).index
            if n != ensemble_class_name and n not in dummy
        ]
        assert all(x in ensemble_estimators for x in sorted_names[:top_n])


def test_ensemble_diversity_strategy():
    """Diversity strategy should pick estimators with low pairwise similarity."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame(np.random.normal(size=(n, 10)))
    y = (X[0] + X[1] * X[2] > 0).astype(int)
    clf = PoniardClassifier(
        estimators={
            "lr": LogisticRegression(max_iter=1000),
            "dt1": DecisionTreeClassifier(max_depth=2),
            "dt2": DecisionTreeClassifier(max_depth=3),
        },
        cv=3,
        random_state=0,
    )
    clf.fit(X, y)
    clf.build_ensemble(
        method="voting",
        strategy="diversity",
        top_n=3,
        X=X,
        y=y,
        ensemble_name="diverse_ens",
        voting="soft",
    )
    clf.fit(X, y)
    results = clf.get_results()
    assert "diverse_ens" in results.index
    ens = clf.get_estimator("diverse_ens")
    member_names = [name for name, _ in ens[-1].estimators]
    assert len(member_names) >= 2


def test_ensemble_top_n_strategy():
    """top_n strategy is the legacy behavior — just take the best N."""
    np.random.seed(42)
    n = 50
    X = pd.DataFrame(np.random.normal(size=(n, 5)))
    y = np.random.choice([0, 1], size=n)
    clf = PoniardClassifier(
        estimators={
            "lr": LogisticRegression(max_iter=1000),
            "dt": DecisionTreeClassifier(),
        },
        cv=2,
        random_state=0,
    )
    clf.fit(X, y)
    clf.build_ensemble(method="voting", strategy="top_n", top_n=2, ensemble_name="topn", voting="soft")
    clf.fit(X, y)
    results = clf.get_results()
    assert "topn" in results.index


def test_ensemble_diversity_fallback_to_top_n():
    """When X/y not provided, diversity falls back to top_n selection."""
    np.random.seed(42)
    n = 50
    X = pd.DataFrame(np.random.normal(size=(n, 5)))
    y = np.random.choice([0, 1], size=n)
    clf = PoniardClassifier(
        estimators={
            "lr": LogisticRegression(max_iter=1000),
            "dt": DecisionTreeClassifier(),
        },
        cv=2,
        random_state=0,
    )
    clf.fit(X, y)
    clf.build_ensemble(method="voting", strategy="diversity", top_n=2, ensemble_name="fallback", voting="soft")
    clf.fit(X, y)
    results = clf.get_results()
    assert "fallback" in results.index


def test_ensemble_diversity_single_estimator():
    """With only 1 non-dummy estimator, diversity returns that one."""
    n = 30
    X = pd.DataFrame(np.random.normal(size=(n, 3)))
    y = np.random.choice([0, 1], size=n)
    clf = PoniardClassifier(estimators={"lr": LogisticRegression(max_iter=1000)}, cv=2, random_state=0)
    clf.fit(X, y)
    selected = clf._select_diverse(top_n=3, sort_by=None, similarity_threshold=0.5, X=X, y=y)
    assert len(selected) >= 1
    assert "lr" in selected


def test_ensemble_invalid_strategy():
    n = 30
    X = pd.DataFrame(np.random.normal(size=(n, 3)))
    y = np.random.choice([0, 1], size=n)
    clf = PoniardClassifier(estimators={"lr": LogisticRegression(max_iter=1000)}, cv=2, random_state=0)
    clf.fit(X, y)
    with pytest.raises(ValueError, match="Strategy"):
        clf.build_ensemble(strategy="invalid", X=X, y=y)


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
    est.fit(x, y)
    result = est.get_predictions_similarity(x, y, on_errors=on_errors)
    assert result.shape == (2, 2)
    assert result.iloc[1, 0] == result.iloc[0, 1]


def test_predictions_similarity_excludes_dummies_by_type():
    """Dummy estimators are excluded by class, regardless of their name."""
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])
    x = pd.DataFrame(np.random.normal(size=(len(y), 5)))
    est = PoniardClassifier(estimators=[LogisticRegression()])
    est.setup(x, y)
    est.add_estimators({"baseline": DummyClassifier(strategy="prior")})
    est.fit(x, y)
    result = est.get_predictions_similarity(x, y)
    assert result.shape == (1, 1)
    assert "baseline" not in result.columns
    assert "LogisticRegression" in result.columns
