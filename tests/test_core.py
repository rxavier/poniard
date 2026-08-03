import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    mean_absolute_percentage_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from poniard import PoniardClassifier, PoniardRegressor
from poniard.error_analysis import ErrorAnalyzer
from poniard.preprocessing import PoniardPreprocessor


@pytest.mark.parametrize(
    "target,metrics,estimators,cv",
    [
        (np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]), None, None, None),
        (
            pd.Series(np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])),
            "accuracy",
            [LogisticRegression()],
            5,
        ),
        (
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]).tolist(),
            ["accuracy", "roc_auc"],
            {"logreg": LogisticRegression(), "rf": RandomForestClassifier()},
            StratifiedKFold(n_splits=2),
        ),
        (
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]),
            {"acc": make_scorer(accuracy_score), "roc": make_scorer(roc_auc_score)},
            [LogisticRegression(), DecisionTreeClassifier()],
            KFold(n_splits=3),
        ),
        (
            np.array([0, 0, 2, 2, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1]),
            None,
            [LogisticRegression(), DecisionTreeClassifier()],
            None,
        ),
    ],
)
def test_classifier_fit(target, metrics, estimators, cv):
    features = pd.DataFrame(np.random.normal(size=(len(target), 5)))
    features.columns = features.columns.astype(str)
    features["strings"] = np.random.choice(["a", "b", "c"], size=len(target))
    features["dates"] = pd.date_range("2020-01-01", periods=len(target))
    clf = PoniardClassifier(estimators=estimators, cv=cv, metrics=metrics, random_state=0)
    clf.setup(features, target)
    clf.fit(features, target)
    results = clf.get_results(return_train_scores=True)
    if not estimators:
        n_estimators = len(clf._default_estimators)
    else:
        n_estimators = len(estimators)
    if isinstance(metrics, str):
        n_metrics = 1
    else:
        n_metrics = len(clf.metrics)
    assert results.isna().sum().sum() == 0
    assert results.shape == (n_estimators + 1, n_metrics * 2 + 4)


@pytest.mark.parametrize(
    "target,metrics,estimators,cv",
    [
        (np.random.normal(size=(20,)), None, None, None),
        (
            pd.Series(np.random.normal(size=(20,))),
            "neg_mean_squared_error",
            [LinearRegression()],
            5,
        ),
        (
            np.random.normal(size=(20,)).tolist(),
            ["neg_mean_squared_error", "neg_mean_absolute_percentage_error"],
            {"linreg": LinearRegression(), "rf": RandomForestRegressor()},
            3,
        ),
        (
            np.random.normal(size=(20,)),
            {
                "mse": make_scorer(mean_squared_error, greater_is_better=False),
                "mape": make_scorer(mean_absolute_percentage_error, greater_is_better=False),
            },
            [LinearRegression(), DecisionTreeRegressor()],
            KFold(n_splits=3),
        ),
    ],
)
def test_regressor_fit(target, metrics, estimators, cv):
    features = pd.DataFrame(np.random.normal(size=(20, 5)))
    features.columns = features.columns.astype(str)
    features["strings"] = np.random.choice(["a", "b", "c"], size=len(target))
    features["dates"] = pd.date_range("2020-01-01", periods=len(target))
    clf = PoniardRegressor(estimators=estimators, cv=cv, metrics=metrics, random_state=0)
    clf.setup(features, target)
    clf.fit(features, target)
    results = clf.get_results(return_train_scores=True)
    if not estimators:
        n_estimators = len(clf._default_estimators)
    else:
        n_estimators = len(estimators)
    if isinstance(metrics, str):
        n_metrics = 1
    else:
        n_metrics = len(clf.metrics)
    assert results.isna().sum().sum() == 0
    assert results.shape == (n_estimators + 1, n_metrics * 2 + 4)


def test_regressor_default_metrics():
    X, y = make_regression(n_samples=60, n_features=4, random_state=42)
    reg = PoniardRegressor(estimators=[LinearRegression()], cv=3, random_state=0)
    reg.setup(pd.DataFrame(X), y)
    assert reg.metrics[0] == "neg_root_mean_squared_error"
    assert "neg_mean_absolute_percentage_error" not in reg.metrics


def test_regressor_positive_target_includes_mape():
    X, y = make_regression(n_samples=60, n_features=4, random_state=42)
    reg = PoniardRegressor(estimators=[LinearRegression()], cv=3, random_state=0)
    reg.setup(pd.DataFrame(X), np.exp(y))
    assert reg.metrics == [
        "neg_root_mean_squared_error",
        "neg_mean_absolute_error",
        "r2",
        "neg_mean_absolute_percentage_error",
    ]


def test_regressor_zero_target_scores_cleanly():
    X, y = make_regression(n_samples=60, n_features=4, random_state=42)
    y_zero = np.where(y > 0, y, 0.0)
    reg = PoniardRegressor(estimators=[LinearRegression()], cv=3, random_state=0)
    reg.fit(pd.DataFrame(X), y_zero)
    results = reg.get_results()
    assert "test_neg_mean_absolute_percentage_error" not in results.columns
    assert not results.isna().any().any()


def test_multilabel_fit():
    import warnings

    X, y = make_multilabel_classification(n_samples=300, n_classes=3, n_labels=3)
    clf = PoniardClassifier(
        estimators={
            "DT": OneVsRestClassifier(DecisionTreeClassifier()),
            "LR": OneVsRestClassifier(LogisticRegression()),
        },
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TargetEncoder is not supported")
        clf.setup(X, y)
        clf.fit(X, y)
    results = clf.get_results(return_train_scores=True)
    assert results.isna().sum().sum() == 0
    assert results.shape == (3, len(clf.metrics) * 2 + 4)


def test_classifier_default_metrics_without_predict_proba():
    from sklearn.svm import LinearSVC

    X, y = make_classification(n_samples=120, n_features=5, random_state=42)
    clf = PoniardClassifier(estimators=[LinearSVC(max_iter=5000)], cv=3, random_state=0)
    clf.setup(pd.DataFrame(X), y)
    assert clf.metrics[0] == "roc_auc"
    assert "neg_log_loss" not in clf.metrics
    assert "average_precision" not in clf.metrics


def test_multioutput_fit():
    import warnings

    X, y = make_regression(n_targets=3)
    clf = PoniardRegressor(
        estimators={
            "DT": MultiOutputRegressor(DecisionTreeRegressor()),
            "LR": MultiOutputRegressor(LinearRegression()),
        },
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TargetEncoder is not supported")
        clf.setup(X, y)
        clf.fit(X, y)
    results = clf.get_results(return_train_scores=True)
    assert results.isna().sum().sum() == 0
    assert results.shape == (3, len(clf.metrics) * 2 + 4)


def test_classifier_binary_default_metrics():
    X, y = make_classification(n_samples=120, n_features=5, random_state=42)
    clf = PoniardClassifier(estimators=[LogisticRegression()], cv=3, random_state=0)
    clf.setup(pd.DataFrame(X), y)
    assert clf.metrics[0] == "roc_auc"
    assert "neg_log_loss" in clf.metrics
    assert "average_precision" in clf.metrics


def test_classifier_multiclass_default_metrics():
    X, y = make_classification(
        n_samples=180, n_features=6, n_classes=3, n_informative=4, n_redundant=1, random_state=42
    )
    clf = PoniardClassifier(estimators=[LogisticRegression()], cv=3, random_state=0)
    clf.setup(pd.DataFrame(X), y)
    assert clf.metrics[0] == "roc_auc_ovr"
    assert "neg_log_loss" in clf.metrics
    assert "average_precision" not in clf.metrics


def test_classifier_multilabel_default_metrics():
    import warnings

    X, y = make_multilabel_classification(n_samples=300, n_classes=3, n_labels=3)
    clf = PoniardClassifier(estimators=[OneVsRestClassifier(LogisticRegression())], random_state=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TargetEncoder is not supported")
        clf.setup(X, y)
    assert clf.metrics[0] == "roc_auc"
    assert "neg_log_loss" in clf.metrics
    assert "average_precision" not in clf.metrics


def test_type_inference():
    x = pd.DataFrame(
        {
            "numeric": [float(i) for i in range(10)],
            "low_cardinality_str": ["a"] * 5 + ["b"] * 5,
            "low_cardinality_int": [1] * 10,
            "high_cardinality_str": [str(x) for x in range(10)],
            "high_cardinality_int": [x for x in range(10)],
            "datetime_H": pd.date_range("2020-01-01", freq="h", periods=10),
            "datetime_D": pd.date_range("2020-01-01", freq="D", periods=10, tz="Europe/Moscow"),
        }
    )
    # Add random nan to 10% per column: https://stackoverflow.com/a/61018279
    for col in x.columns:
        x.loc[x.sample(frac=0.1).index, col] = np.nan
    y = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    preprocessor = PoniardPreprocessor(cardinality_threshold=0.3)
    clf = PoniardClassifier(
        estimators=[LogisticRegression()],
        cv=3,
        custom_preprocessor=preprocessor,
        random_state=0,
    )
    clf.setup(x, y)
    clf.fit(x, y)
    assert all(x in clf.feature_types["numeric"] for x in ["numeric", "high_cardinality_int"])
    assert all(x in clf.feature_types["categorical_high"] for x in ["high_cardinality_str"])
    assert all(
        x in clf.feature_types["categorical_low"]
        for x in ["low_cardinality_str", "low_cardinality_int"]
    )
    assert all(x in clf.feature_types["datetime"] for x in ["datetime_H", "datetime_D"])


def test_predict_recomputes_when_data_changes():
    """predict() on new data must recompute, never return cached predictions."""
    from unittest import mock

    from sklearn.model_selection import cross_val_predict as real_cross_val_predict

    X1 = pd.DataFrame(np.random.normal(size=(30, 2)))
    y1 = pd.Series(np.random.choice([0, 1], size=30))
    clf = PoniardClassifier(
        estimators={"lr": LogisticRegression(max_iter=5000)}, cv=2, random_state=0
    )
    clf.setup(X1, y1)
    clf.fit(X1, y1)

    X2 = pd.DataFrame(np.random.normal(size=(30, 2)))
    y2 = pd.Series(np.random.choice([0, 1], size=30))

    with mock.patch(
        "poniard.estimators.core.cross_val_predict", wraps=real_cross_val_predict
    ) as spy:
        clf.predict(X=X1, y=y1, estimator_names=["lr"])
        first = spy.call_count
        clf.predict(X=X2, y=y2, estimator_names=["lr"])
        second = spy.call_count
        clf.predict(X=X1, y=y1, estimator_names=["lr"])
        third = spy.call_count

    # Fresh pass per call: public predict never reads the cache, so each call
    # (even on already-seen data) runs one CV pass.
    assert first == 1
    assert second == first + 1
    assert third == second + 1


def test_prediction_cache_stores_fingerprint_not_data():
    """The prediction cache must hold a hash of the data, never the data itself."""
    from poniard.estimators.core import _data_fingerprint

    X1 = pd.DataFrame(np.random.normal(size=(30, 2)))
    y1 = pd.Series(np.random.choice([0, 1], size=30))
    clf = PoniardClassifier(
        estimators={"lr": LogisticRegression(max_iter=5000)}, cv=2, random_state=0
    )
    clf.setup(X1, y1)
    clf.fit(X1, y1)
    clf.predict(X=X1, y=y1, estimator_names=["lr"])

    cached = clf._prediction_cache[("lr", "predict")]
    assert cached.fingerprint == _data_fingerprint(X1, y1)
    assert not hasattr(cached, "X")
    assert not hasattr(cached, "y")


def test_analyze_recomputes_after_in_place_mutation():
    """Mutating the input data in place must invalidate cached predictions."""
    from unittest import mock

    from sklearn.model_selection import cross_val_predict as real_cross_val_predict

    X1 = pd.DataFrame(np.random.normal(size=(N := 60, 2)))
    y1 = pd.Series(np.random.choice([0, 1], size=N))
    clf = PoniardClassifier(
        estimators={"lr": LogisticRegression(max_iter=5000)}, cv=2, random_state=0
    )
    clf.setup(X1, y1)
    clf.fit(X1, y1)
    ea = ErrorAnalyzer.from_poniard(clf)

    with mock.patch(
        "poniard.estimators.core.cross_val_predict", wraps=real_cross_val_predict
    ) as spy:
        ea.analyze(X=X1, y=y1)
        first = spy.call_count
        X1.iloc[0, 0] = 999.0
        ea.analyze(X=X1, y=y1)
        second = spy.call_count

    assert second > first
