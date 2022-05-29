import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    roc_auc_score,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)

from poniard import PoniardClassifier, PoniardRegressor


@pytest.mark.parametrize(
    "target,metrics,estimators,cv",
    [
        (np.random.randint(0, 2, (20,)), None, None, None),
        (
            pd.Series(np.random.randint(0, 2, (20,))),
            "accuracy",
            [LogisticRegression()],
            5,
        ),
        (
            np.random.randint(0, 2, (20,)).tolist(),
            ["accuracy", "roc_auc"],
            {"logreg": LogisticRegression(), "rf": RandomForestClassifier()},
            StratifiedKFold(n_splits=2),
        ),
        (
            np.random.randint(0, 2, (20,)),
            {"acc": make_scorer(accuracy_score), "roc": make_scorer(roc_auc_score)},
            [LogisticRegression(), RandomForestClassifier()],
            KFold(n_splits=3),
        ),
        (
            np.random.randint(0, 3, (20,)),
            None,
            [LogisticRegression(), RandomForestClassifier()],
            None,
        ),
    ],
)
def test_classifier_fit(target, metrics, estimators, cv):
    features = pd.DataFrame(np.random.normal(size=(20, 5)))
    clf = PoniardClassifier(
        estimators=estimators, cv=cv, metrics=metrics, random_state=0
    )
    clf.fit(features, target)
    results = clf.show_results()
    if not estimators:
        n_estimators = len(clf._base_estimators)
    else:
        n_estimators = len(estimators) + 1
    if isinstance(metrics, str):
        n_metrics = 1
    else:
        n_metrics = len(clf.metrics_)
    assert results.isna().sum().sum() == 0
    assert results.shape == (n_estimators, n_metrics * 2 + 2)


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
                "mape": make_scorer(
                    mean_absolute_percentage_error, greater_is_better=False
                ),
            },
            [LinearRegression(), RandomForestRegressor()],
            KFold(n_splits=3),
        ),
    ],
)
def test_regressor_fit(target, metrics, estimators, cv):
    features = pd.DataFrame(np.random.normal(size=(20, 5)))
    clf = PoniardRegressor(
        estimators=estimators, cv=cv, metrics=metrics, random_state=0
    )
    clf.fit(features, target)
    results = clf.show_results()
    if not estimators:
        n_estimators = len(clf._base_estimators)
    else:
        n_estimators = len(estimators) + 1
    if isinstance(metrics, str):
        n_metrics = 1
    else:
        n_metrics = len(clf.metrics_)
    assert results.isna().sum().sum() == 0
    assert results.shape == (n_estimators, n_metrics * 2 + 2)


def test_multilabel_fit():
    X, y = make_multilabel_classification(n_classes=3, n_labels=3)
    clf = PoniardClassifier(
        estimators={
            "RF": MultiOutputClassifier(RandomForestClassifier()),
            "LR": MultiOutputClassifier(LogisticRegression()),
        },
        random_state=0,
    )
    clf.fit(X, y)
    results = clf.show_results()
    assert results.isna().sum().sum() == 0
    assert results.shape == (3, 10)


def test_multioutput_fit():
    X, y = make_regression(n_targets=3)
    clf = PoniardRegressor(
        estimators={
            "RF": MultiOutputRegressor(RandomForestRegressor()),
            "LR": MultiOutputRegressor(LinearRegression()),
        },
        random_state=0,
    )
    clf.fit(X, y)
    results = clf.show_results()
    assert results.isna().sum().sum() == 0
    assert results.shape == (3, 10)
