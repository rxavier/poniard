from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from poniard import PoniardClassifier, PoniardRegressor


def test_multiclass_fit():
    X, y = make_classification(n_classes=3, n_informative=5)
    clf = PoniardClassifier(
        estimators=[RandomForestClassifier(), LogisticRegression()],
        random_state=0,
    )
    clf.fit(X, y)
    results = clf.show_results()
    assert results.isna().sum().sum() == 0
    assert results.shape == (3, 12)


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
