import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline

from poniard import PoniardClassifier


def test_add():
    clf = PoniardClassifier()
    clf.add_estimators([ExtraTreesClassifier()])
    clf + {"rf2": RandomForestClassifier()}
    assert len(clf.estimators_) == len(clf._base_estimators) + 2
    assert "rf2" in clf.estimators_
    assert "ExtraTreesClassifier" in clf.estimators_


def test_remove():
    clf = PoniardClassifier()
    clf.remove_estimators(["RandomForestClassifier"])
    clf - ["LogisticRegression"]
    assert len(clf.estimators_) == len(clf._base_estimators) - 2


def test_remove_fitted():
    clf = PoniardClassifier()
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])
    x = pd.DataFrame(np.random.normal(size=(len(y), 5)))
    clf.setup(x, y)
    clf.fit()
    clf.remove_estimators(["RandomForestClassifier"], drop_results=True)
    assert len(clf.estimators_) == len(clf._base_estimators) - 1
    assert clf.show_results().shape[0] == len(clf._base_estimators) - 1
    assert "RandomForestClassifier" not in clf.show_results().index


@pytest.mark.parametrize(
    "include_preprocessor,output_type", [(True, Pipeline), (False, ClassifierMixin)]
)
def test_get(include_preprocessor, output_type):
    clf = PoniardClassifier(estimators=[RandomForestClassifier()])
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])
    x = pd.DataFrame(np.random.normal(size=(len(y), 5)))
    clf.setup(x, y)
    clf.fit()
    estimator = clf.get_estimator(
        "RandomForestClassifier", include_preprocessor=include_preprocessor
    )
    assert isinstance(estimator, output_type)
