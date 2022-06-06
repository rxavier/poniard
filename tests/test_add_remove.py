import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

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
    clf.fit(x, y)
    clf.remove_estimators(["RandomForestClassifier"], drop_results=True)
    assert len(clf.estimators_) == len(clf._base_estimators) - 1
    assert clf.show_results().shape[0] == len(clf._base_estimators) - 1
    assert "RandomForestClassifier" not in clf.show_results().index
