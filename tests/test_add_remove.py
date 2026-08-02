import numpy as np
import pandas as pd
import pytest
from sklearn.base import ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from poniard import PoniardClassifier


def test_add():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])
    x = pd.DataFrame(np.random.normal(size=(len(y), 5)))
    clf = PoniardClassifier().setup(x, y)
    clf.add_estimators([ExtraTreesClassifier()])
    clf.add_estimators({"rf2": RandomForestClassifier()})
    # Dummy is also added
    assert len(clf.pipelines) == len(clf._default_estimators) + 3
    assert "rf2" in clf.pipelines
    assert "ExtraTreesClassifier" in clf.pipelines


def test_remove():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])
    x = pd.DataFrame(np.random.normal(size=(len(y), 5)))
    clf = PoniardClassifier().setup(x, y)
    clf.remove_estimators(["RandomForestClassifier"])
    clf.remove_estimators(["LogisticRegression"])
    # Same amount of estimators because the dummy is added
    assert len(clf.pipelines) == len(clf._default_estimators) - 1


def test_remove_fitted():
    estimators = [RandomForestClassifier(), LogisticRegression()]
    clf = PoniardClassifier(estimators=estimators)
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])
    x = pd.DataFrame(np.random.normal(size=(len(y), 5)))
    clf.setup(x, y)
    clf.fit(x, y)
    clf.remove_estimators(["RandomForestClassifier"], drop_results=True)
    # Dummy baseline is added on top of the explicit estimator list.
    assert len(clf.pipelines) == len(estimators)
    assert clf.get_results().shape[0] == len(estimators)
    assert "RandomForestClassifier" not in clf.get_results().index


@pytest.mark.parametrize(
    "include_preprocessor,output_type", [(True, Pipeline), (False, ClassifierMixin)]
)
def test_get(include_preprocessor, output_type):
    clf = PoniardClassifier(estimators=[RandomForestClassifier()])
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])
    x = pd.DataFrame(np.random.normal(size=(len(y), 5)))
    clf.setup(x, y)
    clf.fit(x, y)
    estimator = clf.get_estimator(
        "RandomForestClassifier", include_preprocessor=include_preprocessor
    )
    assert isinstance(estimator, output_type)


def test_remove_then_readd_refits_estimator():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])
    x = pd.DataFrame(np.random.normal(size=(len(y), 5)))
    clf = PoniardClassifier(estimators=[LogisticRegression()]).setup(x, y)
    clf.fit(x, y)
    clf.remove_estimators(["LogisticRegression"])
    clf.add_estimators({"LogisticRegression": LogisticRegression()})
    clf.fit(x, y)
    assert "LogisticRegression" in clf.get_results().index
