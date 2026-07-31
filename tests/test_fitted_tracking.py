import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from poniard import PoniardClassifier

Y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1])


def _clf():
    x = pd.DataFrame(np.random.normal(size=(len(Y), 5)))
    return PoniardClassifier(
        estimators=[LogisticRegression()], cv=2, random_state=0
    ), x


def test_fit_tracks_pipeline_names():
    clf, x = _clf()
    clf.setup(x, Y)
    clf.fit(x, Y)
    assert clf._fitted_pipeline_names == set(clf.pipelines)


def test_add_estimators_does_not_clear_fitted_set():
    clf, x = _clf()
    clf.setup(x, Y)
    clf.fit(x, Y)
    fitted_before = clf._fitted_pipeline_names.copy()
    clf.add_estimators({"rf": LogisticRegression()})
    assert clf._fitted_pipeline_names == fitted_before
    clf.fit(x, Y)
    assert "rf" in clf._fitted_pipeline_names


def test_reassign_types_clears_fitted_set():
    clf, x = _clf()
    clf.setup(x, Y)
    clf.fit(x, Y)
    assert clf._fitted_pipeline_names
    clf.reassign_types(numeric=[0])
    assert clf._fitted_pipeline_names == set()


def test_add_preprocessing_step_clears_fitted_set():
    clf, x = _clf()
    clf.setup(x, Y)
    clf.fit(x, Y)
    assert clf._fitted_pipeline_names
    clf.add_preprocessing_step(("scaler", StandardScaler()))
    assert clf._fitted_pipeline_names == set()


def test_remove_estimators_leaves_remaining_fitted():
    clf, x = _clf()
    clf.setup(x, Y)
    clf.fit(x, Y)
    clf.remove_estimators(["DummyClassifier"])
    assert "LogisticRegression" in clf._fitted_pipeline_names
