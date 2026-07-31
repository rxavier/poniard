import os
import pickle
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from poniard import PoniardClassifier


def _data():
    n = 50
    X = pd.DataFrame(
        {
            "a": np.random.normal(size=n),
            "b": np.random.normal(size=n),
            "c": np.random.choice(["x", "y"], size=n),
        }
    )
    y = pd.Series(np.random.choice([0, 1], size=n))
    return X, y


def _fitted_clf():
    X, y = _data()
    clf = PoniardClassifier(
        estimators=[LogisticRegression()], cv=2, random_state=0
    )
    clf.setup(X, y)
    clf.fit(X, y)
    return clf, X, y


def test_get_estimator_returns_plain_sklearn_pipeline():
    clf, X, y = _fitted_clf()
    est = clf.get_estimator("LogisticRegression", retrain=True, X=X, y=y)
    assert isinstance(est, Pipeline)
    assert type(est).__module__.startswith("sklearn")


def test_get_estimator_without_preprocessor_is_bare_estimator():
    n = 50
    X = pd.DataFrame(np.random.normal(size=(n, 2)), columns=["a", "b"])
    y = pd.Series(np.random.choice([0, 1], size=n))
    clf = PoniardClassifier(
        estimators=[LogisticRegression()], cv=2, random_state=0
    )
    clf.setup(X, y)
    clf.fit(X, y)
    est = clf.get_estimator(
        "LogisticRegression",
        include_preprocessor=False,
        retrain=True,
        X=X,
        y=y,
    )
    assert isinstance(est, LogisticRegression)


def test_get_estimator_retrain_requires_X_and_y():
    clf, _, _ = _fitted_clf()
    with np.testing.assert_raises_regex(ValueError, "X and y"):
        clf.get_estimator("LogisticRegression", retrain=True)


def test_get_estimator_pickles_without_poniard(tmp_path):
    """A get_estimator() pipeline must pickle and load in a subprocess where
    poniard cannot be imported: the real definition of 'you can delete poniard
    when you're done'."""
    clf, X, y = _fitted_clf()
    est = clf.get_estimator("LogisticRegression", retrain=True, X=X, y=y)

    model_path = tmp_path / "model.pkl"
    X_path = tmp_path / "X.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(est, f)
    with open(X_path, "wb") as f:
        pickle.dump(X, f)

    code = textwrap.dedent(
        """
        import builtins
        import pickle
        import sys

        _real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name == "poniard" or name.startswith("poniard."):
                raise ModuleNotFoundError("poniard is blocked")
            return _real_import(name, *args, **kwargs)

        builtins.__import__ = _blocked

        from sklearn.pipeline import Pipeline

        with open(sys.argv[1], "rb") as f:
            est = pickle.load(f)
        assert isinstance(est, Pipeline), type(est)
        with open(sys.argv[2], "rb") as f:
            X = pickle.load(f)
        pred = est.predict(X)
        assert len(pred) == len(X)
        """
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    subprocess.run(
        [sys.executable, "-c", code, str(model_path), str(X_path)],
        check=True,
        env=env,
        cwd=tmp_path,
    )
