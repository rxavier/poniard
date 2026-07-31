import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from poniard import PoniardClassifier, PoniardRegressor


def _clf_data():
    n = 60
    X = pd.DataFrame(
        {
            "a": np.random.normal(size=n),
            "b": np.random.normal(size=n),
            "c": np.random.choice(["x", "y", "z"], size=n),
        }
    )
    y = pd.Series(np.random.choice([0, 1], size=n))
    return X, y


def _reg_data():
    X = pd.DataFrame(np.random.normal(size=(60, 3)), columns=["a", "b", "c"])
    y = pd.Series(np.random.normal(size=60))
    return X, y


def test_save_load_round_trip_classifier(tmp_path):
    X, y = _clf_data()
    clf = PoniardClassifier(
        estimators=[LogisticRegression()], cv=2, random_state=0
    )
    clf.setup(X, y)
    clf.fit(X, y)
    results_before = clf.get_results()

    path = tmp_path / "clf.joblib"
    clf.save(path)
    loaded = PoniardClassifier.load(path)

    assert isinstance(loaded, PoniardClassifier)
    pd.testing.assert_frame_equal(loaded.get_results(), results_before)
    assert loaded._experiment_results.keys() == clf._experiment_results.keys()


def test_save_load_round_trip_regressor(tmp_path):
    X, y = _reg_data()
    reg = PoniardRegressor(
        estimators=[LinearRegression()], cv=2, random_state=0
    )
    reg.setup(X, y)
    reg.fit(X, y)
    results_before = reg.get_results()

    path = tmp_path / "reg.joblib"
    reg.save(path)
    loaded = PoniardRegressor.load(path)

    assert isinstance(loaded, PoniardRegressor)
    pd.testing.assert_frame_equal(loaded.get_results(), results_before)


def test_loaded_estimator_can_export_pipeline(tmp_path):
    X, y = _clf_data()
    clf = PoniardClassifier(
        estimators=[LogisticRegression()], cv=2, random_state=0
    )
    clf.setup(X, y)
    clf.fit(X, y)
    path = tmp_path / "clf.joblib"
    clf.save(path)
    loaded = PoniardClassifier.load(path)
    model = loaded.get_estimator(
        "LogisticRegression", retrain=True, X=X, y=y
    )
    assert len(model.predict(X)) == len(X)


def test_save_load_after_tuning(tmp_path):
    X, y = _clf_data()
    clf = PoniardClassifier(
        estimators=[LogisticRegression()], cv=2, random_state=0
    )
    clf.setup(X, y)
    clf.fit(X, y)
    clf.tune_estimator(
        "LogisticRegression",
        X,
        y,
        grid={"C": [0.1, 1.0]},
    )
    clf.fit(X, y)
    path = tmp_path / "tuned.joblib"
    clf.save(path)
    loaded = PoniardClassifier.load(path)
    assert "LogisticRegression_tuned" in loaded.pipelines
    assert loaded.get_results().shape[0] == 3
