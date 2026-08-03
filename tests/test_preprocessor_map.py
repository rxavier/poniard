import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from poniard import PoniardClassifier


@pytest.fixture
def X_y():
    X, y = make_classification(n_samples=120, n_features=5, random_state=42)
    return pd.DataFrame(X), y


@pytest.fixture
def fitted(X_y):
    X, y = X_y
    return PoniardClassifier(
        estimators=[LogisticRegression(max_iter=1000)], cv=3, random_state=0
    ).fit(X, y, show_info=False)


def test_registry_has_default_only(X_y):
    X, y = X_y
    clf = PoniardClassifier(estimators=[LogisticRegression()], cv=3, random_state=0)
    clf.setup(X, y, show_info=False)
    assert list(clf.preprocessors) == ["default"]
    assert clf.preprocessor_map == {}
    assert clf.preprocessor is clf.preprocessors["default"]
    assert clf.pipelines["LogisticRegression"].named_steps["preprocessor"] is clf.preprocessor


def test_mapped_estimator_uses_mapped_preprocessor(X_y):
    X, y = X_y
    clf = PoniardClassifier(
        estimators={"mapped": LogisticRegression(), "plain": LogisticRegression()},
        cv=3,
        random_state=0,
    )
    clf.setup(X, y, show_info=False)
    template = make_pipeline(MinMaxScaler())
    clf.add_preprocessor("mm", template)
    clf.set_preprocessor("mapped", "mm")
    assert clf.preprocessor_map == {"mapped": "mm"}
    assert clf.pipelines["mapped"].named_steps["preprocessor"] is clf.preprocessors["mm"]
    assert clf.pipelines["plain"].named_steps["preprocessor"] is clf.preprocessors["default"]
    clf.fit(X, y, show_info=False)
    assert not clf.get_results().isna().any().any()


def test_preprocessor_map_auto_registers_pipeline_instance(X_y):
    X, y = X_y
    clf = PoniardClassifier(
        estimators={"mapped": LogisticRegression(), "plain": LogisticRegression()},
        cv=3,
        random_state=0,
        preprocessor_map={"mapped": make_pipeline(MinMaxScaler())},
    )
    clf.setup(X, y, show_info=False)
    assert "default" in clf.preprocessors
    (prep_name,) = set(clf.preprocessor_map.values())
    assert prep_name != "default"
    assert clf.preprocessor_map == {"mapped": prep_name}
    assert clf.pipelines["mapped"].named_steps["preprocessor"] is clf.preprocessors[prep_name]
    assert clf.pipelines["plain"].named_steps["preprocessor"] is clf.preprocessors["default"]


def test_set_preprocessor_unknown_estimator_raises(X_y, fitted):
    with pytest.raises(KeyError, match="Estimator 'Nope' not found"):
        fitted.set_preprocessor("Nope", "default")


def test_set_preprocessor_unknown_preprocessor_raises(X_y, fitted):
    with pytest.raises(KeyError, match="Preprocessor 'nope' not registered"):
        fitted.set_preprocessor("LogisticRegression", "nope")


def test_preprocessor_map_unknown_name_raises(X_y):
    X, y = X_y
    clf = PoniardClassifier(
        estimators=[LogisticRegression()],
        cv=3,
        random_state=0,
        preprocessor_map={"LogisticRegression": "nonexistent"},
    )
    with pytest.raises(KeyError, match="not registered"):
        clf.setup(X, y, show_info=False)


def test_add_preprocessing_step_targets_specific_preprocessor(X_y):
    X, y = X_y
    clf = PoniardClassifier(estimators=[LogisticRegression()], cv=3, random_state=0)
    clf.setup(X, y, show_info=False)
    clf.add_preprocessor("mm", make_pipeline(MinMaxScaler()))
    clf.add_preprocessing_step(StandardScaler(), preprocessor="mm")
    assert any(isinstance(t, StandardScaler) for _, t in clf.preprocessors["mm"].steps)
    assert not any(isinstance(t, StandardScaler) for _, t in clf.preprocessors["default"].steps)
    with pytest.raises(KeyError, match="Unknown preprocessor"):
        clf.add_preprocessing_step(StandardScaler(), preprocessor="nope")


def test_add_preprocessing_step_all_preserves_legacy_behavior(X_y):
    X, y = X_y
    clf = PoniardClassifier(estimators=[LogisticRegression()], cv=3, random_state=0)
    clf.setup(X, y, show_info=False)
    clf.add_preprocessor("mm", make_pipeline(MinMaxScaler()))
    clf.add_preprocessing_step(StandardScaler())
    assert all(
        any(isinstance(t, StandardScaler) for _, t in prep.steps)
        for prep in clf.preprocessors.values()
    )


def test_reassign_types_preserves_registry(X_y):
    X, y = X_y
    clf = PoniardClassifier(estimators=[LogisticRegression()], cv=3, random_state=0)
    clf.setup(X, y, show_info=False)
    clf.add_preprocessor("mm", make_pipeline(MinMaxScaler()))
    clf.set_preprocessor("LogisticRegression", "mm")
    clf.reassign_types(numeric=list(X.columns))
    assert set(clf.preprocessors) == {"default", "mm"}
    assert clf.preprocessor_map == {"LogisticRegression": "mm"}
    assert (
        clf.pipelines["LogisticRegression"].named_steps["preprocessor"] is clf.preprocessors["mm"]
    )


def test_persistence_round_trip_preserves_map(X_y, tmp_path):
    X, y = X_y
    clf = PoniardClassifier(
        estimators={"mapped": LogisticRegression(), "plain": LogisticRegression()},
        cv=3,
        random_state=0,
    )
    clf.setup(X, y, show_info=False)
    clf.add_preprocessor("mm", make_pipeline(MinMaxScaler()))
    clf.set_preprocessor("mapped", "mm")
    path = tmp_path / "estimator.joblib"
    clf.save(path)
    loaded = PoniardClassifier.load(path)
    assert loaded.preprocessor_map == {"mapped": "mm"}
    assert set(loaded.preprocessors) == {"default", "mm"}
    assert loaded.pipelines["mapped"].named_steps["preprocessor"] is loaded.preprocessors["mm"]


def test_tuning_with_mapped_pipeline(X_y):
    X, y = X_y
    clf = PoniardClassifier(estimators=[LogisticRegression(max_iter=1000)], cv=3, random_state=0)
    clf.setup(X, y, show_info=False)
    clf.add_preprocessor("mm", make_pipeline(MinMaxScaler()))
    clf.set_preprocessor("LogisticRegression", "mm")
    clf.tune_estimator(
        estimator_name="LogisticRegression",
        X=X,
        y=y,
        grid={"preprocessor__minmaxscaler__feature_range": [(0, 1), (0, 2)]},
    )
    assert "LogisticRegression_tuned" in clf.pipelines


def test_native_mapped_hgb_uses_native(X_y):
    X, y = X_y
    clf = PoniardClassifier(
        estimators=[
            HistGradientBoostingClassifier(max_iter=50, random_state=0),
            LogisticRegression(),
        ],
        cv=2,
        random_state=0,
        preprocessor_map={"HistGradientBoostingClassifier": "native"},
    )
    clf.setup(X, y, show_info=False)
    assert "native" in clf.preprocessors
    assert clf.preprocessor_map == {"HistGradientBoostingClassifier": "native"}
    hgb = clf.pipelines["HistGradientBoostingClassifier"]
    assert hgb.named_steps["preprocessor"] is clf.preprocessors["native"]
    estimator = hgb.named_steps["HistGradientBoostingClassifier"]
    assert estimator.categorical_features == "from_dtype"
    lr = clf.pipelines["LogisticRegression"]
    assert lr.named_steps["preprocessor"] is clf.preprocessors["default"]


def test_native_set_preprocessor_on_demand(X_y):
    X, y = X_y
    clf = PoniardClassifier(
        estimators=[HistGradientBoostingClassifier(max_iter=50, random_state=0)],
        cv=2,
        random_state=0,
    )
    clf.setup(X, y, show_info=False)
    assert "native" not in clf.preprocessors
    clf.set_preprocessor("HistGradientBoostingClassifier", "native")
    assert "native" in clf.preprocessors
    assert clf.preprocessor_map == {"HistGradientBoostingClassifier": "native"}
    clf.fit(X, y, show_info=False)
    assert not clf.get_results().isna().any().any()


def test_native_non_hgb_raises(X_y):
    X, y = X_y
    clf = PoniardClassifier(estimators=[LogisticRegression()], cv=2, random_state=0)
    clf.setup(X, y, show_info=False)
    with pytest.raises(ValueError, match="only be mapped to HistGradientBoosting"):
        clf.set_preprocessor("LogisticRegression", "native")
    clf2 = PoniardClassifier(
        estimators=[LogisticRegression()],
        cv=2,
        random_state=0,
        preprocessor_map={"LogisticRegression": "native"},
    )
    with pytest.raises(ValueError, match="only be mapped to HistGradientBoosting"):
        clf2.setup(X, y, show_info=False)


def test_native_e2e_with_nan_categorical():
    X = pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "cat": ["a", "b", np.nan, "a", "b", "c", "a", "b", "a", "c"],
        }
    )
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    clf = PoniardClassifier(
        estimators=[HistGradientBoostingClassifier(max_iter=50, random_state=0)],
        cv=2,
        random_state=0,
        preprocessor_map={"HistGradientBoostingClassifier": "native"},
    )
    clf.fit(X, y, show_info=False)
    assert not clf.get_results().isna().any().any()


def test_native_reassign_types_shared_inference(X_y):
    X, y = X_y
    clf = PoniardClassifier(
        estimators=[HistGradientBoostingClassifier(max_iter=50, random_state=0)],
        cv=2,
        random_state=0,
        preprocessor_map={"HistGradientBoostingClassifier": "native"},
    )
    clf.setup(X, y, show_info=False)
    native = clf.preprocessors["native"].named_steps["type_preprocessor"]
    before = {name for name, _, cols in native.transformers if cols}
    clf.reassign_types(
        numeric=[],
        categorical_high=[],
        categorical_low=list(X.columns),
        datetime=[],
        keep_remainder=False,
    )
    native = clf.preprocessors["native"].named_steps["type_preprocessor"]
    after = {name for name, _, cols in native.transformers if cols}
    assert before == {"numeric_preprocessor"}
    assert after == {"categorical_low_preprocessor"}
    assert clf.preprocessor_map == {"HistGradientBoostingClassifier": "native"}


def test_add_preprocessing_step_targets_native(X_y):
    X, y = X_y
    clf = PoniardClassifier(
        estimators=[HistGradientBoostingClassifier(max_iter=50, random_state=0)],
        cv=2,
        random_state=0,
        preprocessor_map={"HistGradientBoostingClassifier": "native"},
    )
    clf.setup(X, y, show_info=False)
    clf.add_preprocessing_step(StandardScaler(), preprocessor="native")
    assert any(isinstance(t, StandardScaler) for _, t in clf.preprocessors["native"].steps)
    assert not any(isinstance(t, StandardScaler) for _, t in clf.preprocessors["default"].steps)


def test_persistence_round_trip_preserves_native(X_y, tmp_path):
    X, y = X_y
    clf = PoniardClassifier(
        estimators=[HistGradientBoostingClassifier(max_iter=50, random_state=0)],
        cv=2,
        random_state=0,
        preprocessor_map={"HistGradientBoostingClassifier": "native"},
    )
    clf.setup(X, y, show_info=False)
    path = tmp_path / "est.joblib"
    clf.save(path)
    loaded = PoniardClassifier.load(path)
    assert loaded.preprocessor_map == {"HistGradientBoostingClassifier": "native"}
    assert "native" in loaded.preprocessors
    hgb = loaded.pipelines["HistGradientBoostingClassifier"]
    assert hgb.named_steps["preprocessor"] is loaded.preprocessors["native"]


def test_tuning_with_native_pipeline(X_y):
    X, y = X_y
    clf = PoniardClassifier(
        estimators=[HistGradientBoostingClassifier(max_iter=50, random_state=0)],
        cv=2,
        random_state=0,
        preprocessor_map={"HistGradientBoostingClassifier": "native"},
    )
    clf.setup(X, y, show_info=False)
    clf.tune_estimator(
        estimator_name="HistGradientBoostingClassifier",
        X=X,
        y=y,
        grid={"HistGradientBoostingClassifier__learning_rate": [0.01, 0.1]},
    )
    assert "HistGradientBoostingClassifier_tuned" in clf.pipelines
