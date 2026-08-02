import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from poniard import PoniardClassifier, PoniardRegressor
from poniard.preprocessing import PoniardPreprocessor, infer_feature_types
from poniard.preprocessing.datetime import DateLevel, DatetimeEncoder


@pytest.mark.parametrize(
    "X,preprocess,scaler,numeric_imputer,high_cardinality_encoder,include_preprocessor",
    [
        (
            pd.DataFrame(
                {
                    "A": [4, 3, 1, -1, np.nan],
                    "B": [-2, np.nan, 3, 7, 1],
                    "C": list("abcde"),
                    "D": pd.date_range("2020-01-01", freq="ME", periods=5),
                }
            ),
            True,
            None,
            None,
            "target",
            True,
        ),
        (
            pd.DataFrame(
                {
                    "A": [4, 200, 1, -1, np.nan],
                    "B": [-2, np.nan, 3, 7, 1],
                    "C": list("abcde"),
                    "D": pd.date_range("2020-01-01", freq="h", periods=5),
                }
            ),
            True,
            "standard",
            "iterative",
            "ordinal",
            True,
        ),
        (
            pd.DataFrame(
                {
                    "A": [4, 200, 1, -1, np.nan],
                    "B": [-2, np.nan, 3, 7, 1],
                    "C": list("abcde"),
                    "D": pd.date_range("2020-01-01", freq="YE", periods=5),
                }
            ),
            True,
            "robust",
            "simple",
            None,
            True,
        ),
        (
            pd.DataFrame(
                {
                    "A": [4, 200, 1, -1, np.nan],
                    "B": [-2, np.nan, 3, 7, 1],
                    "C": list("abcde"),
                    "D": pd.date_range("2020-01-01", freq="MS", periods=5),
                }
            ),
            True,
            "minmax",
            None,
            "target",
            True,
        ),
        (
            pd.DataFrame({"A": [4, 3, 1, -1, 0], "B": [-2, 1, 3, 7, 1]}),
            False,
            None,
            None,
            "ordinal",
            False,
        ),
    ],
)
def test_preprocessing_classifier(
    X,
    preprocess,
    scaler,
    numeric_imputer,
    high_cardinality_encoder,
    include_preprocessor,
):
    preprocessor = PoniardPreprocessor(
        scaler=scaler,
        numeric_imputer=numeric_imputer,
        high_cardinality_encoder=high_cardinality_encoder,
    )
    estimator = PoniardClassifier(
        estimators=[LogisticRegression()],
        preprocess=preprocess,
        custom_preprocessor=preprocessor,
        cv=2,
        random_state=0,
    )
    y = [0, 1, 0, 1, 0]
    estimator.setup(X, y)
    estimator.fit(X, y)
    assert estimator.get_results().isna().sum().sum() == 0
    train_results = estimator.get_results(return_train_scores=True)
    assert any(c.startswith("train_") for c in train_results.columns)
    assert isinstance(
        estimator.get_estimator("LogisticRegression", include_preprocessor=include_preprocessor),
        BaseEstimator,
    )


@pytest.mark.parametrize(
    "new_step,position,existing_step",
    [
        (SelectKBest(f_regression, k=2), 0, None),
        (
            make_pipeline(SimpleImputer(), SelectKBest(f_regression, k=2)),
            "start",
            StandardScaler(),
        ),
        (
            make_pipeline(SimpleImputer(), SelectKBest(f_regression, k=2)),
            "end",
            make_pipeline(SimpleImputer(), StandardScaler()),
        ),
    ],
)
def test_add_step(new_step, position, existing_step):
    X = pd.DataFrame(
        {
            "A": [4, 3, 1, -1, np.nan],
            "B": [-2, np.nan, 3, 7, 1],
            "C": list("abcde"),
            "D": pd.date_range("2020-01-01", freq="ME", periods=5),
        }
    )
    y = np.random.uniform(0, 1, size=5)
    reg = PoniardRegressor(custom_preprocessor=existing_step).setup(X, y)
    reg.add_preprocessing_step(new_step, position)
    assert isinstance(reg.preprocessor, Pipeline)


def test_feature_names_stable_across_type_compositions():
    base = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": ["a", "b", "a", "b", "a"],
            "D": pd.date_range("2020-01-01", periods=5),
        }
    )
    y = np.array([0, 1, 0, 1, 0])
    pp_full = PoniardPreprocessor().build(X=base, y=y, task="classification")
    pp_numeric = PoniardPreprocessor().build(X=base[["A"]], y=y, task="classification")
    pp_full.preprocessor.fit(base, y)
    pp_numeric.preprocessor.fit(base[["A"]], y)
    full_names = pp_full.preprocessor.get_feature_names_out()
    numeric_names = pp_numeric.preprocessor.get_feature_names_out()
    assert list(full_names) == ["A", "B_b", "D_day", "D_weekday", "D_dayofyear"]
    assert list(numeric_names) == ["A"]


def test_datetime_features_scaled_and_imputed():
    dates = pd.date_range("2020-01-01", freq="h", periods=99).to_list()
    dates.insert(0, pd.NaT)
    X = pd.DataFrame({"D": dates})
    y = np.random.RandomState(0).randint(0, 2, size=100)
    pp = PoniardPreprocessor().build(X=X, y=y, task="classification")
    out = pp.preprocessor.fit_transform(X, y)
    assert not out.isna().any().any()
    assert {"D_day", "D_hour", "D_weekday", "D_dayofyear"} <= set(out.columns)
    for col in ("D_day", "D_hour", "D_weekday", "D_dayofyear"):
        assert np.allclose(out[col].mean(), 0, atol=1e-8)
        assert np.allclose(out[col].std(ddof=0), 1, atol=1e-8)


def test_build_without_data_raises_clear_error():
    pp = PoniardPreprocessor(task="classification")
    with pytest.raises(ValueError, match="X and y must be passed to build"):
        pp.build()


@pytest.mark.parametrize(
    "array,frame",
    [
        (
            np.array(
                [
                    [1.0, np.nan, 3.0],
                    [4.0, 5.0, np.nan],
                    [1.0, 2.0, 3.0],
                    [7.0, 8.0, 9.0],
                    [1.0, 2.0, 3.0],
                ],
                dtype=float,
            ),
            pd.DataFrame(
                [
                    [1.0, np.nan, 3.0],
                    [4.0, 5.0, np.nan],
                    [1.0, 2.0, 3.0],
                    [7.0, 8.0, 9.0],
                    [1.0, 2.0, 3.0],
                ],
                dtype=float,
            ),
        ),
        (
            np.array([[True, False], [False, True], [True, True], [False, False], [True, False]]),
            pd.DataFrame(
                [[True, False], [False, True], [True, True], [False, False], [True, False]]
            ),
        ),
        (
            pd.date_range("2020-01-01", periods=5).to_numpy().reshape(-1, 1),
            pd.DataFrame(pd.date_range("2020-01-01", periods=5).to_numpy().reshape(-1, 1)),
        ),
    ],
)
def test_inference_parity_dataframe_vs_ndarray(array, frame):
    from_array = infer_feature_types(array, numeric_threshold=2, cardinality_threshold=3)
    from_frame = infer_feature_types(frame, numeric_threshold=2, cardinality_threshold=3)
    assert from_array == from_frame


def test_categorical_imputer_constant_creates_missing_category():
    X = pd.DataFrame({"cat": ["a", "b", np.nan, "a", "b"]})
    y = np.array([0, 1, 0, 1, 0])
    pp = PoniardPreprocessor(categorical_imputer="constant").build(X=X, y=y, task="classification")
    out = pp.preprocessor.fit_transform(X, y)
    assert "cat_missing" in out.columns


def test_cyclical_datetime_emits_sin_cos_pairs():
    X = pd.DataFrame({"D": pd.date_range("2020-01-01", freq="h", periods=100)})
    y = np.random.RandomState(0).randint(0, 2, size=100)
    pp = PoniardPreprocessor(cyclical_datetime=True).build(X=X, y=y, task="classification")
    out = pp.preprocessor.fit_transform(X, y)
    cols = set(out.columns)
    assert {"D_hour_sin", "D_hour_cos"} <= cols
    assert {"D_dayofyear_sin", "D_dayofyear_cos"} <= cols


def test_datetime_encoder_cyclical_wrap_around():
    enc = DatetimeEncoder(levels=[DateLevel.HOUR], cyclical=True)
    X = pd.DataFrame({"D": pd.to_datetime(["2020-01-01 23:00", "2020-01-02 00:00"])})
    enc.fit(X)
    out = enc.transform(X)
    assert out.shape == (2, 2)
    assert np.allclose(out[0, 0], out[1, 0], atol=0.5)
    assert np.allclose(out[0, 1], out[1, 1], atol=0.5)
    assert enc.get_feature_names_out() == ["D_hour_sin", "D_hour_cos"]


def test_ordinal_encoder_unknown_yields_nan():
    X = pd.DataFrame({"high": [f"cat_{i}" for i in range(50)]})
    y = np.random.RandomState(0).randint(0, 2, size=50)
    pp = PoniardPreprocessor(high_cardinality_encoder="ordinal", cardinality_threshold=5).build(
        X=X, y=y, task="classification"
    )
    pp.preprocessor.fit(X, y)
    out = pp.preprocessor.transform(pd.DataFrame({"high": ["unseen"]}))
    assert np.isnan(out.iloc[0, 0])
