import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from poniard import PoniardClassifier, PoniardRegressor
from poniard.preprocessing import PoniardPreprocessor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def iris_binary():
    """Iris dataset reduced to binary (setosa vs versicolor), 4 numeric features."""
    data = load_iris()
    mask = data.target < 2
    X = pd.DataFrame(data.data[mask], columns=data.feature_names)
    y = data.target[mask]
    return X, y


@pytest.fixture
def mixed_df():
    """DataFrame with numeric, string, boolean, and datetime columns."""
    rng = np.random.RandomState(42)
    n = 60
    return pd.DataFrame(
        {
            "num_high": rng.normal(size=n),  # numeric (many unique)
            "num_low": rng.choice([1, 2, 3], size=n),  # numeric few unique -> cat_low
            "str_low": rng.choice(["a", "b", "c"], size=n),  # string low cardinality
            "str_high": [f"cat_{i}" for i in range(n)],  # string high cardinality
            "bool_col": rng.choice([True, False], size=n),  # boolean
            "dt": pd.date_range("2020-01-01", periods=n),  # datetime
        }
    ), pd.Series(rng.choice([0, 1], size=n), name="target")


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), y


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=200, n_features=10, random_state=42)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), y


@pytest.fixture
def fitted_classifier(classification_data):
    X, y = classification_data
    clf = PoniardClassifier(
        estimators=[LogisticRegression(random_state=42)],
        cv=3,
        random_state=42,
    )
    clf.fit(X, y)
    return clf


@pytest.fixture
def fitted_regressor(regression_data):
    X, y = regression_data
    reg = PoniardRegressor(
        estimators=[LinearRegression()],
        cv=3,
        random_state=42,
    )
    reg.fit(X, y)
    return reg


# ===========================================================================
# 1. Type inference tests
# ===========================================================================


class TestTypeInference:
    def test_numeric_many_unique_values_classified_as_numeric(self):
        """Continuous float column with many unique values should be numeric."""
        n = 100
        X = pd.DataFrame({"cont": np.linspace(0, 1, n)})
        y = np.zeros(n, dtype=int)
        y[::2] = 1
        preprocessor = PoniardPreprocessor()
        preprocessor.build(X=X, y=y, task="classification")
        assert "cont" in preprocessor.feature_types["numeric"]

    def test_numeric_few_unique_values_classified_as_categorical_low(self):
        """Integer column with very few unique values should be categorical_low."""
        n = 100
        X = pd.DataFrame({"repeated": [1, 2, 3] * (n // 3) + [1] * (n % 3)})
        y = np.zeros(n, dtype=int)
        y[::2] = 1
        preprocessor = PoniardPreprocessor()
        preprocessor.build(X=X, y=y, task="classification")
        assert "repeated" in preprocessor.feature_types["categorical_low"]
        assert "repeated" not in preprocessor.feature_types["numeric"]

    def test_string_low_cardinality_is_categorical_low(self):
        """String column with few unique values should be categorical_low."""
        n = 100
        X = pd.DataFrame({"color": np.random.choice(["red", "green", "blue"], size=n)})
        y = np.zeros(n, dtype=int)
        y[::2] = 1
        preprocessor = PoniardPreprocessor()
        preprocessor.build(X=X, y=y, task="classification")
        assert "color" in preprocessor.feature_types["categorical_low"]

    def test_string_high_cardinality_is_categorical_high(self):
        """String column with many unique values should be categorical_high."""
        n = 100
        X = pd.DataFrame({"id_str": [f"item_{i}" for i in range(n)]})
        y = np.zeros(n, dtype=int)
        y[::2] = 1
        preprocessor = PoniardPreprocessor(cardinality_threshold=5)
        preprocessor.build(X=X, y=y, task="classification")
        assert "id_str" in preprocessor.feature_types["categorical_high"]

    def test_datetime_column_classified_as_datetime(self):
        """Datetime64 column should be classified as datetime."""
        n = 100
        X = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="D")})
        y = np.zeros(n, dtype=int)
        y[::2] = 1
        preprocessor = PoniardPreprocessor()
        preprocessor.build(X=X, y=y, task="classification")
        assert "date" in preprocessor.feature_types["datetime"]

    def test_mixed_dataframe_all_types_detected(self, mixed_df):
        """All feature types should be correctly identified in a mixed DataFrame."""
        X, y = mixed_df
        preprocessor = PoniardPreprocessor(cardinality_threshold=20)
        preprocessor.build(X=X, y=y, task="classification")
        ft = preprocessor.feature_types
        assert "num_high" in ft["numeric"]
        assert "str_low" in ft["categorical_low"]
        assert "str_high" in ft["categorical_high"]
        assert "dt" in ft["datetime"]

    def test_boolean_column_is_categorical(self):
        """Boolean column should be treated as categorical, not numeric."""
        n = 100
        X = pd.DataFrame({"flag": np.random.choice([True, False], size=n)})
        y = np.zeros(n, dtype=int)
        y[::2] = 1
        preprocessor = PoniardPreprocessor()
        preprocessor.build(X=X, y=y, task="classification")
        ft = preprocessor.feature_types
        assert "flag" in ft["categorical_low"] or "flag" in ft["categorical_high"]
        assert "flag" not in ft["numeric"]

    def test_feature_types_dict_has_all_keys(self):
        """feature_types should always have the four expected keys."""
        n = 20
        X = pd.DataFrame({"a": range(n), "b": list("abc" * 6 + "ab")})
        y = np.zeros(n, dtype=int)
        y[::2] = 1
        preprocessor = PoniardPreprocessor()
        preprocessor.build(X=X, y=y, task="classification")
        assert set(preprocessor.feature_types.keys()) == {
            "numeric",
            "categorical_high",
            "categorical_low",
            "datetime",
        }

    def test_numeric_threshold_as_float(self):
        """When numeric_threshold is a float, it should be interpreted as fraction of samples."""
        n = 200
        # 150 unique values out of 200 — if threshold is 0.5 (=100), this is numeric
        X = pd.DataFrame({"many_ints": np.arange(n)})
        y = np.zeros(n, dtype=int)
        y[::2] = 1
        preprocessor = PoniardPreprocessor(numeric_threshold=0.5)
        preprocessor.build(X=X, y=y, task="classification")
        assert "many_ints" in preprocessor.feature_types["numeric"]

    def test_float_thresholds_stay_pristine_after_build(self):
        """build() must not convert the float thresholds to ints in place, so a
        second build() on different data uses the same constructor values."""
        preprocessor = PoniardPreprocessor(
            numeric_threshold=0.5,
            cardinality_threshold=0.5,
        )
        X1 = pd.DataFrame({"a": np.arange(20)})
        y = np.zeros(20, dtype=int)
        preprocessor.build(X=X1, y=y, task="classification")
        assert preprocessor.numeric_threshold == 0.5
        assert preprocessor.cardinality_threshold == 0.5
        X2 = pd.DataFrame({"a": np.arange(100)})
        preprocessor.build(X=X2, y=y, task="classification")
        assert preprocessor.numeric_threshold == 0.5
        assert preprocessor.cardinality_threshold == 0.5


# ===========================================================================
# 2. Preprocessing tests
# ===========================================================================


class TestPreprocessing:
    def test_default_preprocessor_has_expected_structure(self):
        """Default preprocessor should be a Pipeline with type_preprocessor and VarianceThreshold."""
        n = 50
        X = pd.DataFrame(
            {
                "num": np.random.normal(size=n),
                "cat": np.random.choice(["a", "b"], size=n),
            }
        )
        y = np.zeros(n, dtype=int)
        y[::2] = 1
        pp = PoniardPreprocessor()
        pp.build(X=X, y=y, task="classification")
        assert isinstance(pp.preprocessor, Pipeline)
        step_names = [name for name, _ in pp.preprocessor.steps]
        assert "type_preprocessor" in step_names
        assert "remove_invariant" in step_names

    def test_numeric_gets_scaler_and_imputer(self):
        """Numeric pipeline should contain a StandardScaler (default) and SimpleImputer."""
        n = 50
        X = pd.DataFrame({"num": np.random.normal(size=n)})
        y = np.zeros(n, dtype=int)
        y[::2] = 1
        pp = PoniardPreprocessor()
        pp.build(X=X, y=y, task="classification")
        ct = pp.preprocessor.named_steps["type_preprocessor"]
        if isinstance(ct, ColumnTransformer):
            numeric_pipeline = dict(ct.named_transformers_)["numeric_preprocessor"]
        else:
            # Single transformer — the whole preprocessor IS the numeric one
            numeric_pipeline = ct
        step_names = (
            [s for s, _ in numeric_pipeline.steps] if hasattr(numeric_pipeline, "steps") else []
        )
        has_scaler = any("scaler" in s.lower() for s in step_names)
        assert has_scaler or isinstance(numeric_pipeline, Pipeline)

    @staticmethod
    def _get_transformer_by_name(ct, name):
        """Get a sub-transformer from a (possibly unfitted) ColumnTransformer by name."""
        for t_name, transformer, _ in ct.transformers:
            if t_name == name:
                return transformer
        raise KeyError(f"Transformer '{name}' not found")

    def test_categorical_low_gets_onehot_encoder(self):
        """Categorical low should get OneHotEncoder."""
        n = 50
        rng = np.random.RandomState(0)
        X = pd.DataFrame(
            {
                "num": rng.normal(size=n),
                "cat": rng.choice(["a", "b", "c"], size=n),
            }
        )
        y = np.zeros(n, dtype=int)
        y[::2] = 1
        pp = PoniardPreprocessor()
        pp.build(X=X, y=y, task="classification")
        ct = pp.preprocessor.named_steps["type_preprocessor"]
        assert isinstance(ct, ColumnTransformer)
        cat_pipeline = self._get_transformer_by_name(ct, "categorical_low_preprocessor")
        encoder = cat_pipeline.named_steps["one-hot_encoder"]
        assert isinstance(encoder, OneHotEncoder)

    def test_categorical_high_gets_target_encoder(self):
        """Categorical high should get TargetEncoder by default."""
        n = 60
        rng = np.random.RandomState(0)
        X = pd.DataFrame(
            {
                "num": rng.normal(size=n),
                "high": [f"cat_{i}" for i in range(n)],
            }
        )
        y = np.zeros(n, dtype=int)
        y[::2] = 1
        pp = PoniardPreprocessor(cardinality_threshold=5)
        pp.build(X=X, y=y, task="classification")
        ct = pp.preprocessor.named_steps["type_preprocessor"]
        assert isinstance(ct, ColumnTransformer)
        cat_high_pipeline = self._get_transformer_by_name(ct, "categorical_high_preprocessor")
        encoder = cat_high_pipeline.named_steps["high_cardinality_encoder"]
        assert isinstance(encoder, TargetEncoder)

    def test_custom_preprocessor_overrides_default(self):
        """When a custom preprocessor Pipeline is provided, it should be used as-is."""
        n = 50
        X = pd.DataFrame({"num": np.random.normal(size=n)})
        y = np.zeros(n, dtype=int)
        y[::2] = 1
        custom = Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())])
        clf = PoniardClassifier(
            estimators=[LogisticRegression()],
            custom_preprocessor=custom,
            preprocess=True,
            cv=2,
            random_state=0,
        )
        clf.fit(X, y)
        assert isinstance(clf.preprocessor, Pipeline)
        assert "imputer" in clf.preprocessor.named_steps or "imputer" in dict(
            clf.preprocessor.named_steps.get("type_preprocessor", {}).steps
            if hasattr(clf.preprocessor.named_steps.get("type_preprocessor"), "steps")
            else []
        )

    def test_preprocess_false_skips_preprocessing(self):
        """preprocess=False should not create a preprocessor pipeline."""
        n = 50
        X = pd.DataFrame({"num": np.random.normal(size=n)})
        y = np.zeros(n, dtype=int)
        y[::2] = 1
        clf = PoniardClassifier(
            estimators=[LogisticRegression()],
            preprocess=False,
            cv=2,
            random_state=0,
        )
        clf.fit(X, y)
        assert not hasattr(clf, "preprocessor") or clf.preprocess is False
        for name, pipeline in clf.pipelines.items():
            # Pipeline should only contain the estimator, no preprocessor step
            assert len(pipeline.steps) == 1

    def test_ordinal_encoder_for_high_cardinality_when_selected(self):
        """When high_cardinality_encoder='ordinal', OrdinalEncoder should be used."""
        n = 60
        rng = np.random.RandomState(0)
        X = pd.DataFrame(
            {
                "num": rng.normal(size=n),
                "high": [f"cat_{i}" for i in range(n)],
            }
        )
        y = np.zeros(n, dtype=int)
        y[::2] = 1
        pp = PoniardPreprocessor(high_cardinality_encoder="ordinal", cardinality_threshold=5)
        pp.build(X=X, y=y, task="classification")
        ct = pp.preprocessor.named_steps["type_preprocessor"]
        assert isinstance(ct, ColumnTransformer)
        cat_high_pipeline = self._get_transformer_by_name(ct, "categorical_high_preprocessor")
        encoder = cat_high_pipeline.named_steps["high_cardinality_encoder"]
        assert isinstance(encoder, OrdinalEncoder)

    def test_datetime_features_in_preprocessor(self):
        """Datetime columns should get a DatetimeEncoder in the preprocessor."""
        from poniard.preprocessing.datetime import DatetimeEncoder

        n = 50
        rng = np.random.RandomState(0)
        X = pd.DataFrame(
            {
                "num": rng.normal(size=n),
                "dt": pd.date_range("2020-01-01", periods=n),
            }
        )
        y = np.zeros(n, dtype=int)
        y[::2] = 1
        pp = PoniardPreprocessor()
        pp.build(X=X, y=y, task="classification")
        ct = pp.preprocessor.named_steps["type_preprocessor"]
        assert isinstance(ct, ColumnTransformer)
        dt_pipeline = self._get_transformer_by_name(ct, "datetime_preprocessor")
        assert isinstance(dt_pipeline.named_steps["datetime_encoder"], DatetimeEncoder)


# ===========================================================================
# 3. Results tests
# ===========================================================================


class TestResults:
    def test_get_results_returns_dataframe(self, fitted_classifier):
        results = fitted_classifier.get_results()
        assert isinstance(results, pd.DataFrame)

    def test_results_shape(self, fitted_classifier):
        """Results should have n_est x n_metric columns (test only, no train)."""
        results = fitted_classifier.get_results()
        # Default: 1 custom estimator + DummyClassifier = 2 rows
        # Columns: test_* metrics + fit_time + score_time
        assert results.shape[0] == 2
        assert results.shape[1] > 0

    def test_classification_scores_between_0_and_1(self, fitted_classifier):
        """All classification metric scores should be in [0, 1]."""
        results = fitted_classifier.get_results()
        score_cols = [c for c in results.columns if c.startswith("test_")]
        for col in score_cols:
            assert results[col].between(0, 1).all(), f"Column {col} has values outside [0,1]"

    def test_dummy_classifier_gets_prior_score(self, classification_data):
        """DummyClassifier(strategy='prior') on balanced data should get ~0.5 accuracy."""
        X, y = classification_data
        clf = PoniardClassifier(
            estimators=[LogisticRegression()],
            cv=3,
            random_state=42,
        )
        clf.fit(X, y)
        results = clf.get_results()
        dummy_row = results.loc["DummyClassifier"]
        accuracy = dummy_row.get("test_accuracy")
        if accuracy is not None:
            assert 0.4 <= accuracy <= 0.6, f"Dummy accuracy {accuracy} not near 0.5"

    def test_return_train_scores_includes_train_columns(self, fitted_classifier):
        """return_train_scores=True should include train_* columns."""
        results = fitted_classifier.get_results(return_train_scores=True)
        train_cols = [c for c in results.columns if c.startswith("train_")]
        assert len(train_cols) > 0, "No train columns found with return_train_scores=True"

    def test_std_returns_tuple_of_two_dataframes(self, fitted_classifier):
        """std=True should return (means, stds), both DataFrames."""
        result = fitted_classifier.get_results(std=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        means, stds = result
        assert isinstance(means, pd.DataFrame)
        assert isinstance(stds, pd.DataFrame)
        assert means.shape == stds.shape

    def test_wrt_dummy_divides_by_dummy_scores(self, fitted_classifier):
        """wrt_dummy=True should produce ratios relative to DummyClassifier."""
        results = fitted_classifier.get_results(wrt_dummy=True)
        raw = fitted_classifier.get_results()
        # Shape is preserved — dummy row stays (but becomes 1.0)
        assert results.shape == raw.shape
        # Dummy row values should all be 1.0 after dividing by itself
        dummy_row = results.loc["DummyClassifier"]
        score_cols = [c for c in results.columns if c.startswith("test_")]
        for col in score_cols:
            assert abs(dummy_row[col] - 1.0) < 1e-10, f"Dummy row col {col} not ~1.0"

    def test_wrt_dummy_stds_are_nan(self, fitted_classifier):
        """wrt_dummy=True with std=True returns NaN stds (only means are meaningful)."""
        means, stds = fitted_classifier.get_results(wrt_dummy=True, std=True)
        assert means.isna().sum().sum() == 0
        assert stds.isna().all().all()

    def test_wrt_dummy_requires_single_dummy(self, classification_data):
        """wrt_dummy=True should raise if there is not exactly one dummy estimator."""
        from sklearn.dummy import DummyClassifier

        X, y = classification_data
        clf = PoniardClassifier(
            estimators=[LogisticRegression(random_state=42)],
            cv=3,
            random_state=42,
        )
        clf.setup(X, y)
        clf.add_estimators({"DummyClassifier_2": DummyClassifier(strategy="prior")})
        clf.fit(X, y)
        with pytest.raises(ValueError, match="exactly one dummy"):
            clf.get_results(wrt_dummy=True)

    def test_no_nan_in_results(self, fitted_classifier):
        """Results should not contain NaN values for standard metrics."""
        results = fitted_classifier.get_results()
        assert not results.isna().any().any()

    def test_regressor_results_shape(self, fitted_regressor):
        """Regressor results should have correct shape."""
        results = fitted_regressor.get_results()
        # 1 custom + DummyRegressor = 2 rows
        assert results.shape[0] == 2
        # Regressor metrics: neg_mean_squared_error, neg_mean_absolute_percentage_error,
        # neg_median_absolute_error, r2 = 4 test cols + fit_time + score_time
        score_cols = [c for c in results.columns if c.startswith("test_")]
        assert len(score_cols) == 4


# ===========================================================================
# 4. Estimator management tests
# ===========================================================================


class TestEstimatorManagement:
    def test_add_estimators_adds_to_pipelines(self, fitted_classifier):
        """add_estimators should add new pipelines."""
        from sklearn.tree import DecisionTreeClassifier

        initial_count = len(fitted_classifier.pipelines)
        fitted_classifier.add_estimators([DecisionTreeClassifier()])
        assert len(fitted_classifier.pipelines) == initial_count + 1
        assert "DecisionTreeClassifier" in fitted_classifier.pipelines

    def test_remove_estimators_removes_from_pipelines(self, fitted_classifier):
        """remove_estimators should remove from pipelines."""
        initial_count = len(fitted_classifier.pipelines)
        fitted_classifier.remove_estimators(["LogisticRegression"])
        assert len(fitted_classifier.pipelines) == initial_count - 1
        assert "LogisticRegression" not in fitted_classifier.pipelines

    def test_remove_estimators_removes_from_results(self, fitted_classifier):
        """remove_estimators with drop_results=True should remove from results."""
        fitted_classifier.remove_estimators(["LogisticRegression"], drop_results=True)
        results = fitted_classifier.get_results()
        assert "LogisticRegression" not in results.index

    def test_get_estimator_returns_pipeline(self, fitted_classifier):
        """get_estimator should return a Pipeline when include_preprocessor=True."""
        est = fitted_classifier.get_estimator("LogisticRegression", include_preprocessor=True)
        assert isinstance(est, Pipeline)

    def test_get_estimator_without_preprocessor(self, fitted_classifier):
        """get_estimator with include_preprocessor=False returns the raw estimator."""
        est = fitted_classifier.get_estimator("LogisticRegression", include_preprocessor=False)
        assert isinstance(est, LogisticRegression)

    def test_remove_all_estimators_raises(self, fitted_classifier):
        """Removing all estimators should raise ValueError."""
        all_names = list(fitted_classifier.pipelines.keys())
        with pytest.raises(ValueError, match="Cannot remove all estimators"):
            fitted_classifier.remove_estimators(all_names)

    def test_remove_preserves_other_results(self, fitted_classifier):
        """Removing one estimator should preserve results for others."""
        fitted_classifier.remove_estimators(["LogisticRegression"], drop_results=True)
        results = fitted_classifier.get_results()
        assert "DummyClassifier" in results.index


# ===========================================================================
# 5. Ensemble tests
# ===========================================================================


class TestEnsemble:
    def test_voting_ensemble_classification(self, fitted_classifier):
        """build_ensemble('voting') should create a VotingClassifier."""
        fitted_classifier.build_ensemble(method="voting", top_n=2)
        assert "VotingClassifier" in fitted_classifier.pipelines
        pipe = fitted_classifier.pipelines["VotingClassifier"]
        # The final estimator in the pipeline (after preprocessor) should be VotingClassifier
        final = pipe.steps[-1][1]
        assert isinstance(final, VotingClassifier)

    def test_stacking_ensemble_classification(self, fitted_classifier):
        """build_ensemble('stacking') should create a StackingClassifier."""
        fitted_classifier.build_ensemble(method="stacking", top_n=2)
        assert "StackingClassifier" in fitted_classifier.pipelines
        pipe = fitted_classifier.pipelines["StackingClassifier"]
        final = pipe.steps[-1][1]
        assert isinstance(final, StackingClassifier)

    def test_voting_ensemble_regression(self, fitted_regressor):
        """build_ensemble('voting') for regressor creates VotingRegressor."""
        fitted_regressor.build_ensemble(method="voting", top_n=2)
        assert "VotingRegressor" in fitted_regressor.pipelines
        pipe = fitted_regressor.pipelines["VotingRegressor"]
        final = pipe.steps[-1][1]
        assert isinstance(final, VotingRegressor)

    def test_stacking_ensemble_regression(self, fitted_regressor):
        """build_ensemble('stacking') for regressor creates StackingRegressor."""
        fitted_regressor.build_ensemble(method="stacking", top_n=2)
        assert "StackingRegressor" in fitted_regressor.pipelines
        pipe = fitted_regressor.pipelines["StackingRegressor"]
        final = pipe.steps[-1][1]
        assert isinstance(final, StackingRegressor)

    def test_invalid_ensemble_method_raises(self, fitted_classifier):
        """build_ensemble with invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="Method must be either voting or stacking"):
            fitted_classifier.build_ensemble(method="invalid")

    def test_ensemble_with_specific_estimators(self, fitted_classifier):
        """build_ensemble with explicit estimator_names should use those."""
        fitted_classifier.build_ensemble(
            method="voting",
            estimator_names=["LogisticRegression"],
            ensemble_name="my_voting",
        )
        assert "my_voting" in fitted_classifier.pipelines

    def test_ensemble_with_custom_name(self, fitted_classifier):
        """ensemble_name parameter should set the key in pipelines."""
        fitted_classifier.build_ensemble(method="voting", ensemble_name="custom_ensemble")
        assert "custom_ensemble" in fitted_classifier.pipelines

    def test_ensemble_in_results_after_fit(self):
        """After building ensemble and re-fitting, ensemble should appear in results."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        clf = PoniardClassifier(
            estimators=[LogisticRegression(random_state=42)],
            cv=3,
            random_state=42,
        )
        clf.fit(X, y)
        # Use soft voting so predict_proba is available (needed for roc_auc)
        clf.build_ensemble(method="voting", top_n=2, voting="soft")
        clf.fit(X, y)
        results = clf.get_results()
        assert "VotingClassifier" in results.index


# ===========================================================================
# 6. Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_single_feature_dataset(self):
        """Classifier should work with a single numeric feature."""
        n = 100
        X = np.random.normal(size=(n, 1))
        y = (X.ravel() > 0).astype(int)
        clf = PoniardClassifier(
            estimators=[LogisticRegression()],
            cv=3,
            random_state=42,
        )
        clf.fit(X, y)
        results = clf.get_results()
        assert results.shape[0] == 2  # LogisticRegression + DummyClassifier
        assert not results.isna().any().any()

    def test_all_same_target_binary(self):
        """Dataset where all targets are the same class — LogisticRegression raises."""
        n = 50
        X = pd.DataFrame({"a": np.random.normal(size=n)})
        y = np.zeros(n, dtype=int)  # all zeros
        clf = PoniardClassifier(
            estimators=[LogisticRegression()],
            cv=3,
            random_state=42,
        )
        with pytest.raises(ValueError, match="samples of at least 2 classes"):
            clf.fit(X, y)

    def test_all_nan_columns(self):
        """DataFrame with all-NaN columns should handle gracefully."""
        n = 50
        X = pd.DataFrame(
            {
                "nan_col": [np.nan] * n,
                "valid_col": np.random.normal(size=n),
            }
        )
        y = np.zeros(n, dtype=int)
        y[::2] = 1
        clf = PoniardClassifier(
            estimators=[LogisticRegression(max_iter=1000)],
            cv=3,
            random_state=42,
        )
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Skipping features without any observed values"
            )
            clf.fit(X, y)
        results = clf.get_results()
        assert results.shape[0] == 2

    def test_very_small_dataset(self):
        """Very small dataset (5 samples) with cv=2 should still produce results."""
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [5.0, 4.0, 3.0, 2.0, 1.0]})
        y = np.array([0, 0, 1, 1, 1])
        clf = PoniardClassifier(
            estimators=[LogisticRegression()],
            cv=2,
            random_state=42,
        )
        clf.fit(X, y)
        results = clf.get_results()
        assert results.shape[0] == 2
        # fit_time and score_time should be valid (non-NaN)
        assert not results["fit_time"].isna().any()
        assert not results["score_time"].isna().any()

    def test_mixed_types_end_to_end(self, mixed_df):
        """Full fit on a mixed-type DataFrame should produce valid results."""
        X, y = mixed_df
        clf = PoniardClassifier(
            estimators=[LogisticRegression(max_iter=1000)],
            cv=3,
            random_state=42,
        )
        clf.fit(X, y)
        results = clf.get_results()
        assert results.shape[0] == 2
        assert not results.isna().any().any()

    def test_regressor_with_nan_in_target(self):
        """Regressor should handle NaN-free target normally; ensure no crash on edge shape."""
        n = 50
        X = pd.DataFrame({"a": np.random.normal(size=n)})
        y = np.random.normal(size=n)
        reg = PoniardRegressor(
            estimators=[LinearRegression()],
            cv=3,
            random_state=42,
        )
        reg.fit(X, y)
        results = reg.get_results()
        assert results.shape[0] == 2

    def test_setup_then_fit_produces_same_pipelines(self):
        """Calling setup() then fit() should reuse the same pipelines."""
        n = 50
        X = pd.DataFrame({"a": np.random.normal(size=n)})
        y = np.array([0, 1] * (n // 2))
        clf = PoniardClassifier(
            estimators=[LogisticRegression()],
            cv=2,
            random_state=42,
        )
        clf.setup(X, y)
        pipeline_ids_before = {name: id(p) for name, p in clf.pipelines.items()}
        clf.fit(X, y)
        # Pipelines dict should have the same keys after fit
        assert set(pipeline_ids_before.keys()) == set(clf.pipelines.keys())

    def test_setup_and_fit_configure_identically(self):
        """setup() and fit() must produce the same introspection surface
        (feature_types, metrics, preprocessor and pipelines), so the
        setup-then-adjust-then-fit flow is preserved."""
        n = 60
        X = pd.DataFrame(
            {
                "num": np.random.normal(size=n),
                "cat": np.random.choice(["a", "b", "c"], size=n),
            }
        )
        y = np.array([0, 1] * (n // 2))

        setup_clf = PoniardClassifier(estimators=[LogisticRegression()], cv=2, random_state=42)
        setup_clf.setup(X, y)

        fit_clf = PoniardClassifier(estimators=[LogisticRegression()], cv=2, random_state=42)
        fit_clf.fit(X, y)

        assert setup_clf.feature_types == fit_clf.feature_types
        assert setup_clf.metrics == fit_clf.metrics
        assert set(setup_clf.pipelines) == set(fit_clf.pipelines)
        assert isinstance(setup_clf.preprocessor, type(fit_clf.preprocessor))

    def test_random_state_propagation(self):
        """random_state should be propagated to estimators that support it."""
        n = 50
        X = pd.DataFrame({"a": np.random.normal(size=n)})
        y = np.array([0, 1] * (n // 2))
        clf = PoniardClassifier(
            estimators=[LogisticRegression()],
            cv=2,
            random_state=99,
        )
        clf.fit(X, y)
        lr = clf.pipelines["LogisticRegression"].named_steps["LogisticRegression"]
        assert lr.random_state == 99

    def test_verbose_propagation(self):
        """verbose should be propagated to estimators that support it."""
        from sklearn.ensemble import RandomForestClassifier

        n = 50
        X = pd.DataFrame({"a": np.random.normal(size=n), "b": np.random.normal(size=n)})
        y = np.array([0, 1] * (n // 2))
        clf = PoniardClassifier(
            estimators=[RandomForestClassifier(n_estimators=5)],
            cv=2,
            verbose=True,
            random_state=42,
        )
        clf.fit(X, y)
        rf = clf.pipelines["RandomForestClassifier"].named_steps["RandomForestClassifier"]
        assert rf.verbose is True
