from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_multilabel_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from poniard import PoniardClassifier, PoniardRegressor
from poniard.error_analysis import ErrorAnalyzer

N = 60


@pytest.fixture
def binary_data():
    X = pd.DataFrame(
        {
            "a": np.random.normal(size=N),
            "b": np.random.normal(size=N),
            "c": np.random.choice(["x", "y", "z"], size=N),
        }
    )
    y = pd.Series(np.random.choice([0, 1], size=N))
    clf = PoniardClassifier(estimators={"lr": LogisticRegression()}, cv=2, random_state=0)
    clf.setup(X, y)
    clf.fit(X, y)
    return clf, X, y


@pytest.fixture
def multiclass_data():
    X = pd.DataFrame(
        {
            "a": np.random.normal(size=N),
            "b": np.random.normal(size=N),
        }
    )
    y = pd.Series(np.random.choice([0, 1, 2], size=N))
    clf = PoniardClassifier(
        estimators={"lr": LogisticRegression(max_iter=5000)},
        cv=2,
        random_state=0,
    )
    clf.setup(X, y)
    clf.fit(X, y)
    return clf, X, y


@pytest.fixture
def multilabel_data():
    X, y = make_multilabel_classification(n_samples=N, n_features=3, n_classes=3, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    clf = PoniardClassifier(
        estimators={"lr": OneVsRestClassifier(LogisticRegression(max_iter=5000))},
        cv=2,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TargetEncoder is not supported")
        clf.setup(X, y)
        clf.fit(X, y)
    return clf, X, y


@pytest.fixture
def reg_data():
    X = pd.DataFrame(
        {
            "a": np.random.normal(size=N),
            "b": np.random.normal(size=N),
        }
    )
    y = pd.Series(np.random.normal(size=N))
    reg = PoniardRegressor(estimators={"lr": LinearRegression()}, cv=2, random_state=0)
    reg.setup(X, y)
    reg.fit(X, y)
    return reg, X, y


@pytest.fixture
def multioutput_reg_data():
    X, y = make_regression(n_samples=N, n_features=3, n_targets=2, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = pd.DataFrame(y, columns=[f"t{i}" for i in range(y.shape[1])])
    reg = PoniardRegressor(estimators={"lr": LinearRegression()}, cv=2, random_state=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TargetEncoder is not supported")
        reg.setup(X, y)
        reg.fit(X, y)
    return reg, X, y


def _make_ea(clf, estimator_names: str | Sequence[str] = "lr"):
    return ErrorAnalyzer.from_poniard(clf, estimator_names)


class TestFromPoniard:
    def test_construction(self, binary_data):
        clf, _, _ = binary_data
        ea = _make_ea(clf)
        assert ea._has_poniard
        assert ea.task == "classification"
        assert ea.estimator_names == ["lr"]
        assert ea.type_of_target == "binary"

    def test_construction_multiple_estimators(self, binary_data):
        clf, _, _ = binary_data
        ea = _make_ea(clf, estimator_names=["lr", "DummyClassifier"])
        assert ea.estimator_names == ["lr", "DummyClassifier"]

    def test_standalone_construction(self):
        ea = ErrorAnalyzer(task="classification")
        assert not ea._has_poniard
        assert ea.task == "classification"


class TestRankErrors:
    def test_binary(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        result = ea.rank_errors(X=X, y=y)
        assert isinstance(result, dict)
        assert "lr" in result
        assert isinstance(result["lr"], pd.DataFrame)
        assert "error" in result["lr"].columns

    def test_multiclass(self, multiclass_data):
        clf, X, y = multiclass_data
        ea = _make_ea(clf)
        result = ea.rank_errors(X=X, y=y)
        assert "proba_0" in result["lr"].columns

    def test_multilabel(self, multilabel_data):
        clf, X, y = multilabel_data
        ea = _make_ea(clf)
        result = ea.rank_errors(X=X, y=y)
        assert result["lr"].shape[1] >= 3
        assert "error" in result["lr"].columns

    def test_continuous(self, reg_data):
        clf, X, y = reg_data
        ea = _make_ea(clf)
        result = ea.rank_errors(X=X, y=y)
        assert "prediction" in result["lr"].columns
        assert "error" in result["lr"].columns

    def test_continuous_multioutput(self, multioutput_reg_data):
        clf, X, y = multioutput_reg_data
        ea = _make_ea(clf)
        result = ea.rank_errors(X=X, y=y)
        assert "prediction_0" in result["lr"].columns
        assert "error" in result["lr"].columns

    def test_exclude_correct_false(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        result_excluded = ea.rank_errors(X=X, y=y, exclude_correct=True)
        result_all = ea.rank_errors(X=X, y=y, exclude_correct=False)
        assert len(result_all["lr"]) >= len(result_excluded["lr"])

    def test_regression_error_quantile(self, reg_data):
        clf, X, y = reg_data
        ea = _make_ea(clf)
        strict = ea.rank_errors(X=X, y=y, error_quantile=0.1)
        lax = ea.rank_errors(X=X, y=y, error_quantile=0.5)
        assert len(lax["lr"]) > len(strict["lr"])
        assert lax["lr"]["error"].max() >= strict["lr"]["error"].max()

    def test_requires_X_and_y(self, binary_data):
        clf, _, _ = binary_data
        ea = _make_ea(clf)
        with pytest.raises(ValueError, match="X and y"):
            ea.rank_errors()

    def test_standalone_mode(self):
        n = 30
        y = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ]
        )
        predictions = np.zeros(n, dtype=int)
        probas = np.array([[0.9, 0.1]] * 15 + [[0.6, 0.4]] * 15)
        ea = ErrorAnalyzer(task="classification")
        result = ea.rank_errors(y=y, predictions=predictions, probas=probas)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 15
        assert np.allclose(result["error"].values, 0.6)


class TestMergeErrors:
    def test_merge(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf, estimator_names=["lr", "DummyClassifier"])
        errors = ea.rank_errors(X=X, y=y)
        merged = ErrorAnalyzer.merge_errors(errors)
        assert isinstance(merged, pd.DataFrame)
        assert "mean_error" in merged.columns
        assert "freq" in merged.columns
        assert "estimators" in merged.columns
        assert merged["freq"].max() <= len(errors)

    def test_merge_single_dataframe(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        errors = ea.rank_errors(X=X, y=y)["lr"]
        merged = ErrorAnalyzer.merge_errors(errors)
        assert isinstance(merged, pd.DataFrame)
        assert merged["freq"].max() == 1


class TestAnalyzeTarget:
    def test_classification(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        errors = ea.rank_errors(X=X, y=y)
        merged = ErrorAnalyzer.merge_errors(errors)
        analysis = ea.analyze_target(errors_idx=merged.index, y=y)
        assert isinstance(analysis, pd.DataFrame)
        assert not analysis.empty
        assert {"error_count", "target_count", "error_rate"} <= set(analysis.columns)

    def test_regression(self, reg_data):
        clf, X, y = reg_data
        ea = _make_ea(clf)
        errors = ea.rank_errors(X=X, y=y)
        merged = ErrorAnalyzer.merge_errors(errors)
        analysis = ea.analyze_target(errors_idx=merged.index, y=y)
        assert isinstance(analysis, pd.DataFrame)
        assert not analysis.empty
        assert {"error_count", "target_count", "error_rate"} <= set(analysis.columns)

    def test_error_rate_is_count_ratio(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        errors = ea.rank_errors(X=X, y=y)
        merged = ErrorAnalyzer.merge_errors(errors)
        analysis = ea.analyze_target(errors_idx=merged.index, y=y)
        expected = analysis["error_count"] / analysis["target_count"]
        assert np.allclose(analysis["error_rate"].values, expected.values)

    def test_requires_y(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        errors = ea.rank_errors(X=X, y=y)
        merged = ErrorAnalyzer.merge_errors(errors)
        with pytest.raises(ValueError, match="y"):
            ea.analyze_target(errors_idx=merged.index)

    def test_known_error_region(self):
        """Errors confined to one class must be surfaced by error_rate."""
        y = pd.Series(["easy"] * 50 + ["hard"] * 50)
        errors_idx = y.index[y == "hard"]
        ea = ErrorAnalyzer(task="classification")
        ea.type_of_target = "multiclass"
        analysis = ea.analyze_target(errors_idx=errors_idx, y=y)
        assert analysis.loc["hard", "error_count"] == 50
        assert analysis.loc["hard", "error_rate"] == pytest.approx(1.0)
        assert analysis.loc["easy", "error_count"] == 0
        assert analysis.loc["easy", "error_rate"] == 0.0


class TestAnalyzeFeatures:
    def test_all_features(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        errors = ea.rank_errors(X=X, y=y)
        merged = ErrorAnalyzer.merge_errors(errors)
        summary = ea.analyze_features(errors_idx=merged.index, X=X)
        assert isinstance(summary, dict)
        assert len(summary) > 0
        for feature_name, feature_summary in summary.items():
            assert isinstance(feature_summary, pd.DataFrame)

    def test_with_feature_subset_by_name(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        errors = ea.rank_errors(X=X, y=y)
        merged = ErrorAnalyzer.merge_errors(errors)
        summary = ea.analyze_features(errors_idx=merged.index, X=X, features=["a", "b"])
        assert set(summary.keys()) == {"a", "b"}

    def test_with_feature_subset_by_index(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        errors = ea.rank_errors(X=X, y=y)
        merged = ErrorAnalyzer.merge_errors(errors)
        summary = ea.analyze_features(errors_idx=merged.index, X=X, features=[0, 1])
        assert set(summary.keys()) == {"a", "b"}

    def test_with_estimator_name(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        errors = ea.rank_errors(X=X, y=y)
        merged = ErrorAnalyzer.merge_errors(errors)
        summary = ea.analyze_features(
            errors_idx=merged.index,
            X=X,
            y=y,
            estimator_name="lr",
            n_features=2,
        )
        assert isinstance(summary, dict)
        assert len(summary) <= X.shape[1]

    def test_with_estimator_name_n_features_float(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        errors = ea.rank_errors(X=X, y=y)
        merged = ErrorAnalyzer.merge_errors(errors)
        summary = ea.analyze_features(
            errors_idx=merged.index,
            X=X,
            y=y,
            estimator_name="lr",
            n_features=0.5,
        )
        assert isinstance(summary, dict)

    def test_numeric_feature_summary_structure(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        errors = ea.rank_errors(X=X, y=y)
        merged = ErrorAnalyzer.merge_errors(errors)
        summary = ea.analyze_features(errors_idx=merged.index, X=X)
        assert "count" in summary["a"].columns
        assert "mean" in summary["a"].columns

    def test_categorical_feature_has_error_rate(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        errors = ea.rank_errors(X=X, y=y)
        merged = ErrorAnalyzer.merge_errors(errors)
        summary = ea.analyze_features(errors_idx=merged.index, X=X)
        assert "c" in summary
        assert {"correct", "errors", "error_rate"} <= set(summary["c"].columns)

    def test_requires_X(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        errors = ea.rank_errors(X=X, y=y)
        merged = ErrorAnalyzer.merge_errors(errors)
        with pytest.raises(ValueError, match="X"):
            ea.analyze_features(errors_idx=merged.index)

    def test_estimator_name_requires_y(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        errors = ea.rank_errors(X=X, y=y)
        merged = ErrorAnalyzer.merge_errors(errors)
        with pytest.raises(ValueError, match="y"):
            ea.analyze_features(errors_idx=merged.index, X=X, estimator_name="lr")

    def test_estimator_name_requires_from_poniard(self, reg_data):
        clf, X, y = reg_data
        ea = ErrorAnalyzer(task="regression")
        with pytest.raises(ValueError, match="from_poniard"):
            ea.analyze_features(errors_idx=pd.Index([0]), X=X, y=y, estimator_name="lr")

    def test_categorical_known_error_region(self):
        """Errors confined to one category must surface as a high error_rate."""
        n = 100
        X = pd.DataFrame(
            {
                "num": np.random.normal(size=n),
                "cat": np.array(["a", "b"] * 50),
            }
        )
        errors_idx = X.index[X["cat"] == "b"]
        ea = ErrorAnalyzer(task="classification")
        summary = ea.analyze_features(errors_idx=errors_idx, X=X)
        cat_table = summary["cat"]
        assert cat_table.loc["b", "errors"] == 50
        assert cat_table.loc["b", "error_rate"] == pytest.approx(1.0)
        assert cat_table.loc["a", "errors"] == 0
        assert cat_table.loc["a", "error_rate"] == 0.0

    def test_numeric_known_error_region(self):
        """Errors confined to one end of a numeric range must separate the
        mean of the two error groups."""
        X = pd.DataFrame({"num": np.linspace(-5, 5, 100)})
        errors_idx = X.index[X["num"] > 0]
        ea = ErrorAnalyzer(task="classification")
        summary = ea.analyze_features(errors_idx=errors_idx, X=X)
        num_table = summary["num"]
        assert num_table.loc[1, "mean"] > 0
        assert num_table.loc[0, "mean"] < 0

    def test_permutation_importances_deterministic(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        errors = ea.rank_errors(X=X, y=y)
        merged = ErrorAnalyzer.merge_errors(errors)
        first = ea.analyze_features(
            errors_idx=merged.index, X=X, y=y, estimator_name="lr", n_features=2
        )
        second = ea.analyze_features(
            errors_idx=merged.index, X=X, y=y, estimator_name="lr", n_features=2
        )
        assert set(first) == set(second)


class TestAnalyze:
    def test_report_structure(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        report = ea.analyze(X=X, y=y)
        assert set(report) == {
            "ranked_errors",
            "merged_errors",
            "summary",
            "by_target",
            "by_feature",
            "universal_failures",
            "disagreement_set",
            "lift_by_target",
            "lift_by_feature",
        }
        assert "lr" in report["ranked_errors"]
        assert isinstance(report["merged_errors"], pd.DataFrame)
        assert not report["by_target"].empty
        assert len(report["by_feature"]) > 0

    def test_summary_error_rates(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf, estimator_names=["lr", "DummyClassifier"])
        report = ea.analyze(X=X, y=y)
        assert "error_rate" in report["summary"].columns
        expected = len(report["ranked_errors"]["lr"]) / len(y)
        assert report["summary"].loc["lr", "error_rate"] == pytest.approx(expected)

    def test_estimator_names_override(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf, estimator_names=["lr", "DummyClassifier"])
        report = ea.analyze(X=X, y=y, estimator_names="lr")
        assert set(report["ranked_errors"]) == {"lr"}

    def test_regression_report(self, reg_data):
        clf, X, y = reg_data
        ea = _make_ea(clf)
        report = ea.analyze(X=X, y=y)
        assert "error_rate" in report["summary"].columns
        assert not report["by_target"].empty
        assert len(report["by_feature"]) > 0

    def test_universal_failures(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf, estimator_names=["lr", "DummyClassifier"])
        report = ea.analyze(X=X, y=y)
        assert isinstance(report["universal_failures"], pd.DataFrame)
        merged = report["merged_errors"]
        n_estimators = len(report["ranked_errors"])
        expected = merged[merged["freq"] == n_estimators]
        assert len(report["universal_failures"]) == len(expected)

    def test_disagreement_set(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf, estimator_names=["lr", "DummyClassifier"])
        report = ea.analyze(X=X, y=y)
        assert isinstance(report["disagreement_set"], pd.DataFrame)
        merged = report["merged_errors"]
        n_estimators = len(report["ranked_errors"])
        expected = merged[(merged["freq"] > 0) & (merged["freq"] < n_estimators)]
        assert len(report["disagreement_set"]) == len(expected)

    def test_lift_by_target(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        report = ea.analyze(X=X, y=y)
        assert isinstance(report["lift_by_target"], pd.DataFrame)
        assert "lift" in report["lift_by_target"].columns
        assert "error_rate" in report["lift_by_target"].columns

    def test_lift_by_feature(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        report = ea.analyze(X=X, y=y)
        assert isinstance(report["lift_by_feature"], dict)
        for fname, ftable in report["lift_by_feature"].items():
            if "error_rate" in ftable.columns:
                assert "lift" in ftable.columns

    def test_report_bracket_access(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        report = ea.analyze(X=X, y=y)
        assert report["summary"] is report.summary
        assert report["by_target"] is report.by_target

    def test_report_contains(self, binary_data):
        clf, X, y = binary_data
        ea = _make_ea(clf)
        report = ea.analyze(X=X, y=y)
        assert "summary" in report
        assert "nonexistent" not in report

    def test_from_poniard_default_all_non_dummy(self, binary_data):
        clf, X, y = binary_data
        ea = ErrorAnalyzer.from_poniard(clf)
        assert len(ea.estimator_names) >= 1
        for name in ea.estimator_names:
            assert "Dummy" not in name


def test_repr():
    ea = ErrorAnalyzer(task="classification")
    r = repr(ea)
    assert "ErrorAnalyzer" in r


def test_rank_errors_multiclass_non_integer_labels():
    """String class labels must not crash error ranking; proba lookup is positional."""
    ea = ErrorAnalyzer(task="classification")
    ea.type_of_target = "multiclass"
    y = np.array(["cat", "dog", "mouse", "cat", "dog"])
    preds = np.array(["cat", "cat", "mouse", "dog", "dog"])
    probas = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.6, 0.3, 0.1],
            [0.1, 0.1, 0.8],
            [0.5, 0.4, 0.1],
            [0.1, 0.8, 0.1],
        ]
    )
    ranked = ea._rank_errors_multiclass(y, preds, probas)
    # Only rows 1 and 3 are misclassified; row 1 (dog, truth-proba 0.3) ranks first.
    assert list(ranked.index) == [1, 3]
    assert list(ranked["error"]) == pytest.approx([0.7, 0.5])


def test_analyze_multiclass_string_labels():
    """End-to-end error analysis must work with non-integer class labels."""
    X = pd.DataFrame(np.random.normal(size=(N, 2)))
    y = pd.Series(np.random.choice(["a", "b", "c"], size=N))
    clf = PoniardClassifier(
        estimators={"lr": LogisticRegression(max_iter=5000)}, cv=2, random_state=0
    )
    clf.setup(X, y)
    clf.fit(X, y)
    report = ErrorAnalyzer.from_poniard(clf).analyze(X, y)
    assert set(report.summary.index) == {"lr"}
