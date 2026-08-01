from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from poniard import PoniardClassifier

N = 100


@pytest.fixture
def fitted_clf():
    X, y = make_classification(
        n_samples=N, n_features=5, random_state=42, n_informative=3
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    clf = PoniardClassifier(
        estimators={
            "lr": LogisticRegression(max_iter=1000),
            "rf": RandomForestClassifier(n_estimators=10, random_state=0),
        },
        cv=3,
        random_state=0,
    )
    clf.fit(X, y)
    return clf, X, y


class TestCompare:
    def test_compare_returns_dataframe(self, fitted_clf):
        clf, X, y = fitted_clf
        result = clf.compare()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_compare_pairwise(self, fitted_clf):
        clf, X, y = fitted_clf
        result = clf.compare()
        estimator_pairs = result.index.droplevel(0).tolist()
        assert ("lr", "rf") in estimator_pairs

    def test_compare_has_expected_columns(self, fitted_clf):
        clf, X, y = fitted_clf
        result = clf.compare()
        assert "mean_diff" in result.columns
        assert "wins_a" in result.columns
        assert "wins_b" in result.columns
        assert "ties" in result.columns
        assert "p_value" in result.columns

    def test_compare_specific_estimators(self, fitted_clf):
        clf, X, y = fitted_clf
        result = clf.compare(estimators=["lr", "rf"])
        assert len(result) >= 1

    def test_compare_specific_metric(self, fitted_clf):
        clf, X, y = fitted_clf
        metrics = clf.get_results().columns.tolist()
        result = clf.compare(metrics=[metrics[0]])
        assert len(result) >= 1

    def test_compare_excludes_dummies_by_default(self, fitted_clf):
        clf, X, y = fitted_clf
        result = clf.compare()
        for _, a, b in result.index:
            assert "Dummy" not in a
            assert "Dummy" not in b

    def test_compare_single_estimator(self, fitted_clf):
        clf, X, y = fitted_clf
        result = clf.compare(estimators=["lr"])
        assert result.empty

    def test_get_results_has_per_sample_times(self, fitted_clf):
        clf, X, y = fitted_clf
        results = clf.get_results()
        assert "fit_time_per_sample" in results.columns
        assert "score_time_per_sample" in results.columns


class TestPareto:
    def test_pareto_returns_dataframe(self, fitted_clf):
        clf, X, y = fitted_clf
        result = clf.pareto()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_pareto_subset_of_results(self, fitted_clf):
        clf, X, y = fitted_clf
        pareto = clf.pareto()
        results = clf.get_results()
        assert set(pareto.index).issubset(set(results.index))

    def test_pareto_no_dummies(self, fitted_clf):
        clf, X, y = fitted_clf
        pareto = clf.pareto()
        for name in pareto.index:
            assert "Dummy" not in name

    def test_pareto_invalid_metric(self, fitted_clf):
        clf, X, y = fitted_clf
        with pytest.raises(ValueError, match="not found"):
            clf.pareto(metric="nonexistent")

    def test_pareto_by_score_time(self, fitted_clf):
        clf, X, y = fitted_clf
        result = clf.pareto(time_col="score_time")
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "score_time" in result.columns

    def test_pareto_by_per_sample_fit_time(self, fitted_clf):
        clf, X, y = fitted_clf
        result = clf.pareto(time_col="fit_time_per_sample")
        assert isinstance(result, pd.DataFrame)
        assert "fit_time_per_sample" in result.columns

    def test_pareto_by_per_sample_score_time(self, fitted_clf):
        clf, X, y = fitted_clf
        result = clf.pareto(time_col="score_time_per_sample")
        assert isinstance(result, pd.DataFrame)
        assert "score_time_per_sample" in result.columns


class TestBestUnder:
    def test_best_under_returns_name(self, fitted_clf):
        clf, X, y = fitted_clf
        result = clf.best_under(seconds=1000)
        assert isinstance(result, str)

    def test_best_under_is_fast_enough(self, fitted_clf):
        clf, X, y = fitted_clf
        name = clf.best_under(seconds=1000)
        results = clf.get_results()
        assert results.loc[name, "fit_time"] <= 1000

    def test_best_under_no_match_raises(self, fitted_clf):
        clf, X, y = fitted_clf
        with pytest.raises(ValueError, match="No estimator"):
            clf.best_under(seconds=0.0001)

    def test_best_under_picks_best_metric(self, fitted_clf):
        clf, X, y = fitted_clf
        name = clf.best_under(seconds=1000)
        results = clf.get_results()
        metric = results.columns[0]
        under = results[results["fit_time"] <= 1000]
        assert results.loc[name, metric] == under[metric].max()

    def test_best_under_by_score_time(self, fitted_clf):
        clf, X, y = fitted_clf
        name = clf.best_under(seconds=1000, time_col="score_time")
        results = clf.get_results()
        assert results.loc[name, "score_time"] <= 1000

    def test_best_under_by_per_sample_score_time(self, fitted_clf):
        clf, X, y = fitted_clf
        name = clf.best_under(seconds=1.0, time_col="score_time_per_sample")
        results = clf.get_results()
        assert results.loc[name, "score_time_per_sample"] <= 1.0
