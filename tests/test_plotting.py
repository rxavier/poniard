import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from poniard import PoniardClassifier, PoniardRegressor
from poniard.plot import PoniardPlotFactory


@pytest.fixture(scope="module")
def clf_setup():
    n = 60
    X = pd.DataFrame(np.random.normal(size=(n, 4)), columns=list("abcd"))
    y = pd.Series(np.random.choice([0, 1], size=n))
    clf = PoniardClassifier(
        estimators=[LogisticRegression()], cv=2, random_state=0
    )
    clf.setup(X, y)
    clf.fit(X, y)
    return X, y, clf


@pytest.fixture(scope="module")
def reg_setup():
    n = 60
    X = pd.DataFrame(np.random.normal(size=(n, 4)), columns=list("abcd"))
    y = pd.Series(np.random.normal(size=n))
    reg = PoniardRegressor(
        estimators=[LinearRegression()], cv=2, random_state=0
    )
    reg.setup(X, y)
    reg.fit(X, y)
    return X, y, reg


class TestClassifierPlots:
    def test_metrics(self, clf_setup):
        X, y, clf = clf_setup
        fig = PoniardPlotFactory(X, y, clf).metrics()
        assert isinstance(fig, go.Figure)

    def test_metrics_single_metric_no_facet(self, clf_setup):
        X, y, clf = clf_setup
        fig = PoniardPlotFactory(X, y, clf).metrics(metrics="test_accuracy")
        assert isinstance(fig, go.Figure)
        # A single explicitly-selected metric must not produce a metric facet.
        assert fig.layout.grid is None or fig.layout.grid.columns is None

    def test_metrics_bar(self, clf_setup):
        X, y, clf = clf_setup
        fig = PoniardPlotFactory(X, y, clf).metrics(kind="bar")
        assert isinstance(fig, go.Figure)

    def test_overfitness(self, clf_setup):
        X, y, clf = clf_setup
        fig = PoniardPlotFactory(X, y, clf).overfitness()
        assert isinstance(fig, go.Figure)

    def test_permutation_importance(self, clf_setup):
        X, y, clf = clf_setup
        fig = PoniardPlotFactory(X, y, clf).permutation_importance(
            "LogisticRegression", n_repeats=2
        )
        assert isinstance(fig, go.Figure)

    def test_roc_curve(self, clf_setup):
        X, y, clf = clf_setup
        fig = PoniardPlotFactory(X, y, clf).roc_curve()
        assert isinstance(fig, go.Figure)

    def test_confusion_matrix(self, clf_setup):
        X, y, clf = clf_setup
        fig = PoniardPlotFactory(X, y, clf).confusion_matrix("LogisticRegression")
        assert isinstance(fig, go.Figure)

    def test_partial_dependence(self, clf_setup):
        X, y, clf = clf_setup
        fig = PoniardPlotFactory(X, y, clf).partial_dependence(
            "LogisticRegression", feature=0
        )
        assert isinstance(fig, go.Figure)

    def test_full_estimator_analysis(self, clf_setup):
        X, y, clf = clf_setup
        fig = PoniardPlotFactory(X, y, clf).full_estimator_analysis(
            "LogisticRegression"
        )
        assert isinstance(fig, go.Figure)


class TestRegressorPlots:
    def test_metrics(self, reg_setup):
        X, y, reg = reg_setup
        fig = PoniardPlotFactory(X, y, reg).metrics()
        assert isinstance(fig, go.Figure)

    def test_overfitness(self, reg_setup):
        X, y, reg = reg_setup
        fig = PoniardPlotFactory(X, y, reg).overfitness()
        assert isinstance(fig, go.Figure)

    def test_permutation_importance(self, reg_setup):
        X, y, reg = reg_setup
        fig = PoniardPlotFactory(X, y, reg).permutation_importance(
            "LinearRegression", n_repeats=2
        )
        assert isinstance(fig, go.Figure)

    def test_partial_dependence(self, reg_setup):
        X, y, reg = reg_setup
        fig = PoniardPlotFactory(X, y, reg).partial_dependence(
            "LinearRegression", feature=0
        )
        assert isinstance(fig, go.Figure)

    def test_residuals(self, reg_setup):
        X, y, reg = reg_setup
        fig = PoniardPlotFactory(X, y, reg).residuals(["LinearRegression"])
        assert isinstance(fig, go.Figure)

    def test_residuals_histogram(self, reg_setup):
        X, y, reg = reg_setup
        fig = PoniardPlotFactory(X, y, reg).residuals_histogram(
            ["LinearRegression"]
        )
        assert isinstance(fig, go.Figure)

    def test_full_estimator_analysis(self, reg_setup):
        X, y, reg = reg_setup
        fig = PoniardPlotFactory(X, y, reg).full_estimator_analysis(
            "LinearRegression"
        )
        assert isinstance(fig, go.Figure)


class TestNewPlots:
    def test_error_lift_bars(self, clf_setup):
        from poniard.error_analysis import ErrorAnalyzer

        X, y, clf = clf_setup
        ea = ErrorAnalyzer.from_poniard(clf)
        report = ea.analyze(X=X, y=y)
        fig = PoniardPlotFactory(X, y, clf).error_lift_bars(
            lift_by_target=report.lift_by_target
        )
        assert isinstance(fig, go.Figure)

    def test_error_lift_bars_top_n(self, clf_setup):
        from poniard.error_analysis import ErrorAnalyzer

        X, y, clf = clf_setup
        ea = ErrorAnalyzer.from_poniard(clf)
        report = ea.analyze(X=X, y=y)
        fig = PoniardPlotFactory(X, y, clf).error_lift_bars(
            lift_by_target=report.lift_by_target, top_n=1
        )
        assert isinstance(fig, go.Figure)

    def test_error_lift_bars_requires_data(self, clf_setup):
        X, y, clf = clf_setup
        with pytest.raises(ValueError, match="lift_by_target"):
            PoniardPlotFactory(X, y, clf).error_lift_bars()

    def test_similarity_heatmap(self, clf_setup):
        X, y, clf = clf_setup
        fig = PoniardPlotFactory(X, y, clf).similarity_heatmap(X, y)
        assert isinstance(fig, go.Figure)

    def test_similarity_heatmap_on_predictions(self, clf_setup):
        X, y, clf = clf_setup
        fig = PoniardPlotFactory(X, y, clf).similarity_heatmap(
            X, y, on_errors=False
        )
        assert isinstance(fig, go.Figure)

    def test_similarity_heatmap_regressor(self, reg_setup):
        X, y, reg = reg_setup
        fig = PoniardPlotFactory(X, y, reg).similarity_heatmap(X, y)
        assert isinstance(fig, go.Figure)

    def test_time_quality_scatter(self, clf_setup):
        X, y, clf = clf_setup
        fig = PoniardPlotFactory(X, y, clf).time_quality_scatter()
        assert isinstance(fig, go.Figure)

    def test_time_quality_scatter_regressor(self, reg_setup):
        X, y, reg = reg_setup
        fig = PoniardPlotFactory(X, y, reg).time_quality_scatter()
        assert isinstance(fig, go.Figure)

    def test_time_quality_scatter_custom_metric(self, clf_setup):
        X, y, clf = clf_setup
        fig = PoniardPlotFactory(X, y, clf).time_quality_scatter(
            metric="test_accuracy"
        )
        assert isinstance(fig, go.Figure)
