from __future__ import annotations

__all__ = ["PoniardPlotFactory"]

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import auc, confusion_matrix, roc_curve

if TYPE_CHECKING:
    from poniard.estimators.core import PoniardBaseEstimator
from ..utils.estimate import element_to_list_maybe

try:
    import plotly.express as px
    from plotly.graph_objs._figure import Figure
    from plotly.subplots import make_subplots
except ImportError as e:
    raise ImportError(
        "plotly is required for plotting. Install it with: pip install plotly"
    ) from e


class PoniardPlotFactory:
    """Helper class that handles plotting for Poniard Estimators.

    It operates as a standalone plotting object that receives data, an estimator,
    and optional plotly configuration.

    Parameters
    ----------
    X :
        The feature data (DataFrame or array).
    y :
        The target data (Series or array).
    estimator :
        A fitted PoniardBaseEstimator instance (provides results, pipelines, etc.).
    **plot_config :
        Optional plotly config keys: ``template``, ``discrete_colors``, ``font_family``,
        ``font_color``.
    """

    def __init__(
        self,
        X,
        y,
        estimator: PoniardBaseEstimator,
        **plot_config,
    ):
        self._X = X
        self._y = y
        self._estimator = estimator

        self._template = plot_config.get("template", "plotly_white")
        self._discrete_colors = plot_config.get(
            "discrete_colors", px.colors.qualitative.Bold
        )
        self._font_family = plot_config.get("font_family", "Helvetica")
        self._font_color = plot_config.get("font_color", "#8C8C8C")

    def _apply_layout(self, fig: Figure) -> Figure:
        """Apply the configured template, font, margin, and legend to a figure.

        No global plotly state is mutated.
        """
        fig.update_layout(
            template=self._template,
            font={"family": self._font_family, "color": self._font_color},
            margin={"l": 20, "r": 20},
            legend={
                "yanchor": "top",
                "y": -0.2,
                "xanchor": "left",
                "x": 0.0,
                "orientation": "h",
            },
        )
        return fig

    def _px_kwargs(self, kwargs: dict | None = None) -> dict:
        """Build the kwargs dict for `px.*` calls, including plot_config defaults."""
        merged = {
            "template": self._template,
            "color_discrete_sequence": self._discrete_colors,
        }
        if kwargs:
            merged.update(kwargs)
        return merged

    def metrics(
        self,
        kind: str = "strip",
        facet: str = "col",
        metrics: str | Sequence[str] | None = None,
        only_test: bool = True,
        exclude_dummy: bool = True,
        show_means: bool = True,
        **kwargs,
    ) -> Figure:
        """Plot metrics obtained by running `PoniardBaseEstimator.fit`.

        Parameters
        ----------
        kind :
            Either "strip" or "bar". Default "strip".
        facet :
            Either "col" or "row". Default "col".
        metrics :
            String or list of strings. This must follow the names passed to the
            Poniard constructor. For example, if during init a dict of metrics was passed, its
            keys can be passed here. Default None, which plots every estimator metric available.
        only_test :
            Whether to plot only test scores. Default True.
        exclude_dummy :
            Whether to exclude dummy estimators. Default True.
        show_means :
            Whether to plot means along with fold scores. Default True.

        Returns
        -------
        Figure
            Plotly strip or bar plot.
        """
        results = self._estimator._long_results.replace(
            "Classifier|Regressor", "", regex=True
        )
        results = results.loc[~results["Metric"].isin(["fit_time", "score_time"])]
        if only_test:
            results = results.loc[results["Metric"].str.contains("test", case=False)]
        if exclude_dummy:
            results = results.loc[~results["Model"].str.contains("Dummy")]
        if metrics:
            metrics = element_to_list_maybe(metrics)
            metrics = "|".join(metrics)
            results = results.loc[results["Metric"].str.contains(metrics)]
        if not show_means:
            results = results.loc[~(results["Type"] == "Mean")]
        height = 100 * results["Model"].nunique()
        if facet == "col":
            facet_row = None
            facet_col = "Metric" if not metrics or len(metrics) > 1 else None
        else:
            facet_row = "Metric" if not metrics or len(metrics) > 1 else None
            facet_col = None
        if kind == "strip":
            fig = px.strip(
                results,
                y="Model",
                x="Score",
                color="Type" if show_means else None,
                facet_row=facet_row,
                facet_col=facet_col,
                title="Model scores",
                height=height,
                **self._px_kwargs(kwargs),
            )
        else:
            stds = self._estimator._stds.reset_index().melt(id_vars="index")
            stds.columns = ["Model", "Metric", "Score"]
            stds["Model"] = stds["Model"].str.replace(
                "Classifier|Regressor", "", regex=True
            )
            results = results.loc[results["Type"] == "Mean"].merge(
                stds, how="left", on=["Model", "Metric"], suffixes=(None, "_y")
            )
            results = results.rename(columns={"Score_y": "Std"})
            results["Std"] = results["Std"] / 2
            fig = px.bar(
                results,
                y="Model",
                x="Score",
                facet_row=facet_row,
                facet_col=facet_col,
                error_x="Std",
                error_y="Std",
                orientation="h",
                title="Model scores",
                height=height,
                **self._px_kwargs(kwargs),
            )
        fig.update_xaxes(matches=None)
        fig.update_layout(yaxis_title="")
        self._apply_layout(fig)

        return fig

    def overfitness(
        self, metric: str | None = None, exclude_dummy: bool = True
    ) -> Figure:
        """Plot the ratio of test scores to train scores for every estimator.

        Parameters
        ----------
        metric :
            String representing a metric. This must follow the names passed to the
            Poniard constructor. For example, if during init a dict of metrics was passed, one of
            its keys can be passed here. Default None, which plots the first metric.
        exclude_dummy :
            Whether to exclude dummy estimators. Default True.

        Returns
        -------
        Figure
            Plotly strip plot.
        """
        if not metric:
            metric = self._estimator._first_scorer(sklearn_scorer=False)
        results = self._estimator._long_results.replace(
            "Classifier|Regressor", "", regex=True
        )
        results = results.loc[
            (results["Type"] == "Mean") & (results["Metric"].str.contains(metric))
        ]
        if exclude_dummy:
            results = results.loc[~results["Model"].str.contains("Dummy")]
        results = results.pivot(columns="Metric", index="Model", values="Score")
        results = results.loc[:, results.columns.str.contains("train")].div(
            results.loc[:, results.columns.str.contains("test")].squeeze(), axis=0
        )
        results = results.sort_values(results.columns[0])
        fig = px.strip(
            results.reset_index(),
            y="Model",
            x=results.columns[0],
            title=f"{metric} overfitness",
            **self._px_kwargs(),
        )
        fig.update_layout(xaxis_title="Train / test ratio", yaxis_title="")
        self._apply_layout(fig)
        return fig

    def permutation_importance(
        self,
        estimator_name: str,
        n_repeats: int = 10,
        kind: str = "bar",
        **kwargs,
    ) -> Figure:
        """Plot permutation importances for an estimator.

        This shuffles features randomly one at a time and measures the change in the estimator's
        performance. If the feature is important for the model, the estimator's performance
        should decrease (represented by positive values in the plot).
        See the [scikit-learn guide](https://scikit-learn.org/stable/modules/permutation_importance.html).

        Parameters
        ----------
        estimator_name :
            Estimator to include.
        n_repeats :
            How many times to repeat random permutations of a single feature. Default 10.
        kind :
            Either "bar" or "strip". Default "bar". "strip" plots each permutation repetition
            as well as the mean. Bar plots only the mean.
        kwargs :
            Passed to `sklearn.inspection.permutation_importance()`.

        Returns
        -------
        Figure
            Plotly bar or strip plot.
        """
        X_train, X_test, y_train, y_test = self._estimator._train_test_split_from_cv(
            self._X, self._y
        )
        scoring = self._estimator._first_scorer(sklearn_scorer=True)
        estimator = self._estimator.pipelines[estimator_name]
        estimator.fit(X_train, y_train)
        raw_importances = permutation_importance(
            estimator,
            X_test,
            y_test,
            scoring=scoring,
            random_state=self._estimator.random_state,
            n_repeats=n_repeats,
            n_jobs=self._estimator.n_jobs,
            **kwargs,
        )
        if isinstance(X_test, pd.DataFrame):
            index = X_test.columns
        else:
            index = [str(x) for x in range(X_test.shape[1])]
        importances = pd.DataFrame(raw_importances["importances"], index=index)
        importances.rename_axis("Feature", inplace=True)
        importances.reset_index(inplace=True)

        importances = importances.melt(
            id_vars="Feature", var_name="Type", value_name="Importance"
        )
        importances["Type"] = "Repetition"
        aggs = (
            importances.groupby("Feature")["Importance"]
            .agg(Mean=np.mean, Std=np.std)
            .reset_index()
        )
        aggs = aggs.melt(id_vars="Feature", var_name="Type", value_name="Importance")
        importances = pd.concat([importances, aggs])

        title = f"Permutation importances ({estimator_name}, {scoring}, {n_repeats} repeats)"
        if kind == "strip":
            importances = importances.loc[importances["Type"] != "Std"]
            fig = px.strip(
                importances,
                x="Importance",
                y="Feature",
                color="Type",
                title=title,
                **self._px_kwargs(),
            )
        else:
            importances = importances.loc[
                -importances["Type"].isin(["Repetition", "Std"])
            ]
            fig = px.bar(
                importances,
                x="Importance",
                y="Feature",
                title=title,
                **self._px_kwargs(),
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
        self._apply_layout(fig)
        return fig

    def roc_curve(
        self,
        estimator_names: Sequence[str] | None = None,
        response_method: str = "auto",
        **kwargs,
    ) -> Figure:
        """Plot ROC curve with cross validated predictions for multiple estimators.

        Parameters
        ----------
        estimator_names :
            Estimators to include. If None, all estimators are used.
        response_method :
            Either "auto", "predict_proba" or "decision_function". "auto" will try to use
            `predict_proba` if all estimators have it, otherwise it will try `decision_function`
            If there is no common `response_method`, it will raise an error.
        kwargs :
            Passed to `sklearn.metrics.roc_curve()`.

        Returns
        -------
        Figure
            Plotly line plot.
        """
        if self._estimator.poniard_task == "regression":
            raise ValueError("ROC curve is not available for regressors.")
        y = self._y
        if y.ndim > 1:
            raise ValueError("ROC curve is only available for binary classification.")
        results = self._estimator._experiment_results
        estimator_names = element_to_list_maybe(estimator_names)
        if not estimator_names:
            estimator_names = list(results.keys())

        if response_method == "auto":
            if all(
                hasattr(self._estimator.pipelines[estimator], "predict_proba")
                for estimator in estimator_names
            ):
                type_of_prediction = "predict_proba"
            elif all(
                hasattr(self._estimator.pipelines[estimator], "decision_function")
                for estimator in estimator_names
            ):
                type_of_prediction = "decision_function"
            else:
                raise ValueError(
                    "Selected estimators do not have a common response_method (predict_proba or decision_function)."
                )
        else:
            type_of_prediction = response_method
            if not all(
                hasattr(self._estimator.pipelines[estimator], response_method)
                for estimator in estimator_names
            ):
                raise ValueError(
                    f"Selected estimators do not have a common response_method ({response_method})."
                )

        estimator_metrics = []
        for name in estimator_names:
            y_pred = self._estimator._get_or_compute_prediction(
                self._X, self._y, name, type_of_prediction
            )
            if type_of_prediction == "predict_proba":
                y_pred = y_pred[:, 1]
            fpr, tpr, _ = roc_curve(y, y_pred, **kwargs)
            roc_auc = auc(fpr, tpr)
            estimator_metrics.append(
                pd.DataFrame(
                    {
                        "Estimator": name,
                        "False positive rate": fpr,
                        "True positive rate": tpr,
                        "AUC": roc_auc,
                        "Estimator_AUC": f"{name} | AUC: {roc_auc:.2f}",
                    }
                )
            )
        metrics = pd.concat(estimator_metrics)
        fig = px.line(
            metrics,
            x="False positive rate",
            y="True positive rate",
            color="Estimator_AUC",
            title="ROC curve with cross-validated predictions",
            hover_data={
                "Estimator_AUC": False,
                "Estimator": True,
                "True positive rate": ":.2f",
                "False positive rate": ":.2f",
                "AUC": ":.2f",
            },
            **self._px_kwargs(),
        )
        fig.update_layout(
            shapes=[
                {
                    "type": "line",
                    "yref": "y",
                    "xref": "x",
                    "y0": 0,
                    "y1": 1,
                    "x0": 0,
                    "x1": 1,
                    "line": {"dash": "dash"},
                }
            ]
        )
        self._apply_layout(fig)
        return fig

    def confusion_matrix(self, estimator_name: str, **kwargs) -> Figure:
        """Plot confusion matrix with cross validated predictions for a single estimator.

        Parameters
        ----------
        estimator_name :
            Estimator to include.
        kwargs :
            Passed to `sklearn.metrics.confusion_matrix()`.

        Returns
        -------
        Figure
            Plotly image plot.
        """
        if self._estimator.poniard_task == "regression":
            raise ValueError("Confusion matrix is not available for regressors.")
        y = self._y
        y_pred = self._estimator._get_or_compute_prediction(
            self._X, self._y, estimator_name, "predict"
        )
        matrix = confusion_matrix(y, y_pred, **kwargs)
        fig = px.imshow(
            matrix,
            labels={"x": "Predicted", "y": "Ground truth", "color": "Count"},
            color_continuous_scale="Blues",
            text_auto=True,
            title="Confusion matrix with cross-validated predictions",
            **self._px_kwargs(),
        )
        fig.update_yaxes(nticks=len(np.unique(y)) + 1)
        fig.update_xaxes(nticks=len(np.unique(y)) + 1)
        fig.update(layout_coloraxis_showscale=False)
        self._apply_layout(fig)
        return fig

    def partial_dependence(
        self, estimator_name: str, feature: str | int, **kwargs
    ) -> Figure:
        """Plot partial dependence for a single feature of a single estimator.

        In essence, visualize how the target changes within the feature's range.

        Only plots average partial dependence for all samples and not individual samples (ICE).

        Parameters
        ----------
        estimator_name :
            Estimator to include.
        feature :
            Feature for which to plot partial dependence. Can be a pandas column name or index.
        kwargs :
            Passed to `sklearn.inspection.partial_dependence()`.

        Returns
        -------
        Figure
            Plotly line plot.
        """
        y = self._y
        X = self._X
        estimator = self._estimator.pipelines[estimator_name]
        estimator.fit(X, y)
        partial_dep = partial_dependence(
            estimator, X, features=[feature], kind="average", **kwargs
        )
        response = partial_dep["average"].reshape(-1)
        n_values = len(partial_dep["values"][0])
        n_repeats = int(len(response) / n_values)
        values = np.tile(partial_dep["values"][0], n_repeats)
        data = pd.DataFrame({"Target": response, f"Feature: {feature}": values})
        hide_legend = False
        if n_repeats > 1 and self._estimator.poniard_task == "classification":
            data["Class"] = np.repeat(estimator.classes_, n_values)
        elif self._estimator.poniard_task == "classification":
            data["Class"] = 1
        else:
            data["Class"] = "Target"
            hide_legend = True

        fig = px.line(
            data,
            x=f"Feature: {feature}",
            y="Target",
            color="Class",
            title=f"Average partial dependence between feature '{feature}' and target",
            **self._px_kwargs(),
        )
        if hide_legend:
            fig.update_layout(showlegend=False)
        self._apply_layout(fig)
        return fig

    def _build_residuals_data(self, estimator_names: list[str]) -> pd.DataFrame:
        """Build residuals DataFrame for a list of estimators."""
        y = np.array(self._y)
        estimator_names = element_to_list_maybe(estimator_names)
        data = []
        for name in estimator_names:
            y_pred = self._estimator._get_or_compute_prediction(
                self._X, self._y, name, "predict"
            )
            if y.ndim == 1:
                y_2d = np.expand_dims(y, 1)
            else:
                y_2d = y
            if y_pred.ndim == 1:
                y_pred = np.expand_dims(y_pred, 1)
            for i in range(y_2d.shape[1]):
                row = {"Estimator": name, "Target": i}
                if y_2d.shape[1] > 1:
                    row["Predicted"] = y_pred[:, i]
                row["Residuals"] = y_2d[:, i] - y_pred[:, i]
                data.append(pd.DataFrame(row))
        return pd.concat(data)

    def residuals(self, estimator_names: list[str]) -> Figure:
        """Plot regression residuals vs predictions for a list of estimators.

        Parameters
        ----------
        estimator_names :
            Estimators to include.

        Returns
        -------
        Figure
            Residuals plot.
        """
        if self._estimator.poniard_task == "classification":
            raise ValueError("Residuals plot is not available for classifiers.")
        data = self._build_residuals_data(estimator_names)
        fig = px.scatter(
            data,
            x="Predicted" if "Predicted" in data.columns else "Residuals",
            y="Residuals",
            color="Estimator",
            symbol="Target" if "Target" in data.columns else None,
            title="Residuals plot with cross validated predictions",
            **self._px_kwargs(),
        )
        self._apply_layout(fig)
        return fig

    def residuals_histogram(self, estimator_names: list[str]) -> Figure:
        """Plot a histogram of regression residuals for a list of estimators.

        Parameters
        ----------
        estimator_names :
            Estimators to include.

        Returns
        -------
        Figure
            Residuals histogram plot.
        """
        if self._estimator.poniard_task == "classification":
            raise ValueError(
                "Residuals histogram plot is not available for classifiers."
            )
        data = self._build_residuals_data(estimator_names)
        fig = px.histogram(
            data,
            x="Residuals",
            color="Estimator",
            pattern_shape="Target" if "Target" in data.columns else None,
            histnorm="percent",
            barmode="overlay",
            title="Residuals histogram plot with cross validated predictions",
            **self._px_kwargs(),
        )
        self._apply_layout(fig)
        return fig

    def _full_estimator_analysis(
        self, estimator_name: str, height: int = 800, width: int = 800
    ) -> Figure:
        main_scorer = self._estimator._first_scorer(sklearn_scorer=False)
        sorted_means = self._estimator._long_results.query(
            f"Metric == 'test_{main_scorer}' & Type=='Mean'"
        ).sort_values(ascending=False, by="Score")
        estimator_position = sorted_means.set_index("Model").index.get_loc(
            estimator_name
        )
        if estimator_position == 0:
            better_estimator_name = None
            worse_estimator_name = sorted_means.iloc[1, 0]
        elif estimator_position == len(self._estimator.pipelines) - 1:
            better_estimator_name = sorted_means.iloc[
                len(self._estimator.pipelines) - 2, 0
            ]
            worse_estimator_name = None
        else:
            better_estimator_name = sorted_means.iloc[estimator_position - 1, 0]
            worse_estimator_name = sorted_means.iloc[estimator_position + 1, 0]
        estimator_names = [
            x
            for x in [estimator_name, better_estimator_name, worse_estimator_name]
            if x
        ]

        metrics = self._estimator.get_results()
        metric_rankings = metrics.loc[:, ~metrics.columns.str.contains("time")].rank(
            ascending=True
        )
        time_rankings = metrics.loc[:, metrics.columns.str.contains("time")].rank(
            ascending=False
        )
        rankings = pd.concat([metric_rankings, time_rankings], axis=1)
        rank = px.bar(
            rankings.loc[estimator_name, :],
            text_auto=True,
            **self._px_kwargs(),
        )
        rank_title = f"Metrics rank (best={len(self._estimator.pipelines)})"
        rank.update_layout(dict(xaxis_title=None, yaxis_title="Rank"), showlegend=False)

        if self._estimator.poniard_task == "classification":
            if self._estimator.target_info["type_"] == "binary":
                task_1_fig = self.roc_curve(estimator_names=estimator_names)
                task_1_title = "ROC curve w/ CV predictions"
            else:
                task_1_fig = self.metrics(metrics=main_scorer, kind="bar")
                task_1_title = f"{main_scorer} scores"
            task_2_fig = self.confusion_matrix(estimator_name=estimator_name)
            task_2_title = "Confusion matrix w/ CV predictions"
            task_2_fig.update_layout(coloraxis_showscale=False)
        else:
            task_1_fig = self.residuals_histogram(estimator_names=estimator_names)
            task_1_title = "Residuals histogram w/ CV predictions"
            task_2_fig = self.residuals(estimator_names=estimator_names)
            task_2_title = "Residuals plot w/ CV predictions"

        importance_fig = self.permutation_importance(estimator_name=estimator_name)

        plot_array = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                rank_title,
                task_1_title,
                task_2_title,
                "Feature importance",
            ),
        )
        figures = [rank, task_1_fig, task_2_fig, importance_fig]

        row = 1
        col = 1
        for i, figure in enumerate(figures):
            for trace in range(len(figure["data"])):
                plot_array.append_trace(figure["data"][trace], row=row, col=col)
            if col == 2:
                row += 1
                col -= 1
            else:
                col += 1
        plot_array.update_layout(
            title_text=f"{estimator_name} analysis (better={better_estimator_name}, worse={worse_estimator_name})",
            height=height,
            width=width,
        )
        plot_array.update_layout(
            coloraxis={
                "colorbar": {"title": {"text": "Count"}},
                "colorscale": [
                    [0.0, "rgb(247,251,255)"],
                    [0.125, "rgb(222,235,247)"],
                    [0.25, "rgb(198,219,239)"],
                    [0.375, "rgb(158,202,225)"],
                    [0.5, "rgb(107,174,214)"],
                    [0.625, "rgb(66,146,198)"],
                    [0.75, "rgb(33,113,181)"],
                    [0.875, "rgb(8,81,156)"],
                    [1.0, "rgb(8,48,107)"],
                ],
                "showscale": False,
            }
        )
        self._apply_layout(plot_array)
        return plot_array
