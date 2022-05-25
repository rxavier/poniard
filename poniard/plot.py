from typing import List, Union, Optional
from typing import Optional, TYPE_CHECKING

import plotly.io as pio
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.graph_objs._figure import Figure

if TYPE_CHECKING:
    from poniard.core import PoniardBaseEstimator


class PoniardPlotFactory:
    """Helper class that handles plotting for Poniard Estimators."""

    def __init__(
        self,
        template: str = "plotly_white",
        discrete_colors: List[str] = px.colors.qualitative.Bold,
        font_family: str = "Helvetica",
        font_color: str = "#8C8C8C",
    ):
        pio.templates.default = template
        pio.templates["plotly_white"].layout.font = {"family": font_family}
        pio.templates["plotly_white"].layout.font = {"color": font_color}
        pio.templates["plotly_white"].layout.margin = {"l": 20, "r": 20}
        pio.templates["plotly_white"].layout.legend.yanchor = "top"
        pio.templates["plotly_white"].layout.legend.y = -0.2
        pio.templates["plotly_white"].layout.legend.xanchor = "left"
        pio.templates["plotly_white"].layout.legend.x = 0.0
        pio.templates["plotly_white"].layout.legend.orientation = "h"
        px.defaults.color_discrete_sequence = discrete_colors

        self._poniard: Optional["PoniardBaseEstimator"] = None

    def metrics(
        self,
        kind: str = "strip",
        facet: str = "col",
        metrics: Union[str, List[str]] = None,
        only_test: bool = True,
        exclude_dummy: bool = True,
        show_means: bool = True,
        **kwargs,
    ) -> Figure:
        """Plot metrics.

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
        results = self._poniard._long_results
        results = results.loc[~results["Metric"].isin(["fit_time", "score_time"])]
        if only_test:
            results = results.loc[results["Metric"].str.contains("test", case=False)]
        if exclude_dummy:
            results = results.loc[~results["Model"].str.contains("Dummy")]
        if metrics:
            metrics = [metrics] if isinstance(metrics, str) else metrics
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
                **kwargs,
            )
        else:
            stds = self._poniard._stds.reset_index().melt(id_vars="index")
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
                **kwargs,
            )
        fig.update_xaxes(matches=None)
        fig.update_layout(yaxis_title="")
        self._poniard._run_plugin_methods("on_plot", figure=fig, name="scores_plot")
        return fig

    def overfitness(self, metric: Optional[str] = None) -> Figure:
        """Plot the ratio of test scores to train scores for every estimator.

        Parameters
        ----------
        metric :
            String representing a metric. This must follow the names passed to the
            Poniard constructor. For example, if during init a dict of metrics was passed, one of
            its keys can be passed here. Default None, which plots the first metric.

        Returns
        -------
        Figure
            Plotly strip plot.
        """
        if not metric:
            metric = self._poniard._first_scorer(sklearn_scorer=False)
        results = self._poniard._long_results
        results = results.loc[
            (results["Type"] == "Mean") & (results["Metric"].str.contains(metric))
        ]
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
        )
        fig.update_layout(xaxis_title="Train / test ratio", yaxis_title="")
        self._poniard._run_plugin_methods(
            "on_plot", figure=fig, name="overfitness_plot"
        )
        return fig

    def permutation_importances(
        self,
        estimator_names: Union[str, List[str]],
        kind: str = "bar",
        facet: str = "col",
    ) -> Figure:
        """Plot permutation importances for a list of estimators.

        Parameters
        ----------
        estimator_names :
            Estimators to include.
        kind :
            Either "bar" or "strip". Default "bar". "strip" plots each permutation repetition
            as well as the mean. Bar plots only the mean.
        facet :
           Either "col" or "row". Default "col".

        Returns
        -------
        Figure
            Plotly bar or strip plot.
        """
        if isinstance(estimator_names, str):
            estimator_names = [estimator_names]
        importances = {
            estimator: self._poniard._experiment_results[estimator][
                "permutation_importances"
            ]["importances"]
            for estimator in estimator_names
        }
        importances_arr = []
        for estimator, importance_values in importances.items():
            aux = pd.DataFrame(importance_values, index=self._poniard.X.columns)
            aux.rename_axis("Feature", inplace=True)
            aux.reset_index(inplace=True)
            aux.insert(0, "Estimator", estimator)
            importances_arr.append(aux)
        importances = pd.concat(importances_arr)
        importances = importances.melt(
            id_vars=["Estimator", "Feature"], var_name="Type", value_name="Importance"
        )
        importances["Type"] = "Repetition"
        aggs = (
            importances.groupby(["Estimator", "Feature"])["Importance"]
            .agg(Mean=np.mean, Std=np.std)
            .reset_index()
        )
        aggs = aggs.melt(
            id_vars=["Estimator", "Feature"], var_name="Type", value_name="Importance"
        )
        importances = pd.concat([importances, aggs])

        scoring = self._poniard._first_scorer(sklearn_scorer=False)
        repeats = self._poniard._experiment_results[estimator_names[0]][
            "permutation_importances"
        ]["importances"].shape[1]
        title = f"Permutation importances ({scoring}, {repeats} repeats)"
        if kind == "strip":
            if facet == "row":
                facet_row = "Estimator"
                facet_col = None
            else:
                facet_row = None
                facet_col = "Estimator"
            importances = importances.loc[importances["Type"] != "Std"]
            fig = px.strip(
                importances,
                x="Importance",
                y="Feature",
                color="Type",
                facet_col=facet_col,
                facet_row=facet_row,
                title=title,
            )
        else:
            importances = importances.loc[
                -importances["Type"].isin(["Repetition", "Std"])
            ]
            fig = px.bar(
                importances, x="Importance", y="Feature", color="Estimator", title=title
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
        self._poniard._run_plugin_methods(
            "on_plot", figure=fig, name="permutation_importances_plot"
        )
        return fig
