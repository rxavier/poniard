from typing import List, Union, Optional
from typing import Optional, TYPE_CHECKING, Sequence

import plotly.io as pio
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.inspection import partial_dependence, permutation_importance
from plotly.graph_objs._figure import Figure

if TYPE_CHECKING:
    from poniard.estimators.core import PoniardBaseEstimator


class PoniardPlotFactory:
    """Helper class that handles plotting for Poniard Estimators."""

    def __init__(
        self,
        template: str = "plotly_white",
        discrete_colors: List[str] = px.colors.qualitative.Bold,
        font_family: str = "Helvetica",
        font_color: str = "#8C8C8C",
    ):
        self._template = template
        self._discrete_colors = discrete_colors
        self._font_family = font_family
        self._font_color = font_color
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
        metrics: Union[str, Sequence[str]] = None,
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
        self._poniard._run_plugin_method("on_plot", figure=fig, name="scores_plot")
        return fig

    def overfitness(
        self, metric: Optional[str] = None, exclude_dummy: bool = True
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
            metric = self._poniard._first_scorer(sklearn_scorer=False)
        results = self._poniard._long_results
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
        )
        fig.update_layout(xaxis_title="Train / test ratio", yaxis_title="")
        self._poniard._run_plugin_method("on_plot", figure=fig, name="overfitness_plot")
        return fig

    def permutation_importance(
        self,
        estimator_name: str,
        n_repeats: int = 10,
        kind: str = "bar",
        **kwargs,
    ) -> Figure:
        """Plot permutation importances for an estimator.

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
        X_train, X_test, y_train, y_test = self._poniard._train_test_split_from_cv()
        scoring = self._poniard._first_scorer(sklearn_scorer=True)
        estimator = self._poniard._experiment_results[estimator_name]["estimator"][0]
        estimator.fit(X_train, y_train)
        raw_importances = permutation_importance(
            estimator,
            X_test,
            y_test,
            scoring=scoring,
            random_state=self._poniard.random_state,
            n_repeats=n_repeats,
            n_jobs=self._poniard.n_jobs,
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
            )
        else:
            importances = importances.loc[
                -importances["Type"].isin(["Repetition", "Std"])
            ]
            fig = px.bar(importances, x="Importance", y="Feature", title=title)
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
        self._poniard._run_plugin_method(
            "on_plot", figure=fig, name=f"{estimator_name}_permutation_importances_plot"
        )
        return fig

    def roc_curve(
        self,
        estimator_names: Optional[Sequence[str]] = None,
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
        if self._poniard.poniard_task == "regression":
            raise ValueError("ROC curve is not available for regressors.")
        y = self._poniard.y
        if y.ndim > 1:
            raise ValueError("ROC curve is only available for binary classification.")
        results = self._poniard._experiment_results
        if not estimator_names:
            estimator_names = list(results.keys())

        if response_method == "auto":
            if all(
                hasattr(results[estimator]["estimator"][0], "predict_proba")
                for estimator in estimator_names
            ):
                prediction = "predict_proba"
            elif all(
                hasattr(results[estimator]["estimator"][0], "decision_function")
                for estimator in estimator_names
            ):
                prediction = "decision_function"
            else:
                raise ValueError(
                    "Selected estimators do not have a common response_method (predict_proba or decision_function)."
                )
        else:
            prediction = response_method
            if not all(
                hasattr(results[estimator]["estimator"][0], response_method)
                for estimator in estimator_names
            ):
                raise ValueError(
                    f"Selected estimators do not have a common response_method ({response_method})."
                )

        estimator_metrics = []
        for name in estimator_names:
            if name not in results or prediction not in results[name]:
                raise KeyError(f"Predictions have not been computed for {name}.")
            y_pred = results[name][prediction]
            if prediction == "predict_proba":
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
        self._poniard._run_plugin_method("on_plot", figure=fig, name="roc_plot")
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
        if self._poniard.poniard_task == "regression":
            raise ValueError("Confusion matrix is not available for regressors.")
        y = self._poniard.y
        results = self._poniard._experiment_results
        y_pred = results[estimator_name]["predict"]
        matrix = confusion_matrix(y, y_pred, **kwargs)
        fig = px.imshow(
            matrix,
            labels={"x": "Predicted", "y": "Ground truth", "color": "Count"},
            color_continuous_scale="Blues",
            text_auto=True,
            title="Confusion matrix with cross-validated predictions",
        )
        fig.update_yaxes(nticks=len(np.unique(y)) + 1)
        fig.update_xaxes(nticks=len(np.unique(y)) + 1)
        fig.update(layout_coloraxis_showscale=False)
        self._poniard._run_plugin_method("on_plot", figure=fig, name="confusion_matrix")
        return fig

    def partial_dependence(
        self, estimator_name: str, feature: Union[str, int], **kwargs
    ) -> Figure:
        """Plot partial dependence for a single feature of a single estimator.

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
        y = self._poniard.y
        X = self._poniard.X
        results = self._poniard._experiment_results
        estimator = results[estimator_name]["estimator"][0]
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
        if n_repeats > 1 and self._poniard.poniard_task == "classification":
            data["Class"] = np.repeat(estimator.classes_, n_values)
        elif self._poniard.poniard_task == "classification":
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
        )
        if hide_legend:
            fig.update_layout(showlegend=False)
        self._poniard._run_plugin_method(
            "on_plot", figure=fig, name=f"{feature}_partial_dependence_plot"
        )
        return fig

    def __repr__(self):
        return f"""{self.__class__.__name__}(template={self._template},
    discrete_colors={self._discrete_colors}, font_family={self._font_family},
    font_color={self._font_color})
    """

    def __str__(self):
        return f"""{self.__class__.__name__}()"""
