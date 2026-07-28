from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from ..utils.stats import cramers_v


class ResultsMixin:
    """Mixin for result processing and prediction similarity methods."""

    def get_results(
        self,
        return_train_scores: bool = False,
        std: bool = False,
        wrt_dummy: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """Return dataframe containing scoring results. By default returns the mean score and fit
        and score times. Optionally returns standard deviations as well.

        Parameters
        ----------
        return_train_scores :
            If False, only return test scores.
        std :
            Whether to return standard deviation of the scores. Default False.
        wrt_dummy :
            Whether to compute each score/time with respect to the dummy estimator results. Default
            False.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame
            Results
        """
        means = self._means
        stds = self._stds
        if not return_train_scores:
            means = means.loc[
                :, means.columns.str.contains("test_|fit|score", regex=True)
            ]
            stds = stds.loc[:, stds.columns.str.contains("test_|fit|score", regex=True)]
        if wrt_dummy:
            dummy_means = means.loc[means.index.str.contains("Dummy")]
            dummy_stds = stds.loc[stds.index.str.contains("Dummy")]
            means = means / dummy_means.squeeze()
            stds = stds / dummy_stds.squeeze()
        if std:
            return means, stds
        else:
            return means

    def _process_results(self) -> None:
        """Compute mean and standard deviations of  experiment results."""
        results = pd.DataFrame(self._experiment_results).T
        results = results.loc[
            :,
            [
                x
                for x in results.columns
                if x not in ["predict", "predict_proba", "decision_function"]
            ],
        ]
        means = results.apply(lambda x: np.mean(np.stack(x.values), axis=1))
        stds = results.apply(lambda x: np.std(np.stack(x.values), axis=1))
        means = means[list(means.columns[2:]) + ["fit_time", "score_time"]]
        stds = stds[list(stds.columns[2:]) + ["fit_time", "score_time"]]
        self._means = means.sort_values(means.columns[0], ascending=False)
        self._stds = stds.reindex(self._means.index)

    def _process_long_results(self) -> None:
        """Prepare experiment results for plotting."""
        base = pd.DataFrame(self._experiment_results).T
        melted = (
            base.rename_axis("Model")
            .reset_index()
            .melt(id_vars="Model", var_name="Metric", value_name="Score")
            .explode("Score")
        )
        melted["Type"] = "Fold"
        means = melted.groupby(["Model", "Metric"])["Score"].mean().reset_index()
        means["Type"] = "Mean"
        melted = pd.concat([melted, means])
        self._long_results = melted

    def _first_scorer(self, sklearn_scorer: bool) -> str | Callable:
        """Helper method to get the first scoring function or name."""
        if isinstance(self.metrics, Sequence):
            return self.metrics[0]
        elif isinstance(self.metrics, dict):
            if sklearn_scorer:
                return list(self.metrics.values())[0]
            else:
                return list(self.metrics.keys())[0]
        else:
            raise ValueError(
                "self.metrics can only be a sequence of str or dict of str: callable."
            )

    def get_predictions_similarity(
        self,
        X,
        y,
        on_errors: bool = True,
    ) -> pd.DataFrame:
        """Compute correlation/association between cross validated predictions for each estimator.

        This can be useful for ensembling.

        Parameters
        ----------
        X :
            Features.
        y :
            Target.
        on_errors :
            Whether to compute similarity on prediction errors instead of predictions. Default
            True.

        Returns
        -------
        pd.DataFrame
            Similarity.
        """
        if y.ndim > 1:
            raise ValueError("y must be a 1-dimensional array.")
        raw_results = {
            name: self._get_or_compute_prediction(X=X, y=y, estimator_name=name, method="predict")
            for name in self.pipelines.keys()
        }
        results = raw_results.copy()
        for name, result in raw_results.items():
            if on_errors:
                if self.poniard_task == "regression":
                    results[name] = y - result
                else:
                    results[name] = np.where(result == y, 1, 0)
        results = pd.DataFrame(results)
        if self.poniard_task == "classification":
            estimator_names = [x for x in results.columns if x != "DummyClassifier"]
            table = pd.DataFrame(
                data=np.nan, index=estimator_names, columns=estimator_names
            )
            for row, col in itertools.combinations_with_replacement(
                table.index[::-1], 2
            ):
                cramer = cramers_v(results[row], results[col])
                if row == col:
                    table.loc[row, col] = 1
                else:
                    table.loc[row, col] = cramer
                    table.loc[col, row] = cramer
        else:
            table = results.drop("DummyRegressor", axis=1).corr()
        return table
