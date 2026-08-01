from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor

from ..utils.estimate import element_to_list_maybe
from ..utils.stats import cramers_v


class ResultsMixin:
    """Mixin for result processing, comparison, and prediction similarity methods."""

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
            Whether to compute each score/time relative to the dummy estimator
            results. Only the mean ratios are meaningful; standard deviations
            are returned as NaN. Requires exactly one dummy estimator. Default
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
            dummy_names = self._dummy_names()
            if len(dummy_names) != 1:
                raise ValueError(
                    f"wrt_dummy=True requires exactly one dummy estimator, "
                    f"found {len(dummy_names)}: {dummy_names}."
                )
            dummy_means = means.loc[dummy_names[0]]
            means = means / dummy_means
            stds = stds.copy()
            stds[:] = np.nan
        if std:
            return means, stds
        else:
            return means

    def _dummy_names(self) -> list[str]:
        """Names of pipelines whose final estimator is a sklearn dummy."""
        return [
            name
            for name, pipeline in self.pipelines.items()
            if isinstance(pipeline._final_estimator, (DummyClassifier, DummyRegressor))
        ]

    def compare(
        self,
        estimators: str | Sequence[str] | None = None,
        metrics: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Statistical comparison of estimators using paired fold scores.

        For each pair of estimators, computes the mean difference, the number
        of folds where each wins, and a two-sided paired t-test p-value.
        This is **exploratory comparison** — not a paper-grade
        multiple-testing correction.

        Parameters
        ----------
        estimators :
            Subset of estimator names to compare. If None, all non-dummy
            estimators are used.
        metrics :
            Subset of metric names (columns of ``get_results()``). If None,
            the primary metric is used.

        Returns
        -------
        pd.DataFrame
            Pairwise comparison table with columns: mean_diff, wins_a,
            wins_b, ties, p_value. Indexed by ``(estimator_a, estimator_b)``.
        """
        from scipy import stats as sp_stats

        dummy = set(self._dummy_names())
        if estimators is not None:
            names = element_to_list_maybe(estimators)
        else:
            names = [n for n in self._means.index if n not in dummy]
        if metrics is not None:
            metric_cols = element_to_list_maybe(metrics)
        else:
            metric_cols = [self._means.columns[0]]

        all_pairs = []
        for metric in metric_cols:
            fold_data = {}
            for name in names:
                raw = self._experiment_results[name].get(metric)
                if raw is None:
                    continue
                fold_data[name] = np.array(raw)
            for a, b in itertools.combinations(fold_data.keys(), 2):
                sa, sb = fold_data[a], fold_data[b]
                n = min(len(sa), len(sb))
                sa, sb = sa[:n], sb[:n]
                diff = sa - sb
                wins_a = int(np.sum(diff > 0))
                wins_b = int(np.sum(diff < 0))
                ties = int(np.sum(diff == 0))
                if n >= 2:
                    _, p = sp_stats.ttest_rel(sa, sb)
                else:
                    p = np.nan
                all_pairs.append({
                    "metric": metric,
                    "estimator_a": a,
                    "estimator_b": b,
                    "mean_diff": float(np.mean(diff)),
                    "wins_a": wins_a,
                    "wins_b": wins_b,
                    "ties": ties,
                    "p_value": float(p) if not np.isnan(p) else np.nan,
                })
        if not all_pairs:
            return pd.DataFrame()
        df = pd.DataFrame(all_pairs)
        df = df.set_index(["metric", "estimator_a", "estimator_b"])
        return df

    def pareto(
        self, metric: str | None = None, time_col: str = "fit_time"
    ) -> pd.DataFrame:
        """Return the Pareto-optimal set of estimators (best metric vs lowest time).

        An estimator is Pareto-optimal if no other estimator is both faster
        and better on the given metric. Dummy estimators are excluded.

        Parameters
        ----------
        metric :
            Metric column from ``get_results()``. If None, uses the first
            (primary) metric.
        time_col :
            Column name for time. Options:

            - ``"fit_time"`` — total training time per fold (default)
            - ``"score_time"`` — total predict+score time per fold
            - ``"fit_time_per_sample"`` — training time per sample
            - ``"score_time_per_sample"`` — predict time per sample

        Returns
        -------
        pd.DataFrame
            Subset of ``get_results()`` containing only Pareto-optimal
            estimators, sorted by metric descending.

        Examples
        --------
        >>> clf.pareto()                                        # best metric vs training time
        >>> clf.pareto(time_col="score_time_per_sample")        # vs inference time per sample
        """
        results = self.get_results()
        dummy = set(self._dummy_names())
        results = results.loc[~results.index.isin(dummy)]
        if metric is None:
            metric = results.columns[0]
        if metric not in results.columns:
            raise ValueError(f"Metric '{metric}' not found. Available: {list(results.columns)}")
        if time_col not in results.columns:
            raise ValueError(f"Time column '{time_col}' not found.")

        vals = results[[metric, time_col]].copy()
        pareto_mask = pd.Series(True, index=vals.index)
        for i, (name_i, row_i) in enumerate(vals.iterrows()):
            for j, (name_j, row_j) in enumerate(vals.iterrows()):
                if i == j:
                    continue
                if row_j[metric] >= row_i[metric] and row_j[time_col] <= row_i[time_col]:
                    if (row_j[metric] > row_i[metric]) or (row_j[time_col] < row_i[time_col]):
                        pareto_mask.iloc[i] = False
                        break
        return results.loc[pareto_mask].sort_values(metric, ascending=False)

    def best_under(
        self,
        seconds: float = 2.0,
        metric: str | None = None,
        time_col: str = "fit_time",
    ) -> str:
        """Return the name of the best non-dummy estimator under a time budget.

        Parameters
        ----------
        seconds :
            Maximum allowed mean time in seconds. Default 2.0.
        metric :
            Metric to rank by (among those under the time budget). If None,
            uses the first (primary) metric.
        time_col :
            Column name for time. Options:

            - ``"fit_time"`` — total training time per fold (default)
            - ``"score_time"`` — total predict+score time per fold
            - ``"fit_time_per_sample"`` — training time per sample
            - ``"score_time_per_sample"`` — predict time per sample

        Returns
        -------
        str
            Name of the best estimator under the time budget.

        Raises
        ------
        ValueError
            If no estimator fits within the time budget.

        Examples
        --------
        >>> clf.best_under(seconds=0.5)                                   # fast trainer
        >>> clf.best_under(seconds=0.001, time_col="score_time_per_sample")  # fast inference
        """
        results = self.get_results()
        dummy = set(self._dummy_names())
        results = results.loc[~results.index.isin(dummy)]
        if metric is None:
            metric = results.columns[0]
        if time_col not in results.columns:
            raise ValueError(f"Time column '{time_col}' not found.")

        under = results[results[time_col] <= seconds]
        if under.empty:
            raise ValueError(
                f"No estimator with mean {time_col} <= {seconds}s. "
                f"Fastest is {results[time_col].idxmin()} at "
                f"{results[time_col].min():.2f}s."
            )
        return under[metric].idxmax()

    def _process_results(self) -> None:
        """Compute mean and standard deviations of experiment results.

        Also computes per-sample fit and score times when fold sizes are
        available (set by ``PoniardBaseEstimator.fit``).
        """
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
        time_columns = ["fit_time", "score_time"]
        metric_columns = [c for c in means.columns if c not in time_columns]
        means = means[metric_columns + time_columns]
        stds = stds[metric_columns + time_columns]

        # Per-sample times: divide fold times by test fold sizes
        fold_sizes = getattr(self, "_fold_sizes", None)
        if fold_sizes is not None:
            sizes = np.array(fold_sizes, dtype=float)
            per_sample_cols = {}
            for estimator_name in means.index:
                raw_fit = np.array(self._experiment_results[estimator_name].get("fit_time", []))
                raw_score = np.array(self._experiment_results[estimator_name].get("score_time", []))
                fit_ps = np.mean(raw_fit / sizes) if len(raw_fit) == len(sizes) else np.nan
                score_ps = np.mean(raw_score / sizes) if len(raw_score) == len(sizes) else np.nan
                per_sample_cols.setdefault("fit_time_per_sample", []).append(fit_ps)
                per_sample_cols.setdefault("score_time_per_sample", []).append(score_ps)
            for col, vals in per_sample_cols.items():
                means[col] = vals
                stds[col] = np.nan

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
        dummy_names = self._dummy_names()
        if self.poniard_task == "classification":
            estimator_names = [x for x in results.columns if x not in dummy_names]
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
            table = results.drop(dummy_names, axis=1).corr()
        return table
