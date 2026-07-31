from __future__ import annotations

__all__ = ["ErrorAnalyzer"]

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from poniard.estimators.core import PoniardBaseEstimator
from ..preprocessing import PoniardPreprocessor
from ..utils.estimate import element_to_list_maybe, get_target_info
from ..utils.utils import get_kwargs, non_default_repr


class ErrorAnalyzer:
    """Analyze where and why predictive models fail.

    Compare ground truth and predicted target, rank the largest deviations and
    cross-tabulate them against the target and the features. "Error" is defined
    as:

    * binary / multiclass: samples the model got wrong, ranked by
      ``1 - probability_of_truth`` (i.e. how confidently wrong the model is).
    * multilabel: samples the model got wrong on at least one label, ranked by
      the mean per-label deviation.
    * regression / multioutput regression: samples whose absolute residual is
      above a threshold, ranked by residual magnitude. The threshold defaults to
      the 90th percentile of absolute residuals and can be configured via
      ``error_quantile``.

    This class is tightly integrated with `PoniardBaseEstimator` (via
    `from_poniard`) but does not require it.

    Parameters
    ----------
    task :
        The machine learning task. Either 'regression' or 'classification'.
    """

    def __init__(self, task: str):
        self._init_params = get_kwargs()
        self.task = task
        self._poniard: PoniardBaseEstimator | None = None

    @property
    def _has_poniard(self) -> bool:
        return self._poniard is not None

    @classmethod
    def from_poniard(
        cls, poniard: PoniardBaseEstimator, estimator_names: str | Sequence[str]
    ) -> ErrorAnalyzer:
        """Use a Poniard instance to instantiate `ErrorAnalyzer`.

        Automatically sets the task, the estimator names and the target type.

        Parameters
        ----------
        poniard :
            A `PoniardClassifier` or `PoniardRegressor` instance.
        estimator_names :
            Array of estimators for which to compute errors.

        Returns
        -------
        ErrorAnalyzer :
            An instance of the class.
        """
        error_analysis = cls(task=poniard.poniard_task)
        error_analysis._poniard = poniard
        error_analysis.estimator_names = element_to_list_maybe(estimator_names)
        error_analysis.type_of_target = poniard.target_info["type_"]
        return error_analysis

    def _compute_predictions(self, X, y):
        """Compute cross-validated predictions for the selected estimators."""
        predictions = self._poniard.predict(
            X=X, y=y, estimator_names=self.estimator_names
        )
        probas = None
        if self.type_of_target in ["binary", "multilabel-indicator", "multiclass"]:
            probas = self._poniard.predict_proba(
                X=X, y=y, estimator_names=self.estimator_names
            )
        return predictions, probas

    def rank_errors(
        self,
        X: np.ndarray | pd.DataFrame | None = None,
        y: np.ndarray | pd.Series | pd.DataFrame | None = None,
        predictions: np.ndarray | pd.Series | pd.DataFrame | None = None,
        probas: np.ndarray | pd.Series | pd.DataFrame | None = None,
        exclude_correct: bool = True,
        error_quantile: float = 0.1,
    ):
        """Rank samples by error for each estimator.

        When using `ErrorAnalyzer.from_poniard`, `X` and `y` must be passed so
        that cross-validated predictions can be computed. Otherwise, pass
        `predictions` (and `probas` for classification) directly.

        For regression, ``exclude_correct`` keeps only the samples whose
        absolute residual exceeds a threshold, defaulting to the
        ``error_quantile`` (default 0.1) worst residuals. For classification it
        keeps only misclassified samples.

        Parameters
        ----------
        X :
            Features. Required when using `ErrorAnalyzer.from_poniard`.
        y :
            Ground truth target.
        predictions :
            Predicted target.
        probas :
            Predicted probabilities for each class in classification tasks.
        exclude_correct :
            Whether to exclude correctly predicted samples from the ranking.
            Default True.
        error_quantile :
            Fraction of worst residuals kept as errors for regression tasks
            when ``exclude_correct`` is True. Default 0.1.

        Returns
        -------
        dict[str, pd.DataFrame] | pd.DataFrame
            A DataFrame per estimator (keys are estimator names) when built via
            `from_poniard`, or a single DataFrame otherwise. Each DataFrame is
            indexed by sample and includes a `error` column used for ranking.
        """
        if self._has_poniard:
            if X is None or y is None:
                raise ValueError(
                    "X and y must be provided when using `from_poniard`."
                )
            predictions, probas = self._compute_predictions(X, y)
            ranked_errors = {}
            for estimator in self.estimator_names:
                proc_probas = probas[estimator] if probas is not None else probas
                ranked_errors[estimator] = self._rank_single(
                    y,
                    predictions[estimator],
                    proc_probas,
                    exclude_correct,
                    error_quantile,
                )
            return ranked_errors
        self.type_of_target = get_target_info(y, task=self.task)["type_"]
        return self._rank_single(
            y, predictions, probas, exclude_correct, error_quantile
        )

    def _rank_single(
        self, y, predictions, probas, exclude_correct, error_quantile
    ) -> pd.DataFrame:
        """Rank errors for a single set of predictions."""
        return self._target_redirect(self.type_of_target)(
            y, predictions, probas, exclude_correct, error_quantile
        )

    def _target_redirect(self, type_of_target: str):
        """A router for error ranking depending on the type of the target."""
        if type_of_target == "binary":
            return self._rank_errors_binary
        elif type_of_target == "multiclass":
            return self._rank_errors_multiclass
        elif type_of_target == "multilabel-indicator":
            return self._rank_errors_multilabel
        elif type_of_target == "continuous":
            return self._rank_errors_continuous
        elif type_of_target == "continuous-multioutput":
            return self._rank_errors_continuous_multioutput
        else:
            raise NotImplementedError("Type of target could not be determined.")

    def _rank_errors_binary(
        self,
        y,
        predictions,
        probas,
        exclude_correct: bool = True,
        error_quantile: float = 0.1,
    ) -> pd.DataFrame:
        errors = pd.DataFrame(
            {
                "y": y,
                "prediction": predictions,
                "proba_0": probas[:, 0],
                "proba_1": probas[:, 1],
            }
        )
        if exclude_correct:
            errors = errors.query("y != prediction")
        errors = errors.assign(error=(errors["y"] - errors["proba_1"]).abs())
        return errors.sort_values("error", ascending=False)

    def _rank_errors_multiclass(
        self,
        y,
        predictions,
        probas,
        exclude_correct: bool = True,
        error_quantile: float = 0.1,
    ) -> pd.DataFrame:
        data = {"y": y, "prediction": predictions}
        data.update({f"proba_{i}": probas[:, i] for i in range(len(np.unique(y)))})
        errors = pd.DataFrame(data)
        if exclude_correct:
            errors = errors.query("y != prediction")
        errors = errors.assign(
            truth_proba=errors.apply(
                lambda row: row[f"proba_{int(row['y'])}"], axis=1
            )
        )
        errors = errors.assign(error=(1 - errors["truth_proba"]).abs())
        return errors.sort_values("error", ascending=False)

    def _rank_errors_multilabel(
        self,
        y,
        predictions,
        probas,
        exclude_correct: bool = True,
        error_quantile: float = 0.1,
    ) -> pd.DataFrame:
        n_labels = y.shape[1]
        truth = pd.DataFrame(y, columns=[f"y_{i}" for i in range(n_labels)])
        preds = pd.DataFrame(predictions, columns=[f"prediction_{i}" for i in range(n_labels)])
        pro = pd.DataFrame(probas, columns=[f"proba_{i}" for i in range(n_labels)])
        errors = pd.concat([truth, preds, pro], axis=1)
        if exclude_correct:
            keep = ~preds.eq(y).all(axis=1)
            errors = errors.loc[keep]
            truth = truth.loc[keep]
            pro = pro.loc[keep]
        per_label = np.abs(truth.values - pro.values)
        errors = errors.assign(
            **{f"error_{i}": per_label[:, i] for i in range(n_labels)}
        )
        errors = errors.assign(
            error=errors[[f"error_{i}" for i in range(n_labels)]].mean(axis=1)
        )
        return errors.sort_values("error", ascending=False)

    def _rank_errors_continuous(
        self,
        y,
        predictions,
        probas=None,
        exclude_correct: bool = True,
        error_quantile: float = 0.1,
    ) -> pd.DataFrame:
        errors = pd.DataFrame({"y": y, "prediction": predictions})
        errors = errors.assign(error=(errors["y"] - errors["prediction"]).abs())
        if exclude_correct:
            threshold = np.quantile(errors["error"], 1 - error_quantile)
            errors = errors.loc[errors["error"] > threshold]
        return errors.sort_values("error", ascending=False)

    def _rank_errors_continuous_multioutput(
        self,
        y,
        predictions,
        probas=None,
        exclude_correct: bool = True,
        error_quantile: float = 0.1,
    ) -> pd.DataFrame:
        n_targets = y.shape[1]
        truth = pd.DataFrame(y, columns=[f"y_{i}" for i in range(n_targets)])
        preds = pd.DataFrame(predictions, columns=[f"prediction_{i}" for i in range(n_targets)])
        errors = pd.concat([truth, preds], axis=1)
        per_target = pd.DataFrame(
            np.abs(truth.values - preds.values),
            index=errors.index,
            columns=[f"error_{i}" for i in range(n_targets)],
        )
        if exclude_correct:
            thresholds = {
                i: np.quantile(per_target[f"error_{i}"], 1 - error_quantile)
                for i in range(n_targets)
            }
            flagged = pd.DataFrame(
                {
                    i: per_target[f"error_{i}"] > thresholds[i]
                    for i in range(n_targets)
                }
            )
            keep = flagged.any(axis=1)
            errors = errors.loc[keep]
            per_target = per_target.loc[keep]
        errors = pd.concat(
            [errors, per_target.assign(error=per_target.mean(axis=1))], axis=1
        )
        return errors.sort_values("error", ascending=False)

    @staticmethod
    def merge_errors(errors):
        """Merge per-estimator error rankings into a single cross-estimator view.

        Accepts the output of `rank_errors` (a dict of DataFrames) or a single
        DataFrame for a lone model. The result is indexed by sample and reports,
        per sample, how many estimators failed on it, their average error and
        which estimators failed.

        Parameters
        ----------
        errors :
            Output of `rank_errors`, or a single ranked-errors DataFrame.

        Returns
        -------
        pd.DataFrame
            Merged errors indexed by sample.
        """
        if isinstance(errors, pd.DataFrame):
            errors = {"model": errors}
        concatenated = pd.concat(
            [
                frame.assign(estimator=estimator)
                for estimator, frame in errors.items()
            ]
        ).reset_index()
        merged = (
            concatenated.groupby("index")
            .agg(
                mean_error=pd.NamedAgg(column="error", aggfunc="mean"),
                freq=pd.NamedAgg(column="error", aggfunc="size"),
                estimators=pd.NamedAgg(column="estimator", aggfunc=lambda x: list(x)),
            )
            .sort_values(["freq", "mean_error"], ascending=False)
        )
        return merged

    def analyze_target(
        self,
        errors_idx,
        y: np.ndarray | pd.Series | pd.DataFrame | None = None,
        reg_bins: int = 5,
    ) -> pd.DataFrame:
        """Cross-tabulate errors against the target.

        For classification the target is grouped by class; for regression it is
        binned into ``reg_bins`` quantile bins. The result reports how many
        error samples and how many population samples each class/bin contains,
        plus the error rate per class/bin (the interpretation column: where is
        the model over- or under-represented in errors).

        Parameters
        ----------
        errors_idx :
            Index of ranked errors (e.g. the index of `merge_errors` output).
        y :
            Ground truth.
        reg_bins :
            Number of bins in which to place ground truth targets for regression
            tasks. Default 5.

        Returns
        -------
        pd.DataFrame
            Counts and error rate per target class/bin.
        """
        type_of_target = self.type_of_target
        if y is None:
            raise ValueError("`y` must be provided.")
        y = pd.DataFrame(y)
        y_errors = y.loc[errors_idx]

        if type_of_target in ["binary", "multiclass", "multilabel-indicator"]:
            target_names = y.columns.tolist()
        elif type_of_target == "continuous":
            bins = pd.qcut(y.squeeze(), q=reg_bins)
            y = y.assign(bins=bins)
            y_errors = y_errors.assign(bins=bins)
            target_names = "bins"
        elif type_of_target == "continuous-multioutput":
            bins = {
                f"bin_{target}": pd.qcut(y[target], q=reg_bins)
                for target in range(y.shape[1])
            }
            y = y.assign(**bins)
            y_errors = y_errors.assign(**bins)
            target_names = list(bins.keys())
        else:
            raise NotImplementedError("Type of target could not be determined.")
        errors_dist = y_errors.groupby(target_names, observed=True).size()
        target_dist = y.groupby(target_names, observed=True).size()
        output = pd.DataFrame(
            {"error_count": errors_dist, "target_count": target_dist}
        ).fillna(0)
        output["error_rate"] = output["error_count"] / output[
            "target_count"
        ].replace(0, np.nan)
        output = output.sort_values("error_count", ascending=False)
        return output

    def analyze_features(
        self,
        errors_idx,
        X: np.ndarray | pd.Series | pd.DataFrame | None = None,
        y: np.ndarray | pd.Series | pd.DataFrame | None = None,
        features: Sequence[str | int] | None = None,
        estimator_name: str | BaseEstimator | None = None,
        n_features: int | float | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Cross-tabulate errors against the features.

        Numeric features are summarized (count, mean, std, ...) per error group;
        categorical features get an error rate per category. When
        ``estimator_name`` is given (only valid with `from_poniard`), only the
        top ``n_features`` by permutation importance are analyzed, computed on
        the same cross-validation folds the Poniard estimator used.

        Parameters
        ----------
        errors_idx :
            Index of ranked errors.
        X :
            Features array.
        y :
            Target. Required when `estimator_name` is used to compute
            permutation importances.
        features :
            Array of features to analyze. If `None`, all features will be
            analyzed (unless `estimator_name` is given).
        estimator_name :
            Only valid if using `ErrorAnalyzer.from_poniard`. Allows using an
            estimator to compute permutation importances and analyzing only the
            top `n_features`.
        n_features :
            How many features to analyze based on permutation importances. A
            float between 0 and 1 is interpreted as a fraction of all features.

        Returns
        -------
        dict[str, pd.DataFrame]
            Per feature summary.
        """
        if X is None:
            raise ValueError("`X` must be provided.")
        if self._has_poniard:
            feature_types = self._poniard.feature_types.items()
        else:
            feature_types = (
                PoniardPreprocessor(task="placeholder")
                .build(X, np.zeros((X.shape[0],)))
                .feature_types
                .items()
            )
        inverted_feature_types = {}
        for k, v in feature_types:
            for i in v:
                inverted_feature_types[i] = k
        X = pd.DataFrame(X)
        error_mask = X.index.isin(errors_idx).astype(int)
        X = X.assign(error=error_mask)
        columns = [col for col in X.columns if col != "error"]

        if features:
            features_idx = {
                i
                for i, col in enumerate(columns)
                if col in features or i in features
            }
        elif estimator_name:
            if not self._has_poniard:
                raise ValueError(
                    "`estimator_name` is only valid when using `from_poniard`."
                )
            if y is None:
                raise ValueError("`y` must be provided when `estimator_name` is used.")
            importances = self._compute_permutation_importances(X[columns], y, estimator_name)
            sorted_importances_idx = importances.argsort()[::-1]
            if n_features is None:
                n_features = 0.5
            if isinstance(n_features, float):
                assert 0 <= n_features <= 1
                n_features = round(n_features * len(columns))
            features_idx = set(sorted_importances_idx[: max(1, n_features)].tolist())
        else:
            features_idx = set(range(len(columns)))

        summary = {}
        for i, col in enumerate(columns):
            if i not in features_idx:
                continue
            current_feature_type = inverted_feature_types.get(col, "numeric")
            if current_feature_type == "numeric":
                summary[col] = X.groupby("error")[col].describe()
            else:
                crosstab = pd.crosstab(X[col], X["error"], dropna=False)
                crosstab = crosstab.reindex(columns=[0, 1], fill_value=0)
                crosstab.columns = ["correct", "errors"]
                total = crosstab["correct"] + crosstab["errors"]
                crosstab = crosstab.assign(
                    error_rate=(
                        crosstab["errors"] / total.replace(0, np.nan)
                    ).fillna(0)
                )
                summary[col] = crosstab
        return summary

    def _compute_permutation_importances(
        self, X, y, estimator_name: str, n_repeats: int = 10
    ) -> np.ndarray:
        """Compute permutation importances averaged over the Poniard CV folds."""
        model = self._poniard[estimator_name]
        scoring = self._poniard._first_scorer(sklearn_scorer=True)
        random_state = self._poniard.random_state
        cv = self._poniard.cv
        splitter = getattr(cv, "split", None)
        if splitter is None:
            train_idx, test_idx = train_test_split(
                np.arange(len(X)), test_size=0.2, random_state=random_state
            )
            folds = [(train_idx, test_idx)]
        else:
            folds = list(cv.split(X, y))
        per_fold = []
        for train_idx, test_idx in folds:
            model.fit(self._subset(X, train_idx), self._subset(y, train_idx))
            result = permutation_importance(
                model,
                self._subset(X, test_idx),
                self._subset(y, test_idx),
                n_repeats=n_repeats,
                scoring=scoring,
                random_state=random_state,
            )
            per_fold.append(result.importances_mean)
        return np.mean(per_fold, axis=0)

    @staticmethod
    def _subset(data, idx):
        """Index rows of a DataFrame/Series/ndarray by positional index."""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.iloc[idx]
        return data[idx]

    def analyze(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | pd.DataFrame,
        estimator_names: str | Sequence[str] | None = None,
        features: Sequence[str | int] | None = None,
        n_features: int | float | None = None,
        reg_bins: int = 5,
        error_quantile: float = 0.1,
    ) -> dict:
        """Run the full error-analysis workflow and return a single report.

        Convenience wrapper that computes ranked errors per estimator, the merged
        cross-estimator view and the error distributions over the target and the
        features, packaged into one dict.

        Parameters
        ----------
        X :
            Features.
        y :
            Ground truth target.
        estimator_names :
            Estimators to analyze. If None, the estimators selected at
            `from_poniard` time are used.
        features :
            Subset of features to analyze. If None, all features are analyzed.
        n_features :
            Number of features to analyze based on permutation importances
            (only used when the full feature set is not selected explicitly).
        reg_bins :
            Number of bins for regression targets. Default 5.
        error_quantile :
            Fraction of worst residuals kept as errors for regression tasks.
            Default 0.1.

        Returns
        -------
        dict
            Report with keys ``ranked_errors`` (dict of DataFrames),
            ``merged_errors`` (DataFrame), ``summary`` (per-estimator error
            counts and rates), ``by_target`` (DataFrame) and ``by_feature``
            (dict of DataFrames).
        """
        if estimator_names is not None:
            self.estimator_names = element_to_list_maybe(estimator_names)
        ranked = self.rank_errors(X=X, y=y, error_quantile=error_quantile)
        if isinstance(ranked, pd.DataFrame):
            ranked = {"model": ranked}
        merged = self.merge_errors(ranked)
        by_target = self.analyze_target(
            errors_idx=merged.index, y=y, reg_bins=reg_bins
        )
        by_feature = self.analyze_features(
            errors_idx=merged.index,
            X=X,
            y=y,
            features=features,
            n_features=n_features,
        )
        summary = pd.DataFrame(
            {
                estimator: {
                    "n_errors": len(frame),
                    "error_rate": len(frame) / len(y),
                    "mean_error": float(frame["error"].mean()),
                }
                for estimator, frame in ranked.items()
            }
        ).T
        return {
            "ranked_errors": ranked,
            "merged_errors": merged,
            "summary": summary,
            "by_target": by_target,
            "by_feature": by_feature,
        }

    def __repr__(self):
        return non_default_repr(self)
