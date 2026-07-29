from __future__ import annotations

from collections.abc import Sequence

from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)

from ..utils.estimate import element_to_list_maybe


class EnsembleMixin:
    """Mixin for ensemble building methods."""

    def build_ensemble(
        self,
        method: str = "stacking",
        estimator_names: Sequence[str] | None = None,
        top_n: int | None = 3,
        sort_by: str | None = None,
        ensemble_name: str | None = None,
        **kwargs,
    ):
        """Combine estimators into an ensemble.

        By default, orders estimators according to the first metric.

        Parameters
        ----------
        method :
            Ensemble method. Either "stacking" or "voting". Default "stacking".
        estimator_names :
            Names of estimators to include. Default None, which uses `top_n`
        top_n :
            How many of the best estimators to include.
        sort_by :
            Which metric to consider for ordering results. Default None, which uses the first metric.
        ensemble_name :
            Ensemble name when adding to `pipelines`. Default None.
        kwargs :
            Passed to the ensemble class constructor.

        Returns
        -------
        Self.
        """
        if method not in ["voting", "stacking"]:
            raise ValueError("Method must be either voting or stacking.")
        estimator_names = element_to_list_maybe(estimator_names)
        if estimator_names:
            models = [
                (name, self.pipelines[name]._final_estimator)
                for name in estimator_names
            ]
        else:
            if sort_by:
                sorter = sort_by
            else:
                sorter = self._means.columns[0]
            models = [
                (name, self.pipelines[name]._final_estimator)
                for name in self._means.sort_values(sorter, ascending=False).index[
                    :top_n
                ]
            ]
        if method == "voting":
            if self.poniard_task == "classification":
                ensemble = VotingClassifier(
                    estimators=models, verbose=self.verbose, **kwargs
                )
            else:
                ensemble = VotingRegressor(
                    estimators=models, verbose=self.verbose, **kwargs
                )
        else:
            if self.poniard_task == "classification":
                ensemble = StackingClassifier(
                    estimators=models, verbose=self.verbose, cv=self.cv, **kwargs
                )
            else:
                ensemble = StackingRegressor(
                    estimators=models, verbose=self.verbose, cv=self.cv, **kwargs
                )
        ensemble_name = ensemble_name or ensemble.__class__.__name__
        self.add_estimators(estimators={ensemble_name: ensemble})
        return self
