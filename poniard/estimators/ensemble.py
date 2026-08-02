from __future__ import annotations

from collections.abc import Sequence

import numpy as np
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
        strategy: str = "diversity",
        similarity_threshold: float = 0.5,
        X=None,
        y=None,
        **kwargs,
    ):
        """Combine estimators into an ensemble.

        By default uses a diversity-aware selection strategy that greedily
        picks estimators that are both strong and dissimilar on prediction
        errors. Falls back to pure ``top_n`` when similarity cannot be
        computed (fewer than 2 non-dummy estimators).

        Parameters
        ----------
        method :
            Ensemble method. Either "stacking" or "voting". Default "stacking".
        estimator_names :
            Names of estimators to include explicitly. Bypasses selection
            strategy. Default None.
        top_n :
            How many estimators to include. Default 3.
        sort_by :
            Which metric to consider for ordering results. Default None,
            which uses the first metric.
        ensemble_name :
            Ensemble name when adding to `pipelines`. Default None.
        strategy :
            Selection strategy. ``"diversity"`` (default) greedily picks
            estimators that are strong and dissimilar. ``"top_n"`` takes the
            top-N by metric score (legacy behavior).
        similarity_threshold :
            Maximum pairwise similarity (0–1) allowed between ensemble
            members when using ``strategy="diversity"``. Lower values
            enforce more diversity. Default 0.5.
        X :
            Features. Required for ``strategy="diversity"`` to compute
            prediction similarity.
        y :
            Target. Required for ``strategy="diversity"`` to compute
            prediction similarity.
        kwargs :
            Passed to the ensemble class constructor.

        Returns
        -------
        Self.
        """
        if method not in ["voting", "stacking"]:
            raise ValueError("Method must be either voting or stacking.")
        if strategy not in ["diversity", "top_n"]:
            raise ValueError("Strategy must be either 'diversity' or 'top_n'.")
        estimator_names = element_to_list_maybe(estimator_names)
        if estimator_names:
            models = [(name, self.pipelines[name]._final_estimator) for name in estimator_names]
        elif strategy == "diversity" and X is not None and y is not None:
            selected = self._select_diverse(
                top_n=top_n or 3,
                sort_by=sort_by,
                similarity_threshold=similarity_threshold,
                X=X,
                y=y,
            )
            models = [(name, self.pipelines[name]._final_estimator) for name in selected]
        else:
            if sort_by:
                sorter = sort_by
            else:
                sorter = self._means.columns[0]
            dummy = set(self._dummy_names())
            eligible = [
                name
                for name in self._means.sort_values(sorter, ascending=False).index
                if name not in dummy
            ]
            models = [(name, self.pipelines[name]._final_estimator) for name in eligible[:top_n]]
        if method == "voting":
            if self.poniard_task == "classification":
                ensemble = VotingClassifier(estimators=models, verbose=self.verbose, **kwargs)
            else:
                ensemble = VotingRegressor(estimators=models, verbose=self.verbose, **kwargs)
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

    def _select_diverse(
        self,
        top_n: int,
        sort_by: str | None,
        similarity_threshold: float,
        X,
        y,
    ) -> list[str]:
        """Greedily select diverse estimators based on metric rank and
        prediction-error similarity.

        Algorithm:
        1. Rank non-dummy estimators by primary metric (descending).
        2. Start with the best estimator.
        3. For each next candidate, accept it only if its maximum pairwise
           similarity to all already-selected members is below the threshold.
        4. Stop when ``top_n`` is reached or candidates exhausted.
        """
        if sort_by:
            sorter = sort_by
        else:
            sorter = self._means.columns[0]

        dummy = set(self._dummy_names())
        ranked = [
            name
            for name in self._means.sort_values(sorter, ascending=False).index
            if name not in dummy
        ]

        if len(ranked) < 2:
            return ranked[:top_n]

        try:
            sim_matrix = self.get_predictions_similarity(X=X, y=y, on_errors=True)
        except (ValueError, np.linalg.LinAlgError):
            return ranked[:top_n]

        dummy_in_sim = set(sim_matrix.index) & dummy
        if dummy_in_sim:
            sim_matrix = sim_matrix.drop(index=list(dummy_in_sim), columns=list(dummy_in_sim))

        if sim_matrix.empty or len(sim_matrix) < 2:
            return ranked[:top_n]

        selected: list[str] = [ranked[0]]
        for name in ranked[1:]:
            if len(selected) >= top_n:
                break
            if name not in sim_matrix.index:
                continue
            max_sim = max(abs(sim_matrix.loc[name, s]) for s in selected if s in sim_matrix.columns)
            if np.isnan(max_sim) or max_sim <= similarity_threshold:
                selected.append(name)

        return selected
