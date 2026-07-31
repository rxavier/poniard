from __future__ import annotations

from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    RandomizedSearchCV,
)


class TuningMixin:
    """Mixin for hyperparameter tuning methods."""

    def tune_estimator(
        self,
        estimator_name: str,
        X,
        y,
        grid: dict | None = None,
        mode: str = "grid",
        tuned_estimator_name: str | None = None,
        **kwargs,
    ):
        """Hyperparameter tuning for a single estimator, reinserted into the experiment.

        Runs a search on the full pipeline (same preprocessor, same step names), then
        adds the best estimator as a **new** named pipeline so it can be cross-validated
        and compared with everything else — no silent overwrite, no prep drift.

        Poniard ships no default hyperparameter grids: ``grid`` is required.

        Grid keys may be bare parameter names (``{"C": [...]}``) or full pipeline
        keys (``{"LogisticRegression__C": [...]}``). Bare names are prefixed with
        ``{estimator_name}__`` automatically. Keys that already contain ``__`` are
        left unchanged (so preprocessor params like ``preprocessor__...`` work).

        After tuning, inspect the search with `get_tuning_results` and compare the
        new pipeline via `fit` + `get_results` like any other estimator.

        Parameters
        ----------
        estimator_name :
            Estimator to tune (must already exist in `pipelines`).
        X :
            Features.
        y :
            Target.
        grid :
            Hyperparameter grid. Required.
        mode :
            Type of search. Either "grid", "halving" or "random". Default "grid".
        tuned_estimator_name :
            Name for the tuned pipeline. Default ``{estimator_name}_tuned``.
        kwargs :
            Passed to the search class constructor.

        Returns
        -------
        PoniardBaseEstimator
            Self.
        """
        if estimator_name not in self.pipelines:
            raise KeyError(
                f"Unknown estimator {estimator_name!r}. "
                f"Available: {list(self.pipelines)}"
            )
        if not grid:
            raise ValueError(
                "`grid` must be provided: poniard ships no default hyperparameter grids."
            )
        if mode not in ("grid", "halving", "random"):
            raise ValueError('mode must be "grid", "halving", or "random".')

        resolved_grid = self._resolve_tuning_grid(estimator_name, grid)
        estimator = clone(self.pipelines[estimator_name])
        self._pass_instance_attrs(estimator)

        scoring = self._first_scorer(sklearn_scorer=True)
        search_cls = {
            "grid": GridSearchCV,
            "halving": HalvingGridSearchCV,
            "random": RandomizedSearchCV,
        }[mode]
        search_kwargs = {
            "estimator": estimator,
            "scoring": scoring,
            "cv": self.cv,
            "verbose": self.verbose,
            "n_jobs": self.n_jobs,
            **kwargs,
        }
        if mode == "random":
            search = search_cls(
                param_distributions=resolved_grid,
                random_state=self.random_state,
                **search_kwargs,
            )
        elif mode == "halving":
            search = search_cls(
                param_grid=resolved_grid,
                random_state=self.random_state,
                **search_kwargs,
            )
        else:
            search = search_cls(param_grid=resolved_grid, **search_kwargs)

        search.fit(X, y)

        tuned_estimator_name = tuned_estimator_name or f"{estimator_name}_tuned"
        if tuned_estimator_name in self.pipelines:
            raise ValueError(
                f"Pipeline name {tuned_estimator_name!r} already exists. "
                "Pass tuned_estimator_name=... to choose another name."
            )

        best_final = clone(search.best_estimator_._final_estimator)
        self.add_estimators(estimators={tuned_estimator_name: best_final})

        if not hasattr(self, "_tuning_results"):
            self._tuning_results = {}
        self._tuning_results[tuned_estimator_name] = {
            "baseline": estimator_name,
            "mode": mode,
            "grid": resolved_grid,
            "best_params_": dict(search.best_params_),
            "best_score_": float(search.best_score_),
            "scorer": scoring,
            "search": search,
        }
        return self

    @staticmethod
    def _resolve_tuning_grid(estimator_name: str, grid: dict) -> dict:
        """Prefix bare param names with the pipeline step name.

        Keys that already contain ``__`` are treated as full pipeline paths and
        left unchanged.
        """
        resolved = {}
        prefix = f"{estimator_name}__"
        for key, values in grid.items():
            if "__" in key:
                resolved[key] = values
            else:
                resolved[f"{prefix}{key}"] = values
        return resolved

    def get_tuning_results(self, estimator_name: str | None = None) -> dict:
        """Return stored hyperparameter search results from `tune_estimator`.

        Parameters
        ----------
        estimator_name :
            Tuned pipeline name (e.g. ``"LogisticRegression_tuned"``). If None
            and exactly one tune has been run, that result is returned. If None
            and several exist, a dict keyed by tuned name is returned.

        Returns
        -------
        dict
            Per tune: ``baseline``, ``mode``, ``grid``, ``best_params_``,
            ``best_score_``, ``scorer``, and the fitted sklearn ``search``
            object (full escape hatch: ``cv_results_``, etc.).
        """
        results = getattr(self, "_tuning_results", None)
        if not results:
            raise ValueError(
                "No tuning results. Call tune_estimator(...) first."
            )
        if estimator_name is not None:
            if estimator_name not in results:
                raise KeyError(
                    f"No tuning results for {estimator_name!r}. "
                    f"Available: {list(results)}"
                )
            return results[estimator_name]
        if len(results) == 1:
            return next(iter(results.values()))
        return dict(results)
