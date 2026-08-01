# Poniard Tech Debt Roadmap

**Date:** 2026-08-01 · **Scope:** `poniard/` (~4,200 LOC), `tests/` (~2,900 LOC), packaging/CI
**Baseline:** 210 tests passing (113s), `ruff check` clean, product direction set in `ROADMAP.md`

This document complements `ROADMAP.md` (product/features). It tracks *internal* quality:
correctness bugs, architecture debt, code smells, inefficiencies, and tooling drift.

---

## Design constraints (do not "fix" these)

- **`setup()` stays the first-class entry point.** It exists so users can *see* what `fit`
  will do preprocessing-wise and *modify* it before fitting. The preprocessor must never
  become a black box. Any refactor of the configure/fit lifecycle must preserve — and
  sharpen — the `setup() → inspect/modify preprocessor → fit()` contract.
- Lightweight core deps (sklearn/pandas/numpy/scipy/joblib/tqdm); plotly optional.
- Breaking changes are fine (pre-1.0, small user base). Prefer delete over deprecate.

---

## P0 — Verified bugs

All four were reproduced live on 2026-08-01. Each fix must ship with a regression test.

### 1. Multiclass error analysis crashes on non-integer class labels 🔴
`error_analysis.py:280–288` — `_rank_errors_multiclass` does `row[f"proba_{int(row['y'])}"]`,
assuming class labels are `0..n-1` ints. String labels crash with
`ValueError: invalid literal for int(): 'dog'`; non-contiguous int labels silently index the
wrong proba column.

**Fix:** map labels positionally (e.g. `np.searchsorted(np.unique(y), y)` /
`np.take_along_axis`) — this also removes the row-wise `.apply(axis=1)` (see P3 perf).
**Severity:** breaks `ErrorAnalyzer`, the P1 product core, for a common input shape.

### 2. Plotting mutates the stored pipelines 🔴
`plot_factory.py:269–270` (`permutation_importance`) and `plot_factory.py:500–501`
(`partial_dependence`) call `.fit()` **directly on the pipeline stored in
`estimator.pipelines[name]`** — not a clone. After drawing a plot, the pipeline is silently
fitted on a random 80/20 split. Reproduced: `hasattr(pipe, "classes_")` flips `False → True`
after one plot call. `full_estimator_analysis` inherits the problem via
`permutation_importance`.

This directly violates `ROADMAP.md` §7.1 ("side-effect discipline — don't regress"), and
`tests/test_side_effects.py` does not cover the plotting surface.

**Fix:** `sklearn.base.clone` the pipeline before fitting (or route through
`get_estimator(..., retrain=True)`); extend side-effect tests to every plot method that fits.

### 3. Remove → re-add an estimator → it is silently never refit 🔴
`core.py:758–794` + `core.py:399` — `remove_estimators` drops results but leaves the name in
`_fitted_pipeline_names`; `add_estimators` bypasses `_build_pipelines()` (the only place that
resets that set); the next `fit()` filters the re-added pipeline out as "already fitted".
Reproduced: after remove + re-add + fit, `get_results()` contains only `DummyClassifier` —
no error, no warning.

**Fix:** single source of truth for fitted tracking (see P1-A1); `remove_estimators` must
discard the name from the fitted set.

### 4. `metrics()` faceting broken for a single metric
`plot_factory.py:136–147` — `metrics = "|".join(metrics)` reassigns the list to a string,
then `len(metrics) > 1` measures *string length* (always > 1 for real metric names), so
single-metric plots always get a facet wrapper.

**Fix:** compute facet decision from the list before joining; escape the joined pattern
(`re.escape`) since metric names flow into `str.contains` as regex.

---

## P1 — Architecture debt

### A1. `hasattr`-driven implicit state machine
13 `hasattr(self, ...)` + 5 `getattr(self, ..., None)` sites define the object lifecycle
(`_experiment_results`, `_means`, `_stds`, `_fold_sizes`, `_tuning_results`,
`_poniard_preprocessor`, `cv`, `feature_types`, ...). Attributes are born inside `fit`,
`_predict`, `tune_estimator`, `reassign_types` — no single place states what a
configured/fitted estimator *is*. This is the root-cause enabler of bug #3.

**Fix:** initialize every state attribute in `__init__` (or a small `_ExperimentState`
dataclass); make lifecycle transitions (`configured`, `fitted`) explicit and documented.
The public `setup() → mutate → fit()` flow stays exactly as-is.

### A2. `_experiment_results` is a grab-bag
One dict holds CV scores, fit/score times, `predict` arrays and `predict_proba` arrays.
Downstream code excludes predictions by *string-matching keys* (`results.py:270`), and
`_process_long_results` melts prediction vectors into the plotting table (`results.py:299–312`).

**Fix:** two stores — `_cv_results` (scores/times) and `_prediction_cache` (arrays) — with an
explicit, documented cache API. This is *also* the missing infrastructure for `ROADMAP.md`
§1.2 ("avoid recompute traps") and §2.2 ("cache CV predictions"): do them together.

### A3. Circular-import `isinstance` task detection
`core.py:124–140` — `poniard_task` does a deferred `from poniard import ...` + `isinstance`
chain. The subclasses know their own task.

**Fix:** abstract property on `PoniardBaseEstimator`; `PoniardClassifier` /
`PoniardRegressor` return the literal string. Removes the import cycle and the `None` branch
that silently flows through type inference.

### A4. Privates are the real API across modules
Mixins touch core internals freely, and both satellite modules reach deep into estimator
privates: `PoniardPlotFactory` uses `_long_results`, `_stds`, `_experiment_results`,
`_train_test_split_from_cv`, `_first_scorer`; `ErrorAnalyzer` uses `_first_scorer`, `cv`,
`_poniard` internals. Cross-module `_private` access means nothing is private.

**Fix:** define a small documented internal protocol (e.g. `EstimatorView`: results table,
prediction cache, pipelines, primary scorer, task) consumed by plots and error analysis.

### A5. Mutating objects the user owns
`_pass_instance_attrs` setattrs `random_state`/`verbose` on user-passed estimators **and on
the user's CV splitter** (`classification.py:134`, `regression.py:111`);
`add_preprocessing_step` does `existing_preprocessor.steps.insert(...)` in place on a
user-supplied Pipeline (`core.py:694`).

**Fix:** `clone()` first, then configure the clone.

### A6. Mutator-web invalidation is inconsistent (setup itself is *not* debt)
`setup()` remains the canonical first step (see *Design constraints*). The debt is the
inconsistent state-reset semantics *around* it: `_build_pipelines()` resets
`_fitted_pipeline_names`, but `add_estimators`/`remove_estimators` bypass it;
`reassign_types`/`add_preprocessing_step` rebuild pipelines but leave other derived state
untouched; and `fit()` silently auto-configures when `setup()` was never called — a second,
implicit path that duplicates the explicit one.

**Fix:** keep `setup()` public and first; document the `setup() → mutate → fit()` contract;
make every mutator invalidate derived state through one code path; decide explicitly whether
bare `fit()` auto-setup is supported (if yes, document it; if no, fail with a clear error
pointing at `setup()`).

---

## P2 — Code smells

| Smell | Location | Fix |
|---|---|---|
| Dead param `all_estimators` in `_generate_estimator_name`; docstring doesn't match behavior | `core.py:339–353` (4 call sites) | Delete param, fix docstring |
| Dummy-estimator detection duplicated 4×, two hand-rolled instead of `_dummy_names()` | `results.py:66`, `error_analysis.py:135–144`, `plot_factory.py:879–882` | Single helper on the estimator |
| Polars→pandas / array coercion copy-pasted | `core.py:182–189` vs `preprocessing/core.py:241–248` | One util (e.g. `utils.estimate.to_pandas_maybe`) |
| `analyze_features` builds a throwaway `PoniardPreprocessor(task="placeholder")` just to infer feature types | `error_analysis.py:518–523` | Extract pure `infer_feature_types(X)`; preprocessor uses it too |
| `error_lift_bars`: dead public param `estimator_names`; docstring promises fallback behavior the code doesn't have (raises when `None`) | `plot_factory.py:744–780` | Remove param; align docstring or implement the fallback |
| Blanket `except AttributeError` in `_predict` converts any internal estimator bug into NaN predictions | `core.py:511–527` | Pre-check `hasattr(pipeline, method)` instead of catching |
| `roc_curve` validates only `y.ndim > 1` (1-D multiclass slips through); `proba[:, 1]` assumes `classes_[1]` is positive | `plot_factory.py:354–393` | Validate `nunique == 2`; resolve positive class from `classes_` |
| Dead `if TYPE_CHECKING: pass`; dead sklearn `<1.3` `grid_values`/`values` shim (floor is 1.3) | `preprocessing/core.py:35`, `plot_factory.py:506` | Delete |
| sklearn imports inside functions where sklearn is a hard dep | `error_analysis.py:135`, `plot_factory.py:866` | Hoist to module level |
| `from typing import Sequence` (deprecated alias) | `core.py:7`, `classification.py:6`, `regression.py:6` | Use `collections.abc` |
| `_init_params = locals()` capture is fragile (any local added above it leaks into repr) | `core.py:93`, `preprocessing/core.py:92`, `error_analysis.py:99` | Explicit dict of params |

---

## P3 — Inefficiencies

1. **`ErrorAnalyzer._compute_predictions` runs full CV twice per `analyze()`** — once for
   `predict`, once for `predict_proba` (`error_analysis.py:149–159`) — and calls
   `poniard.predict()` directly, **bypassing the prediction cache** entirely. Two `analyze()`
   calls = 4× full cross-validation of every model. Biggest perf bug in the library, sitting
   in the product core.
   **Fix:** route through the prediction cache (P1-A2); derive hard predictions from probas
   (`argmax` over `classes_`) instead of a second CV pass.
2. `_rank_errors_multiclass` row-wise `.apply(axis=1)` — vectorize (bundles with P0-#1).
3. `_process_long_results` explodes prediction arrays into the long scores table when
   predictions exist (`results.py:299–312`) — fixed by P1-A2's store split.
4. `get_predictions_similarity` regression path computes correlations *including* dummies,
   then drops them (`results.py:383`).

---

## P4 — Tooling & packaging drift

- **Ruff version drift (confirmed):** `.pre-commit-config.yaml` pins `v0.11.0`; dev group
  installs `0.16.0`; `ruff format --check` reports **22 files would be reformatted**. CI runs
  `ruff check` but never `ruff format --check`. → Align versions (single source), gate format
  in CI, do one reformat commit.
- **No coverage measurement.** pytest-cov absent; bug #3's code path was clearly unexercised.
  → Add pytest-cov with a ratcheting floor in CI.
- `pyproject.toml` build requires `setuptools-scm>=8.0` but `version = "0.6.0"` is static.
  → Dynamic version from scm, or drop the build dep.
- Dev deps declared twice (`[project.optional-dependencies] dev` + `[dependency-groups] dev`).
  → One source of truth.
- Examples (3 notebooks) not executed in CI → bit-rot as the API diet continues. Consider
  `nbval` or a smoke script.
- 29 warnings in the test run (incl. plotly `append_trace` deprecation used by
  `full_estimator_analysis`, `plot_factory.py:713`) with no warnings gate. → Fix our own
  deprecations; `filterwarnings = ["error"]` scoped appropriately.
- No slow/fast markers for the 113s suite; no `AGENTS.md`.

---

## Delivery plan

| Phase | Theme | Items | Effort |
|---|---|---|---|
| **0 — Stop the bleeding** | P0 bugs #1–#4, each with a regression test | Label-safe multiclass ranking; clone-before-fit in plots + side-effect tests for plotting; fitted-tracking invalidation on remove/add; facet-length fix | **1–2 days** |
| **1 — Explicit state** | P1-A1, A2, A3; root-cause of #3 | State initialized in `__init__`; `_cv_results`/`_prediction_cache` split with documented cache API; `poniard_task` as abstract property; single fitted-tracking owner | **3–4 days** |
| **2 — Tooling guards** | P4 (ruff alignment, format gate, coverage ratchet, version source) | Do the reformat commit *between* phases to keep diffs reviewable | **0.5–1 day** |
| **3 — Performance** | P3 #1–#4 | Cached predictions in `ErrorAnalyzer`; probas→predictions derivation; vectorized ranking; long-table hygiene | **1 day** |
| **4 — Decouple satellites** | P1-A4, A5, A6; P2 smells | `EstimatorView` protocol; clone-before-configure; uniform mutator invalidation honoring the `setup()`-first contract; dead-param/docstring cleanup | **2–3 days** |

**Sequencing notes**

- Phase 2 lands early because it's half a day and immediately guards all later phases.
- Phase 1 *is* the infrastructure for `ROADMAP.md` §1.2/§2.2 (prediction caching) — schedule
  them together rather than building cache twice.
- Phase 4 last because touching core/plot boundaries is safest once state and tests are solid.

## Explicitly out of scope (don't spend here)

- Merging the mixins into one file — cosmetic churn.
- mypy strict — sklearn-typing tax is disproportionate for a library this size; ruff + tests suffice.
- Functional rewrite of `PoniardPlotFactory` — only stop the mutation and private access.
- Any change to the `setup()`-first, inspectable-preprocessor workflow — it's a feature, not debt.
