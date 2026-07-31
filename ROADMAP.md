# Poniard — Improvement Roadmap

## Positioning

Poniard's defensible value proposition is not "answer quick questions" — every
model-comparison library does that. The actual differentiators are:

- Type-aware preprocessing (`PoniardPreprocessor`) with cardinality-driven
  encoder selection, datetime expansion, and a `ColumnTransformer` that is
  inspectable and editable.
- The Dummy baseline + `wrt_dummy` ratio as a built-in sanity check.
- A same-object loop: comparison → tune → ensemble → error analysis, all over
  the same cross-validated fold structure.
- **Error analysis** (`ErrorAnalyzer`) as a first-class concern. Most
  competitors stop at a leaderboard; Poniard keeps going into *where and why*
  the models fail. This is the heart of the library's identity and the bit that
  needs the most investment.
- The preprocessor and pipelines are plain sklearn objects the user owns and
  can export — Poniard is scaffolding you delete.

The pitch to sharpen: **"A transparent, sklearn-native model-comparison
workbench, type-aware preprocessing included, that hands you back a normal
sklearn pipeline when you're done — and tells you where your models are
wrong."**

---

## Priority 1 — Correctness and trust (fix before anything else)

These are bug-class issues that silently erode trust in the parts the library
is proudest of.

### 1.1 Remove global side effects

Every implicit global mutation must go. These are the things that get a library
quietly abandoned after one bad afternoon.

- `poniard/__init__.py` calls `sklearn.set_config(transform_output="pandas")`
  at import time. This affects every other sklearn transformer in the host
  program. Replace with a local `with sklearn.config_context(...)` block
  around the code that actually needs pandas output, or set
  `transform_output="pandas"` only on the `PoniardPreprocessor`'s own
  transformer instances. Importing poniard must have no effect on global state.
- `PoniardPlotFactory.__init__` mutates `plotly.io.templates.default` and
  `plotly.express.defaults.color_discrete_sequence`. Replace by applying the
  template/colors **per figure** (pass `template=...` to each `px.*` call) and
  never touching globals.
- `PoniardPreprocessor(cache_transformations=True)` writes to
  `transformation_cache/` in the current working directory (hardcoded relative
  path) and never cleans up. Use `tempfile.TemporaryDirectory` bound to the
  instance, or accept a path and document it; never default to silently
  creating files in CWD.

### 1.2 Fix bugs in differentiated modules

#### `LinearSVR` / `SVR` grid mismatch
`PoniardRegressor._default_estimators` ships `LinearSVR`, but the predefined
grid in `poniard/utils/hyperparameters.py` is keyed `"SVR`. Calling
`tune_estimator("LinearSVR", ...)` with no grid raises `NotImplementedError`.

Resolution (see also 3.1): the default hparam grids are being deleted entirely,
which eliminates this mismatch. Confirm `LinearSVR` still fits cleanly with no
default grid behavior to break.

#### `ErrorAnalyzer.analyze_features` bugs
Two correctness issues in the most differentiated method of the most
differentiated module:

1. **Positional-vs-name index mismatch.** `most_important_idx` holds column
   *positional* indices (from `importances_mean.argsort()[::-1][:n]`), but the
   loop iterates `for i, feature in enumerate(X.columns)` and checks
   `i in most_important_idx or feature in features`. When `features` is None,
   this sets `most_important_idx=[]` and `features=X.columns`, so the
   `i in most_important_idx` check is dead. When `estimator_name` is given,
   `features=[]` and `most_important_idx` is positional — correct in that
   branch but inconsistent. Unify on **one** representation (positional
   indices) and never compare a positional int to a column name.

2. **`_train_test_split_from_cv()` called without `X, y`.**
   `analyze_features` calls `self._poniard._train_test_split_from_cv()` with
   no arguments, but that method requires `X` and `y`. This would `NameError`
   in the `estimator_name` branch. Pass `X` and `y` through from the
   `analyze_features` call site — or, if `ErrorAnalyzer` was constructed via
   `from_poniard`, read `X`/`y` off the bound poniard estimator (and require
   they be stored there).

### 1.3 Replace `id()`-based fitted-pipeline tracking

`PoniardBaseEstimator.fit` tracks which pipelines have been cross-validated by
storing their Python `id()` in `_fitted_pipeline_ids`. Old pipeline objects can
be garbage-collected and new ones reuse the id, causing `fit` to silently skip
newly added estimators.

Replace with a set of **pipeline names** (the keys of `self.pipelines`). Names
are already the user-facing identity and are stable across
`add_preprocessing_step` / `reassign_types` rebuilds (which should invalidate
the fit set by getting new names or by the rebuild clearing the set).

---

## Priority 2 — Architecture and surface

### 2.1 Eliminate `setup`/`fit` duplication (keep the two-step UX)

The user-facing contract stays: `setup(X, y)` runs type inference and builds
pipelines so the user can introspect (`feature_types`, `preprocessor`,
`pipelines`, `metrics`) and adjust (`reassign_types`,
`add_preprocessing_step`, `add/remove_estimators`) **before** any fitting. Then
`fit(X, y)` cross-validates.

The current problem is ~40 lines of duplicated logic (input conversion,
`target_info`, metrics resolution, preprocessor build, info printing, pipeline
build, CV build) appearing verbatim in both methods.

Resolution: extract the shared "configure" logic into a private method (e.g.
`_configure(X, y, show_info)`) that **both** `setup` and `fit` call. `setup`
calls it and returns. `fit` calls it (idempotent — re-running it with the same
inputs is a no-op, re-running after `reassign_types` rebuilds the pipelines as
it already does) and then proceeds to cross-validate. The two-method UX and
introspection flow are preserved; the duplicate code is gone.

### 2.2 Promote `get_estimator` to the headline exporter

`get_estimator(name, include_preprocessor=True, X=None, y=None, retrain=True)`
is already the "give me a normal sklearn pipeline I own" escape hatch. Push it
to the front of the README and the docstring:

- Document it as the supported way to leave Poniard.
- Guarantee the returned object is a plain `sklearn.pipeline.Pipeline` (or bare
  estimator when `include_preprocessor=False`) with no poniard references.
- Add a test that round-trips: build a poniard estimator, `get_estimator(...,
  retrain=True)`, then `pickle.dumps` the result and confirm it loads without
  poniard installed (run in a subprocess with `PYTHONPATH` stripped) — this is
  the real definition of "you can delete poniard when you're done."

### 2.3 Remove default hyperparameter grids entirely

`poniard/utils/hyperparameters.py` ships predefined grids for the default
estimators plus XGBoost (which isn't a dependency). Problems:

- The `LinearSVR`/`SVR` mismatch above.
- Defaults encode opinions about what to tune, which varies wildly by dataset.
- XGBoost/CatBoost/LightGBM grids are present for a library the user installed
  themselves; the user knows their booster's GPU/n_thread situation better
  than poniard does.

Resolution: **delete `GRID` and the `grid=None` → look-up-`GRID` path in
`tune_estimator`.** Require the user to pass `grid=...` explicitly. This makes
`tune_estimator` a thin convenience wrapper around sklearn's search classes on
top of a poniard pipeline, with no surprise opinionated defaults.

If a user wants the old behavior, they can keep a `hyperparameters.py`-style
dict in their own project. (If we ever want to bring back opinionated grids,
ship them as a *separate* optional module, not as default behavior.)

### 2.4 Reduce the public method surface

Commit to a small, observable-tested core. Cut the operator overloads —
`__add__` / `__sub__` / `__getitem__` — they save no time over
`add_estimators` / `remove_estimators` and `pipelines[name]`, and every method
is a documentation and testing burden. Keep:

- `setup`, `fit`, `get_results`
- `predict`, `predict_proba`, `decision_function` (drop `predict_all`? keep
  only if used by plots/error analysis)
- `add_estimators`, `remove_estimators`, `get_estimator`
- `reassign_types`, `add_preprocessing_step`
- `tune_estimator`, `build_ensemble`
- `get_predictions_similarity`

Everything else is an internal implementation detail.

### 2.5 Tighten the default estimator lists

Defaults are opinions; pick strong ones. `SVC(kernel="linear",
probability=True)` is slow and rarely competitive on real tabular data —
people sit through a tqdm bar and wonder why it's there. Consider:

Classification:
- `LogisticRegression(max_iter=5000)`
- `RandomForestClassifier()`
- `HistGradientBoostingClassifier()`

Regression:
- `LinearRegression()`
- `ElasticNet()`
- `RandomForestRegressor()`
- `HistGradientBoostingRegressor()`

Plus the auto-added Dummy. `KNeighbors`, `DecisionTree`, `GaussianNB`,
`LinearSVR` can stay if you want breadth, but `SVC(kernel="linear")` should
go. Keep the dummy at all costs — it's the `wrt_dummy` value prop.

---

## Priority 3 — Make error analysis a headline

`ErrorAnalyzer` is the module most competitors don't have, and the one most
aligned with how you actually do ML. It needs investment, not deletion.

### 3.1 Test coverage

`ErrorAnalyzer` currently has **zero tests.** This is the scariest gap in the
library. Add tests covering:

- `from_poniard` construction.
- `rank_errors` for binary, multiclass, multilabel, continuous, and
  continuous-multioutput targets — assert the right error metric is selected
  and the returned DataFrame has the right index and columns.
- `merge_errors` aggregation: `mean_error`, `freq`, per-row estimator lists.
- `analyze_target` for classification (error vs target distribution) and
  regression (binned).
- `analyze_features`:
  - without `estimator_name` (just `features=X.columns`),
  - with `estimator_name` and `n_features` (positional-index path),
  - the bug-fixes from 1.2 verified by assertions on the selected features.
- Round-trip: `from_poniard` → `rank_errors` → `merge_errors` →
  `analyze_target` → `analyze_features` on a synthetic dataset where the
  expected error region is known.

### 3.2 Fix the integration bugs (1.2) and make `from_poniard` carry data

`from_poniard(poniard, estimator_names)` should capture `X` and `y` from the
bound poniard estimator so `analyze_features` and `analyze_target` can be
called without re-passing them. Currently `_train_test_split_from_cv()` is
called without `X, y` and would `NameError`. Either:

- store `X`/`y` on `ErrorAnalyzer` at `from_poniard` time (requires the poniard
  estimator to also store them — it receive them in `setup`/`fit` and keep
  refs), or
- change `analyze_*` to require `X`/`y` explicitly and stop pretending the
  no-arg path works.

Storing `X`/`y` on the poniard estimator is also needed for `predict` /
`predict_proba` / `get_estimator(retrain=True)` to be callable without
re-passing data, and is needed for pickling (see 4.1). Prefer that.

### 3.3 Documentation and discoverability

- Add an `ErrorAnalyzer` section to the README.
- Add an example notebook covering the full error-analysis workflow
  (rank errors → merge → analyze by target → analyze by feature → interpret).
- Rename private `_full_estimator_analysis` in `PoniardPlotFactory` — it's a
  user-facing dashboard; either surface it publicly or document it as
  internal. Right now it's the best plot in the library and nobody can find it.

### 3.4 Plotting tests

`PoniardPlotFactory` has zero tests. Add smoke tests that instantiate it via
`poniard[plot]` and assert each method returns a `plotly.graph_objects.Figure`
without raising, on both classification and regression poniard estimators.
These don't need to assert on figure contents — just that the call paths don't
explode. The integration with `_experiment_results` and
`_get_or_compute_prediction` is fragile and only exercised when a user clicks
through a notebook.

---

## Priority 4 — Robustness and completeness

### 4.1 Add `save` / `load`

`joblib` is already a dependency (used only for `Memory` so far). Add
`PoniardBaseEstimator.save(path)` and `PoniardBaseEstimator.load(path)`
classmethods that joblib-dump the whole estimator. The whole pitch is "fast
first pass" — first passes get shown to teammates and re-opened later.
Pickle round-trip should be tested as part of CI: fit → save → load →
`get_results` → assert identical.

Storing `X`/`y` on the estimator (see 3.2) makes this trivial. Decide whether
they should be saved with the estimator or dropped on save (memory leak
concern); a `save(path, include_data=True)` flag is one option.

### 4.2 Replace frame-introspection repr

`get_kwargs` (in `poniard/utils/helpers.py`) introspects `inspect.currentframe`
to read `f_back.f_locals` and reconstruct constructor kwargs for `__repr__`.
It's clever and brittle — breaks if `__init__` is decorated, partial-applied,
or if anyone calls `super().__init__` from anywhere but the direct caller.
`_init_params` is stored on both `PoniardBaseEstimator` and
`PoniardPreprocessor` purely to support this.

Replace by capturing kwargs explicitly in `__init__`:

```python
def __init__(self, **kwargs):
    self._init_params = {**kwargs}
    ...
```

or by enumerating the constructor params and reading them back from `self` at
repr time. Either way, remove the frame walk.

### 4.3 Stop stateful side effects in `PoniardPreprocessor._infer_dtypes`

`_infer_dtypes` mutates `self.numeric_threshold` and
`self.cardinality_threshold` from floats to ints as a side effect, so a second
`build` with different `X` silently uses the converted ints. Keep the
constructor values pristine; compute the int thresholds into local variables
(or a separate `_resolved_numeric_threshold` attribute) and use those.

### 4.4 Decouple `_process_results` from cross_validate column order

`_process_results` reorders columns by `list(means.columns[2:]) + ["fit_time",
"score_time"]`, relying on the convention that columns 0–1 from
`cross_validate`'s output dict are the times. This is brittle against sklearn
output-order changes. Instead, select by name: identify time columns by name
(`fit_time`, `score_time`) and the rest as metrics. No positional assumptions.

### 4.5 Clarify prediction semantics

`_predict` initializes `result = np.empty(y.shape); result[:] = np.nan` for
estimators without `predict_proba`/`decision_function`. For 2-D `y` this
silently fills a same-shape NaN array, possibly masking real issues. Either
raise a clear error ("estimator X does not support method Y") or document the
NaN-fill as intentional and make the shape precise. Test both paths.

---

## Priority 5 — Cleanup

- **Document the `PONIARD_TQDM_LEAVE` env var** in the README (it's currently
  discoverable only by reading source).
- **`get_results(wrt_dummy=True)`**: ratio of stds is not meaningful; either
  drop stds from that view or document it as "ratio of means only." Also the
  multiple-Dummy squeeze case should be tested.
- **`get_predictions_similarity`** excludes dummies by substring match
  (`"DummyClassifier"` / `"DummyRegressor"`), not by `poniard_task`. Switch to
  checking the estimator class for `Dummy*` via isinstance, or just exclude
  by the standard prefix.
- **XGBoost grids** go away with 2.3 — confirm no lingering references.
- **`_fitted_pipeline_ids`** replacement (1.3) — verify
  `add_preprocessing_step` and `reassign_types` clear the fitted set so a
  re-fit after reconfiguring re-runs everything.
- **The `poniard_task` isinstance + deferred import** in `core.py` — leave as
  is for now; it's awkward but works. Revisit only if we ever add a third task
  type (which we won't).

---

## Testing summary

Add tests for (in order of urgency):

| Module | Current coverage | Needed |
|---|---|---|
| `ErrorAnalyzer` | none | full coverage — see 3.1 |
| `PoniardPlotFactory` | none | smoke per method — see 3.4 |
| `tune_estimator` after grid deletion (2.3) | partial | explicit-grid path, error path |
| `save`/`load` round-trip (4.1) | none | fit → save → load → identical results |
| `get_estimator` no-poniard pickling (2.2) | none | subprocess pickle load without poniard |
| `wrt_dummy=True` multiple dummies (4.x) | none | numerical correctness |
| `reassign_types` roundtrip | none | assign → fit → get_results sane |
| `cache_transformations` cleanup | none | temp dir is removed |

---

## Sequencing

Do them in this order — each step unblocks or de-risks the next:

1. **1.1** (side effects) and **1.3** (`id()` → names) — small, mechanical,
   high trust.
2. **1.2** (`ErrorAnalyzer` bugs) and **3.1** (its tests) — fixes the most
   differentiated module and locks it in.
3. **2.3** (delete grids) and **2.5** (tighten default estimators) — sharpens
   the opinionated surface.
4. **2.1** (de-duplicate `setup`/`fit`) — now safe because the fitted-tracking
   is by name.
5. **2.2** (`get_estimator` as exporter) + its pickle-without-poniard test.
6. **4.1** (`save`/`load`), **4.3**, **4.4** — robustness pass.
7. **3.2**–**3.4** and **4.2** — polish the analysis layer.
8. **2.4** and Priority 5 — surface cleanup, last, lowest risk.