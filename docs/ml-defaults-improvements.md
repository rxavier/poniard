# ML defaults improvements

Proposed machine-learning-quality improvements, sorted by priority.
Part 1 is a set of small, independently shippable default changes.
Part 2 is a design spec for per-estimator preprocessors (motivated by
HistGradientBoosting's native NaN/categorical support).

Explicitly out of scope (decided):

- `class_weight="balanced"` as a default (distorts calibration; document as a
  recipe instead).
- Replacing `StandardScaler` as the default scaler (marginal gain, standard is
  the least surprising default).
- Shipping default hyperparameter grids for tuning (existing design decision).

---

## Part 1 — Better defaults

### P0-1. Numeric imputation: mean → median

**Problem.** `SimpleImputer(strategy="mean")` (`preprocessing/core.py:377`)
followed by `StandardScaler` lets outliers dominate both steps. The datetime
pipeline already uses median (P0 preprocessing fix), so numeric is now the
inconsistent outlier.

**Fix.** Default numeric imputer to `strategy="median"`. `numeric_imputer`
gains `"mean"` / `"median"` literals, default `"median"`.

**Test.** Default pipeline imputes numeric NaNs with the median.

### P0-2. Missingness indicators on numeric features

**Problem.** Missingness is often informative ("income not reported"); mean/
median imputation destroys that signal.

**Fix.** `SimpleImputer(..., add_indicator=True)` on the numeric imputer.
Synergy with existing steps: when a column has no missing values, its
indicator is constant and the existing `VarianceThreshold` drops it — zero
cost when useless.

**Test.** A column with NaNs yields a `missingindicator_*` column post-fit; a
column without NaNs yields none.

### P0-3. Regression metrics: drop MAPE, change primary metric

**Problem.** `neg_mean_absolute_percentage_error` (`estimators/regression.py`)
silently produces astronomical values when `y` contains zeros (sklearn
epsilon-floors the denominator), poisoning CV means. Also, the first metric is
the primary sort key for ranking/ensembles (`_means.columns[0]`), and
`neg_mean_squared_error` is outlier-dominated.

**Fix.** Default regression metrics:
`["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]`.
Include `neg_mean_absolute_percentage_error` only when the target is strictly
positive (checkable in `_build_metrics` via `target_info`/`y`).

**Test.** Defaults contain RMSE first and no MAPE; a zero-containing target
still scores cleanly.

### P0-4. Classification metrics: add log-loss, PR AUC for binary

**Problem.** ROC AUC ranks well but rewards confident wrongness; nothing in
the defaults is informative under class imbalance.

**Fix.** Add `neg_log_loss` to all classification default metric lists (all
default classifiers support `predict_proba`), and `average_precision` to the
binary list (`estimators/classification.py::_build_metrics`). Keep `roc_auc`
first to avoid changing ranking behavior in this PR.

**Test.** Default binary metrics include `neg_log_loss` and
`average_precision`; multiclass/multilabel include `neg_log_loss`.

### P1-5. `OneHotEncoder(min_frequency=...)`

**Problem.** Every observed category gets a column; rare categories add noise
and dimensionality.

**Fix.** Add `min_frequency` (e.g. 5) to the default OHE, exposed as a
`PoniardPreprocessor` parameter (`ohe_min_frequency: int | float | None`).
Rare categories collapse into sklearn's infrequent bucket; composes correctly
with the existing `handle_unknown="ignore"`.

**Test.** Categories rarer than the threshold collapse into the infrequent
column; `None` preserves current behavior.

### P1-6. Pass `n_jobs` to KNN

**Problem.** `KNeighborsClassifier/Regressor` are constructed without `n_jobs`
(`classification.py`, `regression.py`), unlike RandomForest.

**Fix.** Pass `n_jobs=self.n_jobs`.

**Test.** Default KNN pipelines carry the estimator's `n_jobs`.

---

## Part 2 — Per-estimator preprocessors (design spec)

### Motivation

One shared preprocessor forces lowest-common-denominator treatment.
HistGradientBoosting handles NaN natively (missing values get their own split
direction — principled, not imputed) and categoricals natively via
`categorical_features="from_dtype"` with pandas `Categorical` columns. The
default pipeline mean-imputes away HGB's missingness signal and encodes
categoricals it could split on directly. Meanwhile linear models genuinely
need imputation + scaling + one-hot. The fix is a mapping from estimators to
preprocessors, not a compromise preprocessor.

### Core idea

Replace the single preprocessor template with a **registry of named
preprocessors** plus an **estimator → preprocessor mapping**:

```python
self.preprocessors: dict[str, Pipeline]   # "default" always exists when preprocess=True
self.preprocessor_map: dict[str, str]     # estimator name -> preprocessor name
```

There is exactly **one consumption point**: `_make_pipeline(name, estimator)`
(`estimators/core.py:389`) resolves
`prep_name = self.preprocessor_map.get(name, "default")` and builds
`Pipeline([("preprocessor", self.preprocessors[prep_name]), (name, estimator)])`.
Both `_build_pipelines` and `add_estimators` already funnel through
`_make_pipeline`, so no other call site changes. Pipeline step name stays
`"preprocessor"` everywhere, so tuning grid keys (`preprocessor__...`),
`get_estimator`, and save/load are unaffected.

### API

```python
PoniardClassifier(
    ...,
    preprocessor_map={"HistGradientBoostingClassifier": "native"},  # or a Pipeline instance
)

# after setup(), before fit():
est.add_preprocessor("sparse_friendly", my_pipeline)      # register a custom template
est.set_preprocessor("LogisticRegression", "sparse_friendly")     # (re)assign
est.preprocessor_map                                        # inspect
```

- `preprocessor_map` values: a registered name (`"default"`, `"native"`, or
  user-registered) or an actual Pipeline/Transformer (auto-registered under a
  generated name).
- `set_preprocessor` validates: estimator exists in `pipelines`, preprocessor
  name is registered. Called pre-`fit`, it rebuilds that estimator's pipeline
  via `_make_pipeline`.
- Unknown estimator or preprocessor names → `KeyError` listing valid names.

### Backwards compatibility

- Default behavior is byte-identical: `preprocessors == {"default": ...}`,
  empty map, every pipeline uses `"default"`.
- `self.preprocessor` becomes a property aliasing
  `self.preprocessors["default"]` (with a setter), so `reassign_types`,
  `add_preprocessing_step`, and external code keep working.

### The `"native"` profile

Lives in `PoniardPreprocessor` as `profile: Literal["default", "native"]`,
switching `_setup_transformers` (keeps one class and the shared
type-inference machinery):

- numeric → `passthrough` (HGB learns NaN split directions; no scaling needed
  for trees)
- categorical (low *and* high cardinality) →
  `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)`
  then a small `_ToCategorical` transformer casting output columns to pandas
  `category` dtype
- datetime → `DatetimeEncoder` integers, passthrough
- no `VarianceThreshold`, no scaling

Estimator coupling: `set_preprocessor(name, "native")` validates the mapped
estimator is `HistGradientBoosting*` and auto-sets
`categorical_features="from_dtype"` on the cloned estimator via `set_params`
(same spirit as `_pass_instance_attrs`); document the side effect. Mapping
`"native"` to anything else → `ValueError`.

**Open question to verify in implementation:** sklearn's exact NaN semantics
for `from_dtype` categorical columns (expected: NaN → treated as missing,
handled natively). Lock it down with a unit test using NaNs in a categorical
column.

### Setup flow changes (`_configure`, `estimators/core.py:257`)

1. Build the `"default"` preprocessor exactly as today
   (`_build_preprocessor`), computing `feature_types` once.
2. For each additional registered PoniardPreprocessor profile:
   `build(X=X, y=y, task=..., target_info=..., feature_types=self.feature_types)`
   — reuses the P0 `build(feature_types=...)` parameter, so type inference
   runs exactly once and all profiles agree on it. `reassign_types` therefore
   propagates by rebuilding each PoniardPreprocessor-backed profile with the
   new `feature_types`; user-supplied Pipeline templates are left untouched.
3. Cache/memory: each preprocessor keeps its own `memory`; `_make_pipeline`
   uses the *mapped* preprocessor's memory instead of the global
   `self._memory`.
4. `_print_setup_info`: unchanged (inference is shared), plus one line listing
   non-default mappings when any exist.

### Ensembles

`build_ensemble` currently takes `self.pipelines[name]._final_estimator`
(bare estimators) and the ensemble gets wrapped in the shared preprocessor —
wrong once members need different preprocessors.

- If all selected members map to the **same** preprocessor: current behavior
  (bare members, ensemble wrapped in that preprocessor). Efficient —
  preprocessing runs once.
- If members are **mixed**: use each member's full pipeline as the ensemble
  member (sklearn ensembles accept Pipelines) and add the ensemble itself
  with no outer preprocessor. Requires an internal
  `_make_pipeline(..., include_preprocessor: bool | None = None)` knob
  (`None` = resolve from mapping, `False` = bare).

### `add_preprocessing_step`

Gains `preprocessor: str | Sequence[str] = "all"` selecting which registered
templates receive the step. `"all"` preserves today's global behavior;
unknown names → `KeyError`.

### What does not change

- `EstimatorView` protocol (`pipelines`, `feature_types`) — plotting and error
  analysis only see finished pipelines, which already contain the right
  preprocessor.
- Results, metrics, CV, tuning (grid keys keep the `"preprocessor"` step
  name), prediction cache, `get_estimator` (returns the pipeline with its
  mapped preprocessor), save/load (new dict attributes pickle fine).

### Test plan

| Area | Test |
| --- | --- |
| Mapping | Mapped estimator's pipeline uses the native template; unmapped use default |
| Validation | `set_preprocessor` raises on unknown estimator/preprocessor and on non-HGB + `"native"` |
| Native E2E | Mixed-type data with NaNs → mapped HGB cross-validates and scores |
| Auto-param | Mapped HGB clone has `categorical_features="from_dtype"` |
| Shared inference | `reassign_types` rebuilds all PoniardPreprocessor profiles |
| Step targeting | `add_preprocessing_step(preprocessor="native")` leaves default untouched |
| Ensemble | Mixed-preprocessor ensemble fits and scores |
| Persistence | save/load round-trip preserves `preprocessor_map` |
| Tuning | `tune_estimator` with `preprocessor__...` keys works on a mapped pipeline |

### Suggested sequencing

1. Part 1 P0 items (independent, small).
2. Registry + mapping with `"default"` only (mechanical refactor, no behavior
   change, full back-compat test pass).
3. `"native"` profile + HGB wiring + NaN-semantics verification test.
4. Ensemble mixed-preprocessor support.
