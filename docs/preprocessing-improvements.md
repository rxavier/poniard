# Preprocessing improvements

Proposed improvements for `poniard/preprocessing/`, sorted by priority.
P0 items change observable behavior (some are latent bugs); P1 fixes input
parity; P2 adds opt-in modeling quality; P3 is hygiene.

Explicitly out of scope (decided):

- ID-like column detection (uniqueness is not a reliable signal, e.g. GDP-like floats).
- Coercing object columns to datetime (user's responsibility).
- Sparse one-hot output (dense OHE on low-cardinality features is intentional).

---

## P0 — Correctness / consistency

### 1. Stable output feature names regardless of type composition

**Problem.** `PoniardPreprocessor.build()` (`core.py:314-317`) strips empty
transformers and, when only one remains, replaces the whole `ColumnTransformer`
with the inner transformer. But with ≥2 feature types, `ColumnTransformer`
defaults to `verbose_feature_names_out=True`, so outputs are prefixed
(`numeric_preprocessor__age`); with a single type they are unprefixed (`age`).
Adding one categorical column silently renames every downstream feature,
breaking feature-importance comparisons, SHAP, and any name-keyed consumer.

**Fix.**

- Set `verbose_feature_names_out=False` on every `ColumnTransformer` built in
  `build()`.
- Stop unwrapping single-transformer `ColumnTransformer`s (or build the
  transformer list conditionally *before* constructing the
  `ColumnTransformer`, instead of mutating `.transformers` after the fact).
- Add a test asserting identical feature names for the same columns whether or
  not other feature types are present.

### 2. Datetime features bypass scaling (and use mode imputation)

**Problem.** The datetime pipeline is `DatetimeEncoder →
SimpleImputer(most_frequent)` (`core.py:404-412`). Encoded outputs mix scales
(year ≈ 2020, dayofyear 1–366, month 1–12) and reach estimators unscaled,
while numeric features are scaled — inconsistent for regularized and
distance-based models. Mode imputation is also statistically odd for what are
effectively numeric features.

**Fix.** Append the configured scaler (and use `median` imputation) to the
datetime pipeline, or route encoded datetime features through the numeric
pipeline. Add a test asserting datetime-derived features are scaled.

### 3. Dead `custom_preprocessor` parameter

**Problem.** `PoniardPreprocessor.__init__` accepts `custom_preprocessor` and
stores it in `_init_params` (`core.py:169, 183`) but never uses it — the
estimator layer handles custom preprocessors itself.

**Fix.** Remove it from the signature and `_init_params`, or actually honor it
inside `build()`. Prefer removal (single responsibility).

### 4. `build()` without data raises a confusing error

**Problem.** Calling `build()` before any data is set surfaces
`NotImplementedError("Both X and y need to be passed to _setup_data.")`
(`core.py:339`), referencing a private method.

**Fix.** Raise `ValueError("X and y must be passed to build() (or set by a
previous build() call).")`.

### 5. Implicit external-mutation contract for `feature_types`

**Problem.** `estimators/core.py:718-719` sets
`_poniard_preprocessor.feature_types` directly and calls `build()`; `build()`
then relies on `try/except AttributeError` (`core.py:241-247`) to skip
inference. `inferred_types_df` is only produced as a side effect of
`_infer_dtypes()`, so the two attributes can drift out of sync.

**Fix.** Make it explicit: add a `feature_types` parameter to `build()` (or a
property setter) that also refreshes `inferred_types_df`, and have
`estimators/core.py` use it instead of poking the attribute.

---

## P1 — Type inference parity

### 6. Unify the pandas/numpy paths in `infer_feature_types`

**Problem.** The two branches (`core.py:79-116`) duplicate the logic and
disagree on edge cases:

- pandas `nunique()` ignores NaN; `np.unique` counts it — the same column can
  classify differently depending on the input container.
- Booleans: `pd.api.types.is_numeric_dtype(bool)` is True, while
  `np.issubdtype(bool, np.number)` is False, so they take different paths.

**Fix.** Coerce array input to a DataFrame once (reuse `coerce_input`) and keep
a single inference path. Add parity tests: same data as DataFrame vs ndarray
must yield identical `feature_types`, including columns with NaN and bools.

---

## P2 — Modeling quality (opt-in, no default behavior change)

### 7. Configurable categorical imputer (missingness as a category)

**Problem.** Low-cardinality categoricals use
`SimpleImputer(strategy="most_frequent")` (`core.py:376`), which hides
missingness. `strategy="constant", fill_value="missing"` lets OHE create a
missingness indicator, which is often informative.

**Fix.** Expose a `categorical_imputer` parameter mirroring `numeric_imputer`,
defaulting to current behavior.

### 8. Cyclical encoding for datetime features

**Problem.** Periodic levels (hour, month, weekday, dayofyear) are emitted as
plain integers; the wrap-around (hour 23 → 0) is invisible to models.

**Fix.** Add an opt-in `cyclical` flag to `DatetimeEncoder` that emits sin/cos
pairs for periodic levels. Compose with P0-2 (scaling) so pairs land in a
sensible range.

### 9. `OrdinalEncoder` unknown sentinel can collide with real codes

**Problem.** `unknown_value=99999` (`core.py:367, 373`) is a real encodable
integer; at very high cardinality it can collide with a legitimate code.

**Fix.** Use `unknown_value=np.nan` (supported by sklearn) so unknowns surface
as missing downstream.

### 10. Expose `TargetEncoder` knobs (optional)

**Problem.** `TargetEncoder(cv=3)` is hard-coded (`core.py:370`); `cv`,
`smooth`, and `target_type` may need tuning for small or regression targets.

**Fix.** Accept a dict of overrides or allow passing a configured
`TargetEncoder` instance (already possible via `TransformerMixin` input — in
that case just document it and skip this item).

---

## P3 — Hygiene

### 11. `DatetimeEncoder.get_feature_names_out` returns a `list`

Sklearn convention is `np.ndarray` of str (`datetime.py:133-141`). Return
`np.asarray(feature_names)`.

### 12. `DatetimeEncoder` string-dtype check misses pandas `string` dtype

`X.dtypes.iloc[0] in (object, str)` (`datetime.py:74, 119`) does not match
pandas 2.x `StringDtype`; use `pd.api.types.is_string_dtype` (excluding
categorical) so string-typed date columns are parsed by `pd.to_datetime`
before validation.

### 13. Lazy-import `IterativeImputer`

The experimental enable import runs at module import time
(`core.py:13`) even when the iterative imputer is never used. Move it inside
the `numeric_imputer == "iterative"` branch.

---

## Tests to add (mapped to items)

| Item | Test |
| --- | --- |
| 1 | Feature names identical across type compositions |
| 2 | Datetime-derived features are scaled; imputation strategy respected |
| 6 | ndarray vs DataFrame inference parity (incl. NaN and bool columns) |
| 7 | `categorical_imputer="constant"` produces a missingness OHE column |
| 9 | Unknown category at transform time yields NaN, not 99999 |
| 11 | `get_feature_names_out` returns `np.ndarray` of str |
| 12 | `string`-dtype datetime column is parsed and encoded |

(Existing coverage lives in `tests/test_preprocessing.py`.)
