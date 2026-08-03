# Poniard in depth

This guide walks through the whole library: the experiment lifecycle, how
preprocessing and cross-validation work, error analysis semantics, statistical
comparison, tuning, ensembles, plotting, and export. It assumes you have read the
[README](../README.md) and are comfortable with scikit-learn.

If you prefer code, the runnable examples in [`examples/`](../examples/) cover the
same ground and are executed in CI.

---

## The experiment lifecycle

A Poniard experiment has three phases, and `setup` is deliberately the first
step: you should always be able to **see** what `fit` will do to your data and
modify it before anything is cross-validated.

```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer

clf = PoniardClassifier()

# 1. Configure: infer types, build preprocessing, CV, pipelines
clf.setup(X, y)

# 2. Inspect and modify (optional)
print(clf.feature_types)            # what got inferred
clf.reassign_types(numeric=["age"], categorical_low=["city"])
clf.add_preprocessing_step(("log_transform", FunctionTransformer(np.log1p)))

# 3. Fit: cross-validate every pipeline
clf.fit(X, y)
```

`fit(X, y)` will also configure if `setup` was not called, but the setup-first
flow exists so the preprocessor is never a black box.

### Mutators

- `reassign_types(numeric=..., categorical_high=..., categorical_low=..., datetime=..., keep_remainder=True)`
  — override type inference. Omitted features keep their inferred type unless
  `keep_remainder=False`.
- `add_preprocessing_step(step, position="end")` — insert a transformer into the
  preprocessor pipeline (`"start"`, `"end"`, or an integer index).
- `add_estimators(...)` / `remove_estimators(names, drop_results=True)` — change
  the estimator set. These are the supported ways to mutate `pipelines`: they
  handle naming collisions, attribute propagation, and result bookkeeping.

Mutators that rebuild pipelines (type reassignment, preprocessing steps) reset
the fitted tracking; `add`/`remove` do not.

---

## Preprocessing

Type inference classifies every feature as one of:

| Inferred type | Condition | Pipeline |
|---|---|---|
| `numeric` | number-like with `nunique > numeric_threshold` | imputer (`median` default, or `mean`/`iterative`) with missingness indicator + scaler (`standard`/`minmax`/`robust`) |
| `categorical_low` | not numeric, `nunique <= cardinality_threshold` | most-frequent imputer + `OneHotEncoder(drop="if_binary")` |
| `categorical_high` | not numeric, `nunique > cardinality_threshold` | most-frequent imputer + `TargetEncoder` (ordinal for multioutput targets) |
| `datetime` | datetime dtype | `DatetimeEncoder` + most-frequent imputer |

Thresholds can be integers (a count) or floats (a fraction of rows). Defaults:
`numeric_threshold=0.1`, `cardinality_threshold=20`. A `VarianceThreshold` step
removes invariant features at the end.

The preprocessor is a plain `sklearn.pipeline.Pipeline` configured to output
pandas DataFrames (`set_output(transform="pandas")`), without touching sklearn's
global config — imports of `poniard` never mutate sklearn globals.

To build your own, use `PoniardPreprocessor` directly or pass a custom one:

```python
from poniard.preprocessing import PoniardPreprocessor

pp = PoniardPreprocessor(
    scaler="robust",
    high_cardinality_encoder="ordinal",
    numeric_imputer="iterative",
    cache_transformations=True,  # joblib.Memory-backed caching across estimators
)
clf = PoniardClassifier(custom_preprocessor=pp)
```

`preprocess=False` skips preprocessing entirely (raw data goes straight to the
estimators).

---

## Cross-validation and results

`fit` cross-validates every pipeline on the **same** folds with
`cross_validate(return_train_score=True)`. The splitter defaults to
`StratifiedKFold(shuffle=True)` for binary/multiclass classification and
`KFold(shuffle=True)` otherwise, with 5 folds. Pass an integer or any sklearn
splitter to override:

```python
clf = PoniardClassifier(cv=KFold(n_splits=10, shuffle=True, random_state=0))
```

A `DummyClassifier` / `DummyRegressor` baseline is added automatically unless you
already included one.

### `get_results`

```python
clf.get_results()                       # mean test scores + fit/score times
clf.get_results(std=True)               # also return the fold standard deviations
clf.get_results(return_train_scores=True)
clf.get_results(wrt_dummy=True)         # scores relative to the dummy baseline
```

Columns depend on how metrics were given:

- `metrics=["accuracy", "f1"]` → `test_accuracy`, `train_accuracy`, `f1`, ...
- `metrics={"acc": accuracy_score, "f1": f1_score}` → `acc`, `f1`, ...

Time columns are `fit_time`, `score_time`, and (when fold sizes are known)
`fit_time_per_sample`, `score_time_per_sample`.

---

## Error analysis

`ErrorAnalyzer` is the product core. "Error" is defined per target type:

- **binary / multiclass** — misclassified samples, ranked by
  `1 - probability_of_truth` (how confidently wrong the model is).
- **multilabel** — samples wrong on at least one label, ranked by mean per-label
  deviation.
- **regression / multioutput** — samples whose absolute residual exceeds a
  threshold (default the 90th percentile), ranked by residual magnitude.

```python
from poniard.error_analysis import ErrorAnalyzer

ea = ErrorAnalyzer.from_poniard(clf)   # all non-dummy estimators by default
report = ea.analyze(X, y)
```

`report` is an `ErrorReport` (a dataclass; indexable like a dict):

| Attribute | Meaning |
|---|---|
| `ranked_errors` | per estimator, samples ranked by error |
| `merged_errors` | per sample: how many estimators failed, mean error, which ones |
| `summary` | per estimator: error count, error rate, mean error |
| `universal_failures` | samples every selected estimator got wrong |
| `disagreement_set` | samples where models split (correct + wrong) |
| `by_target` / `by_feature` | error distributions over target classes/bins and features |
| `lift_by_target` / `lift_by_feature` | error rate per class/bin/feature value ÷ global error rate; values > 1 are over-represented in errors |

You can also run the steps individually:

```python
ranked = ea.rank_errors(X, y)
merged = ErrorAnalyzer.merge_errors(ranked)
ea.analyze_target(errors_idx=merged.index, y=y, reg_bins=5)
ea.analyze_features(
    errors_idx=merged.index, X=X, y=y,
    estimator_name="LogisticRegression", n_features=3,   # top features by permutation importance
)
```

`n_features` can be an integer or a float fraction (e.g. `0.5`).

### Caching

Cross-validated predictions are cached and reused, so calling `analyze()` again
on the same data costs nothing:

```python
ea.analyze(X, y)   # computes CV predictions once
ea.analyze(X, y)   # reuses them
```

---

## Statistical comparison

Fold-mean leaderboards hide noise. `compare()` runs paired tests on the
per-fold scores already collected during `fit`:

```python
clf.compare()                              # all non-dummy estimators, primary metric
clf.compare(estimators=["RandomForestClassifier", "LogisticRegression"])
clf.compare(metrics=["test_f1", "test_roc_auc"])
```

The result is a table indexed by `(metric, estimator_a, estimator_b)` with
`mean_diff`, `wins_a`, `wins_b`, `ties`, and a paired t-test `p_value`.

> **Honesty note:** this is exploratory comparison. CV folds are not independent
> and no multiple-testing correction is applied — treat p-values as directional
> evidence, not paper-grade inference.

---

## Time vs quality

Practical model choice, not just peak metric:

```python
clf.pareto()                                   # Pareto-optimal set: no model is both faster and better
clf.pareto(time_col="score_time_per_sample")   # vs inference time
clf.best_under(seconds=0.5)                    # best model with mean fit_time <= 0.5s
clf.best_under(seconds=0.001, time_col="score_time_per_sample")
```

Available time columns: `fit_time`, `score_time`, `fit_time_per_sample`,
`score_time_per_sample`. Dummy estimators are excluded.

---

## Hyperparameter tuning

`tune_estimator` runs a search on the **same** pipeline (same preprocessor, same
step names) and adds the winner as a **new** named pipeline (default
`{name}_tuned`), so it is cross-validated and compared like any other estimator —
no prep drift, no silent overwrite.

```python
clf.tune_estimator("LogisticRegression", X, y, grid={"C": [0.01, 0.1, 1.0, 10.0]})
clf.fit(X, y)
clf.get_results()
clf.get_tuning_results("LogisticRegression_tuned")  # best_params_, best_score_, grid, fitted search
```

- Poniard ships **no default grids** — `grid` is required.
- Bare keys (`{"C": [...]}`) are prefixed with the estimator step automatically;
  keys containing `__` (e.g. `preprocessor__...`) are used as-is.
- `mode="grid"` (default), `"random"`, or `"halving"`; extra kwargs are passed to
  the sklearn search class.

---

## Ensembles

`build_ensemble` combines estimators into a voting/stacking ensemble and adds it
to `pipelines` as a normal named estimator.

```python
clf.build_ensemble(method="stacking", X=X, y=y)   # default: diversity-aware selection
```

The default `strategy="diversity"` greedily picks estimators that are both
strong and dissimilar on prediction errors, using the same similarity machinery
as `get_predictions_similarity`. Pass `strategy="top_n"` for the legacy
best-N-by-metric behavior, or `estimator_names=[...]` to bypass selection.

```python
clf.fit(X, y)      # the ensemble joins the results table
```

---

## Plotting

Plotting is a separate module (`pip install poniard[plot]`). The
`PoniardPlotFactory` is a standalone object — it receives the data and estimator,
and applies per-figure config without mutating any plotly global or the estimator
itself:

```python
from poniard.plot import PoniardPlotFactory

plotter = PoniardPlotFactory(
    X, y, clf,
    template="plotly_white",
    discrete_colors=["#636EFA", "#EF553B"],
)

plotter.metrics()
plotter.metrics(kind="bar", metrics=["test_f1"])   # single metric -> no facet
plotter.overfitness()
plotter.roc_curve()                                # classification, binary only
plotter.confusion_matrix("LogisticRegression")
plotter.residuals(["LinearRegression"])            # regression
plotter.residuals_histogram(["LinearRegression"])
plotter.permutation_importance("LogisticRegression", n_repeats=10)
plotter.partial_dependence("LogisticRegression", feature=0)

# Diagnostic plots
plotter.similarity_heatmap(X, y)                   # pairwise prediction similarity
plotter.error_lift_bars(lift_by_target=report.lift_by_target)
plotter.time_quality_scatter()

# Single-figure dashboard for one estimator
plotter.full_estimator_analysis("LogisticRegression")
```

Plot methods that need a fitted model fit a **clone** — your experiment state is
never touched by drawing a figure.

---

## Export and persistence

`get_estimator` is the supported way to leave Poniard. It returns a plain
scikit-learn object with no poniard references:

```python
pipeline = clf.get_estimator("LogisticRegression")                # unfitted clone
pipeline = clf.get_estimator("LogisticRegression", retrain=True, X=X, y=y)  # fitted on full data
raw = clf.get_estimator("LogisticRegression", include_preprocessor=False)
```

Save and restore whole fitted experiments (results included) with joblib:

```python
clf.save("clf.joblib")
loaded = PoniardClassifier.load("clf.joblib")
loaded.get_results()   # identical to the saved experiment
```

---

## Under the hood

A few design decisions worth knowing:

- **State is explicit.** Every attribute exists from construction; `setup`/`fit`
  mutate it rather than creating attributes implicitly. Derived results
  (`_means`, `_long_results`, ...) are `None` until `fit` produces them.
- **Predictions are cached, not the data.** Cross-validated predictions are
  reused across calls to error analysis and plots, keyed by a **fingerprint** of
  the input values — the cache holds no reference to your data, so it can't pin
  datasets in memory, and mutating the data in place invalidates it. Public
  `predict()` / `predict_proba()` always compute fresh.
- **Side-effect discipline.** Poniard clones user-supplied estimators, CV
  splitters, and custom preprocessors before configuring them. Your objects are
  never mutated; sklearn globals are never touched.
- **`setup` first.** The preprocessor is inspectable and modifiable between
  `setup` and `fit` — it is never a black box.
