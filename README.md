# Poniard

<p align="center">
<img src="https://raw.githubusercontent.com/rxavier/poniard/main/logo.png" alt="Poniard logo" title="Poniard" width="50%"/>
</p>

> A poniard /ˈpɒnjərd/ or poignard (Fr.) is a long, lightweight
> thrusting knife ([Wikipedia](https://en.wikipedia.org/wiki/Poignard)).

Poniard is a scikit-learn companion for **multi-model diagnostics**. Compare models to get oriented, then answer *where they fail, whether differences are real, and what to try next* — then export a plain sklearn object and leave.

Not AutoML. Not end-to-end. Every feature earns its place.

## Installation

```bash
pip install poniard
```

With plotting support:

```bash
pip install poniard[plot]
```

## Quick start

```python
from sklearn.datasets import make_classification
from poniard import PoniardClassifier

X, y = make_classification(n_samples=200, n_features=10, random_state=42)

clf = PoniardClassifier()
clf.fit(X, y)      # cross-validate all estimators
clf.get_results()  # comparison table
```

## Error analysis — the reason to install

`ErrorAnalyzer` answers *where and why* your models fail. Build it from a
fitted `PoniardClassifier` / `PoniardRegressor` and run the full workflow with
a single call:

```python
from poniard.error_analysis import ErrorAnalyzer

ea = ErrorAnalyzer.from_poniard(clf)  # all non-dummy estimators by default
report = ea.analyze(X, y)
```

The `report` is a structured `ErrorReport` containing:

- **`universal_failures`** — samples every model got wrong
- **`disagreement_set`** — samples where models split (useful for ensembling)
- **`lift_by_target`** — per class/bin, error rate relative to the global rate
- **`lift_by_feature`** — per feature value, error rate relative to the global rate
- **`ranked_errors`** — per estimator, samples sorted by error magnitude
- **`merged_errors`** — cross-estimator view: frequency and mean error per sample
- **`summary`** — per estimator: error count, error rate, mean error
- **`by_target`** / **`by_feature`** — error distributions

```python
report.universal_failures   # what's toxic to every model
report.lift_by_target       # which classes are over-represented in errors
report.disagreement_set     # where models disagree (ensembling candidates)
```

How errors are defined:

- **Classification**: misclassified samples, ranked by `1 - probability of the truth` (how confidently wrong the model is).
- **Regression**: samples whose absolute residual exceeds a threshold (default: 90th percentile), ranked by residual magnitude.

## Statistical comparison

Stop pretending fold-mean leaderboards are truth. `compare()` runs paired
tests on cross-validation folds:

```python
clf.compare()
# pairwise: mean_diff, wins_a, wins_b, ties, p_value
```

## Time vs quality

Pick "good enough and cheap" in one call:

```python
clf.pareto()                                              # best metric vs fit_time
clf.pareto(time_col="score_time_per_sample")              # vs inference time per sample
clf.best_under(seconds=0.5)                               # best metric where fit_time <= 0.5s
clf.best_under(seconds=0.001, time_col="score_time_per_sample")  # fast inference
```

Available time columns: `fit_time`, `score_time`, `fit_time_per_sample`, `score_time_per_sample`.

## Exporting a model (leaving Poniard)

`get_estimator` is the supported way to leave Poniard. It returns a plain
scikit-learn `Pipeline` (or a bare estimator with
`include_preprocessor=False`) with **no poniard references** — you can save it,
deploy it, or keep working on it without Poniard installed:

```python
model = clf.get_estimator("LogisticRegression", retrain=True, X=X, y=y)
# model is a fitted sklearn.pipeline.Pipeline you fully own
```

## Hyperparameter tuning (stays in the experiment)

`tune_estimator` runs a search on the **same** preprocessor/pipeline, then adds
the winner as a new named estimator (default `{name}_tuned`) so it can be
cross-validated and compared with everything else — no prep drift, no overwrite.

No default grids: you always pass `grid`. Bare param names are fine; they are
prefixed with the estimator step automatically:

```python
clf.tune_estimator("LogisticRegression", X, y, grid={"C": [0.1, 1.0, 10.0]})
clf.fit(X, y)  # CV the tuned pipeline into the results table
clf.get_results()
clf.get_tuning_results("LogisticRegression_tuned")  # best_params_, search, ...
```

## Plotting

Plotting is a separate module (requires `pip install poniard[plot]`):

```python
from poniard.plot import PoniardPlotFactory

plotter = PoniardPlotFactory(X, y, clf)
plotter.metrics()
plotter.roc_curve()
plotter.confusion_matrix("LogisticRegression")
plotter.permutation_importance("LogisticRegression")

# Single-figure dashboard: metric rankings, model comparison, ROC/confusion
# matrix (or residuals for regression) and permutation importance in one view
plotter.full_estimator_analysis("LogisticRegression")
```

## Estimator naming

Each estimator gets a name automatically (its class name). You can override with tuple syntax:

```python
# Tuple override
clf = PoniardClassifier(estimators=[('my_lr', LogisticRegression())])
# pipelines: {'my_lr': ..., 'DummyClassifier': ...}

# Duplicates → collision handling
clf = PoniardClassifier(estimators=[
    LogisticRegression(max_iter=1000),
    LogisticRegression(C=0.1),
])
# pipelines: {'LogisticRegression': ..., 'LogisticRegression_2': ..., 'DummyClassifier': ...}
```

## Features

- **Error analysis**: Universal failures, disagreement sets, lift vs baseline — find *where and why* models fail
- **Statistical comparison**: Paired fold tests to see if A really beats B
- **Time-quality tradeoff**: Pareto front and best-under-budget helpers
- **Automatic type inference**: Detects numeric, categorical, and datetime features
- **Built-in preprocessing**: Imputation, encoding, scaling via a configurable pipeline
- **Cross-validated comparison**: Fits multiple estimators with cross-validation and collects results
- **Hyperparameter tuning**: Grid, random, and halving search for any estimator
- **Ensemble building**: Create ensembles from fitted estimators
- **Plotting**: Metrics comparison, ROC curves, confusion matrices, feature importance (optional, requires plotly)

## Environment variables

- `PONIARD_TQDM_LEAVE` — set to `"True"` to keep the progress bars on screen
  after fitting/searching completes (instead of clearing them). Default `"False"`.

## Python support

3.10, 3.11, 3.12, 3.13 — tested on Linux, macOS, and Windows.

## Development

```bash
git clone https://github.com/rxavier/poniard.git
cd poniard
uv sync --dev
uv run pytest
```

## License

MIT
