# Poniard

<p align="center">
<img src="https://raw.githubusercontent.com/rxavier/poniard/main/logo.png" alt="Poniard logo" title="Poniard" width="50%"/>
</p>

> A poniard /ˈpɒnjərd/ or poignard (Fr.) is a long, lightweight
> thrusting knife ([Wikipedia](https://en.wikipedia.org/wiki/Poignard)).

Poniard is a scikit-learn companion for **multi-model diagnostics**. Fit a handful
of models, cross-validate them side by side, then answer the questions that
actually matter:

- *Where do the models fail, and why?*
- *Is model A really better than B, or is that fold noise?*
- *Which model is good enough and cheap enough?*

Then export a plain `sklearn` pipeline you own and leave. Not AutoML. Not
end-to-end. Every feature earns its place.

## Installation

```bash
pip install poniard
```

Plotting is an optional extra:

```bash
pip install poniard[plot]
```

## Quick start

```python
from sklearn.datasets import make_classification
from poniard import PoniardClassifier

X, y = make_classification(n_samples=200, n_features=10, random_state=42)

clf = PoniardClassifier()
clf.fit(X, y)      # type-inference, preprocessing, and CV for every estimator
clf.get_results()  # leaderboard with a dummy baseline
```

## The core loop

Poniard is built around one loop: **compare → explain → decide → export**.

### 1. Compare

```python
clf = PoniardClassifier()                 # or PoniardRegressor()
clf.fit(X, y)
clf.get_results()                         # mean scores, fit/score times
```

Every estimator is cross-validated on the same folds, with a `DummyClassifier`
/ `DummyRegressor` baseline included automatically.

### 2. Explain — error analysis

`ErrorAnalyzer` answers *where and why* your models fail. One call returns a
structured `ErrorReport`:

```python
from poniard.error_analysis import ErrorAnalyzer

report = ErrorAnalyzer.from_poniard(clf).analyze(X, y)

report.universal_failures   # samples every model got wrong
report.disagreement_set     # samples where models split (ensembling candidates)
report.lift_by_target       # classes/bins over-represented in errors (lift > 1)
report.lift_by_feature      # feature values over-represented in errors
```

### 3. Decide

```python
clf.compare()                    # paired fold tests: is A really better than B?
clf.pareto()                     # best metric vs training time (Pareto front)
clf.best_under(seconds=0.5)      # best model within a time budget
```

### 4. Export

```python
model = clf.get_estimator("LogisticRegression", retrain=True, X=X, y=y)
# a fitted sklearn.pipeline.Pipeline with no poniard references — deploy it
```

## Feature overview

| Area | What you get |
|---|---|
| **Error analysis** | Universal failures, disagreement sets, lift vs baseline — ranked per sample, sliced by target and features |
| **Statistical comparison** | Paired fold tests so you stop trusting fold-mean leaderboards |
| **Time / quality** | Pareto front and best-under-budget helpers |
| **Preprocessing** | Automatic numeric / categorical / datetime type inference, imputation, encoding, scaling |
| **Tuning** | Grid, random, and halving search that re-enters the experiment as a new named estimator |
| **Ensembles** | Diversity-aware voting / stacking built from your fitted estimators |
| **Plotting** | Metrics, ROC, confusion matrices, residuals, feature importance (optional, requires `[plot]`) |

## Docs and examples

- **[In-depth guide](docs/guide.md)** — the full workflow, preprocessing internals,
  error-analysis semantics, tuning, ensembles, plotting, and export.
- **[Examples](examples/)** — runnable scripts, executed in CI so they never go stale:
  - `examples/00_getting_started.py` — fit, results, predictions
  - `examples/01_error_analysis.py` — full failure-forensics workflow
  - `examples/02_plotting.py` — the plotting API

```bash
python examples/00_getting_started.py
```

## Estimator naming

Estimators are named by class name, with collision suffixes (`_2`, `_3`, ...).
Override names with tuple syntax:

```python
clf = PoniardClassifier(estimators=[("my_lr", LogisticRegression())])
# pipelines: {'my_lr': ..., 'DummyClassifier': ...}
```

## Environment variables

- `PONIARD_TQDM_LEAVE` — set to `"True"` to keep progress bars on screen after
  fitting completes. Default `"False"`.

## Python support

3.10–3.13, tested on Linux, macOS, and Windows.

## Development

```bash
git clone https://github.com/rxavier/poniard.git
cd poniard
uv sync --dev
uv run pytest
```

## License

MIT
