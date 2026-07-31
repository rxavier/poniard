# Poniard

<p align="center">
<img src="https://raw.githubusercontent.com/rxavier/poniard/main/logo.png" alt="Poniard logo" title="Poniard" width="50%"/>
</p>

> A poniard /ˈpɒnjərd/ or poignard (Fr.) is a long, lightweight
> thrusting knife ([Wikipedia](https://en.wikipedia.org/wiki/Poignard)).

Poniard is a scikit-learn companion library that streamlines the process of fitting different machine learning models and comparing them.

It can be used to provide quick answers to questions like these:

- What is the reasonable range of scores for this task?
- Is a simple and explainable linear model enough or should I work with forests and gradient boosters?
- Are the features good enough as is or should I work on feature engineering?
- How much can hyperparameter tuning improve metrics?
- Do I need to work on a custom preprocessing strategy?

This is not meant to be an end-to-end solution, and you should keep working on your models after you are done with Poniard.

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
clf.setup(X, y)    # configure: type inference, preprocessing, pipelines
# optionally: clf.add_estimators(...), clf.reassign_types(...), etc.
clf.fit(X, y)      # cross-validate all estimators
clf.get_results()  # comparison table
```

## Exporting a model (leaving Poniard)

`get_estimator` is the supported way to leave Poniard. It returns a plain
scikit-learn `Pipeline` (or a bare estimator with
`include_preprocessor=False`) with **no poniard references** — you can save it,
deploy it, or keep working on it without Poniard installed:

```python
model = clf.get_estimator("LogisticRegression", retrain=True, X=X, y=y)
# model is a fitted sklearn.pipeline.Pipeline you fully own
```

Without `retrain=True`, the returned pipeline is an unfitted clone you can
inspect. Use it to extract any estimator from the comparison — defaults,
hyperparameter-optimized ones after `tune_estimator`, or ensemble members.

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

## Error analysis

`ErrorAnalyzer` answers *where and why* your models fail. Build it from a
fitted `PoniardClassifier` / `PoniardRegressor` and run the full workflow with
a single call:

```python
from poniard.error_analysis import ErrorAnalyzer

ea = ErrorAnalyzer.from_poniard(clf, estimator_names=["LogisticRegression", "RandomForestClassifier"])
report = ea.analyze(X, y)  # X, y = the data you fitted on
```

`report` contains:

- `ranked_errors` — per estimator, samples sorted by error magnitude
- `merged_errors` — per sample, how many estimators failed and their average error
- `summary` — per estimator: number of errors and error rate
- `by_target` — error counts and error rate per target class/bin
- `by_feature` — per feature, the distribution of errors across its values

The individual steps are also exposed:

```python
ranked = ea.rank_errors(X, y)                       # per-estimator ranked errors
merged = ErrorAnalyzer.merge_errors(ranked)         # cross-estimator view
ea.analyze_target(errors_idx=merged.index, y=y)     # errors vs target distribution
ea.analyze_features(errors_idx=merged.index, X=X)   # errors vs feature values
```

How errors are defined:

- **Classification**: misclassified samples, ranked by `1 - probability of the
  truth` (how confidently wrong the model is). Multilabel targets rank by the
  mean per-label deviation.
- **Regression**: samples whose absolute residual exceeds a threshold, ranked by
  residual magnitude. The threshold defaults to the 90th percentile of residuals
  and can be configured with `error_quantile` in `rank_errors` / `analyze`.

## Estimator naming

Each estimator gets a name automatically (its class name). You can override with tuple syntax:

```python
# Single of each class → class names
clf = PoniardClassifier(estimators=[LogisticRegression(), SVC()])
# pipelines: {'LogisticRegression': ..., 'SVC': ..., 'DummyClassifier': ...}

# Duplicates → collision handling
clf = PoniardClassifier(estimators=[
    LogisticRegression(max_iter=1000),
    LogisticRegression(C=0.1),
])
# pipelines: {'LogisticRegression': ..., 'LogisticRegression_2': ..., 'DummyClassifier': ...}

# Tuple override
clf = PoniardClassifier(estimators=[('my_lr', LogisticRegression())])
# pipelines: {'my_lr': ..., 'DummyClassifier': ...}
```

## Features

- **Automatic type inference**: Detects numeric, categorical, and datetime features
- **Built-in preprocessing**: Imputation, encoding, scaling via a configurable pipeline
- **Cross-validated comparison**: Fits multiple estimators with cross-validation and collects results
- **Hyperparameter tuning**: Grid, random, and halving search for any estimator
- **Ensemble building**: Create ensembles from fitted estimators
- **Error analysis**: Rank prediction errors, and analyze them against the target and features to find *where and why* models fail
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
