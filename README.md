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

## Plotting

Plotting is a separate module (requires `pip install poniard[plot]`):

```python
from poniard.plot import PoniardPlotFactory

plotter = PoniardPlotFactory(X, y, clf)
plotter.metrics()
plotter.roc_curve()
plotter.confusion_matrix("LogisticRegression")
plotter.permutation_importance("LogisticRegression")
```

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
- **Plotting**: Metrics comparison, ROC curves, confusion matrices, feature importance (optional, requires plotly)

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
