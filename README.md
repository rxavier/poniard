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

## Quick start

```python
from sklearn.datasets import make_classification
from poniard import PoniardClassifier

X, y = make_classification(n_samples=200, n_features=10, random_state=42)
clf = PoniardClassifier()
clf.setup(X, y)
clf.fit()
clf.get_results()
```

## Features

- **Automatic type inference**: Detects numeric, categorical, and datetime features
- **Built-in preprocessing**: Imputation, encoding, scaling via a configurable pipeline
- **Cross-validated comparison**: Fits multiple estimators with cross-validation and collects results
- **Hyperparameter tuning**: Optuna-based tuning for any estimator
- **Ensemble building**: Create ensembles from fitted estimators
- **Plotting**: Metrics comparison, ROC curves, confusion matrices, feature importance, and more
- **Plugin system**: Extend with Weights & Biases logging, pandas-profiling reports, etc.

## Examples

See the [examples/](examples/) directory for Jupyter notebooks demonstrating the library:

- [Getting started](examples/00_getting_started.ipynb)
- [End-to-end example](examples/03_end_to_end_example.ipynb)

## Development

```bash
git clone https://github.com/rxavier/poniard.git
cd poniard
uv sync --dev
uv run pytest
```

## License

MIT
