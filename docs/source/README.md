<p align="center"><img src="https://raw.githubusercontent.com/rxavier/poniard/main/logo.png" alt="Poniard logo" title="Poniard" width="50%"/></p>

# Introduction
> A poniard /ˈpɒnjərd/ or poignard (Fr.) is a long, lightweight thrusting knife ([Wikipedia](https://en.wikipedia.org/wiki/Poignard)).

Poniard is a scikit-learn companion library that streamlines the process of fitting different machine learning models and comparing them. It is meant to measure how "challenging" a problem/dataset is, which types of models work well for the task, and help decide which algorithm to choose.

This is not meant to be end to end solution, and you definitely should keep on working on your models after you are done with Poniard.

The core functionality has been tested to work on Python 3.7 through 3.10 on Linux systems, and from
3.8 to 3.10 on macOS.

# Installation

Stable version:

```bash
pip install poniard
```

Dev version with most up to date changes:

```bash
pip install git+https://github.com/rxavier/poniard.git@develop#egg=poniard
```

# Documentation

Check the full docs at [Read The Docs](https://poniard.readthedocs.io/en/latest/index.html).



# Usage/features

## Basics
The API was designed with tabular regression and classification tasks in mind, and it should also work with time series tasks provided an appropiate cross validation strategy is used (don't shuffle!)

The usual Poniard flow is:
1. Define some estimators.
2. Define some metrics.
3. Define a cross validation strategy.
4. Fit everything.
5. Show the results.

Poniard provides sane defaults for 1, 2 and 3, so in most cases you can just do...

```python
from poniard import PoniardClassifier

pnd = PoniardClassifier()
pnd.setup(X_train, y_train)
pnd.fit()
pnd.get_results()
```

... and get a nice table showing the average of each metric in all folds for every model, including fit and score times (thanks, scikit-learn `cross_validate` function!)

|                                |   test_roc_auc |   test_accuracy |   test_precision |   test_recall |   test_f1 |   fit_time |   score_time |
|:-------------------------------|---------------:|----------------:|-----------------:|--------------:|----------:|-----------:|-------------:|
| LogisticRegression             |       0.995456 |        0.978916 |         0.975411 |      0.991549 |  0.983351 | 0.0455776  |   0.00492911 |
| SVC                            |       0.994139 |        0.975408 |         0.975111 |      0.985955 |  0.980477 | 0.0135186  |   0.00860071 |
| HistGradientBoostingClassifier |       0.994128 |        0.970129 |         0.967263 |      0.985955 |  0.976433 | 1.02461    |   0.0249005  |
| XGBClassifier                  |       0.994123 |        0.970129 |         0.967554 |      0.985915 |  0.976469 | 0.0533132  |   0.00457506 |
| RandomForestClassifier         |       0.992264 |        0.964881 |         0.964647 |      0.980282 |  0.972192 | 0.0795071  |   0.00955095 |
| GaussianNB                     |       0.98873  |        0.9297   |         0.940993 |      0.949413 |  0.9443   | 0.00533643 |   0.00397215 |
| KNeighborsClassifier           |       0.98061  |        0.964881 |         0.955018 |      0.991628 |  0.972746 | 0.00236444 |   0.0104116  |
| DecisionTreeClassifier         |       0.920983 |        0.926223 |         0.941672 |      0.94108  |  0.941054 | 0.00618615 |   0.00303702 |
| DummyClassifier                |       0.5      |        0.627418 |         0.627418 |      1        |  0.771052 | 0.00240407 |   0.00279222 |

Alternatively, you can also get a nice plot of your different metrics by using the `estimator.plot.metrics()` method.

You may be wondering why does `setup()` exist and need to be called before `fit()` (which in turn doesn't take any arguments, unlike what you're used to in sklearn). Basically, because Poniard performs type inference to build the preprocessor, decoupling `fit()` allows the user to check whether the preprocessor is appropiate and change it if necessary before training models, which could take long.
## Type inference
Poniard uses some basic heuristics to infer the data types.

Float and integer columns are defined as numeric if the number of unique values is greater than indicated by the `categorical_threshold` parameter.

String/object/categorical columns are assumed to be categorical.

Datetime features are processed separately with a custom encoder.

For categorical features, high and low cardinality is defined by the `cardinality_threshold` parameter. Only low cardinality categorical features are one-hot encoded.

## Ensembles
Poniard makes it easy to combine various estimators in stacking or voting ensembles. The base esimators can be selected according to their performance (top-n) or chosen by their names.

Poniard also reports how similar the predictions of the estimators are, so ensembles with different base estimators can be built. A basic correlation table of the cross-validated predictions is built for regression tasks, while [Cramér's V](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V) is used for classification.

By default, it computes this similarity of prediction errors instead of the actual predictions; this helps in building ensembles with good scoring estimators and uncorrelated errors.

## Hyperparameter optimization
The `tune_estimator` method can be used to optimize the hyperparameters of a given estimator, either by passing a grid of parameters or using the inbuilt ones available for default estimators. The tuned estimator will be added to the list of estimators and will be scored the next time `fit()` is called.

```python
from poniard import PoniardRegressor

pnd = PoniardRegressor()
pnd.setup(x, y)
pnd.fit()
pnd.get_results()
pnd.tune_estimator("RandomForestRegressor", mode="grid")
pnd.fit() # This will only fit new estimators
```
## Plotting
The `plot` accessor provides several plotting methods based on the attached Poniard estimator instance. These Plotly plots are based on a default template, but can be modified by passing a different `PoniardPlotFactory` to the Poniard `plot_options` argument.

## Plugin system
The `plugins` argument in Poniard estimators takes a plugin or list of plugins that subclass `BasePlugin`. These plugins have access to the Poniard estimator instance and hook onto different sections of the process, for example, on setup start, on fit end, on remove estimator, etc.

This makes it easy for third parties to extend Poniard's functionality.

Two plugins are baked into Poniard for now:
1. Weights and Biases: logs your data to an artifact, plots, runs wandb scikit-learn analysis, saves model artifacts, etc.
2. Pandas Profiling: generates an HTML report of the features and target. If the Weights and Biases plugin is present, also logs this report to the wandb run.

The requirements for these plugins are not included in the base Poniard dependencies, so you can safely ignore them if you don't intend to use them.

# Design philosophy

## Not another dependency
We try very hard to avoid cluttering the environment with stuff you won't use outside of this library. Poniard's dependencies are:

1. scikit-learn (duh)
2. pandas
3. XGBoost
4. Plotly
5. tqdm
6. That's it!

Apart from `tqdm` and possibly `Plotly`, all dependencies most likely were going to be installed anyway, so Poniard's added footprint should be small.

## We don't do that here (AutoML)
Poniard tries not to take control away from the user. As such, it is not designed to perform 2 hours of feature engineering and selection, try every model under the sun together with endless ensembles and select the top performing model according to some metric.

Instead, it strives to abstract away some of the boilerplate code needed to fit and compare a number of models and allows the user to decide what to do with the results.

Poniard can be your first stab at a prediction problem, but it definitely shouldn't be your last one.

## Opinionated with a few exceptions
While some parameters can be modified to control how variable type inference and preprocessing are performed, the API is designed to prevent parameter proliferation.

## Cross validate all the things
Everything in Poniard is run with cross validation by default, and in fact no relevant functionality can be used without cross validation.

## Use baselines
A dummy estimator is always included in model comparisons so you can gauge whether your model is better than a dumb strategy.

## Fast TTFM (time to first model)
Preprocessing tries to ensure that your models run successfully without significant data munging. By default, Poniard imputes missing data and one-hot encodes or target encodes (depending on cardinality) inferred categorical variables, which in most cases is enough for scikit-learn algorithms to fit without complaints. Additionally, it scales numeric data and drops features with a single unique value.

# Similar projects
Poniard is not a groundbreaking idea, and a number of libraries follow a similar approach.

**[ATOM](https://github.com/tvdboom/ATOM)** is perhaps the most similar library to Poniard, albeit with a different approach to the API.

**[LazyPredict](https://github.com/shankarpandala/lazypredict)** is similar in that it runs multiple estimators and provides results for various metrics. Unlike Poniard, by default it tries most scikit-learn estimators, and is not based on cross validation.

**[PyCaret](https://github.com/pycaret/pycaret)** is a whole other beast that includes model explainability, deployment, plotting, NLP, anomaly detection, etc., which leads to a list of dependencies several times larger than Poniard's, and a much higher level of abstraction.
