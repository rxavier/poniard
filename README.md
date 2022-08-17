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
The API was designed with regression and classification tasks in mind, but it should also work with time series tasks provided an appropiate cross validation strategy is used (don't shuffle!)

The usual Poniard flow is:
1. Define some estimators.
2. Define some metrics.
3. Define a cross validation strategy.
4. Fit everything.
5. Print the results.

Poniard provides sane defaults for 1, 2 and 3, so in most cases you can just do...

```python
from poniard import PoniardClassifier

pnd = PoniardClassifier(random_state=0)
pnd.setup(X_train, y_train)
pnd.fit()
pnd.show_results()
```

... and get a nice table showing the average of each metric in all folds for every model, including fit and score times (thanks, scikit-learn `cross_validate` function!)

|                                |   test_roc_auc |   train_roc_auc |   test_accuracy |   train_accuracy |   test_precision |   train_precision |   test_recall |   train_recall |   test_f1 |   train_f1 |   fit_time |   score_time |
|:-------------------------------|---------------:|----------------:|----------------:|-----------------:|-----------------:|------------------:|--------------:|---------------:|----------:|-----------:|-----------:|-------------:|
| LogisticRegression             |          1     |         1       |            0.95 |           0.9925 |         0.935664 |          0.985366 |          0.98 |           1    |  0.953863 |   0.992593 | 0.125515   |    0.0132173 |
| RandomForestClassifier         |          1     |         1       |            0.99 |           1      |         0.981818 |          1        |          1    |           1    |  0.990476 |   1        | 0.30499    |    0.254084  |
| GaussianNB                     |          0.998 |         1       |            0.99 |           0.99   |         0.981818 |          0.980488 |          1    |           1    |  0.990476 |   0.990123 | 0.00988617 |    0.0146274 |
| HistGradientBoostingClassifier |          0.99  |         1       |            0.98 |           1      |         0.963636 |          1        |          1    |           1    |  0.980952 |   1        | 0.100336   |    0.0139906 |
| XGBClassifier                  |          0.988 |         1       |            0.98 |           1      |         0.981818 |          1        |          0.98 |           1    |  0.97995  |   1        | 0.236586   |    0.0198627 |
| DecisionTreeClassifier         |          0.98  |         1       |            0.98 |           1      |         0.981818 |          1        |          0.98 |           1    |  0.97995  |   1        | 0.00964909 |    0.0208005 |
| LinearSVC                      |          0.966 |         1       |            0.92 |           1      |         0.900513 |          1        |          0.96 |           1    |  0.925205 |   1        | 0.00792336 |    0.0122485 |
| KNeighborsClassifier           |          0.93  |         0.97975 |            0.87 |           0.9275 |         0.837483 |          0.894992 |          0.92 |           0.97 |  0.874865 |   0.930467 | 0.00526905 |    0.262831  |
| DummyClassifier                |          0.5   |         0.5     |            0.5  |           0.5    |         0        |          0        |          0    |           0    |  0        |   0        | 0.00288043 |    0.0212384 |

Alternatively, you can also get a nice plot of your different metrics by using the `estimator.plot.metrics()` method.

You may be wondering why does `setup()` exist and need to be called before `fit()` (which in turn doesn't take any arguments, unlike what you're used to in sklearn). Basically, because Poniard performs type inference to build the preprocessor, separating `fit()` allows the user to check whether the preprocessor is correct and to change it if necessary.
## Type inference
Poniard uses some basic heuristics to infer the data types.

Float and integer columns are defined as numeric if the number of unique values is greater than indicated by the `categorical_threshold` parameter.

String/object/categorical columns are assumed to be categorical.

Datetime features are processed separately with a custom encoder.

For categorical features, high and low cardinality is defined by the `cardinality_threshold` parameter. Only low cardinality categorical features are one-hot encoded.

## Ensembles
Poniard makes it easy to combine various estimators in stacking or voting ensembles. The base esimators can be selected according to their performance (top-n) or chosen by their names.

Poniard also reports how similar the predictions of the estimators are, so ensembles with different base estimators can be built. A basic correlation table of the cross-validated predictions is built for regression tasks, while [Cramér's V](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V) is used for classification.

By default, it computes this similarity of prediction errors instead of the actual predictions; this helps in building ensembles with good scoring estimators and uncorrelated errors, which in principle and hopefully should lead to a "wisdom of crowds" kind of situation.

## Hyperparameter optimization
The `tune_estimator` method can be used to optimize the hyperparameters of a given estimator, either by passing a grid of parameters or using the inbuilt ones available for default estimators. The tuned estimator will be added to the list of estimators and will be scored the next time `fit()` is called.

```python
from poniard import PoniardRegressor

pnd = PoniardRegressor(random_state=0)
pnd.setup(x, y)
pnd.fit()
pnd.show_results()
pnd.tune_estimator("RandomForestRegressor", mode="grid")
pnd.fit() # This will only fit new estimators
pnd.show_results()
```
|                               |   test_neg_mean_squared_error |   train_neg_mean_squared_error |   test_neg_mean_absolute_percentage_error |   train_neg_mean_absolute_percentage_error |   test_neg_median_absolute_error |   train_neg_median_absolute_error |    test_r2 |   train_r2 |   fit_time |   score_time |
|:------------------------------|------------------------------:|-------------------------------:|------------------------------------------:|-------------------------------------------:|---------------------------------:|----------------------------------:|-----------:|-----------:|-----------:|-------------:|
| LinearRegression              |                      -8051.97 |                   -1.21019e-25 |                                  -7.31808 |                               -4.15903e-14 |                         -56.3348 |                      -2.19114e-13 |  0.661716  |   1        | 0.0114767  |   0.00424609 |
| ElasticNet                    |                     -13169.4  |                -2128.96        |                                  -4.59068 |                               -0.938363    |                         -70.3958 |                     -26.1924      |  0.4942    |   0.919408 | 0.00304666 |   0.00537877 |
| HistGradientBoostingRegressor |                     -13203.5  |                 -748.071       |                                  -7.05925 |                               -0.957685    |                         -87.9605 |                     -14.9543      |  0.404863  |   0.971676 | 0.450184   |   0.0156513  |
| LinearSVR                     |                     -14668.2  |                -5691.34        |                                 -10.7275  |                               -0.238392    |                         -83.5699 |                      -9.55763     |  0.453208  |   0.785703 | 0.00706563 |   0.00295434 |
| RandomForestRegressor_tuned   |                     -17411.9  |                -2934.28        |                                  -1.97944 |                               -2.31562     |                         -86.9236 |                     -35.0583      |  0.332384  |   0.889133 | 0.338669   |   0.124866   |
| RandomForestRegressor         |                     -18330    |                -2631.47        |                                  -3.34435 |                               -1.20557     |                        -103.777  |                     -34.6603      |  0.148666  |   0.899722 | 0.459331   |   0.123703   |
| XGBRegressor                  |                     -18563.8  |                   -1.17836e-07 |                                  -6.24788 |                               -2.9186e-05  |                         -86.2496 |                      -0.000179579 |  0.283574  |   1        | 0.490598   |   0.0165236  |
| KNeighborsRegressor           |                     -22388.9  |               -16538.6         |                                  -5.35881 |                               -5.40109     |                        -109.728  |                     -86.218       |  0.105827  |   0.374221 | 0.0043016  |   0.127281   |
| DummyRegressor                |                     -27480.4  |               -26460           |                                  -1.57572 |                               -1.89635     |                        -119.372  |                    -110.842       | -0.0950711 |   0        | 0.00351734 |   0          |
| DecisionTreeRegressor         |                     -40700.6  |                    0           |                                 -24.6131  |                                0           |                        -151.028  |                       0           | -0.562759  |   1        | 0.0193477  |   0.00560312 |

## Plotting
The `plot` accessor provides several plotting methods based on the attached Poniard estimator instance. These Plotly plots are based on a default template, but can be modified by passing a different `PoniardPlotFactory` to the Poniard `plot_options` argument.

## Plugin system
The `plugins` argument in Poniard estimators takes a plugin or list of plugins that subclass `BasePlugin`. These plugins have access to the Poniard estimator instance and hook onto different sections of the process, for example, on setup start, on fit end, on remove estimator, etc.

This makes it easy for third parties to extend Poniard's functionality.

Two plugins are baked into Poniard.
1. Weights and Biases: logs your data, plots, runs wandb scikit-learn analysis, saves model artifacts, etc.
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

**[PyCaret](https://github.com/pycaret/pycaret)** is a whole other beast that includes model explainability, deployment, plotting, NLP, anomaly detection, etc., which leads to a list of dependencies several times larger than Poniard's, and a more complicated API.
