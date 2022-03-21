<p style="text-align:center"><img src="https://raw.githubusercontent.com/rxavier/poniard/main/logo.png" alt="Poniard logo" title="Poniard" width="50%"/></p>

# Introduction
> A poniard /ˈpɒnjərd/ or poignard (Fr.) is a long, lightweight thrusting knife ([Wikipedia](https://en.wikipedia.org/wiki/Poignard))

Poniard is a scikit-learn companion library that tries to streamline the process of trying out different machine learning models and comparing them. It is meant to measure how "challenging" a problem/dataset is, which types of models work well for the task, and help decide which algorithm to choose.

This is not meant to be end to end solution, and you definitely should keep on working on your models after you are done with Poniard.

# Design philosophy

## Not another dependency
We try very hard not to clutter the environment with stuff you won't use outside of this library. Poniard's dependencies are:

1. scikit-learn (duh)
2. pandas
3. XGBoost
4. tqdm
5. That's it!

Apart from `tqdm`, all dependencies most likely were going to be installed anyway, so Poniard's added footprint should be minimal.

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
Preprocessing tries to ensure that your models run successfully without significant data munging. By default, Poniard imputes missing data and one-hot encodes (or ordinal encodes) inferred categorical variables, which in most cases is enough for scikit-learn algorithms to fit without complaints.

# Installation

```bash
pip install poniard
```

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
pnd.score_estimators(X_train, y_train)
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

## Type inference
Poniard uses some basic heuristics to infer the data types.

Float columns are assumed to be numeric and string/object/categorical columns are assumed to be categorical.

Integer columns are defined as numeric if the number of unique values is greater than indicated by the `categorical_threshold` parameter. If Poniard detects integer columns, it will suggest casting to either float or string to avoid guessing.

For categorical features, high and low cardinality is defined by the `cardinality_threshold` parameter. Only low cardinality categorical features are one-hot encoded.

## Ensembles
Poniard makes it easy to combine various estimators in stacking or voting estimators. The base esimators can be selected according to their performance or chosen by their names.

Poniard also reports how similar the predictions of the base estimators are, so ensembles with different base esimators can be built. A basic correlation table of the cross-validated predictions is built for regression tasks, while [Cramer's V](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V) is used for classification.

## Hyperparameter optimization
The `tune_estimator` method can be used to optimize the hyperparameters of a given estimator, either by passing a grid of parameters or using the inbuilt ones available for default estimators. Optionally, the tuned estimator can be added to the list of estimators and scored, which will add it to results tables.

```python
from poniard import PoniardRegressor

pnd = PoniardRegressor(random_state=0)
pnd.score_estimators(x, y)
pnd.show_results()
pnd.tune_estimator("RandomForestRegressor", x, y, mode="grid", add_to_estimators=True)
pnd.score_estimators(x, y) # This will only fit new estimators
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