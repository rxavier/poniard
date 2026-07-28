from importlib.metadata import version

from sklearn import set_config

from poniard.estimators.classification import PoniardClassifier
from poniard.estimators.regression import PoniardRegressor

set_config(transform_output="pandas")

__version__ = version("poniard")
__all__ = ["PoniardClassifier", "PoniardRegressor"]
