from importlib.metadata import version

from poniard.estimators.classification import PoniardClassifier
from poniard.estimators.regression import PoniardRegressor

__version__ = version("poniard")
__all__ = ["PoniardClassifier", "PoniardRegressor"]
