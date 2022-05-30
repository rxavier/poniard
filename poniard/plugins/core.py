from typing import Optional, TYPE_CHECKING
from abc import ABC

if TYPE_CHECKING:
    from poniard.estimators.core import PoniardBaseEstimator


class BasePlugin(ABC):
    """Base plugin class. New plugins should inherit from this class."""

    def __init__(self):
        self._poniard: Optional["PoniardBaseEstimator"] = None

    def on_setup_start(self):
        pass

    def on_setup_end(self):
        pass

    def on_fit_start(self):
        pass

    def on_fit_end(self):
        pass

    def on_plot(self):
        pass

    def on_get_estimator(self):
        pass

    def on_remove_estimators(self):
        pass

    def _check_plugin_used(self, plugin_cls_name: str):
        if any(x.__class__.__name__ == plugin_cls_name for x in self._poniard.plugins):
            return True
        else:
            return False
