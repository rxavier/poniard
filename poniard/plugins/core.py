from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from poniard.core import PoniardBaseEstimator


class BasePlugin(object):
    """Base plugin class. New plugins should inherit from this class."""

    def __init__(self):
        self.poniard: Optional["PoniardBaseEstimator"] = None

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
