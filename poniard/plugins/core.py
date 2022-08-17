from typing import Optional, TYPE_CHECKING
from abc import ABC

from plotly.graph_objs._figure import Figure
from sklearn.base import BaseEstimator

if TYPE_CHECKING:
    from poniard.estimators.core import PoniardBaseEstimator


class BasePlugin(ABC):
    """Base plugin class. New plugins should inherit from this class."""

    def __init__(self):
        self._poniard: Optional["PoniardBaseEstimator"] = None

    def on_setup_start(self):
        """Called during setup start."""
        pass

    def on_setup_data(self):
        """Called after X and y have been set."""
        pass

    def on_infer_types(self):
        """Called after type inference."""
        pass

    def on_setup_preprocessor(self):
        """Called after preprocessor construction."""
        pass

    def on_setup_end(self):
        """Called after setup is complete."""
        pass

    def on_fit_start(self):
        """Called during fit start."""
        pass

    def on_fit_end(self):
        """Called after fitting is complete."""
        pass

    def on_plot(self, figure: Figure, name: str):
        """Called when a plot is created."""
        pass

    def on_get_estimator(self, estimator: BaseEstimator, name: str):
        """Called when an estimator is selected."""
        pass

    def on_add_estimators(self):
        """Called after adding an estimator."""
        pass

    def on_remove_estimators(self):
        """Called after removing an estimator."""
        pass

    def on_add_preprocessing_step(self):
        """Called after adding a preprocessing step."""
        pass

    def on_reassign_types(self):
        """Called after reassigning types."""
        pass

    def _check_plugin_used(self, plugin_cls_name: str):
        """Check if another plugin is present. If it is, return its instance. Else, return False."""
        plugin_names = [x.__class__.__name__ for x in self._poniard.plugins]
        check = any(x == plugin_cls_name for x in plugin_names)
        if check:
            return self._poniard.plugins[plugin_names.index(plugin_cls_name)]
        else:
            return False
