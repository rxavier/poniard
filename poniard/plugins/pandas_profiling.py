from typing import Union
from pathlib import Path

import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

from poniard.plugins.core import BasePlugin


class PandasProfilingPlugin(BasePlugin):
    """Pandas Profiling plugin. Kwargs from the constructor are passed to `ProfileReport()`.

    Parameters
    ----------
    title :
        Report title.
    explorative :
        Enable explorative mode. Default False.
    minimal :
        Enable minimal mode. Default True.
    html_path :
        Path where the HTML report will be saved. Default is the title of the report.
    """

    def __init__(
        self,
        title: str = "pandas_profiling_report",
        explorative: bool = False,
        minimal: bool = True,
        html_path: Union[str, Path] = None,
        **kwargs
    ):
        self.title = title
        self.explorative = explorative
        self.minimal = minimal
        self.html_path = html_path or Path(self.title + ".html")
        self.kwargs = kwargs or {}

    def on_setup_end(self) -> None:
        """Create Pandas Profiling HTML report."""
        X, y = self._poniard.X, self._poniard.y
        try:
            dataset = pd.concat([X, y], axis=1)
        except TypeError:
            if y.ndim == 1:
                dataset = np.concatenate([X, np.expand_dims(y, 1)], axis=1)
            else:
                dataset = np.concatenate([X, y], axis=1)
            dataset = pd.DataFrame(dataset)
        self.report = ProfileReport(
            df=dataset,
            minimal=self.minimal,
            explorative=self.explorative,
            **self.kwargs
        )
        self.report.to_file(self.html_path)
        self._log_to_wandb_if_available()
        self.report.to_notebook_iframe()
        return

    def _log_to_wandb_if_available(self):
        if self._check_plugin_used("WandBPlugin"):
            with open(self.html_path) as report:
                import wandb

                wandb.log({"profile_report": wandb.Html(report)})
        return
