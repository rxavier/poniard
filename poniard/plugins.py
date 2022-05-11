from __future__ import annotations
from typing import Optional

import wandb
import pandas as pd
import numpy as np


class BasePlugin(object):
    def on_setup_end(self):
        pass


class WandBPlugin(BasePlugin):
    def __init__(
        self, project: Optional[str] = None, entity: Optional[str] = None, **kwargs
    ):
        self.project = project
        self.entity = entity
        wandb.init(project=project, entity=entity, **kwargs)

    def on_setup_end(self):
        X = self.poniard_instance.X
        y = self.poniard_instance.y
        try:
            dataset = pd.concat([X, y], axis=1)
        except TypeError:
            if y.ndim == 1:
                dataset = np.concatenate([X, np.expand_dims(y, 1)], axis=1)
            else:
                dataset = np.concatenate([X, y], axis=1)
            dataset = pd.DataFrame(dataset)
        table = wandb.Table(dataframe=dataset)
        wandb.log({"dataset": table})
        parameters = {
            "estimators": self.poniard_instance.estimators_,
            "metrics": self.poniard_instance.metrics_,
            "cv": self.poniard_instance.cv,
            "preprocess": self.poniard_instance.preprocess,
            "preprocessor": self.poniard_instance.preprocessor_,
            "custom_preprocessor": self.poniard_instance.custom_preprocessor,
            "numeric_imputer": self.poniard_instance.numeric_imputer,
            "scaler": self.poniard_instance.scaler,
            "numeric_threshold": self.poniard_instance.numeric_threshold,
            "cardinality_threshold": self.poniard_instance.cardinality_threshold,
            "verbosity": self.poniard_instance.verbose,
            "n_jobs": self.poniard_instance.n_jobs,
            "random_state": self.poniard_instance.random_state,
        }
        wandb.config.update(parameters)
        return
