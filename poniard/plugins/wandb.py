from typing import Optional, Iterable
from pathlib import Path

import wandb
import joblib
import pandas as pd
import numpy as np
from plotly.graph_objs._figure import Figure
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from poniard.plugins.core import BasePlugin


class WandBPlugin(BasePlugin):
    """Weights and Biases plugin. Kwargs from the constructor are passed to `wandb.init()`.

    Parameters
    ----------
    project :
        Name of the Weights and Biases project.
    entity :
        Name of the Weights and Biases entity (username).
    """

    def __init__(
        self, project: Optional[str] = None, entity: Optional[str] = None, **kwargs
    ):
        self.project = project
        self.entity = entity
        wandb.init(project=project, entity=entity, **kwargs)

    def build_config(self):
        return {
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

    def on_setup_end(self):
        config = self.build_config()
        wandb.config.update(config)

        X, y = self.poniard_instance.X, self.poniard_instance.y
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
        return

    def on_plot(self, figure: Figure, name: str):
        wandb.log({name: figure})
        return

    def on_fit_end(self):
        results = self.poniard_instance.show_results().reset_index()
        results.rename(columns={"index": "Estimator"}, inplace=True)
        table = wandb.Table(dataframe=results)
        wandb.log({"results": table})
        return

    def on_get_estimator(self, estimator: BaseEstimator, name: str):
        X, y = self.poniard_instance.X, self.poniard_instance.y
        saved_model_path = Path(wandb.run.dir, f"{name}.joblib")
        # This works fine for sklearn and sklearn-like models only.
        try:
            check_is_fitted(estimator)
        except NotFittedError:
            estimator.fit(X, y)
        joblib.dump(estimator, saved_model_path)
        artifact = wandb.Artifact(name=f"saved_estimators", type="model")
        artifact.add_file(saved_model_path.as_posix(), name=name)
        wandb.log_artifact(artifact)

        if isinstance(self.poniard_instance.cv, (int, Iterable)):
            cv_params_for_split = {}
        else:
            cv_params_for_split = {
                k: v
                for k, v in vars(self.poniard_instance.cv).items()
                if k in ["shuffle", "random_state"]
            }
            stratify = (
                y
                if "Stratified" in self.poniard_instance.cv.__class__.__name__
                else None
            )
            cv_params_for_split.update({"stratify": stratify})
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, **cv_params_for_split
        )
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        estimator_type = self.poniard_instance._check_estimator_type()
        if estimator_type == "classifier":
            labels = y_test.unique()
            wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)
            if hasattr(estimator, "predict_proba"):
                y_probas = estimator.predict_proba(X_test)
                wandb.sklearn.plot_roc(y_test, y_probas, labels)
                wandb.sklearn.plot_precision_recall(y_test, y_probas, labels)
            wandb.sklearn.plot_calibration_curve(estimator, X_train, y_train, name)
            if isinstance(estimator, Pipeline):
                estimator = estimator[-1]
            wandb.sklearn.plot_feature_importances(estimator, pd.DataFrame(X).columns)
        else:
            wandb.sklearn.plot_residuals(estimator, X_train, y_train)
            wandb.sklearn.plot_outlier_candidates(estimator, X_train, y_train)
        return

    def on_remove_estimators(self):
        self.on_fit_end()
        config = self.build_config()
        wandb.config.update(config)
        return
