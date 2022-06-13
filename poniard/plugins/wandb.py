from typing import Optional
from pathlib import Path

import wandb
import joblib
import pandas as pd
import numpy as np
from plotly.graph_objs._figure import Figure
from sklearn.base import BaseEstimator
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

    def build_config(self) -> dict:
        """Helper method that builds a config dict from the poniard instance."""
        return {
            "estimators": self._poniard.estimators_,
            "metrics": self._poniard.metrics_,
            "cv": self._poniard.cv_,
            "preprocess": self._poniard.preprocess,
            "preprocessor": self._poniard.preprocessor_,
            "custom_preprocessor": self._poniard.custom_preprocessor,
            "numeric_imputer": self._poniard.numeric_imputer,
            "scaler": self._poniard.scaler,
            "numeric_threshold": self._poniard.numeric_threshold,
            "cardinality_threshold": self._poniard.cardinality_threshold,
            "verbosity": self._poniard.verbose,
            "n_jobs": self._poniard.n_jobs,
            "random_state": self._poniard.random_state,
        }

    def on_setup_end(self) -> None:
        """Log config and dataset."""
        config = self.build_config()
        wandb.config.update(config)

        X, y = self._poniard.X, self._poniard.y
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
        """Log plots."""
        wandb.log({name: figure})
        return

    def on_fit_end(self):
        """Log results table."""
        results = self._poniard.show_results().reset_index()
        results.rename(columns={"index": "Estimator"}, inplace=True)
        table = wandb.Table(dataframe=results)
        wandb.log({"results": table})
        return

    def on_get_estimator(self, estimator: BaseEstimator, name: str):
        """Log fitted estimator on full data and log multiple estimator plots provided by WandB."""
        X, y = self._poniard.X, self._poniard.y
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

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self._poniard._train_test_split_from_cv()
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        estimator_type = self._poniard._check_estimator_type()
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
        """Log config and results table."""
        self.on_fit_end()
        config = self.build_config()
        wandb.config.update(config)
        return
