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
        self.run = wandb.init(project=project, entity=entity, **kwargs)

    def build_config(self) -> dict:
        """Helper method that builds a config dict from the poniard instance."""
        return {
            "pipelines": list(self._poniard.estimators_.values()),
            "metrics": self._poniard.metrics,
            "cv": self._poniard.cv,
            "preprocess": self._poniard.preprocess,
            "preprocessor": self._poniard.preprocessor,
            "custom_preprocessor": self._poniard.custom_preprocessor,
            "numeric_imputer": self._poniard.numeric_imputer,
            "scaler": self._poniard.scaler,
            "numeric_threshold": self._poniard.numeric_threshold,
            "cardinality_threshold": self._poniard.cardinality_threshold,
            "high_cardinality_encoder": self._poniard.high_cardinality_encoder,
            "verbosity": self._poniard.verbose,
            "n_jobs": self._poniard.n_jobs,
            "random_state": self._poniard.random_state,
        }

    def on_setup_end(self) -> None:
        """Log config."""
        config = self.build_config()
        wandb.config.update(config)
        return

    def on_setup_data(self) -> None:
        """Log dataset"""
        X, y = self._poniard.X, self._poniard.y
        try:
            dataset = pd.concat([X, y], axis=1)
        except TypeError:
            dataset = np.column_stack([X, y])
            dataset = pd.DataFrame(dataset)
        table = wandb.Table(dataframe=dataset)
        artifact = wandb.Artifact(name="dataset", type="dataset")
        artifact.add(table, "Dataset")
        wandb.log_artifact(artifact)
        return

    def on_infer_types(self):
        """Log inferred types."""
        table = wandb.Table(dataframe=self._poniard.inferred_types_)
        wandb.log({"Inferred types": table})
        return

    def on_setup_preprocessor(self) -> None:
        """Log preprocessor's HTML repr."""
        wandb.log(
            {"Preprocessor": wandb.Html(self._poniard.preprocessor._repr_html_())}
        )
        return

    def on_plot(self, figure: Figure, name: str):
        """Log plots."""
        wandb.log({name: figure})
        return

    def on_fit_end(self):
        """Log results table."""
        results = self._poniard.get_results().reset_index()
        results.rename(columns={"index": "Estimator"}, inplace=True)
        table = wandb.Table(dataframe=results)
        wandb.log({"Results": table})
        for test_metric in results.columns[results.columns.str.startswith("test_")]:
            aux_table = wandb.Table(dataframe=results[["Estimator", test_metric]])
            wandb.log(
                {
                    f"{test_metric} plot": wandb.plot.bar(
                        aux_table, label="Estimator", value=test_metric
                    )
                }
            )
        return

    def on_get_estimator(self, estimator: BaseEstimator, name: str):
        """Save fitted estimator as artifact."""
        X, y = self._poniard.X, self._poniard.y
        saved_model_path = Path(wandb.run.dir, f"{name}.joblib")
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self._poniard._train_test_split_from_cv()
        estimator.fit(X_train, y_train)
        joblib.dump(estimator, saved_model_path)
        artifact = wandb.Artifact(name=f"saved_estimators", type="model")
        artifact.add_file(saved_model_path.as_posix(), name=name)
        wandb.log_artifact(artifact)

        wandb.log({f"{name} pipeline": wandb.Html(estimator._repr_html_())})

        # Wandb complains about missing and non-numeric features despite the estimator
        # having everything to deal with them, so we comment out each function that takes X.
        # y_pred = estimator.predict(X_test)
        # estimator_type = self._poniard.poniard_task
        # if estimator_type == "classification":
        #     labels = np.unique(y_test)
        #     wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)
        #     if hasattr(estimator, "predict_proba"):
        #         y_probas = estimator.predict_proba(X_test)
        #         wandb.sklearn.plot_roc(y_test, y_probas, labels)
        #         wandb.sklearn.plot_precision_recall(y_test, y_probas, labels)

        # wandb.sklearn.plot_learning_curve(estimator, X_train, y_train)
        # wandb.sklearn.plot_calibration_curve(estimator, X_train, y_train, name)

        # Remove temporarily so we can figure out how to get column names.
        # if isinstance(estimator, Pipeline):
        #     estimator = estimator[-1]
        # wandb.sklearn.plot_feature_importances(estimator, pd.DataFrame(X).columns)
        # else:
        # wandb.sklearn.plot_residuals(estimator, X_train, y_train)
        # wandb.sklearn.plot_outlier_candidates(estimator, X_train, y_train)
        return

    def on_remove_estimators(self):
        """Log config and results table."""
        self.on_fit_end()
        self.on_setup_end()
        return

    def on_add_estimators(self):
        """Rerun logging at setup end."""
        self.on_setup_end()
        return

    def on_add_preprocessing_step(self):
        """Rerun logging at setup end."""
        self.on_setup_end()
        self.on_setup_preprocessor()
        return

    def on_reassign_types(self):
        """Rerun logging at setup end."""
        self.on_setup_end()
        self.on_setup_preprocessor()
        return
