from __future__ import annotations

import collections

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..utils.estimate import get_target_info

__all__ = ['TargetEncoder']

def check_input(X):
    X_ = check_array(X, dtype=None, ensure_2d=True, ensure_all_finite=False)
    if X_.dtype.kind in {"U", "S"}:
        if np.any(X_ == "nan"):
            return check_array(
                np.array(X, dtype=object),
                dtype=None,
                ensure_2d=True,
                ensure_all_finite=False,
            )
    return X_

class TargetEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features considering the effect that it has in the
    target variable.

    Note that implementation and docstrings are largely taken from
    [Dirty Cat](https://github.com/dirty-cat/dirty_cat/blob/master/dirty_cat/target_encoder.py).

    Parameters
    ----------
    task :
        The type of problem. Either "classification" or "regression".
    handle_unknown :
        Either "error" or "ignore". Whether to raise an error or ignore if a unknown
        categorical feature is present during transform (default is to raise). If 'ignore',
        unknown categories will be set to the mean of the target.
    handle_missing :
        Either "error" or "". Whether to raise an error or impute with blank string "" if missing
        values (NaN) are present during fit (default is to impute).
    """

    def __init__(self, task: str, handle_unknown="error", handle_missing=""):
        self.task = task
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing

    def fit(
        self,
        X: pd.DataFrame | np.ndarray | list,
        y: pd.DataFrame | np.ndarray | list,
    ) -> TargetEncoder:
        """Fit the TargetEncoder to X.

        Parameters
        ----------
        X :
            The data to determine the categories of each feature.
        y :
            The associated target vector.

        Returns
        -------
        TargetEncoder
            Fitted TargetEncoder.
        """
        self.type_of_target_ = get_target_info(y, self.task)["type_"]
        if isinstance(X, pd.DataFrame):
            self.colnames_ = X.columns
        X = check_input(X)
        self.n_features_in_ = X.shape[1]
        X = X.astype(str)
        if self.handle_missing not in ["error", ""]:
            template = "handle_missing should be either 'error' or " "'', got %s"
            raise ValueError(template % self.handle_missing)
        if hasattr(X, "iloc") and X.isna().values.any():
            if self.handle_missing == "error":
                msg = (
                    "Found missing values in input data; set "
                    "handle_missing='' to encode with missing values"
                )
                raise ValueError(msg)
            else:
                X = X.fillna(self.handle_missing)
        elif not hasattr(X, "dtype") and isinstance(X, list):
            X = np.asarray(X, dtype=object)
        if hasattr(X, "dtype"):
            mask = pd.isna(X)
            if mask.any():
                if self.handle_missing == "error":
                    msg = (
                        "Found missing values in input data; set "
                        "handle_missing='' to encode with missing values"
                    )
                    raise ValueError(msg)
                else:
                    X[mask] = self.handle_missing

        if self.handle_unknown not in ["error", "ignore"]:
            template = "handle_unknown should be either 'error' or " "'ignore', got %s"
            raise ValueError(template % self.handle_unknown)
        X_temp = check_array(X, dtype=None)
        if not hasattr(X, "dtype") and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp

        n_features = X.shape[1]

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for j in range(n_features):
            le = self._label_encoders_[j]
            Xj = X[:, j]
            le.fit(Xj)
        self.categories_ = [le.classes_ for le in self._label_encoders_]
        self.n_ = len(y)
        if self.type_of_target_ == "binary" and np.array(y).dtype.kind not in "biufc":
            y = LabelEncoder().fit_transform(y)
        if self.type_of_target_ in ["continuous", "binary"]:
            self.Eyx_ = [
                {cat: np.mean(y[X[:, j] == cat]) for cat in self.categories_[j]}
                for j in range(len(self.categories_))
            ]
            self.Ey_ = np.mean(y)
            self.counter_ = {j: collections.Counter(X[:, j]) for j in range(n_features)}
        if self.type_of_target_ == "multiclass":
            self.classes_ = np.unique(y)

            self.Eyx_ = {
                c: [
                    {
                        cat: np.mean((y == c)[X[:, j] == cat])
                        for cat in self.categories_[j]
                    }
                    for j in range(len(self.categories_))
                ]
                for c in self.classes_
            }
            self.Ey_ = {c: np.mean(y == c) for c in self.classes_}
            self.counter_ = {j: collections.Counter(X[:, j]) for j in range(n_features)}
        self.k_ = {j: len(self.counter_[j]) for j in self.counter_}
        return self

    def transform(self, X: pd.DataFrame | np.ndarray | list):
        """Transform X using specified encoding scheme.

        Parameters
        ----------
        X :
            The data to encode.
        Returns
        -------
        X :
            Transformed input.
        """
        check_is_fitted(self, attributes=["n_features_in_"])
        X = check_input(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Number of features in the input data ({X.shape[1]}) does not match the number of features "
                f"seen during fit ({self.n_features_in_})."
            )
        X = X.astype(str)
        if hasattr(X, "iloc") and X.isna().values.any():
            if self.handle_missing == "error":
                msg = (
                    "Found missing values in input data; set "
                    "handle_missing='' to encode with missing values"
                )
                raise ValueError(msg)
            else:
                X = X.fillna(self.handle_missing)
        elif not hasattr(X, "dtype") and isinstance(X, list):
            X = np.asarray(X, dtype=object)
        if hasattr(X, "dtype"):
            mask = pd.isna(X)
            if mask.any():
                if self.handle_missing == "error":
                    msg = (
                        "Found missing values in input data; set "
                        "handle_missing='' to encode with missing values"
                    )
                    raise ValueError(msg)
                else:
                    X[mask] = self.handle_missing

        X_temp = check_array(X, dtype=None)
        if not hasattr(X, "dtype") and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp

        n_features = X.shape[1]
        X_int = np.zeros_like(X, dtype=int)
        X_mask = np.ones_like(X, dtype=bool)

        for i in range(n_features):
            Xi = X[:, i]
            valid_mask = np.isin(Xi, self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == "error":
                    diff = np.unique(X[~valid_mask, i])
                    msg = (
                        "Found unknown categories {0} in column {1}"
                        " during transform".format(diff, i)
                    )
                    raise ValueError(msg)
                else:
                    X_mask[:, i] = valid_mask
                    Xi = Xi.copy()
                    Xi[~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(Xi)

        out = []

        for j, cats in enumerate(self.categories_):
            unqX = np.unique(X[:, j])
            encoder = {x: 0 for x in unqX}
            if self.type_of_target_ in ["continuous", "binary"]:
                for x in unqX:
                    if x not in cats:
                        Eyx = 0
                    else:
                        Eyx = self.Eyx_[j][x]
                    lambda_n = self._compute_lambda(self.counter_[j][x], self.n_ / self.k_[j])
                    encoder[x] = lambda_n * Eyx + (1 - lambda_n) * self.Ey_
                x_out = np.zeros((len(X[:, j]), 1))
                for i, x in enumerate(X[:, j]):
                    x_out[i, 0] = encoder[x]
                out.append(x_out.reshape(-1, 1))
            if self.type_of_target_ == "multiclass":
                x_out = np.zeros((len(X[:, j]), len(self.classes_)))
                lambda_n = {x: 0 for x in unqX}
                for x in unqX:
                    lambda_n[x] = self._compute_lambda(
                        self.counter_[j][x], self.n_ / self.k_[j]
                    )
                for k, c in enumerate(np.unique(self.classes_)):
                    for x in unqX:
                        if x not in cats:
                            Eyx = 0
                        else:
                            Eyx = self.Eyx_[c][j][x]
                        encoder[x] = lambda_n[x] * Eyx + (1 - lambda_n[x]) * self.Ey_[c]
                    for i, x in enumerate(X[:, j]):
                        x_out[i, k] = encoder[x]
                out.append(x_out)
        out = np.hstack(out)
        return out

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get feature names for output."""
        colnames_ = getattr(self, "colnames_", None)
        if colnames_ is not None:
            if self.type_of_target_ in ["continuous", "binary"]:
                output = colnames_.tolist()
            else:
                output = []
                for name in colnames_:
                    unique = self.classes_
                    names = [f"{name}_{level}" for level in unique]
                    output.extend(names)
            return output

    @staticmethod
    def _compute_lambda(x, n):
        return x / (x + n)
