"""Wrapper for sklearn's LinearRegression model."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union, Sequence

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator


class ModelWrapper(ABC):
    """Wrapper for sklearn's LinearRegression model."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        self.model = BaseEstimator()
        self.indexes = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ModelWrapper:
        """Fit the model to the data."""
        self.indexes = y.index
        self.model.fit(X.loc[self.indexes], y)

        return self

    def predict(
        self, X: pd.DataFrame, fh: Union[int, Sequence[int], np.ndarray[int]]
    ) -> pd.Series:
        """Make predictions."""
        if self.indexes is None:
            raise ValueError("Model has not been fitted yet.")
        fit_index = X.index.isin(self.indexes)
        X_predict = X.loc[~fit_index].iloc[: len(fh)]
        if len(X_predict) < len(fh):
            raise ValueError(
                "The number of forecast points exceeds the number of points in the input series."
            )
        return pd.Series(self.model.predict(X_predict))
