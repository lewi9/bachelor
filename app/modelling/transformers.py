"""Transformers for ForecastingPipeline."""
from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class CustomTransformer(ABC):
    """
    Interface for custom transformers to ForecastingPipeline.
    """

    def fit(self, y: pd.Series) -> CustomTransformer:
        """Fit transformer."""
        return self

    @abstractmethod
    def transform(self, y: pd.Series) -> pd.Series:
        """Transform series."""

    def fit_transform(self, y: pd.Series) -> pd.Series:
        """Fit and transform series."""
        return self.fit(y).transform(y)


class ImputerBackTime(CustomTransformer):
    """
    Imputer that fill nans with value from X hours before.
    """

    def __init__(self, period_to_take_value_from: int = 24):
        """
        Parameters
        ----------
        period_to_take_value_from : int
            Nans will be filled with value from period_to_take_value_from hours before.
        """
        self.period_to_take_value_from = period_to_take_value_from

    def transform(self, y: pd.Series) -> pd.Series:
        """
        Returns series with imputed values.
        """
        y_imputed = y.copy()
        for i in range(self.period_to_take_value_from, len(y)):
            y_imputed.iloc[i] = (
                y_imputed.iloc[i]
                if not np.isnan(y_imputed.iloc[i])
                else y_imputed.iloc[i - self.period_to_take_value_from]
            )
        return y_imputed


class ImputerPandas(CustomTransformer):
    """
    Impute values with pandas.interpolate
    """

    def __init__(self, method: str = "linear"):
        """
        Parameters
        ----------
        method : str
            Method of interpolation to fill missing values.
            Check pandas.interpolate.
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
        """
        self.method = method

    def transform(self, y: pd.Series) -> pd.Series:
        return y.interpolate(self.method)


class NanDropper(CustomTransformer):
    """
    Drop leading nans from series.
    """

    def fit(self, y: pd.Series) -> NanDropper:
        nans = y.isna().sum()
        series = y.iloc[:nans]
        if not series.isna().all():
            raise ValueError(
                "There are nans in the middle of series."
                "NanDropper can remove only leading nans."
            )
        return self

    def transform(self, y: pd.Series) -> pd.Series:
        return y.dropna()


class CompletnessFilter(CustomTransformer):
    """
    Check completness of series.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def transform(self, y: pd.Series) -> pd.Series:
        if y.isna().sum() > self.threshold * len(y):
            raise ValueError("There is too much missing data.")
        return y


class DormantFilter(CustomTransformer):
    """
    Check if sensor is active
    """

    def __init__(self, period: int):
        self.period = period

    def transform(self, y: pd.Series) -> pd.Series:
        if y.iloc[-self.period :].isna().all():
            raise ValueError("Sensor is inactive.")
        return y
