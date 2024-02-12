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
        """Default fit method. It will be called before transform method."""
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
        Initialize ImputerBackTime with period to take value from.

        Parameters
        ----------
        period_to_take_value_from : int
            Period in hours to take value from.
            Example: period_to_take_value_from=24, then if value is missing at 25th hour,
            it will be imputed with value from 1st hour. If value is missing at 26th hour,
            it will be imputed with value from 2nd hour.
        """
        self.period_to_take_value_from = period_to_take_value_from

    def transform(self, y: pd.Series) -> pd.Series:
        """
        Returns series with imputed values.

        Parameters
        ----------
        y : pd.Series
            Series to impute.
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
        Initialize ImputerPandas with method of interpolation.

        Parameters
        ----------
        method : str
            Method of interpolation to fill missing values.
            Check pandas.interpolate.
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
        """
        self.method = method

    def transform(self, y: pd.Series) -> pd.Series:
        """
        Impute values with pandas.interpolate

        Parameters
        ----------
        y : pd.Series
            Series to impute.
        """
        return y.interpolate(self.method)


class NanDropper(CustomTransformer):
    """
    Drop missing values at the beginning of series.
    """

    def fit(self, y: pd.Series) -> NanDropper:
        """
        Check if missing values are only at the beginning of series.
        """
        nans = y.isna().sum()
        series = y.iloc[:nans]
        if not series.isna().all():
            raise ValueError(
                "There are nans in the middle of series."
                "NanDropper can missing values only at the beginning of series."
            )
        return self

    def transform(self, y: pd.Series) -> pd.Series:
        """
        Return series without missing values at the beginning.
        """
        return y.dropna()


class CompletnessFilter(CustomTransformer):
    """
    Check completness of series.
    """

    def __init__(self, threshold: float):
        """
        Parameters
        ----------
        threshold : float
            Threshold of completness. If series has more missing values than threshold, raise error.
        """
        self.threshold = threshold

    def transform(self, y: pd.Series) -> pd.Series:
        """
        Check if series has more missing values than threshold.

        Parameters
        ----------
        y : pd.Series
            Series to check.

        Raises
        ------
        ValueError
            If series has more missing values than threshold.
        """
        if y.isna().sum() > self.threshold * len(y):
            raise ValueError("There is too much missing data.")
        return y


class DormantFilter(CustomTransformer):
    """
    Check if sensor is active
    """

    def __init__(self, period: int):
        """
        Check if sensor is active in last period hours.

        Parameters
        ----------
        period : int
            Period in hours.
        """
        self.period = period

    def transform(self, y: pd.Series) -> pd.Series:
        """
        Check if sensor is active in last period hours.
        Sensor is active if it has at least one non-nan value in last period hours.

        Parameters
        ----------
        y : pd.Series
            Series to check.
        """
        if y.iloc[-self.period :].isna().all():
            raise ValueError("Sensor is inactive.")
        return y


class ValuePositiver(CustomTransformer):
    """
    Make all values positive.
    """

    def transform(self, y: pd.Series) -> pd.Series:
        """
        Make all values positive.

        Parameters
        ----------
        y : pd.Series
            Series to transform.
        """
        new_y = y.copy()
        new_y.loc[y[y < 0].index] = 0
        return new_y
