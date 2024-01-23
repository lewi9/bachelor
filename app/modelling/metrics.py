"""Metrics for evaluating model performance."""

import pandas as pd


def mae(a: pd.Series, b: pd.Series) -> float:
    """Mean absolute error."""
    return (a - b).abs().mean()


def rmse(a: pd.Series, b: pd.Series) -> float:
    """Root mean square error."""
    return ((a - b) ** 2).mean() ** 0.5


def rme(a: pd.Series, b: pd.Series) -> float:
    """Root mean error."""
    return (a - b).mean() ** 0.5


def mse(a: pd.Series, b: pd.Series) -> float:
    """Mean square error."""
    return ((a - b) ** 2).mean()
