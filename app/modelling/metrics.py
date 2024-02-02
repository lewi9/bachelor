"""Metrics for evaluating model performance."""

import pandas as pd


def mae(a: pd.Series, b: pd.Series) -> float:
    """Mean absolute error."""
    return (a - b).abs().mean()


def rmse(a: pd.Series, b: pd.Series) -> float:
    """Root mean square error."""
    return ((a - b) ** 2).mean() ** 0.5


def maxae(a: pd.Series, b: pd.Series) -> float:
    """Max absolute error."""
    return (a - b).abs().max()
