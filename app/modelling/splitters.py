"""Module contains custom splitters for time series data."""

from abc import ABC, abstractmethod
from typing import Union, Sequence

import numpy as np
import pandas as pd


class BaseWindowSplitter(ABC):
    """Base class for window splitters."""

    def __init__(
        self,
        window_length: int,
        fh: Union[int, Sequence[int], np.ndarray[int]],
        step_length: int,
    ):
        """
        Parameters
        ----------
        window_length : int
            Length of the window.
        fh : Union[int, Sequence[int], np.ndarray[int]]
            Forecast horizon.
        step_length : int
            Step length.
        """
        self._window_length = window_length
        self._fh = fh
        self._step_length = step_length

    @property
    def fh(self) -> Union[int, Sequence[int], np.ndarray[int]]:
        """Forecast horizon."""
        return self._fh

    @property
    def window_length(self) -> int:
        """Length of the window."""
        return self._window_length

    @property
    def step_length(self) -> int:
        """Step length."""
        return self._step_length

    @abstractmethod
    def split(self, y: pd.Series, forecasts: int) -> (Sequence[int], Sequence[int]):
        """Split a single series into windows."""


class ExpandingWindowSplitter(BaseWindowSplitter):
    """Expanding window splitter."""

    def __init__(
        self,
        fh: Union[int, Sequence[int], np.ndarray[int]],
        step_length: int,
    ):
        """
        Parameters
        ----------
        fh : Union[int, Sequence[int], np.ndarray[int]]
            Forecast horizon.
        step_length : int
            Step length.
        """
        super().__init__(0, fh, step_length)

    def split(self, y: pd.Series, forecasts: int) -> (Sequence[int], Sequence[int]):
        """
        Split a single series into windows.

        Parameters
        ----------
        y : pd.Series
            Series to split.
        X : pd.DataFrame
            Exogenous variables.
        forecasts : int
            Number of forecasts to make.
        """
        y_len = len(y)
        len_fh = len(self.fh)
        for i in range(forecasts):
            yield (
                list(range(0, y_len - i * self.step_length)),
                list(range(0, y_len + len_fh - i * self.step_length)),
            )


class SlidingWindowSplitter(BaseWindowSplitter):
    """Sliding window splitter."""

    def split(self, y: pd.Series, forecasts: int) -> (Sequence[int], Sequence[int]):
        """
        Split a single series into windows.

        Parameters
        ----------
        y : pd.Series
            Series to split.
        forecasts : int
            Number of forecasts to make.
        """
        y_len = len(y)
        len_fh = len(self.fh)
        for i in range(forecasts):
            yield (
                list(range(0, y_len - i * self.step_length))[-self.window_length :],
                list(range(0, y_len + len_fh - i * self.step_length))[
                    -self.window_length - len_fh :
                ],
            )
