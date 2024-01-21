"""Definition of the selector interface."""

from abc import ABC, abstractmethod
from typing import Optional

from app.modelling.task_modelling import TaskModelling


class Selector(TaskModelling, ABC):
    """
    Interface for selectors.
    """

    def __init__(
        self,
        forecast_data_dir: str,
        result_path: str,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        forecast_data_dir : str
            Path to directory with forecast files.
        result_path : str
            Path to file where selection results will be saved.
        min_date : str, optional
            Min date of data that will be used for selection, by default None
            Expected format: YYYY-MM-DD HH
        max_date : str, optional
            Max date of data that will be used for selection, by default None
            Expected format: YYYY-MM-DD HH
        """
        super().__init__()
        self.forecast_data_dir = forecast_data_dir
        self.result_path = result_path
        self.min_date = min_date
        self.max_date = max_date

    @abstractmethod
    def _run(self) -> None:
        """
        Generate output file with the result.
        """
