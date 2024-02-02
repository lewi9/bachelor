"""AggModelStrategyTask class."""

import logging
import os
import shutil
from typing import Callable, Union

import pandas as pd

from app.data_managers.namespaces import column_names_ns, date_formats_ns
from app.data_managers.types import ForecastingPipelineMode
from app.utils.task import Task


class AggModelStrategyTask(Task):
    """Class for aggregation model strategy task."""

    def __init__(
        self,
        forecast_dir: str,
        result_dir: str,
        aggregation_function: Union[Callable, str],
        mode: ForecastingPipelineMode,
    ):
        """
        Parameters
        ----------
        forecast_dir : str
            Directory with forecast data.
        result_dir : str
            Directory for result data.
        aggregation_function : Union[Callable, str]
            Aggregation function.
        mode : ForecastingPipelineMode
            Mode for forecasting pipeline.
        """
        super().__init__()
        self.forecast_dir = forecast_dir
        self.result_dir = result_dir
        self.aggregation_function = aggregation_function
        self.mode = mode

    def _run(self):
        """Run task."""
        if not self._handle_result_dir():
            return

        files = os.listdir(self.forecast_dir)
        for file_name in files:
            forecast = pd.read_csv(os.path.join(self.forecast_dir, file_name))
            forecast[column_names_ns.TIME] = pd.to_datetime(
                forecast[column_names_ns.TIME],
                format=date_formats_ns.FORECASTING_PIPELINE_DATETIME_FORMAT,
            )

            forecast = forecast[
                [
                    column_names_ns.DEPENDENT_VARIABLE_NAME,
                    column_names_ns.TIME,
                    column_names_ns.VALUE,
                ]
            ]

            forecast = forecast.groupby(
                [column_names_ns.DEPENDENT_VARIABLE_NAME, column_names_ns.TIME],
                as_index=False,
            ).agg(self.aggregation_function)
            forecast[column_names_ns.FORECASTER] = "AGGREGATION"
            forecast[column_names_ns.SPLITTER] = "AGGREGATION"

            forecast.to_csv(
                os.path.join(self.result_dir, file_name), index=False, header=True
            )

    def _handle_result_dir(self) -> bool:
        """
        Handle result dir.
        """
        flag = os.path.exists(self.result_dir)

        if flag and self.mode == "recreate":
            logging.info("Result dir already exists. It will be recreated.")
            shutil.rmtree(self.result_dir)
            os.mkdir(self.result_dir)
            return True

        if flag and self.mode is None:
            logging.info(
                "Result dir already exists. Forecasting Pipeline will be skipped."
            )
            return False

        if not flag:
            os.mkdir(self.result_dir)
            return True

        raise ValueError(f"Mode {self.mode} is not supported.")
