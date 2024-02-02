"""Module for the BestModelStrategyTask class."""

import logging
import os
import shutil
from collections import defaultdict

import pandas as pd
import ray

from app.data_managers.namespaces import column_names_ns, date_formats_ns
from app.data_managers.types import ForecastingPipelineMode, SeriesMetric
from app.utils.task import Task


def _load_data(
    transformed_data_path: str, forecast_path: str
) -> (pd.DataFrame, pd.DataFrame):
    """
    Parameters
    ----------
    transformed_data_path : str
        Path to the transformed data.
    forecast_path : str
        Path to the forecast data.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Tuple with transformed data and forecast.
    """
    transformed_data = pd.read_csv(transformed_data_path)
    forecast = pd.read_csv(forecast_path)

    transformed_data[column_names_ns.TIME] = pd.to_datetime(
        transformed_data[column_names_ns.TIME],
        format=date_formats_ns.FORECASTING_PIPELINE_DATETIME_FORMAT,
    )
    forecast[column_names_ns.TIME] = pd.to_datetime(
        forecast[column_names_ns.TIME],
        format=date_formats_ns.FORECASTING_PIPELINE_DATETIME_FORMAT,
    )

    return transformed_data, forecast


@ray.remote
def _run_remote(file_name: str, **kwargs):
    """Run task remotely."""
    logging.info("Start processing file %s.", file_name)

    transformed_data_path = os.path.join(kwargs["transformed_data_dir"], file_name)
    forecast_path = os.path.join(kwargs["forecast_dir"], file_name)
    output_path = os.path.join(kwargs["result_dir"], file_name)
    metric = kwargs["metric"]
    last_evaluation_date = kwargs["last_evaluation_date"]

    # Load data
    transformed_data, forecast = _load_data(transformed_data_path, forecast_path)

    forecast_subset = forecast.query(
        f"{column_names_ns.TIME} <= '{last_evaluation_date}'"
    )
    forecast_subset_min_date = forecast_subset[column_names_ns.TIME].min()

    transformed_data_subset = transformed_data.query(
        f"{column_names_ns.TIME} <= '{last_evaluation_date}'"
    ).query(f"{column_names_ns.TIME} >= '{forecast_subset_min_date}'")

    dependent_variables = set(forecast[column_names_ns.DEPENDENT_VARIABLE_NAME])
    forecasters = set(forecast[column_names_ns.FORECASTER])
    splitters = set(forecast[column_names_ns.SPLITTER])

    results = defaultdict(dict)

    for dependent_variable in dependent_variables:
        actuals = transformed_data_subset.query(
            f"{column_names_ns.DEPENDENT_VARIABLE_NAME} == '{dependent_variable}'"
        )
        actuals = actuals.set_index(column_names_ns.TIME)
        actuals = actuals.loc[:, column_names_ns.VALUE]
        for forecaster in forecasters:
            for splitter in splitters:

                forecast_subset_filtered = (
                    forecast_subset.query(
                        f"{column_names_ns.DEPENDENT_VARIABLE_NAME} == '{dependent_variable}'"
                    )
                    .query(f"{column_names_ns.FORECASTER} == '{forecaster}'")
                    .query(f"{column_names_ns.SPLITTER} == '{splitter}'")
                )

                if forecast_subset.empty:
                    continue

                forecast_subset_filtered = forecast_subset_filtered.set_index(
                    column_names_ns.TIME
                )
                forecast_subset_filtered = forecast_subset_filtered.loc[
                    :, column_names_ns.VALUE
                ]
                results[dependent_variable][
                    f"{forecaster}{column_names_ns.SEPARATOR}{splitter}"
                ] = metric(actuals, forecast_subset_filtered)

    output = []
    for dependent_variable, values in results.items():
        minimal_key = min(values, key=values.get)
        output.append(
            forecast.query(
                f"{column_names_ns.DEPENDENT_VARIABLE_NAME} == '{dependent_variable}'"
            )
            .query(
                f"{column_names_ns.FORECASTER} == '{minimal_key.split(column_names_ns.SEPARATOR)[0]}'"
            )
            .query(
                f"{column_names_ns.SPLITTER} == '{minimal_key.split(column_names_ns.SEPARATOR)[1]}'"
            )
            .query(f"{column_names_ns.TIME} > '{last_evaluation_date}'")
        )
    if output:
        output = pd.concat(output)
        output.to_csv(output_path, index=False)


class BestModelStrategyTask(Task):
    """Task for finding the best model strategy."""

    def __init__(
        self,
        transformed_data_dir: str,
        forecast_dir: str,
        result_dir: str,
        metric: SeriesMetric,
        last_evaluation_date: str,
        mode: ForecastingPipelineMode = "recreate",
    ):
        """
        Parameters
        ----------
        transformed_data_dir : str
            Path to the directory with transformed data.
        forecast_dir : str
            Path to the directory with forecast data.
        result_dir : str
            Path to the directory with result data.
        metric : SeriesMetric
            Metric for model evaluation.
        last_evaluation_date : str
            The forecasts after this date will be in the output.
            E.g. "2020-01-01 00" (format: "%Y-%m-%d %H").
            Then the forecasts after 2020-01-01 00 will be in the result.
            The previous forecast will be used for the model evaluation.
        mode : ForecastingPipelineMode
            Mode for the handling of the result directory.
            Check app.data_managers.types.ForecastingPipelineMode for more details.
        """
        super().__init__()
        self.transformed_data_dir = transformed_data_dir
        self.forecast_dir = forecast_dir
        self.result_dir = result_dir
        self.metric = metric
        self.last_evaluation_date = pd.to_datetime(
            last_evaluation_date,
            format=date_formats_ns.BEST_MODEL_STRATEGY_DATETIME_FORMAT,
        )
        self.mode = mode

    def _run(self):
        """Run task."""

        if not self._handle_result_dir():
            return

        files = os.listdir(self.forecast_dir)
        ray.init()
        results_ref = [
            _run_remote.remote(file_name=file_name, **self.__dict__)
            for file_name in files
        ]
        ray.get(results_ref)
        ray.shutdown()

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
