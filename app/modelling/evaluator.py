"""Module for evaluation of forecasting results."""

import logging
import os
import pandas as pd

from app.data_managers.namespaces import column_names_ns
from app.data_managers.types import EvaluatorMetric
from app.modelling.task_modelling import TaskModelling


class Evaluator(TaskModelling):
    """
    Class for evaluation of forecasting results.
    """

    def __init__(
        self,
        forecast_data_dir: str,
        transformed_data_dir: str,
        result_path: str,
        metric: EvaluatorMetric,
    ):
        """
        Paramters
        ---------
        forecast_data_dir : str
            Path to directory with forecast files.
        transformed_data_dir : str
            Path to directory with actual data, that has been used for forecasting.
        result_path : str
            Path to file where evaluation results will be saved.
        metric : EvaluatorMetric
            Metric that will be used to evaluate forecasting results.
        """
        super().__init__()
        self.forecast_data_dir = forecast_data_dir
        self.transformed_data_dir = transformed_data_dir
        self.result_path = result_path
        self.metric = metric

    def _run(self):
        """
        Evaluate forecasting results with defined metric in __init__.
        """
        output = []
        files = os.listdir(self.forecast_data_dir)
        for file_name in files:
            forecast = pd.read_csv(os.path.join(self.forecast_data_dir, file_name))
            try:
                actual_data = pd.read_csv(
                    os.path.join(self.transformed_data_dir, file_name)
                )
            except OSError:
                logging.warning(
                    "File %s has not been found in %s.",
                    file_name,
                    self.transformed_data_dir,
                )
                continue

            evaluation = self._evaluate(forecast=forecast, actual_data=actual_data)
            evaluation[column_names_ns.FILE_NAME] = file_name
            logging.info("%s has been evaluated.", file_name)
            output.append(evaluation)
        pd.DataFrame.from_records(output).to_csv(self.result_path, index=False)

    def _evaluate(self, forecast: pd.DataFrame, actual_data: pd.DataFrame) -> dict:
        """
        Evaluate metric for specified forecast and actual data.
        """
        # Cut actual data to period with forecast.
        actual_data_cut = actual_data[
            (actual_data[column_names_ns.TIME] >= forecast[column_names_ns.TIME].min())
            & (
                actual_data[column_names_ns.TIME]
                <= forecast[column_names_ns.TIME].max()
            )
        ]
        names = forecast[column_names_ns.PREDICTED].unique()
        result = {}
        # Iterate over predicted variable names
        for name in names:
            selected_forecast = forecast[forecast[column_names_ns.PREDICTED] == name]
            # Iterate over models that has been used to obtain forecasts.
            for model in selected_forecast[column_names_ns.MODEL]:
                specify_forecast = selected_forecast[
                    selected_forecast[column_names_ns.MODEL] == model
                ]
                merged = actual_data_cut.merge(
                    specify_forecast, on=column_names_ns.TIME, how="inner"
                )
                result[f"{name}{column_names_ns.SEPARATOR}{model}"] = self.metric(
                    merged[name], merged[column_names_ns.VALUE]
                )
        return result
