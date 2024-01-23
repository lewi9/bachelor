"""Module to select best model for each predicted variable per time series."""
import os
from typing import Optional

import pandas as pd

from app.data_managers.namespaces import column_names_ns
from app.modelling.selector import Selector


class BestSelector(Selector):
    """
    Select best model for each predicted variable per time series.
    It must be run after evaluation task, it uses evaluation results to select the best model.
    """

    def __init__(
        self,
        forecast_data_dir: str,
        evaluation_path: str,
        result_path: str,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        forecast_data_dir : str
            Path to directory with forecast files.
        evaluation_path : str
            Path to file with evaluation results.
        result_path : str
            Path to file where selection results will be saved.
        min_date : str, optional
            Min date of data that will be used for selection, by default None
            Expected format: YYYY-MM-DD HH
        max_date : str, optional
            Max date of data that will be used for selection, by default None
            Expected format: YYYY-MM-DD HH
        """
        super().__init__(
            forecast_data_dir=forecast_data_dir,
            result_path=result_path,
            min_date=min_date,
            max_date=max_date,
        )
        self.evaluation_df = pd.read_csv(evaluation_path)

    def _run(self):
        """
        Select best model for each predicted variable per time series.

        Returns
        -------
        pd.DataFrame
            Dataframe with columns for each combination of predicted variable and model.
            The values are calculated metric.
        """
        output = []
        best_model_columns = set()

        # Create set of predicted variables
        predicted_variables = {
            x.split(column_names_ns.SEPARATOR)[0]
            for x in self.evaluation_df.columns
            if column_names_ns.SEPARATOR in x
        }

        for predicted_variable in predicted_variables:
            # Filter columns for predicted variable
            columns = [
                column
                for column in self.evaluation_df.columns
                if column.startswith(f"{predicted_variable}{column_names_ns.SEPARATOR}")
            ]

            # Save the best model for predicted variable
            self.evaluation_df[
                f"{column_names_ns.BEST_MODEL}{column_names_ns.SEPARATOR}{predicted_variable}"
            ] = self.evaluation_df[columns].idxmin(axis=1)

            # Add column name to set of best model columns
            best_model_columns.add(
                f"{column_names_ns.BEST_MODEL}{column_names_ns.SEPARATOR}{predicted_variable}"
            )

        for _, row in self.evaluation_df.iterrows():
            # Get the forecast
            file_name = row[column_names_ns.FILE_NAME]
            forecast = pd.read_csv(os.path.join(self.forecast_data_dir, file_name))

            # Add column with sensor id
            forecast[column_names_ns.ID] = file_name.split(".")[0]

            # Filter forecast by date
            forecast[column_names_ns.TIME] = forecast[column_names_ns.TIME].apply(
                lambda date_time: date_time[:-6]
            )
            min_date = self.min_date or forecast[column_names_ns.TIME].min()
            max_date = self.max_date or forecast[column_names_ns.TIME].max()

            forecast = forecast.query(
                f"'{min_date}' <= {column_names_ns.TIME} <= '{max_date}'"
            )

            # Get the best model for each predicted variable
            forecasts = [
                forecast.query(
                    f"{column_names_ns.PREDICTED} == '{variable}' & "
                    f"{column_names_ns.FORECASTER} == '{model}'"
                )
                for variable, model in [
                    row[col].split(column_names_ns.SEPARATOR)
                    for col in best_model_columns
                ]
            ]

            # Concatenate forecasts
            if forecasts:
                output.append(pd.concat(forecasts, ignore_index=True))

        if output:
            pd.concat(output, ignore_index=True).to_csv(self.result_path, index=False)
            return
        pd.DataFrame(columns=column_names_ns.FORECAST_COLUMNS).to_csv(
            self.result_path, index=False
        )
