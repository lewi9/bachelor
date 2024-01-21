"""Module containing AverageSelector class."""

import os

import pandas as pd

from app.data_managers.namespaces import column_names_ns, values_ns
from app.modelling.selector import Selector


class AverageSelector(Selector):
    """
    Create results as average of all models.
    Select the data based on the min and max date.
    """

    def _run(self) -> None:
        """
        Generate output file with the result.
        """
        output = []

        for file_name in os.listdir(self.forecast_data_dir):
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

            # Drop model column
            forecast = forecast.drop(columns=column_names_ns.MODEL)

            # Calculate average for each predicted variable
            forecasts = forecast.groupby(
                forecast.columns.drop(column_names_ns.VALUE).to_list(), as_index=False
            ).mean()

            forecasts[column_names_ns.MODEL] = values_ns.AVERAGE_MODEL

            output.append(forecasts)

        if output:
            pd.concat(output, ignore_index=True).to_csv(self.result_path, index=False)
            return
        pd.DataFrame(columns=column_names_ns.FORECAST_COLUMNS).to_csv(
            self.result_path, index=False
        )
