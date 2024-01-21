"""Module with classes to transform data."""

import logging
import os

import pandas as pd

from app.data_managers.namespaces import column_names_ns, date_formats_ns
from app.data_managers.task_etl import TaskETL


class CreateOneDatasetTask(TaskETL):
    """
    Concatenate data from all sensors and weather data to one file.
    """

    def __init__(
        self, weather_data_dir: str, sensor_data_dir: str, output_file_path: str
    ):
        """
        Parameters
        ----------
        weather_data_dir : str
            Path to directory with ingested Weather data
        sensor_data_dir : str
            Path to directory with ingested sensor data
        one_data_file_path : str
            Path, where raw concatanated data will be saved
        """
        super().__init__()
        self.weather_data_dir = weather_data_dir
        self.sensor_data_dir = sensor_data_dir
        self.output_file_path = output_file_path

    def _run(self):
        """
        Creates one file with data of all sensors and whole weather data.
        It merges two files with the same name on column that contains time.
        Columns ID and TIME are index of the result pd.DataFrame

        It iterates over files with weather.
        """
        dfs = []
        weather_data_files = os.listdir(self.weather_data_dir)
        for file_name in weather_data_files:
            logging.info("process file: %s", file_name)

            meteo = pd.read_csv(os.path.join(self.weather_data_dir, file_name))
            try:
                sensor = pd.read_csv(os.path.join(self.sensor_data_dir, file_name))
            except OSError:
                logging.warning(
                    "There is no sensor data for sensor: %s", file_name.split(".")[0]
                )
                continue

            meteo[column_names_ns.TIME] = pd.to_datetime(
                meteo[column_names_ns.TIME],
                format=date_formats_ns.WEATHER_DATETIME_FORMAT,
            )
            sensor[column_names_ns.TIME] = pd.to_datetime(
                sensor[column_names_ns.TIME],
                format=date_formats_ns.SENSOR_DATETIME_FORMAT,
            )

            data = meteo.merge(sensor, on=[column_names_ns.TIME], how="left")
            data[column_names_ns.ID] = file_name.split(".")[0]
            data = data.set_index([column_names_ns.ID, column_names_ns.TIME])

            dfs.append(data)

        to_save = pd.concat(dfs)
        drop_columns = list(filter(lambda x: "Unnamed:" in x, to_save.columns))
        logging.info("Drop columns: %s", drop_columns)
        logging.info("Saving to csv file")
        to_save.drop(columns=drop_columns).to_csv(self.output_file_path)


class CreateStandarizedWideFormatTask(TaskETL):
    """
    Create standarized wide format of data that will be used in forecasting.
    """

    def __init__(
        self,
        one_data_file_path: str,
        features_path: str,
        result_dir: str,
        min_time: str,
        max_time: str,
        forecast_period: int,
    ):
        """
        Parameters
        ----------
        one_data_file_path : str
            Path to one data dataframe file.
        features_path : str
            Path to dataframe with one column - 'feature' and feature column names as values.
        result_dir : str
            Path to dir, where prepared DataModel per sensor will be stored.
        min_time : str
            Start time of actual data, format YYYY-MM-DD HH
        max_time : str
            End time of actual data, format YYYY-MM-DD HH
        forecast_period : int
            Forecast period in hours
        """
        super().__init__()
        self.one_data_file_path = one_data_file_path
        self.features_path = features_path
        self.result_dir = result_dir
        self.min_time = pd.to_datetime(min_time, format=date_formats_ns.MIN_TIME_FORMAT)
        self.max_time = pd.to_datetime(
            max_time, format=date_formats_ns.MAX_TIME_FORMAT
        ) + pd.DateOffset(hours=forecast_period)

    def _run(self):
        """
        Create standarized wide format of data that will be used in forecasting.
        """
        logging.info("Prepare wide format of data")
        data = self._prepare_wide_format(
            dataframe_path=self.one_data_file_path, features_path=self.features_path
        )
        logging.info("Resample time to hourly")
        for sensor_id in data[column_names_ns.ID].unique():
            selected_data = data[data[column_names_ns.ID] == sensor_id].drop(
                columns=column_names_ns.ID
            )
            output = self._resample_time(selected_data)
            output.to_csv(os.path.join(self.result_dir, f"{sensor_id}.csv"))

    def _prepare_wide_format(
        self,
        dataframe_path: str,
        features_path: str,
    ) -> pd.DataFrame:
        """
        Load dataframe from file and select features that names are saved in features file.

        Parameters
        ----------
        dataframe_path : str
            Path to one data dataframe file.
        features_path : str
            Path to dataframe with one column - 'feature' and feature column names as values.

        Returns
        -------
        pd.DataFrame
            Dataframe with selected features.
        """
        features = pd.read_csv(features_path, usecols=[column_names_ns.FEATURES])
        data = pd.read_csv(dataframe_path)
        return data[
            list(features.values.flatten()) + [column_names_ns.ID, column_names_ns.TIME]
        ]

    def _resample_time(
        self,
        input_dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Resamples data to hourly data.

        Parameters
        ----------
        input_dataframe : pd.DataFrame
            Dataframe with columns: TIME.

        Returns
        -------
        pd.DataFrame
            Dataframe resampled to hourly data - filled empty with np.nan.
        """
        data = input_dataframe.copy()

        # Remove utc and convert to datetime
        data[column_names_ns.TIME] = data[column_names_ns.TIME].apply(
            lambda date: date.split("+")[0]
        )
        data[column_names_ns.TIME] = pd.to_datetime(
            data[column_names_ns.TIME], format="%Y-%m-%d %H:%M:%S"
        )

        # Add row with min_time
        if self.min_time < data[column_names_ns.TIME].min():
            row = pd.DataFrame({column_names_ns.TIME: [self.min_time]})
            data = pd.concat([row, data], ignore_index=True)

        # Add row with max_time
        if self.max_time > data[column_names_ns.TIME].max():
            row = pd.DataFrame({column_names_ns.TIME: [self.max_time]})
            data = pd.concat([data, row], ignore_index=True)

        # Cut data
        data = data[
            (data[column_names_ns.TIME] >= self.min_time)
            & (data[column_names_ns.TIME] <= self.max_time)
        ]

        # Set index
        data = data.set_index(
            pd.DatetimeIndex(data[column_names_ns.TIME]), drop=True
        ).drop(columns=column_names_ns.TIME)

        # Resample data to hours
        result = data.resample("H").asfreq()
        result = result.reset_index()
        return result
