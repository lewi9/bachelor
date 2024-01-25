"""Module with classes to transform data."""

import logging
import os

import pandas as pd

from app.data_managers.namespaces import column_names_ns, date_formats_ns
from app.utils.task import Task


class TransformSensorMeteoDatasetsTask(Task):
    """Transform data from ingested to transformed - standardized - format."""

    def __init__(
        self,
        sensor_data_dir: str,
        weather_data_dir: str,
        result_dir: str,
        features_path: str,
        min_time: str,
        max_time: str,
    ):
        """
        Parameters
        ----------
        sensor_data_dir : str
            Path to directory with ingested sensor data
        weather_data_dir : str
            Path to directory with ingested Weather data
        result_dir : str
            Path to directory where transformed data will be saved
        features_path : str
            Path to dataframe with one column - 'feature' and feature column names as values.
        min_time : str
            Start time of actual data, format YYYY-MM-DD HH
        max_time : str
            End time of actual data, format YYYY-MM-DD HH
        """
        super().__init__()
        self.weather_data_dir = weather_data_dir
        self.sensor_data_dir = sensor_data_dir
        self.result_dir = result_dir
        self.features = pd.read_csv(features_path, usecols=[column_names_ns.FEATURES])
        self.min_time = pd.to_datetime(min_time, format=date_formats_ns.MIN_TIME_FORMAT)
        self.max_time = pd.to_datetime(max_time, format=date_formats_ns.MAX_TIME_FORMAT)

    def _run(self):
        """
        Transform data from ingested to transformed - standardized - format.

        1) Merge weater and sensor data.
        2) Resample data to hourly data.
        3) Melt data to semi-long format.
        4) Save data to csv.

        It iterates over files with weather.
        """
        weather_data_files = os.listdir(self.weather_data_dir)
        for file_name in weather_data_files:
            logging.info("process file: %s", file_name)
            try:
                data = self._merge_data(file_name)
            except OSError:
                continue
            data = data[
                list(self.features.values.flatten())
                + [column_names_ns.TIME]
                + column_names_ns.ENDO_COLUMNS
            ]
            data = self._resample_time(data)
            id_vars = data.columns.drop(column_names_ns.ENDO_COLUMNS).to_list()
            data = data.melt(
                id_vars=id_vars,
                value_vars=column_names_ns.ENDO_COLUMNS,
                var_name=column_names_ns.DEPENDENT_VARIABLE_NAME,
                value_name=column_names_ns.VALUE,
            )
            data.to_csv(
                os.path.join(self.result_dir, file_name),
                index=False,
            )

    def _merge_data(self, file_name: str) -> pd.DataFrame:
        """
        Merge meteo and sensor data.

        Parameters
        ----------
        file_name : str
            Name of file with data.
        """
        meteo = pd.read_csv(os.path.join(self.weather_data_dir, file_name))
        try:
            sensor = pd.read_csv(os.path.join(self.sensor_data_dir, file_name))
        except OSError as exc:
            logging.warning(
                "There is no sensor data for sensor: %s", file_name.split(".")[0]
            )
            raise exc
        meteo[column_names_ns.TIME] = pd.to_datetime(
            meteo[column_names_ns.TIME],
            format=date_formats_ns.WEATHER_DATETIME_FORMAT,
        )
        sensor[column_names_ns.TIME] = pd.to_datetime(
            sensor[column_names_ns.TIME],
            format=date_formats_ns.SENSOR_DATETIME_FORMAT,
        )
        data = meteo.merge(sensor, on=[column_names_ns.TIME], how="left")
        drop_columns = list(filter(lambda x: "Unnamed:" in x, data.columns))
        logging.info("Drop columns: %s", drop_columns)
        return data.drop(columns=drop_columns)

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
