"""Module with classes for data extraction from json files."""

import json
import logging
import os
from typing import Iterable

import pandas as pd

from app.data_managers.namespaces import (
    column_names_ns,
    sensor_json_ns,
    weather_json_ns,
)
from app.data_managers.types import ExtractionMode
from app.utils.task import Task


class SensorDataExtractionTask(Task):
    """
    Responsible for load sensors data from json files and store results in csv files.
    As the result file per sensor is created or data is appended.
    """

    def __init__(
        self, input_dir: Iterable, result_dir: str, mode: ExtractionMode = "append"
    ):
        """
        Parameters
        ----------
        input_dir : Iterable
            Iterable of directories that contains data in json files.
            It will process them in order.
        result_dir : str
            Directory, where data should be stored in csv files
        mode : ExtractionMode, default='append'
            If 'append', Extractor will append data to existing files in result_dir.
            If 'overwrite', Extractor will overwrite files in result_dir, if exist.
            If None, Extractor will not extract if file exists.
        """
        super().__init__()
        if isinstance(input_dir, str):
            input_dir = (input_dir,)
        self.input_dir = input_dir
        self.result_dir = result_dir
        self.mode = mode

    def _run(self):
        """
        Obtain data from json files and store them in result_dir.
        """

        for directory in self.input_dir:
            if not os.path.isdir(directory):
                logging.warning("%s doesn't exist", directory)
                continue

            for file_name in os.listdir(directory):
                sensor_id = self._obtain_id_from_file_name(file_name=file_name)
                if sensor_id == "":
                    logging.warning(
                        "File name - %s - is in bad format, cannot obtain id of sensor",
                        file_name,
                    )
                    continue

                try:
                    with open(
                        os.path.join(directory, file_name), encoding="utf-8"
                    ) as f:
                        data = json.load(f)
                except OSError:
                    logging.warning("Cannot load that file as json: %s ", file_name)
                    continue

                try:
                    df = pd.DataFrame(data[sensor_json_ns.HISTORY])
                except KeyError:
                    logging.warning(
                        "There is no historical data for sensor data in that file: %s",
                        file_name,
                    )
                    continue

                result = self._normalize_df(df)

                loaded_data = pd.DataFrame()
                result_file_name = f"{sensor_id}.csv"
                output_path = os.path.join(self.result_dir, result_file_name)

                if result_file_name in os.listdir(self.result_dir):
                    if self.mode == "append":
                        logging.info("Append data to file: %s", result_file_name)
                        loaded_data = pd.read_csv(output_path)
                        result = pd.concat([loaded_data, result], ignore_index=True)
                        result = result.drop_duplicates(
                            subset=column_names_ns.TIME, keep="last"
                        )
                        result = result.reset_index(drop=True)
                    elif self.mode == "overwrite":
                        logging.info("Overwrite file: %s", result_file_name)
                    else:
                        logging.warning(
                            "File %s already exist. Data will be not extracted.",
                            result_file_name,
                        )
                        continue

                result.to_csv(output_path, index=False)

    def _obtain_id_from_file_name(self, file_name: str) -> str:
        """
        Different files have different names. There are 3 cases:
            YYYY-mm-dd-ID
            YYYY-mm-dd-ID.json
            ID.json
        That code snippet allow to obtain sensor id from file name.
        There is no sensor id inside of file.
        """
        file_name = file_name.split("-")[-1]
        return file_name.split(".")[0]

    def _normalize_df(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data from nested json to more usefull form.
        """
        df = data.copy()
        columns = [
            column
            for column in (
                sensor_json_ns.TILL_DATE_TIME,
                sensor_json_ns.INDEXES,
                sensor_json_ns.STANDARDS,
            )
            if column in df.columns
        ]
        if columns:
            df = df.drop(columns=columns)

        left_df = (
            df.drop(columns=sensor_json_ns.VALUES)
            if sensor_json_ns.VALUES in df.columns
            else df
        )
        right_df = df[sensor_json_ns.VALUES].apply(
            lambda values: pd.Series(values, dtype="object")
        )

        df = pd.concat(
            (left_df, right_df),
            axis=1,
        )

        columns = df.columns
        result = pd.DataFrame()

        for column in columns:
            series = df[column].apply(pd.Series)
            series = series.rename(columns={sensor_json_ns.VALUE: series.iloc[0, 0]})
            result = pd.concat([result, series], axis=1)

        columns = result.columns.to_list()
        columns[0] = column_names_ns.TIME
        result.columns = columns

        if sensor_json_ns.NAME in result.columns:
            result = result.drop(columns=[sensor_json_ns.NAME])

        if 0 in result.columns:
            result = result.drop(columns=[0])

        return result


class WeatherDataExtractionTask(Task):
    """
    Responsible for load weather data from json files and store results in csv files.
    As the result file per sensor is created or data is appended.
    """

    def __init__(
        self,
        input_dir: Iterable,
        result_dir: str,
        lat_lon_file_path: str,
        mode: ExtractionMode = "append",
    ):
        """
        Parameters
        ----------
        input_dir : Iterable
            Iterable of directories that contains data in json files.
            It will process them in order.
        result_dir : str
            Directory, where data should be stored in csv files
        lat_lon_file_path : str
            Path to file with lattitude and longtitude of sensors. The id column is mandatory.
        mode : ExtractionMode, default='append'
            If 'append', Extractor will append data to existing files in result_dir.
            If 'overwrite', Extractor will overwrite files in result_dir, if exist.
            If None, Extractor will not extract if file exists.
        """
        super().__init__()
        if isinstance(input_dir, str):
            input_dir = (input_dir,)
        self.input_dir = input_dir
        self.result_dir = result_dir
        self.lat_lon_file_path = lat_lon_file_path
        self.mode = mode

    def _run(self):
        """
        Obtain data from json files and store them in result_dir.
        """
        sensors = pd.read_csv(self.lat_lon_file_path)
        for directory in self.input_dir:
            if not os.path.isdir(directory):
                logging.warning("%s doesn't exist", directory)
                continue

            for file_name in os.listdir(directory):
                try:
                    df = pd.read_json(os.path.join(directory, file_name))
                except OSError:
                    logging.warning(
                        "File is corrupted, empty or doesn't exist: %s", file_name
                    )
                    continue

                sensor_id = self._obtain_id_from_file_name(
                    file_name=file_name, sensors=sensors
                )

                df = df[weather_json_ns.HOURLY]
                result = pd.DataFrame(columns=df.index)
                for name in df.index:
                    result[name] = df.loc[name]

                loaded_data = pd.DataFrame()
                result_file_name = f"{sensor_id}.csv"
                output_path = os.path.join(self.result_dir, result_file_name)

                if result_file_name in os.listdir(self.result_dir):
                    if self.mode == "append":
                        logging.info("Append data to file: %s", result_file_name)
                        loaded_data = pd.read_csv(output_path)
                        result = pd.concat([loaded_data, result], ignore_index=True)
                        result = result.drop_duplicates(
                            subset=column_names_ns.TIME, keep="last"
                        )
                        result = result.reset_index(drop=True)
                    elif self.mode == "overwrite":
                        logging.info("Overwrite file: %s", result_file_name)
                    else:
                        logging.warning(
                            "File %s already exist. Data will be not extracted.",
                            result_file_name,
                        )
                        continue

                result.to_csv(output_path, index=False)

    def _obtain_id_from_file_name(self, file_name: str, sensors: pd.DataFrame) -> str:
        """
        Different files have different names. There are 2 cases:
            YYYY-mm-dd-<index-in-sensors-file>.json
            id.0.json
        That code snippet allow to obtain sensor id from file name.
        There is no sensor id inside of file.
        """
        if "-" in file_name:
            index = file_name.split("-")[-1].split(".")[0]
            return sensors[column_names_ns.ID].iloc[int(index)]
        return file_name.split(".")[0]
