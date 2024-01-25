"""Module with classes for downloading data from external sources."""

import json
import logging
import os
import requests

import pandas as pd

from app.data_managers.namespaces import column_names_ns
from app.data_managers.types import DownloadMode
from app.utils.task import Task


class DownloadSensorDataTask(Task):
    """
    Download sensors data from AIRLY rest API
    """

    def __init__(
        self,
        id_file_path: str,
        result_dir: str,
        api_key: str,
        mode: DownloadMode = None,
        timeout: int = 10,
    ):
        """
        Parameters
        ----------
        id_file_path : str
            Path to file with airly ids of sensors in id column.
        result_dir : str
            Dir where sensor data will be saved (one file for one sensor, result_dir is dir).
        api_key : str
            Unique api_key to authorize yourself in AIRLY rest API.
        mode : DownloadMode, default=None
            If None, Downloader will attempt to create result_dir.
            If this is not possible, it will not download data.
            If 'overwrite', Downloader will overwrite files in result_dir, if exist.
        timeout : int, default=10
            Timeout for request.
        """
        super().__init__()
        self.id_file_path = id_file_path
        self.result_dir = result_dir
        self.api_key = api_key
        self.mode = mode
        self.timeout = timeout

    def _run(self):
        """
        Download sensor data from last 24 hours.
        It uses AIRLY rest API.
        """
        ids = (
            pd.read_csv(self.id_file_path, names=[column_names_ns.ID])
            .loc[:, column_names_ns.ID]
            .to_numpy()
        )
        headers = {"Accept": "application/json", "apikey": self.api_key}

        if _create_result_dir(self.result_dir, self.mode):
            return

        for identifier in ids:
            response = requests.get(
                f"https://airapi.airly.eu/v2/measurements/installation?installationId={identifier}",
                headers=headers,
                timeout=self.timeout,
            )
            data = response.json()

            with open(
                os.path.join(self.result_dir, f"{identifier}.json"),
                "w",
                encoding="utf-8",
            ) as outfile:
                json.dump(data, outfile)


class DownloadMeteoDataTask(Task):
    """
    Download meteo data.
    """

    def __init__(
        self,
        lat_lon_file_path: str,
        result_dir: str,
        mode: DownloadMode = None,
        past_days: int = 3,
        forecast_days: int = 3,
        timeout: int = 10,
    ):
        """
        Parameters
        ----------
        lat_lon_file_path : str
            Path to file with lattitude and longtitude of sensors. The id column is mandatory.
        result_dir : str
            Dir where sensor data will be saved (one file for one sensor, result_dir is dir).
        mode : DownloadMode, optional
            If None, Downloader will attempt to create result_dir.
            If this is not possible, it will not download data.
            If 'overwrite', Downloader will overwrite files in result_dir, if exist.
        past_days : int, default=3
            Number of past days to download.
        forecast_days : int, default=3
            Number of forecast days to download.
        timeout : int, default=10
            Timeout for request.
        """
        super().__init__()
        self.lat_lon_file_path = lat_lon_file_path
        self.result_dir = result_dir
        self.mode = mode
        self.past_days = past_days
        self.forecast_days = forecast_days
        self.timeout = timeout

    def _run(self):
        """
        Download meteo data from Open Meteo API.
        """
        installations_points = pd.read_csv(self.lat_lon_file_path)

        if _create_result_dir(self.result_dir, self.mode):
            return

        for _, row in installations_points.iterrows():
            url_part_1 = (
                f"https://api.open-meteo.com/v1/forecast?latitude={row[column_names_ns.LAT]}"
                f"&longitude={row[column_names_ns.LON]}"
                "&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,"
                "precipitation_probability,precipitation,rain,showers,snowfall,snow_depth,"
                "freezinglevel_height,weathercode,pressure_msl,surface_pressure,cloudcover,cloudcover_low,"
                "cloudcover_mid,cloudcover_high,visibility,evapotranspiration,et0_fao_evapotranspiration,"
                "vapor_pressure_deficit,cape,windspeed_10m,windspeed_80m,windspeed_120m,windspeed_180m,"
                "winddirection_10m,winddirection_80m,winddirection_120m,winddirection_180m,windgusts_10m,"
                "temperature_80m,temperature_120m,temperature_180m,soil_temperature_0cm,soil_temperature_6cm,"
                "soil_temperature_18cm,soil_temperature_54cm,soil_moisture_0_1cm,soil_moisture_1_3cm,soil_moisture_3_9cm,"
                "soil_moisture_9_27cm,soil_moisture_27_81cm,shortwave_radiation,direct_radiation,diffuse_radiation,"
                "direct_normal_irradiance,terrestrial_radiation,shortwave_radiation_instant,direct_radiation_instant,"
                "diffuse_radiation_instant,direct_normal_irradiance_instant,terrestrial_radiation_instant&"
            )
            url_part_2 = (
                f"past_days={self.past_days}&forecast_days={self.forecast_days}"
            )
            url = url_part_1 + url_part_2
            response = requests.get(url, timeout=self.timeout)
            data = response.json()
            with open(
                os.path.join(self.result_dir, f"{row[column_names_ns.ID]}.json"),
                "w",
                encoding="utf-8",
            ) as outfile:
                json.dump(data, outfile)


def _create_result_dir(result_dir: str, mode: DownloadMode = None) -> int:
    """
    Create result_dir if not exist.

    Parameters
    ----------
    result_dir : str
        Dir where sensor data will be saved (one file for one sensor, result_dir is dir).
    mode : DownloadMode, default=None
        If None, Downloader will attempt to create result_dir.
        If this is not possible, it will not download data.
        If 'overwrite', Downloader will overwrite files in result_dir, if exist.
    """
    if os.path.exists(result_dir):
        if mode != "overwrite":
            logging.warning(
                "Dir %s exists. Cannot download data to this directory without 'overwrite' mode",
                result_dir,
            )
            return 1
        logging.info("Dir %s exists. Overwriting files inside.", result_dir)
    else:
        logging.info("Dir %s does not exist. Creating.", result_dir)
        os.makedirs(result_dir)

    return 0
