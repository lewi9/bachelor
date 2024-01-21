"""Main script for the project."""

import logging
import os
import pickle
from datetime import date, datetime, timezone
from typing import OrderedDict

import numpy as np
import pandas as pd
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.croston import Croston
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arch import StatsForecastGARCH
from sktime.forecasting.varmax import VARMAX
from sktime.split import ExpandingWindowSplitter, SlidingWindowSplitter
from sktime.transformations.series.outlier_detection import HampelFilter

from app.data_managers.download_tasks import (
    DownloadMeteoDataTask,
    DownloadSensorDataTask,
)
from app.data_managers.task_etl_pipeline import TaskETLPipeline
from app.data_managers.extraction_tasks import (
    SensorDataExtractionTask,
    WeatherDataExtractionTask,
)
from app.data_managers.namespaces import data_ns
from app.data_managers.transformation_tasks import (
    CreateOneDatasetTask,
    CreateStandarizedWideFormatTask,
)

# from app.interpolation.distance_matrix_task import DistanceMatrixTask
# from app.interpolation.task_interpolation_pipeline import TaskInterpolationPipeline

from app.modelling.average_selector import AverageSelector
from app.modelling.best_selector import BestSelector
from app.modelling.evaluator import Evaluator
from app.modelling.forecasting_pipeline import ForecastingPipeline
from app.modelling.metrics import mae
from app.modelling.task_modelling_pipeline import TaskModellingPipeline
from app.modelling.transformers import (
    CompletnessFilter,
    DormantFilter,
    ImputerBackTime,
    ImputerPandas,
    NanDropper,
)
from app.utils.process_pipeline import ProcessPipeline

DOWNLOAD_DATA = 0
EXCTRACT_DATA = 0
TRANSFORM_DATA = 0

FORECASTING_PIPELINE = 0
EVALUATOR = 0
SELECTOR = 0

# Disabled for now
CREATE_DISTANCE_MATRIX = 0

FULL_EXTRACTION = 0
SKIP_CREATING_ONE_DATA_FILE = 0

FORECAST_PERIOD = 48
FORECASTING_PIPELINE_MODE = "recreate"

# Disabled for now
DISTANCE_METRICS = ("euclidean",)

# SETUP
logging.getLogger().setLevel(logging.INFO)
today = date.today()
now = datetime.now(tz=timezone.utc) - pd.DateOffset(hours=1)
now_str = now.strftime("%Y-%m-%d %H")

logging.info("Started program: %s %s", today, now_str)

with open("api_key.pkl", "rb") as f:
    api_key = pickle.load(f)
    logging.info("API key loaded: %s", api_key)


etl_tasks = OrderedDict()

if DOWNLOAD_DATA:
    etl_tasks |= {
        "get_sensor_data": DownloadSensorDataTask(
            id_file_path=data_ns.ID_FILE_PATH,
            result_dir=f"data/{today}",
            api_key=api_key,
            mode=None,
            timeout=10,
        ),
        "get_meteo_data": DownloadMeteoDataTask(
            lat_lon_file_path=data_ns.LAT_LON_FILE_PATH,
            result_dir=f"weather/{today}",
            mode=None,
            past_days=3,
            forecast_days=3,
            timeout=10,
        ),
    }

if EXCTRACT_DATA:
    input_dir_sensor = (
        f"data/{today}"
        if not FULL_EXTRACTION
        else map(lambda x: "data/" + x, os.listdir("data"))
    )
    input_dir_weather = (
        f"weather/{today}"
        if not FULL_EXTRACTION
        else map(lambda x: "weather/" + x, os.listdir("weather"))
    )

    etl_tasks |= {
        "extract_sensor_data": SensorDataExtractionTask(
            input_dir=input_dir_sensor,
            result_dir=data_ns.SENSOR_DATA_DIR,
            mode="append",
        ),
        "extract_weather_data": WeatherDataExtractionTask(
            input_dir=input_dir_weather,
            result_dir=data_ns.WEATHER_DATA_DIR,
            lat_lon_file_path=data_ns.LAT_LON_FILE_PATH,
            mode="append",
        ),
    }


if TRANSFORM_DATA:
    if not SKIP_CREATING_ONE_DATA_FILE:
        etl_tasks |= {
            "create_one_data_file": CreateOneDatasetTask(
                weather_data_dir=data_ns.WEATHER_DATA_DIR,
                sensor_data_dir=data_ns.SENSOR_DATA_DIR,
                output_file_path=data_ns.ONE_DATA_FILE,
            )
        }

    etl_tasks |= {
        "create_standarized_wide_format": CreateStandarizedWideFormatTask(
            one_data_file_path=data_ns.ONE_DATA_FILE,
            features_path=data_ns.FEATURES_PATH,
            result_dir=data_ns.TRANSFORMED_DATA_DIR,
            min_time="2023-02-27 00",
            max_time=now_str,
            forecast_period=FORECAST_PERIOD,
        )
    }


modelling_tasks = OrderedDict()

# Forecasting pipeline
if FORECASTING_PIPELINE:
    transformers = [
        DormantFilter(period=FORECAST_PERIOD + 48),
        CompletnessFilter(0.5),
        ImputerBackTime(period_to_take_value_from=24),
        HampelFilter(window_length=72),
        ImputerPandas(method="linear"),
        NanDropper(),
    ]
    forecasters = {
        # "ARIMA": ARIMA(),
        "CROSTON_0.1": Croston(smoothing=0.1),
        "CROSTON_0.2": Croston(smoothing=0.2),
        "CROSTON_0.5": Croston(smoothing=0.5),
        "CROSTON_0.8": Croston(smoothing=0.8),
        "NAIVE_DRIFT_24H": NaiveForecaster(strategy="drift", window_length=24, sp=24),
        "NAIVE_DRIFT_FORECAST_PERIOD": NaiveForecaster(
            strategy="drift", sp=FORECAST_PERIOD, window_length=FORECAST_PERIOD
        ),
        "NAIVE_LAST_24H": NaiveForecaster(strategy="last", window_length=24, sp=24),
        "NAIVE_LAST_FORECAST_PERIOD": NaiveForecaster(
            strategy="last", sp=FORECAST_PERIOD
        ),
        "NAIVE_MEAN_24H": NaiveForecaster(strategy="mean", window_length=24, sp=24),
        "NAIVE_MEAN_FORECAST_PERIOD": NaiveForecaster(
            strategy="mean", sp=FORECAST_PERIOD
        ),
        "NAIVE_MEAN_WEEK": NaiveForecaster(strategy="mean", window_length=168, sp=168),
        "StatsForecastGARCH": StatsForecastGARCH(),
        "COMPLEX_GARCH": StatsForecastGARCH(p=5, q=5),
        "VARMAX": VARMAX(low_memory=True, suppress_warnings=True),
    }
    splitters = {
        "ExpandingWindowSplitter": ExpandingWindowSplitter(
            fh=np.arange(1, FORECAST_PERIOD + 1), step_length=FORECAST_PERIOD
        ),
        "SlidingWindowSplitter180D": SlidingWindowSplitter(
            fh=np.arange(1, FORECAST_PERIOD + 1),
            step_length=FORECAST_PERIOD,
            window_length=24 * 180,
        ),
    }

    modelling_tasks |= {
        "Forecasting Pipeline": ForecastingPipeline(
            input_dir=data_ns.TRANSFORMED_DATA_DIR,
            result_dir=data_ns.FORECAST_RESULT_DIR,
            transformers=transformers,
            forecasters=forecasters,
            splitters=splitters,
            max_forecasts=6,
            exo_filler="mean",
            mode=FORECASTING_PIPELINE_MODE,
        )
    }

# Evaluate results
"""
if EVALUATOR:
    modelling_tasks |= {
        "Evaluator": Evaluator(
            forecast_data_dir=data_ns.FORECAST_RESULT_DIR,
            transformed_data_dir=data_ns.TRANSFORMED_DATA_DIR,
            result_path=data_ns.EVALUATION_PATH,
            metric=mae,
        )
    }

# Select best model for each sensor
if SELECTOR:
    modelling_tasks |= {
        "BestSelector": BestSelector(
            forecast_data_dir=data_ns.FORECAST_RESULT_DIR,
            evaluation_path=data_ns.EVALUATION_PATH,
            result_path=data_ns.SELECTION_FILE,
            min_date=now_str,
            max_date=None,
        )
    }
"""

if SELECTOR:
    modelling_tasks |= {
        "AverageSelector": AverageSelector(
            forecast_data_dir=data_ns.FORECAST_RESULT_DIR,
            result_path=data_ns.SELECTION_FILE,
            min_date=now_str,
            max_date=None,
        )
    }


# TODO: ...
# interpolation_tasks = OrderedDict()

# if CREATE_DISTANCE_MATRIX:
#    interpolation_tasks |= {
#        "Create Distance Matrix": DistanceMatrixTask(
#            grid_path=data_ns.GRID_FILE_PATH,
#            stations_path=data_ns.LAT_LON_FILE_PATH,
#            grid_epsg=2180,
#            output_path=data_ns.DISTANCE_MATRIX_GRID_FILE,
#            metrics=DISTANCE_METRICS,
#        )
#    }


# Publishing tasks


pipelines = OrderedDict()
pipelines |= {
    "ETL Pipeline": TaskETLPipeline(tasks=etl_tasks),
    "Modelling Pipeline": TaskModellingPipeline(tasks=modelling_tasks),
    #    "Interpolation Pipeline": TaskInterpolationPipeline(tasks=interpolation_tasks),
}

process_pipeline = ProcessPipeline(pipelines=pipelines)
process_pipeline.run()


logging.info(
    "Each task execution time: %s", process_pipeline.get_tasks_execution_time()
)
logging.info("Total execution time: %s", process_pipeline.execution_time)
