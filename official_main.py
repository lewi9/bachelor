"""Main script for the project."""

import logging
import os
import pickle
from datetime import date, datetime, timezone
from typing import OrderedDict

import numpy as np
import pandas as pd
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.croston import Croston
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arch import StatsForecastGARCH
from sktime.transformations.series.outlier_detection import HampelFilter

from app.data_managers.download_tasks import (
    DownloadMeteoDataTask,
    DownloadSensorDataTask,
)
from app.data_managers.extraction_tasks import (
    SensorDataExtractionTask,
    WeatherDataExtractionTask,
)
from app.data_managers.namespaces import data_ns
from app.data_managers.transformation_tasks import TransformSensorMeteoDatasetsTask

from app.modelling.agg_model_strategy_task import AggModelStrategyTask
from app.modelling.best_model_strategy_task import BestModelStrategyTask
from app.modelling.forecasting_pipeline_task import ForecastingPipelineTask
from app.modelling.metrics import mae, maxae, rmse
from app.modelling.models import LinReg, RandomForrest, RegressionTree
from app.modelling.splitters import ExpandingWindowSplitter, SlidingWindowSplitter
from app.modelling.transformers import (
    CompletnessFilter,
    DormantFilter,
    ImputerBackTime,
    ImputerPandas,
    NanDropper,
)
from app.utils.evaluator import Evaluator
from app.utils.process_pipeline import ProcessPipeline
from app.utils.task_pipeline import TaskPipeline

DOWNLOAD_DATA = 1
EXCTRACT_DATA = 1
TRANSFORM_DATA = 0

FORECASTING_PIPELINE = 0
BEST_MODEL_STRATEGY = 0
AGG_MODEL_STRATEGY = 0

PICKLE_EVALUATION = 1

FULL_EXTRACTION = 0

FORECAST_PERIOD = 48
FORECASTING_PIPELINE_MODE = "recreate"

# SETUP
logging.getLogger().setLevel(logging.INFO)

today = date.today()
now = datetime.now(tz=timezone.utc) - pd.DateOffset(hours=1)
now_with_forecast_period = now + pd.DateOffset(hours=FORECAST_PERIOD)

now_str = now.strftime("%Y-%m-%d %H")
now_with_forecast_period_str = now_with_forecast_period.strftime("%Y-%m-%d %H")

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
    etl_tasks |= {
        "transform_data": TransformSensorMeteoDatasetsTask(
            sensor_data_dir=data_ns.SENSOR_DATA_DIR,
            weather_data_dir=data_ns.WEATHER_DATA_DIR,
            result_dir=data_ns.TRANSFORMED_DATA_DIR,
            features_path=data_ns.FEATURES_PATH,
            min_time="2023-03-20 00",
            max_time=now_with_forecast_period_str,
        )
    }


modelling_tasks = OrderedDict()

forecast_horizon = np.arange(1, FORECAST_PERIOD + 1)

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
        # "PROPHET": Prophet(),
        # "PROPHET_DEFAULT": Prophet(
        #    freq="H",
        #    add_seasonality=True,
        #    daily_seasonality=True,
        #    weekly_seasonality=True,
        #    yearly_seasonality=True,
        # ),
        # "PROPHET_MULT_SEAS": Prophet(
        #    freq="H",
        #    add_seasonality=True,
        #    daily_seasonality=True,
        #    yearly_seasonality=True,
        #    seasonality_mode="multiplicative",
        # ),
        # "ARIMA_1": ARIMA(),
        # "ARIMA_2": ARIMA((1, 1, 0)),
        # "ARIMA_3": ARIMA((1, 1, 1)),
        # "SARIMAX_Y": SARIMAX((1, 1, 1), (1, 1, 1, 24 * 365)),
        # "SARIMAX_D": SARIMAX((1, 0, 0), (1, 0, 0, 24)),
        # "SARIMAX_D1": SARIMAX((1, 1, 1), (1, 1, 1, 24)),
        # "SARIMAX_D2": SARIMAX((1, 1, 2), (1, 1, 2, 24)),
        # "SARIMAX_D3": SARIMAX((1, 0, 1), (1, 0, 1, 24)),
        # "SARIMAX_D4": SARIMAX((2, 1, 1), (2, 1, 1, 24)),
        "CROSTON_0.01": Croston(smoothing=0.01),
        "CROSTON_0.1": Croston(smoothing=0.1),
        "CROSTON_0.2": Croston(smoothing=0.2),
        "CROSTON_0.5": Croston(smoothing=0.5),
        "CROSTON_0.8": Croston(smoothing=0.8),
        "NAIVE_DRIFT": NaiveForecaster(strategy="drift"),
        "NAIVE_LAST_24H": NaiveForecaster(strategy="last", sp=24),
        "NAIVE_LAST_48H": NaiveForecaster(strategy="last", sp=48),
        "NAIVE_MEAN_30D_SP24H": NaiveForecaster(
            strategy="mean", window_length=30 * 24, sp=24
        ),
        "NAIVE_MEAN_60D_SP24H": NaiveForecaster(
            strategy="mean", window_length=60 * 24, sp=24
        ),
        "NAIVE_MEAN_14D_SP24H": NaiveForecaster(
            strategy="mean", window_length=14 * 24, sp=24
        ),
        "NAIVE_MEAN_7D_SP24H": NaiveForecaster(
            strategy="mean", window_length=7 * 24, sp=24
        ),
        "NAIVE_MEAN_3D_SP24H": NaiveForecaster(
            strategy="mean", window_length=3 * 24, sp=24
        ),
        "NAIVE_MEAN_30D": NaiveForecaster(strategy="mean", window_length=30 * 24),
        "NAIVE_MEAN_14D": NaiveForecaster(strategy="mean", window_length=14 * 24),
        "NAIVE_MEAN_7D": NaiveForecaster(strategy="mean", window_length=7 * 24),
        "NAIVE_MEAN_3D": NaiveForecaster(strategy="mean", window_length=3 * 24),
        "StatsForecastGARCH": StatsForecastGARCH(),
        "COMPLEX_GARCH": StatsForecastGARCH(p=5, q=5),
        "LinReg": LinReg(),
        "RandomForrest": RandomForrest(),
        "RegressionTree": RegressionTree(),
    }

    splitters = {
        "ExpandingWindowSplitter": ExpandingWindowSplitter(
            fh=forecast_horizon, step_length=FORECAST_PERIOD
        ),
        "SlidingWindowSplitter": SlidingWindowSplitter(
            fh=forecast_horizon,
            step_length=FORECAST_PERIOD,
            window_length=FORECAST_PERIOD * 60,
        ),
    }

    modelling_tasks |= {
        "Forecasting Pipeline": ForecastingPipelineTask(
            input_dir=data_ns.TRANSFORMED_DATA_DIR,
            result_dir=data_ns.FORECAST_RESULT_DIR,
            max_forecasts=3,
            transformers=transformers,
            forecasters=forecasters,
            splitters=splitters,
            last_valid_actual="2024-01-23 15",
            exo_filler="mean",
            mode=FORECASTING_PIPELINE_MODE,
            parallel_batch=20,
        )
    }

if BEST_MODEL_STRATEGY:
    modelling_tasks |= {
        "Best Model Strategy": BestModelStrategyTask(
            transformed_data_dir=data_ns.TRANSFORMED_DATA_DIR,
            forecast_dir=data_ns.FORECAST_RESULT_DIR,
            result_dir=data_ns.SELECTED_DATA_DIR,
            metric=rmse,
            last_evaluation_date="2024-01-23 15",
            mode="recreate",
        )
    }

if not BEST_MODEL_STRATEGY and AGG_MODEL_STRATEGY:
    modelling_tasks |= {
        "Agg Model Strategy": AggModelStrategyTask(
            forecast_dir=data_ns.FORECAST_RESULT_DIR,
            result_dir=data_ns.SELECTED_DATA_DIR,
            aggregation_function="mean",
            mode="recreate",
        )
    }

pipelines = OrderedDict()
pipelines |= {
    "ETL Pipeline": TaskPipeline(tasks=etl_tasks),
    "Modelling Pipeline": TaskPipeline(tasks=modelling_tasks),
}

process_pipeline = ProcessPipeline(pipelines=pipelines)
process_pipeline.run()


logging.info(
    "Each task execution time: %s", process_pipeline.get_tasks_execution_time()
)
logging.info("Total execution time: %s", process_pipeline.execution_time)

ev = Evaluator(
    selected_data_dir=data_ns.SELECTED_DATA_DIR,
    reference_data_dir=data_ns.TRANSFORMED_DATA_DIR,
    metrics=[mae, rmse, maxae],
)

results = ev.evaluate()
results["execution_time"] = process_pipeline.execution_time.seconds

logging.info("Results: %s", results)

if PICKLE_EVALUATION:
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
        logging.info("Results saved to file: results.pkl")
