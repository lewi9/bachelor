"""Tuner class for tuning the system."""
from __future__ import annotations
import os
from typing import Callable, Iterable, Optional, OrderedDict, Sequence, Union

from pymoo.core.problem import Problem
from sktime.forecasting.base import BaseForecaster
from sktime.split.base import BaseWindowSplitter
from sktime.transformations.base import BaseTransformer

from app.data_managers.namespaces import data_ns
from app.data_managers.transformation_tasks import (
    CreateOneDatasetTask,
    CreateStandarizedWideFormatTask,
)
from app.data_managers.task_etl_pipeline import TaskETLPipeline
from app.modelling.average_selector import AverageSelector
from app.modelling.best_selector import BestSelector
from app.modelling.evaluator import Evaluator
from app.modelling.forecasting_pipeline import ForecastingPipeline
from app.modelling.task_modelling_pipeline import TaskModellingPipeline
from app.modelling.transformers import CustomTransformer, NanDropper


class FullTuner:
    """Full tuner class for tuning the whole system."""

    def __init__(
        self,
        models: OrderedDict[str, BaseForecaster],
        splitters: OrderedDict[str, BaseWindowSplitter],
        transformer_setups: OrderedDict[
            str, list[Union[BaseTransformer, CustomTransformer]]
        ],
        forecast_period: int,
        validation_metrics: OrderedDict[str, Callable],
        n_repeats: int = 3,
        internal_metrics: Optional[OrderedDict[str, Callable]] = None,
        max_forecasts: int = 1,
        validation_dates: Optional[Iterable[str]] = None,
    ):
        """
        Initialize the tuner.

        Parameters
        ----------
        models : dict[str, BaseForecaster]
            Dictionary with models to select from. Based on them tuner will select
            the models that will be used in the system.
        splitters : dict[str, BaseWindowSplitter]
            Dictionary with splitters to select from. Based on them tuner will select
            the splitters that will be used in the system.
        internal_metrics : dict[str, Callable], default=None
            If specified, the tuner will include "best model" strategy.
            The internal_metrics will be used to evaluate the models to selecte the best one
            according to the metric.
            If None, the "best_model" strategy will be not used, by default it is None.
        transformer_setups : dict[str, list[Union[BaseTransformer, CustomTransformer]]]
            Dictionary with list of transformers. The best configuration will be selected
            for the system. The constant part of transformers - activity filters should be
            provided in the set_activity_filters() method. The NanDropper will be added
            automatically at the end of the list.
        forecast_period : int
            The parameter is used to create the forecast horizon.
            It allows to reduce the search space. It also allows to regulate the time of system work.
            The output of the system will be the forecast for the next forecast_period.
            The forecast_period should be doubled value of time beetwen scheduled system retrainings.
        max_forecasts : int, default=1
            The maximum number of forecasts that will be created by the system.
            The parameter will be used if any internal metric is used.
            In the other case, the parameter will be equal to 1.
            It allows to reduce the search space. It also allows to regulate the time of system work.
        validation_metrics : dict[str, Callable]
            Dictionary with metrics that will be used to evaluate the performance of the system.
            Additional metric - time - will be added automatically.
        n_repeats : int, default=3
            Number of repeats - how many pipelines will be created with the same configuration to evaluate them.
            The tuner will select the best configuration based on the average performance of pipelines.
            Parameter is ignored if validation_dates is not None.
        validation_dates : Optional[Iterable[str]], default=None
            Dates that will be used to evaluate the system.
            The dates should be in the format "YYYY-MM-DD HH".
            If None, they will be created based on n_repeats parameter.
            The dataset will be divided into n_repeats parts.
            The last date of each part will be used as validation date.
        """
        self.models = models
        self.splitters = splitters

        self.internal_metrics = internal_metrics
        self.max_forecasts = max_forecasts if internal_metrics is not None else 1
        if internal_metrics and max_forecasts == 1:
            raise ValueError(
                "Parameter max_forecasts must be greater than 1 if internal_metrics is not None."
            )

        self.forecast_period = forecast_period
        self.n_repeats = n_repeats
        self.transformer_setups = transformer_setups
        self.validation_metrics = validation_metrics
        self.validation_dates = validation_dates

        self.features_path = None
        self.sensor_data_dir = None
        self.weather_data_dir = None
        self.temp_dir = None
        self.paths_flag = 0

        self.minimum_time = None
        self.minimum_time_flag = 0

        self.activity_filters = None
        self.activity_filters_flag = 0

    def generate_features_file(self, data: str, output: str) -> None:
        """
        TODO: support function for generating features file.
        Generate file with features. They will be used as exogenous variables.

        Parameters
        ----------
        data : str
            Path to the data file.
        output : str
            Path to the output file.
        """
        # Run create one dataset task.
        # Generate features file.
        raise NotImplementedError

    def set_file_paths(
        self,
        features_path: str,
        sensor_data_dir: str,
        weather_data_dir: str,
        temp_dir: str,
    ) -> None:
        """
        Set file paths that will be used in the system tuning.
        It is necessary to set them before tuning.

        Parameters
        ----------
        features_path : str
            Path to the features file.
        sensor_data_dir : str
            Path to the sensor data directory.
        weather_data_dir : str
            Path to the weather data directory.
        temp_dir : str
            Path to the temporary directory.
        """
        self.features_path = features_path
        self.sensor_data_dir = sensor_data_dir
        self.weather_data_dir = weather_data_dir
        self.temp_dir = temp_dir
        self.paths_flag = 1

    def set_minimum_time(self, minimum_time: str) -> None:
        """
        Set minimum time that will be used in the system tuning.
        This date indicated the beginning of the dataset.

        Parameters
        ----------
        minimum_time : str
            Minimum time that will be used in the system tuning.
            The date should be in the format "YYYY-MM-DD HH".
        """
        self.minimum_time = minimum_time
        self.minimum_time_flag = 1

    def set_activity_filters(self, activity_filter: Sequence) -> None:
        """
        Set activity filters that will be used in the system tuning.
        The filter will be applied at the beginning of forecasting pipeline.
        E.g DormantFilter, CompletenessFilter.

        Parameters
        ----------
        activity_filters : Sequence
            Activity filters that will be used in the system tuning.
        """
        self.activity_filters = activity_filter
        self.activity_filters_flag = 1

    def tune(self) -> None:
        """Tune the system."""

        if self.paths_flag == 0:
            raise ValueError(
                "File paths are not set." "Use the set_file_paths() method to set them."
            )

        if self.minimum_time_flag == 0:
            raise ValueError(
                "Minimum time is not set."
                "Use the set_minimum_time() method to set it."
            )

        if self.activity_filters_flag == 0:
            raise ValueError(
                "Activity filters are not set."
                "Use the set_activity_filters() method to set them."
            )

    @property
    def _problem(self) -> Problem:
        """
        Create the problem for optimization.
        """
        return self._build_problem()

    def _build_problem(self):
        """Build the problem for optimization."""
        return self.SystemOptimizationProblem(self)

    class SystemOptimizationProblem(Problem):
        """Class for optimization problem. Implements the pymoo Problem class."""

        SELECTOR_STRATEGIES = 1
        ADDITIONAL_PARAMETERS = 3

        def __init__(self, tuner_instance: FullTuner):
            """Initialize the problem based on the tuner instance."""

            self.tuner_instance = tuner_instance

            self.len_models = len(self.tuner_instance.models)
            self.len_splitters = len(self.tuner_instance.splitters)
            self.len_transformers = len(self.tuner_instance.transformer_setups)
            self.len_internal_metrics = len(self.tuner_instance.internal_metrics)
            self.len_metrics = len(self.tuner_instance.validation_metrics)

            n_var = self.len_models + self.len_splitters + self.ADDITIONAL_PARAMETERS
            n_obj = self.len_metrics + 1

            xl = [0] * n_var
            xu = [1] * (n_var - self.ADDITIONAL_PARAMETERS) + [
                self.len_transformers - 1,
                self.len_internal_metrics + self.SELECTOR_STRATEGIES - 1,
                self.tuner_instance.max_forecasts,
            ]

            super().__init__(
                n_var=n_var, n_obj=n_obj, n_ieq_constr=1, xl=xl, xu=xu, vtype=int
            )

        def _evaluate(self, x, out, *args, **kwargs):
            """Evaluate the solution."""
            f = [[] for _ in range(self.len_metrics + 1)]

            for index, solution in enumerate(x):
                for validation_date in self.tuner_instance.validation_dates:
                    models = {
                        model_name: model
                        for model_index, model_name, model in enumerate(
                            self.tuner_instance.models.items()
                        )
                        if solution[model_index]
                    }
                    splitters = {
                        splitter_name: splitter
                        for splitter_index, splitter_name, splitter in enumerate(
                            self.tuner_instance.splitters.items()
                        )
                        if solution[self.len_models + splitter_index]
                    }
                    transformer_setup = self.tuner_instance.transformer_setups[
                        solution[self.len_models + self.len_splitters]
                    ]

                    strategy = solution[self.len_models + self.len_splitters + 1]
                    n_forecasts = solution[self.len_models + self.len_splitters + 2]

                    pipeline_dict = self._get_pipeline_dict(
                        index=index,
                        models=models,
                        splitters=splitters,
                        transformer_setup=transformer_setup,
                        strategy=strategy,
                        n_forecasts=n_forecasts,
                        validation_date=validation_date,
                    )

                    # Add pipeline evaluation to pipeline dict

                    # Collect the data and insert them to f array

                    # Constraint - the time

        def _get_pipeline_dict(
            self,
            index: int,
            models: dict,
            splitters: dict,
            transformer_setup: dict,
            strategy: int,
            n_forecasts: int,
            validation_date: str,
        ) -> dict:
            """Construct the pipeline dictionary."""
            one_data_file_path = os.path.join(
                self.tuner_instance.temp_dir, f"{index}_{data_ns.ONE_DATA_FILE}"
            )
            transformed_data_dir = os.path.join(
                self.tuner_instance.temp_dir, f"{index}_{data_ns.RESULT_DIR}"
            )
            forecast_dir = os.path.join(
                self.tuner_instance.temp_dir, f"{index}_{data_ns.FORECAST_RESULT_DIR}"
            )
            evaluation_path = os.path.join(
                self.tuner_instance.temp_dir, f"{index}_{data_ns.EVALUATION_PATH}"
            )
            selection_path = os.path.join(
                self.tuner_instance.temp_dir, f"{index}_{data_ns.SELECTION_FILE}"
            )

            etl_tasks = OrderedDict()

            etl_tasks |= {
                "create_one_data_file": CreateOneDatasetTask(
                    weather_data_dir=self.tuner_instance.weather_data_dir,
                    sensor_data_dir=self.tuner_instance.sensor_data_dir,
                    output_file_path=one_data_file_path,
                )
            }

            etl_tasks |= {
                "create_standarized_wide_format": CreateStandarizedWideFormatTask(
                    one_data_file_path=one_data_file_path,
                    features_path=self.tuner_instance.features_path,
                    result_dir=transformed_data_dir,
                    min_time=self.tuner_instance.minimum_time,
                    max_time=validation_date,
                    forecast_period=self.tuner_instance.forecast_period,
                )
            }

            modelling_tasks = OrderedDict()

            transformers = (
                self.tuner_instance.activity_filters
                + list(transformer_setup.values())
                + [NanDropper()]
            )

            modelling_tasks |= {
                "Forecasting Pipeline": ForecastingPipeline(
                    input_dir=transformed_data_dir,
                    result_dir=forecast_dir,
                    transformers=transformers,
                    forecasters=models,
                    splitters=splitters,
                    max_forecasts=n_forecasts,
                    exo_filler="mean",
                    mode="recreate",
                )
            }

            if strategy == 0:
                modelling_tasks |= {
                    "Average Selector": AverageSelector(
                        forecast_data_dir=forecast_dir,
                        result_path=selection_path,
                        min_date=validation_date,
                        max_date=None,
                    )
                }

            else:
                modelling_tasks |= {
                    "Evaluator": Evaluator(
                        forecast_data_dir=forecast_dir,
                        transformed_data_dir=transformed_data_dir,
                        result_path=evaluation_path,
                        metric=self.tuner_instance.internal_metrics[
                            strategy - self.SELECTOR_STRATEGIES
                        ],
                    )
                }

                modelling_tasks |= {
                    "BestSelector": BestSelector(
                        forecast_data_dir=forecast_dir,
                        evaluation_path=evaluation_path,
                        result_path=selection_path,
                        min_date=validation_date,
                        max_date=None,
                    )
                }

            return {
                "ETL Pipeline": TaskETLPipeline(tasks=etl_tasks),
                "Modelling Pipeline": TaskModellingPipeline(tasks=modelling_tasks),
            }
