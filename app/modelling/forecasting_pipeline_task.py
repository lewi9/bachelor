"""Module to perform forecasting on data."""

import itertools
import logging
import os
import shutil
import warnings
from typing import Iterable, Sequence, Union

import numpy as np
import pandas as pd
import ray
from sktime.forecasting.base import BaseForecaster
from sktime.transformations.series.outlier_detection import HampelFilter

from app.data_managers.namespaces import column_names_ns, date_formats_ns
from app.data_managers.types import ExoFiller, ForecastingPipelineMode
from app.modelling.splitters import BaseWindowSplitter
from app.utils.task import Task


@ray.remote
def _forecast_data(
    y: pd.Series,
    X: pd.Series,
    forecaster: BaseForecaster,
    forecaster_name: str,
    splitter_name: str,
    fh: Union[int, Sequence[int], np.ndarray[int]],
) -> Union[pd.DataFrame, str]:
    """
    Run forecasting on data.

    Parameters
    ----------
    y : pd.Series
        Endogenous variable.
    X : pd.Series
        Exogenous variable.
    forecaster : BaseForecaster
        Forecaster that will be used to perform forecasting.
    forecaster_name : str
        Name of the forecaster.
    splitter_name : str
        Name of the splitter.
    fh : Union[int, Sequence[int], np.ndarray[int]]
        Forecasting horizon.

    Returns
    -------
    pd.DataFrame | str
        Dataframe with forecasts or error message.
    """
    try:
        forecaster.fit(
            y=y,
            X=X,
        )
    except (ValueError, RuntimeError):
        logging.exception("FORECASTER_FIT_ERROR")
        return "FORECASTER_FIT_ERROR"

    try:
        prediction = forecaster.predict(fh=fh, X=X)
    except (ValueError, RuntimeError):
        logging.exception("FORECASTER_PREDICT_ERROR")
        return "FORECASTER_PREDICT_ERROR"

    dates = pd.date_range(start=y.index[-1], periods=len(fh) + 1, freq="H")[1:]

    result = pd.DataFrame(
        {
            column_names_ns.VALUE: prediction.reset_index(drop=True),
            column_names_ns.TIME: dates,
        }
    )

    result[column_names_ns.FORECASTER] = forecaster_name
    result[column_names_ns.SPLITTER] = splitter_name

    return result


@ray.remote
def _process_data(
    y: pd.Series,
    X: pd.Series,
    transformers: list,
    exo_filler: ExoFiller,
    forecasters: dict[str, BaseForecaster],
    splitter_name: str,
    fh: Union[int, Sequence[int], np.ndarray[int]],
) -> Iterable[Union[pd.DataFrame, str]]:
    """
    Process data to perform forecasting.

    Parameters
    ----------
    y : pd.Series
        Endogenous variable.
    X : pd.Series
        Exogenous variable.
    transformers : list
        List of transformers that will be run on data.
        They will be applied in order from list to endo and exo.
    exo_filler : ExoFiller
        Method to fill exo nans after transformers are applied.
        Only "mean" is supported now.
    forecasters : dict[str, BaseForecaster]
        Dictionary with forecasters. Key is the string with name
        of the forecaster. The value is forecaster.
    splitter_name : str
        Name of splitter that will be used to split data.
    fh : Union[int, Sequence[int], np.ndarray[int]]
        Forecasting horizon.

    Returns
    -------
    Iterable[Union[pd.DataFrame, str]]
        Iterable with forecasts or error messages.
    """
    for transformer in transformers:
        if isinstance(transformer, HampelFilter):
            warnings.filterwarnings("ignore")
        try:
            y = transformer.fit_transform(y)
        except ValueError:
            logging.exception("Y_TRANSFORMER_ERROR")
            return "Y_TRANSFORMER_ERROR"
        try:
            X = X.apply(transformer.fit_transform, axis=0)
        except ValueError:
            logging.exception("X_TRANSFORMER_ERROR")
            return "X_TRANSFORMER_ERROR"
        warnings.filterwarnings("always")

    if exo_filler == "mean":
        for column in X.columns:
            X[column] = X[column].fillna(X[column].mean())

    results_ref = [
        _forecast_data.remote(
            y=y,
            X=X,
            forecaster=forecaster,
            forecaster_name=forecaster_name,
            splitter_name=splitter_name,
            fh=fh,
        )
        for forecaster_name, forecaster in forecasters.items()
    ]
    return ray.get(results_ref)


def _run_remote(
    file_name: str,
    input_dir: str,
    result_dir: str,
    max_forecasts: int,
    transformers: list,
    forecasters: dict[str, BaseForecaster],
    splitters: dict[str, BaseWindowSplitter],
    last_valid_actual: str,
    exo_filler: ExoFiller = "mean",
    **kwargs,
):
    """
    Run pipeline on data to perform forecasts.
    """

    def _read_data(directory: str, file_name: str) -> pd.DataFrame:
        """
        Read data from file and prepare them to forecasting process.

        Parameters
        ----------
        directory : str
            Path to directory with prepared files.
        file_name : str
            Name of file that will be read.

        Returns
        -------
        pd.DataFrame
            Data read from file.
        """
        # Read from file
        data = pd.read_csv(os.path.join(directory, file_name))

        # Change time to datetime
        data[column_names_ns.TIME] = pd.to_datetime(
            data[column_names_ns.TIME],
            format=date_formats_ns.FORECASTING_PIPELINE_DATETIME_FORMAT,
        )

        # Set datetime index and assign Freq
        with_index = data.set_index(column_names_ns.TIME)
        return with_index

    predictions = []
    data = _read_data(directory=input_dir, file_name=file_name)
    dependent_variables = data[column_names_ns.DEPENDENT_VARIABLE_NAME].unique()
    for dependent_variable in dependent_variables:
        to_process = data.query(f"DEPENDENT_VARIABLE_NAME == '{dependent_variable}'")
        to_process = to_process.drop(columns=column_names_ns.DEPENDENT_VARIABLE_NAME)
        X = to_process.drop(columns=column_names_ns.VALUE)
        y = to_process.loc[:last_valid_actual, column_names_ns.VALUE]
        y.index.freq = pd.infer_freq(y.index)
        predicted_ref = [
            _process_data.remote(
                y=y.iloc[index_y],
                X=X.iloc[index_X],
                transformers=transformers,
                exo_filler=exo_filler,
                forecasters=forecasters,
                splitter_name=splitter_name,
                fh=splitter.fh,
            )
            for splitter_name, splitter in splitters.items()
            for index_y, index_X in splitter.split(y=y, forecasts=max_forecasts)
        ]
        predicted = ray.get(predicted_ref)
        chained_predicted = itertools.chain(*predicted)
        filtered_predicted = filter(
            lambda x: isinstance(x, pd.DataFrame), chained_predicted
        )
        predictions.extend(list(filtered_predicted))

    if not predictions:
        logging.warning("File %s cannot be processed", file_name)
        return

    result = pd.concat(predictions, ignore_index=True)
    result.to_csv(os.path.join(result_dir, file_name), index=False)


class ForecastingPipelineTask(Task):
    """
    Pipeline to perform forecasts on prepared data.
    """

    def __init__(
        self,
        input_dir: str,
        result_dir: str,
        max_forecasts: int,
        transformers: list,
        forecasters: dict[str, BaseForecaster],
        splitters: dict[str, BaseWindowSplitter],
        last_valid_actual: str,
        exo_filler: ExoFiller = "mean",
        mode: ForecastingPipelineMode = "recreate",
        parallel_actors: int = 10,
    ):
        """
        Paramaters
        ----------
        input_dir : str
            Path to directory with prepared data files.
        result_dir : str
            Path to directory, where forecasts will be saved.
        max_forecasts : int
            Maximum number of forecasts that will be performed.
        transformers : list
            List of transformers that will be run on data.
            They will be applied in order from list to endo and exo.
        forecasters : dict[str, BaseForecaster]
            Dictionary with forecasters. Key is the string with name
            of the forecaster. The value is forecaster.
        splitters : dict[str, BaseWindowSplitter]
            Dictionary with splitters. Key is the string with name
            of the splitter. The value is splitter.
        last_valid_actual : str
            Last valid actual date. Forecasts will be performed from this date.
            The proper format is "YYYY-MM-DD HH".
        exo_filler : ExoFiller
            Method to fill exo nans after transformers are applied.
            Only "mean" is supported now.
        mode : ForecastingPipelineMode
            Mode to controll how pipeline will be run.
            If "recreate" then all files in result_dir will be deleted.
            If None then pipeline will check if dir exists and skip it if it does.
        parallel_actors : int
            Number of actors that will be used to perform forecasting.
            The cluster will be created with this number of actors.
            After the actors are processed, the cluster will be closed.
            The recreating of cluster is needed to avoid memory leaks.
            I don't know how to choose this number.
        """
        super().__init__(parallel=True)
        self.input_dir = input_dir
        self.result_dir = result_dir
        self.max_forecasts = max_forecasts
        self.transformers = transformers
        self.forecasters = forecasters
        self.splitters = splitters
        self.last_valid_actual = pd.to_datetime(
            last_valid_actual, format=date_formats_ns.LAST_VALID_ACTUAL_TIME_FORMAT
        )
        self.exo_filler = exo_filler
        self.mode = mode
        self.parallel_actors = parallel_actors

    def _run(self):
        """
        Run pipeline on data to perform forecasts.
        """

        if not self._handle_result_dir():
            return

        files = os.listdir(self.input_dir)
        counter = 0
        while files:
            try:
                self._call_remote_pipeline(files[: self.parallel_actors])
            except:
                logging.info("Error occured. Trying again.")
                counter += 1
                if counter < 5:
                    continue
            counter = 0
            files = files[self.parallel_actors :]

    def _call_remote_pipeline(self, file_names: list[str]) -> None:
        """
        Call actors to perform forecasting.

        Parameters
        ----------
        file_names : list[str]
            List with names of files that will be processed.
        """
        ray.init()
        pipelines = [
            _run_remote.remote(  # pylint: disable=E1101:no-member
                file_name=file_name, **self.__dict__
            )
            for file_name in file_names
        ]
        results_ref = [pipeline.run.remote() for pipeline in pipelines]
        ray.get(results_ref)
        ray.shutdown()

    def _handle_result_dir(self) -> bool:
        """
        Handle result dir.
        """
        flag = os.path.exists(self.result_dir)

        if flag and self.mode == "recreate":
            logging.info("Result dir already exists. It will be recreated.")
            shutil.rmtree(self.result_dir)
            os.mkdir(self.result_dir)
            return True

        if flag and self.mode is None:
            logging.info(
                "Result dir already exists. Forecasting Pipeline will be skipped."
            )
            return False

        if not flag:
            os.mkdir(self.result_dir)
            return True

        raise ValueError(f"Mode {self.mode} is not supported.")
