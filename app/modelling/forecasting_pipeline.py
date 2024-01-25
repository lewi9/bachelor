"""Module to perform forecasting on data."""

import logging
import os
import shutil
import warnings

import pandas as pd
from sktime.forecasting.base import BaseForecaster
from sktime.transformations.series.outlier_detection import HampelFilter

from app.data_managers.namespaces import column_names_ns, date_formats_ns
from app.data_managers.types import ExoFiller, ForecastingPipelineMode
from app.modelling.splitters import BaseWindowSplitter
from app.utils.task import Task


class ForecastingPipeline(Task):
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
        """
        super().__init__()
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

    def _run(self):
        """
        Run pipeline on data to perform forecasts.
        """

        if not self._handle_result_dir():
            return

        files = os.listdir(self.input_dir)
        for file_name in files:
            logging.info("Processing %s", file_name)
            predictions = []

            data = self._read_data(directory=self.input_dir, file_name=file_name)
            dependent_variables = data[column_names_ns.DEPENDENT_VARIABLE_NAME].unique()
            for dependent_variable in dependent_variables:
                to_process = data.query(
                    f"DEPENDENT_VARIABLE_NAME == '{dependent_variable}'"
                )
                to_process = to_process.drop(
                    columns=column_names_ns.DEPENDENT_VARIABLE_NAME
                )

                try:
                    y, X = self._apply_transformers(data=to_process)
                except ValueError as exc:
                    logging.warning(
                        "File %s, variable %s, cannot be processed, because %s",
                        file_name,
                        dependent_variable,
                        exc,
                    )
                    continue

                forecast = self._apply_forecasters(y=y, X=X)
                forecast[column_names_ns.DEPENDENT_VARIABLE_NAME] = dependent_variable
                predictions.append(forecast)
            if not predictions:
                logging.warning("File %s cannot be processed", file_name)
                continue

            result = pd.concat(predictions, ignore_index=True)
            result.to_csv(os.path.join(self.result_dir, file_name), index=False)

            logging.info("%s processed with success", file_name)

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

    def _read_data(self, directory: str, file_name: str) -> pd.DataFrame:
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

        # Remove index column if exists
        if "Unnamed: 0" in data.columns:
            data = data.drop(columns="Unnamed: 0")

        # Change time to datetime
        data[column_names_ns.TIME] = pd.to_datetime(
            data[column_names_ns.TIME],
            format=date_formats_ns.FORECASTING_PIPELINE_DATETIME_FORMAT,
        )

        # Set datetime index and assign Freq
        with_index = data.set_index(column_names_ns.TIME)
        return with_index

    def _apply_transformers(self, data: pd.DataFrame) -> (pd.Series, pd.DataFrame):
        """
        Cut endo data to last valid actual date.
        Enforce index to be datetime with freq "H".
        Apply transformers on data.
        Use special method to impute exo data after transformers are applied.
        The last case is implemented due to some problems with exo data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to apply transformers on.

        Returns
        -------
        pd.Series
            Endo data after transformers are applied.
        pd.DataFrame
            Exo data after transformers are applied.
        """
        X = data.drop(columns=column_names_ns.VALUE)
        y = data[column_names_ns.VALUE]
        y.index.freq = "H"
        y = y.loc[: self.last_valid_actual]

        for transformer in self.transformers:
            if isinstance(transformer, HampelFilter):
                warnings.filterwarnings("ignore")
            y = transformer.fit_transform(y)
            X = X.apply(transformer.fit_transform, axis=0)
            warnings.filterwarnings("always")

        if self.exo_filler == "mean":
            for column in X.columns:
                X[column] = X[column].fillna(X[column].mean())

        return y, X

    def _apply_forecasters(self, y: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply forecasters to perform forecasting
        """
        results = []
        for splitter_name, splitter in self.splitters.items():
            for forecaster_name, forecaster in self.forecasters.items():
                for indexes in splitter.split(y=y, forecasts=self.max_forecasts):
                    y_train = y.iloc[indexes]
                    try:
                        forecaster.fit(
                            y=y_train.reset_index(drop=True),
                            X=X.reset_index(drop=True),
                        )
                    except (ValueError, RuntimeError) as exc:
                        logging.warning(
                            "Model %s cannot be fitted, because %s",
                            forecaster_name,
                            exc,
                        )
                        continue

                    try:
                        prediction = forecaster.predict(
                            fh=splitter.fh,
                            X=X.reset_index(drop=True),
                        )
                    except (ValueError, RuntimeError) as exc:
                        logging.warning(
                            "Model %s cannot be predicted, because %s",
                            forecaster_name,
                            exc,
                        )
                        continue

                    dates = pd.date_range(
                        start=y.iloc[indexes].index[-1],
                        periods=len(splitter.fh) + 1,
                        freq="H",
                    )[1:]

                    result = pd.DataFrame(
                        {
                            column_names_ns.VALUE: prediction.reset_index(drop=True),
                            column_names_ns.TIME: dates,
                        }
                    )

                    result[column_names_ns.FORECASTER] = (
                        forecaster_name + "_" + splitter_name
                    )
                    results.append(result)

        if results:
            return pd.concat(results, ignore_index=True)

        return pd.DataFrame(
            columns=[
                column_names_ns.VALUE,
                column_names_ns.TIME,
                column_names_ns.FORECASTER,
            ]
        )
