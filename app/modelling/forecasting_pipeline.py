"""Module to perform forecasting on data."""

import logging
import os
import shutil
import warnings

import pandas as pd
from sktime.split.base import BaseWindowSplitter
from sktime.forecasting.base import BaseForecaster

from app.data_managers.namespaces import column_names_ns
from app.data_managers.types import ExoFiller, ForecastingPipelineMode
from app.modelling.task_modelling import TaskModelling


class _DataStructure:
    """
    Data class used to pasa data beetwen methods.
    """

    def __init__(
        self,
        endo: dict[str, pd.Series],
        exo: pd.DataFrame,
    ):
        self.endo = endo
        self.exo = exo


class ForecastingPipeline(TaskModelling):
    """
    Pipeline to perform forecasts on prepared data.
    """

    def __init__(
        self,
        input_dir: str,
        result_dir: str,
        transformers: list,
        forecasters: dict[str, BaseForecaster],
        splitters: dict[str, BaseWindowSplitter],
        max_forecasts: int = 3,
        exo_filler: ExoFiller = "mean",
        mode: ForecastingPipelineMode = "recreate",
    ):
        """
        Paramaters
        ----------
        input_dir : str
            Path to directory with prepared data files.
            Prepared data files should contains prediction indexes.
        result_dir : str
            Path to directory, where forecasts will be saved.
        transformers : list
            List of transformers that will be run on data.
        forecasters : dict[str, BaseForecaster]
            Dictionary with forecasters. Key is the string with name
            of the forecaster. The value is forecaster.
        splitters : dict[str, BaseWindowSplitter]
            Splitters to split data into train and test sets.
        max_forecasts : int
            Parameter to controll number of forecasts - only the one forecast is performed ahead,
            the rest of them may be used to evaluate model.
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
        self.transformers = transformers
        self.forecasters = forecasters
        self.splitters = splitters
        self.max_forecasts = max_forecasts
        self.exo_filler = exo_filler
        self.mode = mode

    def _run(self):
        """
        Run pipeline on data to perform forecasts.
        """
        flag = os.path.exists(self.result_dir)
        if flag and self.mode == "recreate":
            logging.info("Result dir already exists. It will be recreated.")
            shutil.rmtree(self.result_dir)
            os.mkdir(self.result_dir)
        elif flag and self.mode is None:
            logging.info(
                "Result dir already exists. Forecasting Pipeline will be skipped."
            )
            return
        elif not flag:
            os.mkdir(self.result_dir)
        else:
            raise ValueError(f"Mode {self.mode} is not supported.")

        files = os.listdir(self.input_dir)
        for file_name in files:
            try:
                logging.info("Processing %s", file_name)
                data_structure = self._read_data(
                    directory=self.input_dir, file_name=file_name
                )
                data_structure = self._apply_transformers(data_structure)
                predictions = self._apply_forecasters(data_structure=data_structure)
                predictions.to_csv(
                    os.path.join(self.result_dir, file_name), index=False
                )
                logging.info("%s processed with success", file_name)
            except Exception as exc:
                logging.warning("%s cannot be processed, because %s", file_name, exc)

    def _read_data(self, directory: str, file_name: str) -> _DataStructure:
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
        _DataStructure
            _DataStructure with series to forecast and exo.
        """
        # Read from file
        data = pd.read_csv(os.path.join(directory, file_name))

        # Remove index column if exists
        if "Unnamed: 0" in data.columns:
            data = data.drop(columns="Unnamed: 0")

        # Change time to datetime
        data[column_names_ns.TIME] = pd.to_datetime(
            data[column_names_ns.TIME], format="%Y-%m-%d %H:%M:%S"
        )

        # Set datetime index and assign Freq
        with_index = data.set_index(column_names_ns.TIME)
        with_index.index.freq = "H"

        endo_dict = {
            column_name: with_index[column_name]
            for column_name in column_names_ns.ENDO_COLUMNS
        }

        return _DataStructure(
            endo=endo_dict,
            exo=with_index.drop(columns=list(endo_dict.keys())),
        )

    def _apply_transformers(self, data_structure: _DataStructure) -> _DataStructure:
        """
        Apply transformers on data.
        """
        endo_dict = data_structure.endo
        exo = data_structure.exo.copy()
        warnings.filterwarnings("ignore")

        for step in self.transformers:
            endo_dict = {
                name: step.fit_transform(series) for name, series in endo_dict.items()
            }
            exo = exo.apply(step.fit_transform, axis=0)

        if self.exo_filler == "mean":
            for column in exo.columns:
                exo[column] = exo[column].fillna(value=exo[column].mean())

        warnings.filterwarnings("always")
        return _DataStructure(endo=endo_dict, exo=exo)

    def _apply_forecasters(self, data_structure: _DataStructure) -> pd.DataFrame:
        """
        Apply forecasters to perform forecasting
        """
        warnings.filterwarnings("ignore")
        endo_dict = data_structure.endo
        exo = data_structure.exo
        results = []

        for splitter_name, splitter in self.splitters.items():
            fh = splitter.get_fh()
            for series_name, series in endo_dict.items():
                for forecaster_name, forecaster in self.forecasters.items():
                    logging.info("Processing %s with %s", series_name, forecaster_name)
                    for train, test in list(splitter.split(series))[
                        -self.max_forecasts :
                    ]:
                        try:
                            forecaster.fit(
                                series.reset_index(drop=True).iloc[train],
                                X=exo.reset_index(drop=True).iloc[train],
                            )
                            prediction = forecaster.predict(
                                fh=fh, X=exo.reset_index(drop=True).iloc[test]
                            )
                        except Exception as exc:
                            logging.warning(exc)
                            continue
                        dates = pd.date_range(
                            start=series.index[train[-1]],
                            periods=len(fh) + 1,
                            freq="H",
                        )[1:]
                        result = pd.DataFrame(
                            {
                                column_names_ns.VALUE: prediction.reset_index(
                                    drop=True
                                ),
                                column_names_ns.TIME: dates,
                            }
                        )

                        result[column_names_ns.MODEL] = (
                            forecaster_name + "_" + splitter_name
                        )
                        result[column_names_ns.PREDICTED] = series_name
                        results.append(result)

            warnings.filterwarnings("always")
            return pd.concat(results, ignore_index=True)
