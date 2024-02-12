"""Module contains task for postprocessing. Run transformers over the forecasted data."""

import os
from typing import Iterable

import pandas as pd

from app.data_managers.namespaces import column_names_ns
from app.utils.task import Task


class PostprocessTask(Task):
    """
    Task for postprocessing. Run transformers over the forecasted data.
    Apply changes inplace.
    """

    def __init__(self, forecasts_result_dir: str, transformers: Iterable):
        """
        Parameters
        ----------
        forecasts_result_dir : str
            Directory where the forecasted data is stored.
        transformers : Iterable
            List of transformers to run over the forecasted data.
        """
        super().__init__()
        self.forecasts_result_dir = forecasts_result_dir
        self.transformers = transformers

    def _run(self):

        files = os.listdir(self.forecasts_result_dir)

        for file_name in files:
            file_path = os.path.join(self.forecasts_result_dir, file_name)
            output = []
            data = pd.read_csv(file_path)

            dependent_variables = set(data[column_names_ns.DEPENDENT_VARIABLE_NAME])

            for dependent_variable in dependent_variables:
                y = data[
                    data[column_names_ns.DEPENDENT_VARIABLE_NAME] == dependent_variable
                ].loc[:, column_names_ns.VALUE]

                for transformer in self.transformers:
                    y = transformer.fit_transform(y)
                res = data.loc[y.index, :]
                res[column_names_ns.VALUE] = y
                output.append(res)

            if output:
                pd.concat(output, ignore_index=True).to_csv(file_path, index=False)
