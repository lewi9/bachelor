"""Class to evaluate the system performance"""

import os
from collections import OrderedDict
from typing import Sequence

import numpy as np
import pandas as pd

from app.data_managers.namespaces import column_names_ns
from app.data_managers.types import SeriesMetric


class Evaluator:
    """Class to evaluate the system performance"""

    def __init__(
        self,
        selected_data_dir: str,
        reference_data_dir: str,
        metrics: Sequence[SeriesMetric],
    ):
        """
        Parameters
        ----------
        selected_data_dir : str
            Directory with selected data.
        reference_data_dir : str
            Directory with reference data.
        metrics : Sequence[SeriesMetric]
            Sequence of metrics.
        """
        self.selected_data_dir = selected_data_dir
        self.reference_data_dir = reference_data_dir
        self.metrics = metrics
        self._compared = 0

    def evaluate(self) -> OrderedDict[str, float]:
        """Evaluate the system performance"""
        self._compared = 0
        files = os.listdir(self.selected_data_dir)

        reference_data = []
        selected_data = []

        dependent_variable_names = set()

        for file_name in files:
            selected_df = pd.read_csv(
                os.path.join(self.selected_data_dir, file_name),
                usecols=[
                    column_names_ns.TIME,
                    column_names_ns.VALUE,
                    column_names_ns.DEPENDENT_VARIABLE_NAME,
                ],
            )
            reference_df = pd.read_csv(
                os.path.join(self.reference_data_dir, file_name),
                usecols=[
                    column_names_ns.TIME,
                    column_names_ns.VALUE,
                    column_names_ns.DEPENDENT_VARIABLE_NAME,
                ],
            )

            selected_df_min_time = selected_df[column_names_ns.TIME].min()
            selected_df_max_time = selected_df[column_names_ns.TIME].max()

            reference_df_subset = reference_df.query(
                f"time >= '{selected_df_min_time}'"
            ).query(f"time <= '{selected_df_max_time}'")

            selected_df[column_names_ns.ID] = file_name.split(".")[0]
            reference_df_subset[column_names_ns.ID] = file_name.split(".")[0]

            dependent_variable_names |= set(
                selected_df[column_names_ns.DEPENDENT_VARIABLE_NAME]
            )

            selected_df = selected_df.set_index(
                [
                    column_names_ns.ID,
                    column_names_ns.TIME,
                    column_names_ns.DEPENDENT_VARIABLE_NAME,
                ]
            )
            reference_df_subset = reference_df_subset.set_index(
                [
                    column_names_ns.ID,
                    column_names_ns.TIME,
                    column_names_ns.DEPENDENT_VARIABLE_NAME,
                ]
            )

            selected_data.append(selected_df)
            reference_data.append(reference_df_subset)

        selected_data = pd.concat(selected_data)
        reference_data = pd.concat(reference_data)

        self._compared = np.sum(selected_data.notna() & reference_data.notna()) / max(
            len(selected_data), len(reference_data)
        )

        evaluation = OrderedDict()

        for metric in self.metrics:
            for dependent_variable in dependent_variable_names:
                evaluation[f"{dependent_variable}_{str(metric)}"] = metric(
                    selected_data.xs(
                        dependent_variable,
                        level=column_names_ns.DEPENDENT_VARIABLE_NAME,
                    ),
                    reference_data.xs(
                        dependent_variable,
                        level=column_names_ns.DEPENDENT_VARIABLE_NAME,
                    ),
                )

        return evaluation

    @property
    def compared(self) -> int:
        """Return the number of compared values"""
        return self._compared
