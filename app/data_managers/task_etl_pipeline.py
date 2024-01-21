"""ETL Pipeline class."""
from typing import OrderedDict

from app.data_managers.task_etl import TaskETL
from app.utils.task_pipeline import BaseTaskPipeline


class TaskETLPipeline(BaseTaskPipeline):
    """Pipeline for ETL tasks."""

    def __init__(self, tasks: OrderedDict[str, TaskETL]):
        """
        Parameters
        ----------
        tasks : OrderedDict[str, TaskETL]
            Ordered dictionary of tasks to run.
        """
        super().__init__(tasks=tasks)
