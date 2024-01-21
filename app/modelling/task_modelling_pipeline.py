"""Modelling Pipeline class."""
from typing import OrderedDict

from app.modelling.task_modelling import TaskModelling
from app.utils.task_pipeline import BaseTaskPipeline


class TaskModellingPipeline(BaseTaskPipeline):
    """Pipeline for Modelling tasks."""

    def __init__(self, tasks: OrderedDict[str, TaskModelling]):
        """
        Parameters
        ----------
        tasks : OrderedDict[str, TaskModelling]
            Ordered dictionary of tasks to run.
        """
        super().__init__(tasks=tasks)
