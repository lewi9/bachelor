"""Interpolation Pipeline class."""
from typing import OrderedDict

from app.interpolation.task_interpolation import TaskInterpolation
from app.utils.task_pipeline import BaseTaskPipeline


class TaskInterpolationPipeline(BaseTaskPipeline):
    """Pipeline for Modelling tasks."""

    def __init__(self, tasks: OrderedDict[str, TaskInterpolation]):
        """
        Parameters
        ----------
        tasks : OrderedDict[str, TaskInterpolation]
            Ordered dictionary of tasks to run.
        """
        super().__init__(tasks=tasks)
