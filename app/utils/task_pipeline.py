"""Pipeline for running multiple tasks."""
import datetime
import logging
from typing import OrderedDict

import numpy as np

from app.utils.task import Task


class TaskPipeline:
    """Pipeline for running multiple tasks."""

    def __init__(self, tasks: OrderedDict[str, Task]):
        """
        Parameters
        ----------
        tasks : OrderedDict[str, Task]
            Ordered dictionary of tasks to run.
        """
        self.tasks = tasks
        self._execution_time = {}

    def run(self):
        """Run all tasks in pipeline."""
        for task_name, task in self.tasks.items():
            logging.info("Run task: %s", task_name)
            task.run()
            self._execution_time[task_name] = task.time_elapsed

    def get_execution_time(self):
        """Return execution time of each task."""
        return self._execution_time

    @property
    def execution_time(self):
        """Return total execution time."""
        return np.sum(list(self._execution_time.values())) or datetime.timedelta(0)
