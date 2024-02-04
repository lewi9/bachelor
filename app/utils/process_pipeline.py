"""Pipeline of Pipelines. One Pipeline to rule them all."""

import logging
import datetime
from typing import OrderedDict

import numpy as np

from app.utils.task_pipeline import TaskPipeline


class ProcessPipeline:
    """Pipeline for running multiple tasks."""

    def __init__(self, pipelines: OrderedDict[str, TaskPipeline]):
        """
        Parameters
        ----------
        pipelines : OrderedDict[str, TaskPipeline]
            Ordered dictionary of pipelines to run.
        """
        self.pipelines = pipelines
        self._tasks_execution_time = {}
        self._pipelines_execution_time = {}

    def run(self):
        """Run all tasks in pipeline."""
        for pipeline_name, pipeline in self.pipelines.items():
            logging.info("Run pipeline: %s", pipeline_name)
            pipeline.run()
            self._tasks_execution_time[pipeline_name] = pipeline.get_execution_time()
            self._pipelines_execution_time[pipeline_name] = pipeline.execution_time

    def get_tasks_execution_time(self):
        """Return execution time of each task."""
        return self._tasks_execution_time

    def get_execution_time(self):
        """Return execution time of each pipeline."""
        return self._pipelines_execution_time

    @property
    def execution_time(self) -> datetime.timedelta:
        """Return total execution time."""
        return np.sum(list(self._pipelines_execution_time.values()))
