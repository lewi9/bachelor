"""Module contains abstract class for interpolation task execution."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime

from app.utils.task import Task


class TaskInterpolation(Task, ABC):
    """Abstract class for interpolation task execution."""

    def run(self):
        """Run task."""
        logging.info("Start %s interpolation task.", self.__class__.__name__)
        start_time = datetime.now()
        self._run()
        end_time = datetime.now()
        logging.info("Finish %s interpolation task.", self.__class__.__name__)

        self.time_elapsed = end_time - start_time
        logging.info("Time elapsed: %s", self.time_elapsed)

    @abstractmethod
    def _run(self):
        ...
