"""Module contains abstract class for task execution."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime

import ray


class Task(ABC):
    """Abstract class for task execution."""

    def __init__(self, parallel: bool = False):
        self.time_elapsed = None
        self.parallel = parallel

    def run(self):
        """Run task."""
        logging.info("Start %s task.", self.__class__.__name__)
        start_time = datetime.now()

        self._run()

        end_time = datetime.now()
        logging.info("Finish %s task.", self.__class__.__name__)

        self.time_elapsed = end_time - start_time
        logging.info("Time elapsed: %s", self.time_elapsed)

    @abstractmethod
    def _run(self):
        ...
