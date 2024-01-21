"""Module contains abstract class for task execution."""

import logging
from abc import ABC, abstractmethod


class Task(ABC):
    """Abstract class for task execution."""

    def __init__(self):
        self.time_elapsed = None

    def run(self):
        """Run task."""
        logging.info("Start %s task.", self.__class__.__name__)
        self._run()
        logging.info("Finish %s task.", self.__class__.__name__)

    @abstractmethod
    def _run(self):
        ...
