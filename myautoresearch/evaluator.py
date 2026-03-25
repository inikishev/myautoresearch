import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import utils
from .logger import Logger


class Metric:
    def __init__(self, name: str, value, maximize: bool | None, is_main: bool, display: bool):
        self.name = name
        self.value = value
        self.maximize = maximize
        self.is_main = is_main
        self.display = display

    @property
    def is_metric(self):
        return isinstance(self.maximize, bool)

    def error(self):
        if not self.is_metric:
            raise RuntimeError(f'"{self.name}" is not a metric and doesn\'t have an error.')

        assert isinstance(self.value, (int, float))
        if self.maximize: return -self.value
        return self.value

    def to_tuple(self):
        return (self.name, self.value, self.maximize, self.is_main, self.display)

    @classmethod
    def from_tuple(cls, t: tuple[str, float, bool, bool, bool]):
        return cls(*t)


class Evaluator(ABC):
    """Evaluator runs in ``current/{run_name}`` folder. All files from that folder are copied to ``runs`` on submission."""
    def __init__(self, argv: list[str] = sys.argv):
        self.object = utils.import_object(argv)
        self._metrics: dict[str, Metric] = {}
        self._logger = Logger()
        self._feasibility = []

    @abstractmethod
    def run(self) -> None:
        """Run."""

    def log_step(self, step: int, metric: str, value: Any):
        self._logger.log(step, metric, value)

    def log_final(self, metric: str, value: Any, maximize: bool | None, is_main: bool = True, display: bool = True):
        self._metrics[metric] = Metric(
            name = metric,
            value = value,
            maximize = maximize,
            is_main = is_main,
            display = display,
        )

    def set_infeasible(self, reason: str):
        self._feasibility.append({"feasible": False, "reason": reason})

    def save(self):
        self._logger.save("logger.npz")
        utils.write_json({k: v.to_tuple() for k,v in self._metrics.items()}, "metrics.json")
        utils.write_json(self._feasibility, "feasibility.json")


class FinishedRun:
    status: Literal["unsubmitted", "submitted", "discarded", "current"]
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.info = utils.read_json(self.path / "info.json")

        if (self.path / "metrics.json").exists():
            self.metrics = {k: Metric.from_tuple(v) for k, v in utils.read_json(self.path / "metrics.json").items()}

        else:
            self.metrics = {}