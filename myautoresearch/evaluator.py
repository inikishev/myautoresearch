import fcntl
import sys
import time
import warnings
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import click

from . import _utils
from .logger import Logger


class Evaluator(ABC):
    def __init__(self, argv: list[str] = sys.argv):
        self.object, self._args = _utils.import_object(argv)
        self._metrics: dict[str, _utils.Metric] = {}
        self.history = Logger()
        self._feasibility = []

        self.file_logger = _utils.get_logger("debug.log")
        """Silently log to a file."""

    @abstractmethod
    def evaluate(self) -> None:
        """Evaluates ``self.object`` and logs the metrics. Use ``click.echo`` to display any additional information useful to the AI agent."""

    def log_step(self, step: int, metric: str, value: Any):
        """Log an intermediate numeric metric, like train loss. If any are logged, the logger will be saved
        to working directory, and a short instruction for the agent will be displayed
        after evaluating a run on how to inspect it."""
        self.history.log(step, metric, value)

    def log_final(
        self,
        metric: str,
        value: Any,
        maximize: bool | None,
        is_main: bool = True,
        display_value: bool = True,
        display_rank: bool = True,
        display_leaderboard: bool = True,
        display_summary: bool = True,
        weight: float = 1,
    ):
        """Log a final metric. At least one main metric must be logged so that solutions can be compared.

        Args:
            metric: Name of the metric.
            value: Value of the metric.
            maximize: `True` if higher is better, `False` if lower is better, `None` if not numeric.
            is_main: At least one metric should be main to rank the runs.
                If there are multiple main metrics, an average rank is computed from their ranks.
                But it is usually a good idea to manually design a formula to compute a final score,
                and use that score as the only main metric. Defaults to True.
            display_value: Show this metric after a run is evaluated.
                In most cases it is fine to set this to True on all metrics. Defaults to True.
            display_rank: Show this metric rank and name of best run by this metric after a run is evaluated.
                Defaults to True.
            display_leaderboard: Show this metric for all other runs in the leaderboard after a run is evaluated.
                Keep the number of metrics in the leaderboard under 4 to make it more readable. Defaults to True.
            display_summary: Show this metric for all submited runs in the summary shown when agent runs `mar start`.
                Keep the number of metrics in the summary under 10 to avoid filling the context with large number of submissions.
            weight: This metric's weight for computing average rank from main metrics. Defaults to 1.0.
        """
        self._metrics[metric] = _utils.Metric(
            name = metric,
            value = value,
            maximize = maximize,
            is_main = is_main,
            display_value = display_value,
            display_rank = display_rank,
            display_leaderboard = display_leaderboard,
            display_summary = display_summary,
            weight = weight,
        )

    def set_infeasible(self, reason: str):
        """Mark this run as infeasible and specify a reason that AI will see."""
        self._feasibility.append({"feasible": False, "reason": reason})

    def save(self):
        if len(self.history) > 0:
            self.history.save("logger.npz")
        _utils.write_json({k: v.to_tuple() for k,v in self._metrics.items()}, "metrics.json")
        _utils.write_json(self._feasibility, "feasibility.json")

def run(evaluator: Evaluator):
    root = Path(evaluator._args.root)

    try:
        with open(root / 'eval.lock', 'w', encoding='utf-8') as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                _utils.cleanup_orphans()
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as e:
                    raise RuntimeError("Another evaluation script is already running and couldn't be terminated.") from e

            evaluator.evaluate()
            evaluator.save()

    finally:
        os.remove(root / 'eval.lock')
