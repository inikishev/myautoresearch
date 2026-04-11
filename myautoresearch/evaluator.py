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
    """Base class evaluation logic. Children must define ``evaluate`` method."""
    def __init__(self, argv: list[str] = sys.argv):
        self.object, self._args = _utils.import_object(argv)
        self._metrics: dict[str, _utils.Metric] = {}
        self.history = Logger()
        self._feasibility = []

        self.file_logger = _utils.get_file_logger("debug.log")
        """Silently log to a file."""

    @abstractmethod
    def evaluate(self) -> None:
        """Evaluates ``self.object`` and logs the metrics. Use ``click.echo`` to display any additional information useful to the AI agent."""

    def log_step(self, step: int, metric: str, value: Any) -> None:
        """Log an intermediate numeric metric, like train loss. If any are logged, and
        ``copy_logger`` configuration optin is enabled, the logger will be saved
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
        weight: float = 1.0,
    ) -> None:
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
                Keep the number of metrics in the summary under 10 to avoid filling the context when number of submissions is large.
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

    def set_infeasible(self, reason: str) -> None:
        """Mark this run as infeasible and specify a reason that the AI agent will see."""
        self._feasibility.append({"feasible": False, "reason": reason})

    def _save(self):
        if len(self.history) > 0:
            self.history.save("logger.npz")
        _utils.write_json({k: v.to_tuple() for k,v in self._metrics.items()}, "metrics.json")
        _utils.write_json(self._feasibility, "feasibility.json")

def run(evaluator: Evaluator):
    with _utils.no_stack_trace():

        root = Path(evaluator._args.root)

        if (root / "eval.lock").exists():
            _utils.cleanup_orphans()

        try:
            with open(root / 'eval.lock', 'w', encoding='utf-8') as f:
                # Create a lock, and make sure no existing evaluation scripts are running
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    _utils.cleanup_orphans()
                    try:
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as e:
                        raise RuntimeError("Another evaluation script is already running and couldn't be terminated.") from e

                evaluator.evaluate()
                evaluator._save()

        finally:
            os.remove(root / 'eval.lock')
