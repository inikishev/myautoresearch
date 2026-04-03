"""Internal utils"""
import argparse
import importlib.util
import json
import logging
import math
import os
import shutil
import signal
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal
from ruamel.yaml import YAML

import click
import numpy as np
import psutil
import yaml
from scipy.stats import rankdata
from contextlib import contextmanager

class NoStackTraceException(Exception):
    """Use this exception with ``no_stack_trace`` context to hide stack trace where it is not needed to save tokens."""

@contextmanager
def no_stack_trace():
    """Context manager to display ``NoStackTraceException`` errors without stack trace"""
    try:
        yield
    except NoStackTraceException as e:
        click.echo(f"ERROR: {e}")

def read_json(file: str | os.PathLike):
    with open(file, "r", encoding='utf-8') as f:
        return json.load(f)

def write_json(obj, file: str | os.PathLike):
    with open(file, "w", encoding='utf-8') as f:
        json.dump(obj, f, sort_keys=False, indent=4, ensure_ascii=False)

def _get_yaml_reader():
    """preserves comments"""
    reader = YAML()
    reader.preserve_quotes = True  # keeps existing quote styles
    reader.indent(mapping=2, sequence=4, offset=2) # sets consistent indents
    return reader

def read_yaml(file: str | os.PathLike):
    reader = _get_yaml_reader()
    with open(file, "r", encoding='utf-8') as f:
        return reader.load(f)

def write_yaml(obj, file: str | os.PathLike):
    reader = _get_yaml_reader()
    with open(file, "w", encoding='utf-8') as f:
        reader.dump(obj, f)

def import_object(argv: list[str]):

    # argv will have
    # --file
    # --object
    # --root
    # --name

    if "__myautoresearch_evaluate__.py" in argv: argv.remove("__myautoresearch_evaluate__.py")
    parser = argparse.ArgumentParser(description="Evaluates scripts.")
    parser.add_argument('-f', "--file", type=str)
    parser.add_argument('-o', "--object", type=str)
    parser.add_argument('-r', "--root", type=str)
    parser.add_argument('-n', "--name", type=str)
    args = parser.parse_args(argv)

    # Create spec
    spec = importlib.util.spec_from_file_location("_myautoresearch_run", args.file)
    assert spec is not None
    assert spec.loader is not None

    # Create module from spec and register to sys modules
    module = importlib.util.module_from_spec(spec)
    sys.modules["_myautoresearch_run"] = module

    # Execute module and return object
    spec.loader.exec_module(module)
    object = getattr(module, args.object)
    if object is None:
        raise NoStackTraceException(f"Submitted file doesn't contain `{object}` or it is None.")
    return object, args

def format_value(x):
    if isinstance(x, np.ndarray) and x.size == 1: x = x.item()
    if isinstance(x, float): return f"{x:.8g}"
    return str(x)

def read_text(file: str | os.PathLike):
    with open(file, "r", encoding='utf-8') as f:
        return f.read()

def write_text(text: str, file: str | os.PathLike):
    with open(file, "w", encoding='utf-8') as f:
        f.write(text)


def make_valid_filename(s: str):
    c = ''.join(c for c in s if c.isalnum() or c in (" .,-()_"))[:100]
    while not c[0].isalnum(): c = c[1:]
    while not c[-1].isalnum(): c = c[:-1]
    return c

def get_cwd():
    try:
        return Path(os.getcwd())
    except FileNotFoundError:
        cwd = Path(psutil.Process(os.getpid()).cwd())
        click.echo(f"os.getcwd() failed. Changing dir to psutil cwd: {cwd}")
        os.chdir(cwd)
        return cwd


class Metric:
    def __init__(self, name: str, value, maximize: bool | None, is_main: bool, display_value: bool, display_rank: bool, display_leaderboard: bool, display_summary: bool, weight: float):
        self.name = name
        self.value = value
        self.maximize = maximize
        self.is_main = is_main
        self.display_value = display_value
        self.display_rank = display_rank
        self.display_leaderboard = display_leaderboard
        self.display_summary = display_summary
        self.weight = weight

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
        return (self.name, self.value, self.maximize, self.is_main, self.display_value, self.display_rank, self.display_leaderboard, self.display_summary, self.weight)

    @classmethod
    def from_tuple(cls, t: tuple[str, float, bool, bool, bool, bool, bool, bool, float]):
        return cls(*t)



class FinishedRun:
    status: Literal["unsubmitted", "submitted", "discarded", "current"]
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

        if (self.path / "info.json").exists():
            self.info = read_json(self.path / "info.json")
        else:
            self.info = {}

        if (self.path / "metrics.json").exists():
            self.metrics = {k: Metric.from_tuple(v) for k, v in read_json(self.path / "metrics.json").items()}

        else:
            self.metrics = {}

def process_metrics(runs: list[FinishedRun]):
    # Find ranks of each run by each main metric
    # First we need to get a list of main metrics
    main_metric_names = set()
    value_metric_names = set()
    rank_metric_names = set()
    weights = {}
    for run in runs:
        main_metric_names.update(name for name, metric in run.metrics.items() if metric.is_main)
        value_metric_names.update(name for name, metric in run.metrics.items() if metric.display_value)
        rank_metric_names.update(name for name, metric in run.metrics.items() if metric.display_rank)
        weights.update({metric.name: metric.weight for metric in run.metrics.values()})

    # Load values of rank metrics for each run
    rank_errors = defaultdict(list)
    for run in runs:
        for metric in sorted(rank_metric_names):
            if metric in run.metrics:
                rank_errors[metric].append(run.metrics[metric].error())
            else:
                rank_errors[metric].append(np.nan)

    # Compute ranks (index of run in `runs` is index of rank in each list)
    ranks = {k: rankdata(v, method='dense', nan_policy='omit') for k,v in rank_errors.items()}

    if len(ranks) > 1 and len(main_metric_names) > 1:
        mean_ranks = []
        for i, run in enumerate(runs):
            run_ranks = []
            weight_sum = 0
            for metric_name in sorted(main_metric_names):
                weight = weights[metric_name]
                weight_sum += weight
                if metric_name in run.metrics:
                    run_ranks.append(ranks[metric_name][i] * weight)
                else:
                    run_ranks.append(len(runs) * weight)

            mean_ranks.append(np.sum(run_ranks) / weight_sum)

        total_ranks = rankdata(mean_ranks, method='dense', nan_policy='omit')

    elif len(main_metric_names) == 1:
        total_ranks = ranks[list(main_metric_names)[0]]
        mean_ranks = None

    else:
        mean_ranks = total_ranks = None

    return main_metric_names, value_metric_names, ranks, mean_ranks, total_ranks


def find_run_dir_by_name(name: str, root: Path) -> Path:
    # need to find name in either submitted or unsubmitted
    target_run_dir = None
    all_names = []
    for status in ("unsubmitted", "submitted"):
        dir = root / "runs" / status
        for run_dir in dir.iterdir():
            run_name = read_json(run_dir / "info.json")["name"]
            if run_name == name:
                target_run_dir = run_dir
                break
            all_names.append(run_name)

        if target_run_dir is not None:
            break

    if target_run_dir is None:
        raise NoStackTraceException(f"A run with name '{name}' doesn't exist, make sure 'name' is identical to one "
                                f"passed to `run` command. The following runs exist: {all_names}")

    return target_run_dir

def get_file_logger(file):
    logger = logging.getLogger("myautoresearch")

    for handler in logger.handlers:
        logger.removeHandler(handler)

    logger.setLevel(1)
    handler = logging.FileHandler(file)
    handler.setLevel(1)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger


def cleanup_orphans(script_name: str = "__myautoresearch_evaluate__.py", warn=True):
    """Cleans up unterminated evaluation scripts. Some cli tools like iflow don't terminate them correctly,
    and that causes them to use resources in the background and results in incorrect timings in benchmarks."""
    current_pid = os.getpid()

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            if any(script_name in arg for arg in cmdline):
                if proc.info['pid'] != current_pid:

                    if warn:
                        click.echo(
                            f"WARNING: another evaluation script was already running (PID: {proc.info['pid']}) and has been terminated. This can happen if you run multiple evaluations in parallel which is not allowed as that would affect their recorded runtime, or if `mar evaluate` was force-killed by the shell command tool. If possible, increase or disable shell command tool timeout to prevent this from happening.")
                    else:
                        click.echo(f"Another evaluation script was already running (PID: {proc.info['pid']}) and has been terminated.")

                    proc.send_signal(signal.SIGTERM)
                    proc.wait(timeout=3)

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

        except psutil.TimeoutExpired:
            proc.kill()


def maybe_int(x):
    if math.isfinite(x): return int(x)
    return x

def maybe_strip(s):
    if isinstance(s, str): return s.strip()
    return s

def get_root_and_config() -> tuple[Path, dict]:
    """Checks if command is ran from workdir; checks if a run was suddenly terminated and terminates old evaluations;
    Returns root path and config."""
    cwd = get_cwd()
    root = cwd.parent

    if not ((root / "config.yaml").exists() and (root / "task.md").exists() and (root / "runs").is_dir()):
        raise RuntimeError(NoStackTraceException(f"All commands must be ran from working directory, but current directory is {cwd}."))

    # Load config
    with open(root / "config.yaml", encoding='utf-8') as f:
        config: dict = yaml.safe_load(f)

    # Check that work dir is correct
    work_dir_name = config.get("work_dir", "workdir")
    if os.path.normpath(cwd) != os.path.normpath(root / work_dir_name):
        raise NoStackTraceException(f'All commands must be ran from working directory ({root / work_dir_name}), but current directory is {cwd}')

    # Check if eval.lock exists from a terminated bash command
    if (root / "eval.lock").exists():
        cleanup_orphans(warn=True)
        (root / "eval.lock").unlink()

    return root, config

def copy_contents(src: str | os.PathLike, dst: str | os.PathLike):
    src = Path(src)
    dst = Path(dst)
    for item in os.listdir(src):
        if os.path.isfile(src / item): shutil.copy2(src / item, dst / item)
        elif os.path.isdir(src / item): shutil.copytree(src / item, dst / item)
