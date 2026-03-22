import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import polars as pl
import psutil
import yaml
from scipy.stats import rankdata

from . import utils
from .evaluate_template import EVALUATE_TEMPLATE
from .evaluator import Evaluator, FinishedRun, Metric

def _toint(x):
    if math.isfinite(x): return int(x)
    return x

def _get_root_and_config() -> tuple[Path, dict]:
    root = utils.get_cwd().parent

    # Load config
    with open(root / "config.yaml", encoding='utf-8') as f:
        config: dict = yaml.safe_load(f)

    # Check that work dir is correct
    work_dir_name = config.get("work_dir", "workdir")
    if os.path.normpath(os.getcwd()) != os.path.normpath(root / work_dir_name):
        raise RuntimeError(f'Working directory must be {root / work_dir_name}, but it is {os.getcwd()}')

    return root, config


def mar_init(work_dir_name: str = "workdir", author: str | None = None, max_time: float | None = None, timeout: float | None = None, top_k: int = 10, n_neighbors: int =2):
    root = utils.get_cwd()
    (root / work_dir_name).mkdir(exist_ok=True)
    (root / "current").mkdir(exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    (root / "discarded").mkdir(exist_ok=True)
    (root / "template").mkdir(exist_ok=True)

    if not(root / "config.yaml").exists():

        def _yaml_field(s):
            if isinstance(s, str): return f'"{s}"'
            if s is None: return "null"
            return s

        config_yaml = f'work_dir: "{work_dir_name}"\nauthor: {_yaml_field(author)}\nmax_time: {_yaml_field(max_time)}\ntimeout: {_yaml_field(timeout)}\ntop_k: {top_k}\nn_neighbors: {n_neighbors}'
        utils.write_text(config_yaml, root / "config.yaml")

    if not (root / "task.md").exists(): utils.write_text("# Task", root / "task.md")
    if not (root / "API.md").exists(): utils.write_text("# API", root / "API.md")
    if not (root / "initialize.py").exists(): utils.write_text("# initialization script", root / "initialize.py")
    if not (root / "evaluate.py").exists(): utils.write_text(EVALUATE_TEMPLATE, root / "evaluate.py")


def _process_metrics(runs: list[FinishedRun]):
    # Find ranks of each run by each main metric
    # First we need to get a list of main metrics
    main_metric_names = set()
    display_metric_names = set()
    for run in runs:
        main_metric_names.update(name for name, metric in run.metrics.items() if metric.is_main)
        display_metric_names.update(name for name, metric in run.metrics.items() if metric.display)

    # Load values of main metrics for each run
    main_errors = defaultdict(list)
    for run in runs:
        for metric in sorted(main_metric_names):
            if metric in run.metrics:
                main_errors[metric].append(run.metrics[metric].error())
            else:
                main_errors[metric].append(np.nan)

    # Compute ranks (index of run in `runs` is index of rank in each list)
    ranks = {k: rankdata(v, method='dense', nan_policy='omit') for k,v in main_errors.items()}

    if len(ranks) > 1:
        mean_ranks = []
        for i, run in enumerate(runs):
            run_ranks = []
            for metric_name in sorted(main_metric_names):
                if metric_name in run.metrics:
                    run_ranks.append(ranks[metric_name][i])
                else:
                    run_ranks.append(len(runs))

            mean_ranks.append(np.mean(run_ranks))

        total_ranks = rankdata(mean_ranks, method='dense', nan_policy='omit')

    else:
        mean_ranks = total_ranks = None

    return main_metric_names, display_metric_names, ranks, mean_ranks, total_ranks

def mar_summary():
    root, config = _get_root_and_config()

    # Load runs
    runs = [FinishedRun(r) for r in (root / "runs").iterdir()]

    if len(runs) == 0:
        return "# Runs\n\nNo runs evaluated yet!"

    runs.sort(key = lambda x: x.info["start_sec"]) # sort by start time

    main_metric_names, display_metric_names, ranks, mean_ranks, total_ranks = _process_metrics(runs)

    # ## Run 1: {name}

    # {datetime} (submitted by {author})

    # {description}

    # {results}

    # feasible: {feasible}

    # metrics:
    # - {metric}: {value} (rank {rank}/{n_runs})

    summary = "# Runs\n\n The following runs have been submitted previously:"
    for i, run in enumerate(runs):
        name = run.info["name"]
        dt = run.info["start_dt"]
        author = run.info["author"]
        description = run.info["description"]
        result = run.info["result"]
        feasible = run.info["feasible"]
        infeasibility_reasons = [d["reason"] for d in run.info["feasibility"] if d["feasible"] is False]

        metrics_s = ""

        if (mean_ranks is not None) and (total_ranks is not None):

            mean_rank = mean_ranks[i]
            total_rank = total_ranks[i]
            metrics_s = f'{metrics_s}\n- rank: {total_rank}\n- mean rank: {utils.format_value(mean_rank)}'

        for metric_name in sorted(display_metric_names):
            if metric_name in run.metrics:
                metric = run.metrics[metric_name]
                metrics_s = f"{metrics_s}\n- {metric_name}: {utils.format_value(metric.value)}"
                if metric_name in ranks:
                    rank = ranks[metric_name][i]
                    metrics_s = f'{metrics_s} (ranked {rank}/{np.nanmax(ranks[metric_name])})'

        if run.info["baseline"]: author_s = " (baseline)"
        elif author is not None: author_s = f" (submitted by {author})"
        else: author_s = ""
        if feasible: feasibility_s = "feasible: True"
        else: feasibility_s = f"feasible: False ({infeasibility_reasons})"
        result_s = "" if result == "" else f"\n\nresult: {result}"
        summary = f"{summary}\n\n## {i+1}. {name}\n\n{dt}{author_s}\n\ndescription: {description}{result_s}\n\n{feasibility_s}\n\nmetrics:{metrics_s}"

    return summary

MAR_INSTRUCTION = """# Instruction

To evaluate a run, use the `mar evaluate` command in the terminal. Pass the following flags:

--file TEXT: name of the python file to submit for evaluation, e.g. "algorithm.py".

--object TEXT: name of the item (class, object, variable) that will be imported from specified file and evaluated. The object is imported as `from <file> import <object>`.

--name TEXT: unique name for this run.

--description TEXT: the description that will be shown in previously submitted runs summary if this run is be submitted. Ideally it should contain all information necessary to recreate your algorithm. Be very concise and try to fit it in as few sentences as possible.


After running the command you will see the results. Keep trying to improve your algorithm and beat the current best run. Finally, submit your best run using `mar submit` command in the terminal. Always submit your best attempt, even if it failed, since it will be useful to know what works and what doesn't. Pass the following flags:

--name TEXT: the same name as you passed to `mar evaluate`. You can list all names via `mar list current`.

--result TEXT: describe results of your experiments - did your idea work, did it beat current leader, can it be improved. This will be shown in in previously submitted runs summary next to the description. The summary already shows all metric values, don't duplicate them here. Be very concise and try to fit it in as few sentences as possible."""

def mar_prompt():
    root, config = _get_root_and_config()

    task = utils.read_text(root / "task.md")
    api = utils.read_text(root / "API.md")

    return f"{task}\n\n{mar_summary()}\n\n{api}\n\n{MAR_INSTRUCTION}"



def mar_start():
    root, config = _get_root_and_config()

    # Move runs from ``current`` to ``discarded``
    if len(os.listdir(root / "current")) > 0:
        shutil.copytree(root / "current", root / "discarded", dirs_exist_ok=True)
        shutil.rmtree(root / "current")
        (root / "current").mkdir()

    # Clear the workdir and reset from template
    work_dir_name: str = config.get("work_dir", "workdir")
    shutil.rmtree(root / work_dir_name)
    shutil.copytree(root / "template", root / work_dir_name)
    os.chdir(root / work_dir_name) # otherwise we get no such file or directory on getcwd

    # Run initialization script
    if (root / "initialize.py").exists():
        subprocess.run([sys.executable, str(root / "initialize.py")], check=True)


def mar_evaluate(file: str, object: str, name: str, description: str, extra_files: tuple[str, ...] = (), baseline=False):
    """
    1. Creates a new dir in ``runs`` folder.
    2. Copies ``file`` to ``run/__source__.py``, ``evaluate.py`` to ``run/__evaluate__.py``, and ``include`` preserving their file names.
    3. Runs the experiment in the run folder, capturing outputs to ``STDOUT.log`` and ``STDERR.log``.
    4. Outputs STDOUT to console.
    5. Saves ``info.json``
    """
    root, config = _get_root_and_config()

    # Check than `name` is unique
    runs_dir = root / "runs"
    runs_dir.mkdir(exist_ok=True)
    if name in os.listdir(runs_dir):
        raise FileExistsError(f"A run with name '{name}' has already been submitted, make sure the name is unique. "
                              f"The following runs are submitted: {os.listdir(runs_dir)}")

    current_names = mar_list_names("current")
    if name in current_names:
        raise FileExistsError(f"A run with name '{name}' already exists, make sure the name is unique. "
                              f"The following names are used: {current_names}")

    # Create a new dir to run in
    nanos = time.time_ns()
    dt = datetime.fromtimestamp(nanos / 1e9)
    dir = utils.make_valid_filename(f"{name}-{dt.strftime('%Y-%m-%d %H-%M-%S')}-{(nanos % 1e9):09.0f}")
    eval_dir = root / "current" / dir
    shutil.copytree(root / "template", eval_dir)

    # Copy everything to it
    work_dir_name = config.get("work_dir", "workdir")
    shutil.copy2(root / work_dir_name / file, eval_dir / "__source__.py") # copy the algo
    shutil.copy2(root / "evaluate.py", eval_dir / "__evaluate__.py") # copy the scoring
    for extra in extra_files: # copy extra files
        if os.path.isfile(extra): shutil.copy2(root / work_dir_name / extra, eval_dir / extra)
        elif os.path.isdir(extra): shutil.copytree(root / work_dir_name / extra, eval_dir / extra)
        else: raise FileNotFoundError(f"Path passed to `extra_files` doesn't exist: {extra}")

    # Run
    timeout = config.get("timeout", None)
    if timeout is not None and timeout <= 0: timeout = None
    start_sec = time.time()
    result = None
    finished = False

    with open(eval_dir / "STDOUT.log", "a", encoding='utf-8') as STDOUT:
        with open(eval_dir / "STDERR.log", "a", encoding='utf-8') as STDERR:
            cwd = os.getcwd()
            try:
                os.chdir(eval_dir)
                result = subprocess.run(
                    [
                        sys.executable,
                        '__evaluate__.py',
                        "--file", '__source__.py',
                        "--object", f'{object}',
                    ],
                    check = True,
                    text = True,
                    stdout = STDOUT,
                    stderr = STDERR,
                    timeout = timeout
                )

                with open(eval_dir / "STDOUT.log", "r", encoding='utf-8') as f: click.echo(f.read())
                finished = True

            except subprocess.TimeoutExpired:
                with open(eval_dir / "STDOUT.log", "r", encoding='utf-8') as f: click.echo(f.read())
                click.echo(f"ERROR: script runtime exceeded the timeout of {timeout} sec, "
                           "process has been terminated early.")

            except subprocess.CalledProcessError:
                with open(eval_dir / "STDOUT.log", "r", encoding='utf-8') as f: click.echo(f.read())
                with open(eval_dir / "STDERR.log", "r", encoding='utf-8') as f:
                    click.echo(f"ERROR: script failed with the following exception:\n{f.read()}\n")

            except Exception as e:
                with open(eval_dir / "STDOUT.log", "r", encoding='utf-8') as f: click.echo(f.read())
                click.echo(f"ERROR: execution failed with the following exception:\n{e}\n")
                with open(eval_dir / "STDERR.log", "r", encoding='utf-8') as f: click.echo(f"STDERR:\n{f.read()}\n")

            finally:
                os.chdir(cwd)

    time_sec = time.time() - start_sec
    click.echo(f"Run finished in {time_sec:.2f} seconds.")

    max_time = config.get("max_time", None)
    if max_time is not None and max_time <= 0: max_time = None
    feasibility = []

    if (eval_dir / "feasibility.json").exists():
        feasibility = utils.read_json(eval_dir / "feasibility.json")

    if not finished:
        feasibility.append({"feasible": False, "reason": "Failed to execute the script."})

    elif max_time is not None and time_sec > max_time:
        feasibility.append({"feasible": False, "reason": f"Script runtime {time_sec:.2f} sec. exceeded maximum allowed runtime of {max_time} sec."})

    infeasibility_reasons = [d["reason"] for d in feasibility if d["feasible"] is False]
    feasible = len(infeasibility_reasons) == 0


    # Save info
    info = {
        "file": file,
        "extra_files": extra_files,
        "object_name": object,
        "name": name,
        "description": description,
        "baseline": baseline,
        "feasible": feasible,
        "feasibility": feasibility,
        "author": config.get("author", None),
        "args": result.args if result is not None else None,
        "start_dt": dt.strftime('%Y-%m-%d %H-%M-%S'),
        "start_sec": start_sec,
        "time_sec": time_sec,
        "finished": finished,
        "config": config,
    }

    utils.write_json(info, eval_dir / "info.json")

    # metrics:
    # - {metric}: {value} (rank {rank}/{n_runs})

    # Current leaderboard:
    # - 1. {run_name}: {metric1=value1}, {metric2=value2}, mean_rank={mean_rank}
    # and so on
    # ...
    # and here it shows 2 neighbors below and above


    if feasible:
        runs = [FinishedRun(r) for r in (root / "runs").iterdir()]
        for r in runs: r.submitted = True
        current_runs = [FinishedRun(r) for r in (root / "current").iterdir() if os.path.normpath(r) != os.path.normcase(eval_dir)]
        for r in current_runs: r.submitted = False
        runs.extend(current_runs)
        this_run = FinishedRun(eval_dir)
        this_run.submitted = False
        runs.append(this_run)
        runs.sort(key = lambda x: x.info["start_sec"]) # sort by start time

        main_metric_names, display_metric_names, ranks, mean_ranks, total_ranks = _process_metrics(runs)
        if total_ranks is None: total_ranks = next(iter(ranks.values()))

        click.echo("Metrics:")
        if mean_ranks is not None:
            mean_rank = mean_ranks[-1]
            total_rank = total_ranks[-1]
            click.echo(f'- rank: {_toint(total_rank)}/{_toint(np.nanmax(total_ranks))}')
            click.echo(f'- mean rank: {utils.format_value(mean_rank)}')

        for metric in this_run.metrics.values():
            if metric.display:
                s = f"- {metric.name}: {utils.format_value(metric.value)}"
                if metric.name in ranks:
                    s = f"{s} (ranked {_toint(ranks[metric.name][-1])}/{_toint(np.nanmax(ranks[metric.name]))})"
                click.echo(s)

        top_k = config.get("top_k", 10)


        def echo_run_metrics(run: FinishedRun, this:bool):
            name = run.info['name']
            if not run.submitted:
                name = f'{name} (unsubmitted)'
            s = f"{_toint(rank)}. {name}: "
            for metric in run.metrics.values():
                if metric.display:
                    s = f"{s}{metric.name}={utils.format_value(metric.value)}, "

            if mean_ranks is not None:
                mean_rank = mean_ranks[index]
                s = f"{s}mean_rank={utils.format_value(mean_rank)}, "

            s = s[:-2]

            if this:
                s = f"{s} <- current run"

            click.echo(s)

        click.echo("\nCurrent leaderboard:")
        within_top_k = False
        sorted_runs = []
        this_i = 0
        argsort = np.argsort(total_ranks)
        for i, (rank, index) in enumerate(zip(total_ranks[argsort], argsort), start=1):
            run: FinishedRun = runs[index]
            if run is this_run:
                this_i = i - 1
            sorted_runs.append(run)

            if i <= top_k:
                if run is this_run: within_top_k = True
                echo_run_metrics(run, run is this_run)

        if not within_top_k:
            click.echo("...")

            n_neigbors = config.get("n_neighbors", 2)
            for i, (rank, index) in enumerate(zip(total_ranks, np.argsort(total_ranks)), start=1):

                if abs(i - this_i) < n_neigbors:
                    run: FinishedRun = runs[index]
                    echo_run_metrics(run, run is this_run)

    if len(infeasibility_reasons) > 0:
        click.echo("WARNING: Run is not feasible for the following reasons:")
        for reason in infeasibility_reasons:
            click.echo(reason)

def mar_list_names(dir="current"):
    root, config = _get_root_and_config()
    evals_dir = root / dir
    all_names = []
    for run in evals_dir.iterdir():
        all_names.append(utils.read_json(run / "info.json")["name"])
    return all_names


def mar_submit(name: str, result: str):
    """Moves specified run from ``current`` to ``runs`` folder, adding ``result`` to ``info.json``.
    This run will now appear in run lists and visualizations."""
    root, config = _get_root_and_config()

    evals_dir = root / "current"
    target_run = None
    all_names = []
    for run in evals_dir.iterdir():
        run_name = utils.read_json(run / "info.json")["name"]
        if run_name == name:
            target_run = run
            break
        all_names.append(run_name)

    if target_run is None:
        raise FileNotFoundError(f"A run with name '{name}' doesn't exist, make sure 'name' is identical to one "
                                f"passed to `run` command. The following runs exist: {all_names}")

    shutil.move(target_run, root / "runs" / name)

    # Add summary to info
    info = utils.read_json(root / "runs" / name /  "info.json")
    info["result"] = result
    utils.write_json(info, root / "runs" / name /  "info.json")

    click.echo(f'Run "{name}" has been saved to "{root / "runs" / name}".')
