import fcntl
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Literal

import click
import numpy as np

from . import _utils, prompts
from .evaluate_template import EVALUATE_TEMPLATE
from .logger import Logger

DEFAULT_CONFIG = dict(
    work_dir = "workdir",
    author = None,
    max_time = None,
    timeout = None,
    top_k = 10,
    n_neighbors = 2,
    copy_logger = False,
)
FLOAT_CONFIG_KEYS = ("max_time", "timeout")
INT_CONFIG_KEYS = ("top_k", "n_neighbors")
BOOL_CONFIG_KEYS = ("copy_logger", )

def mar_init(preset: str | None = None):
    """Initializes a new project."""
    # TODO: add presets

    root = _utils.get_cwd()

    # workdir
    (root / "workdir").mkdir(exist_ok=True)

    # runs
    (root / "runs").mkdir(exist_ok=True)
    (root / "runs" / "unsubmitted").mkdir(exist_ok=True)
    (root / "runs" / "submitted").mkdir(exist_ok=True)
    (root / "runs" / "discarded").mkdir(exist_ok=True)

    # templates
    (root / "templates").mkdir(exist_ok=True)
    (root / "templates" / "workdir").mkdir(exist_ok=True)
    (root / "templates" / "eval").mkdir(exist_ok=True)

    # readme
    if not (root / "README.md").exists():
        _utils.write_text(
            "<!-- This file is not shown to the AI agent. You can fill in the prompt template and copy it from here. -->\n## Prompt:\nYour goal will be to develop <describe the task>. Run `mar start` in the shell and follow the instructions.",
            root / "README.md"
        )

    # prompts
    if not (root / "task.md").exists(): _utils.write_text("## Task\n\n## API\n\n## Evaluation\n\n", root / "task.md")

    # scripts
    (root / "scripts").mkdir(exist_ok=True)
    if not (root / "scripts" / "initialize.py").exists():
        _utils.write_text("# initialization script", root /  "scripts" / "initialize.py")

    if not (root /  "scripts" / "evaluate.py").exists():
        _utils.write_text(EVALUATE_TEMPLATE, root /  "scripts" / "evaluate.py")

    # config
    if not(root / "config.yaml").exists():
        _utils.write_yaml(DEFAULT_CONFIG, root / "config.yaml")


# Note: Summary is meant to show submitted runs, because it is quite verbose.
# All runs can be viewed via leaderboard which doesn't show descriptions.

def mar_summary():
    """Returns sumbitted runs and their descriptions"""

    root, config = _utils.get_root_and_config()

    # Load runs
    runs = [_utils.FinishedRun(r) for r in (root / "runs" / "submitted").iterdir()]
    runs = [r for r in runs if len(r.info) > 0]

    if len(runs) == 0:
        return "No runs have been submitted yet!"

    runs.sort(key = lambda x: x.info["start_sec"]) # sort by start time

    main_metric_names, display_metric_names, ranks, mean_ranks, total_ranks = _utils.process_metrics(runs)

    # ## Run 1: {name}

    # {datetime} (submitted by {author})

    # {description}

    # {results}

    # feasible: {feasible}

    # metrics:
    # - {metric}: {value} (rank {rank}/{n_runs})

    summary = "The following runs have been submitted previously:"
    for i, run in enumerate(runs):
        name = run.info["name"]
        # dt = run.info["start_dt"]
        author = run.info["author"]
        description = _utils.maybe_strip(run.info["description"])
        result: str | None = _utils.maybe_strip(run.info["result"])
        feasible = run.info["feasible"]
        infeasibility_reasons = [d["reason"] for d in run.info["feasibility"] if d["feasible"] is False]

        metrics_s = ""

        if (mean_ranks is not None) and (total_ranks is not None):

            mean_rank = mean_ranks[i]
            total_rank = total_ranks[i]
            metrics_s = f'{metrics_s}\n- rank: {_utils.maybe_int(total_rank)}\n- weighted mean rank: {_utils.format_value(mean_rank)}'

        for metric_name in sorted(display_metric_names):
            if metric_name in run.metrics:
                metric = run.metrics[metric_name]
                if metric.display_summary:
                    metrics_s = f"{metrics_s}\n- {metric_name}: {_utils.format_value(metric.value)}"
                    if metric_name in ranks:
                        rank = ranks[metric_name][i]
                        metrics_s = f'{metrics_s} (ranked {_utils.maybe_int(rank)}/{_utils.maybe_int(np.nanmax(ranks[metric_name]))})'

        if run.info["baseline"]: author_s = " (baseline)"
        elif author is not None: author_s = f" (submitted by {author})"
        else: author_s = ""

        if feasible: feasibility_s = ""
        else: feasibility_s = f"\n\nRun is not feasible: {infeasibility_reasons}"

        description_s = f"\n\ndescription: {description}" if description else ""

        result_s = f"\n\nresult: {result}" if result else ""

        summary = f"{summary}\n\n## {i+1}. {name}{author_s}{description_s}{result_s}{feasibility_s}\n\nmetrics:{metrics_s}"

    return summary

def mar_display_leaderboard(status: Literal["unsubmitted", "submitted", "discarded", "all"] = "all", current_run_dir=None):
    """Displays the leaderboard."""
    root, config = _utils.get_root_and_config()

    runs = []
    if status in ("submitted", "all"):
        runs = [_utils.FinishedRun(r) for r in (root / "runs" / "submitted").iterdir()]
        for r in runs:
            r.status = 'submitted'

    if status in ("unsubmitted", "all"):
        this_s = os.path.normcase(current_run_dir) if current_run_dir is not None else ""
        unsubmitted_runs = [
            _utils.FinishedRun(r) for r in (root / "runs" / "unsubmitted").iterdir() if os.path.normpath(r) != this_s]
        for r in unsubmitted_runs:
            r.status = 'unsubmitted'
        runs.extend(unsubmitted_runs)

    if status == "discarded":
        runs = []
        for session_dir in (root / "runs" / "discarded").iterdir():
            runs.extend(_utils.FinishedRun(r) for r in session_dir.iterdir())

        for r in runs:
            r.status = 'discarded'

    current_run = None
    if current_run_dir is not None:
        current_run = _utils.FinishedRun(current_run_dir)
        current_run.status = "current"
        runs.append(current_run)

    runs = [r for r in runs if len(r.info) > 0]
    runs.sort(key = lambda x: x.info["start_sec"]) # sort by start time

    main_metric_names, display_metric_names, ranks, mean_ranks, total_ranks = _utils.process_metrics(runs)
    if len(ranks) == 0:
        click.echo("No runs evaluated yet!")
        return

    if total_ranks is None: total_ranks = next(iter(ranks.values()))

    if current_run is not None:

        # Display metrics for current run, which is always evaluated last
        click.echo("Metrics:")
        if mean_ranks is not None:
            mean_rank = mean_ranks[-1]
            total_rank = total_ranks[-1]
            click.echo(f'- rank: {_utils.maybe_int(total_rank)}/{_utils.maybe_int(np.nanmax(total_ranks))}')
            click.echo(f'- weighted mean rank: {_utils.format_value(mean_rank)}')

        for metric in current_run.metrics.values():
            if metric.display_value:
                s = f"- {metric.name}: {_utils.format_value(metric.value)}"
                if (metric.display_rank) and (metric.name in ranks):
                    best_by_metric = runs[np.nanargmin(ranks[metric.name])].info["name"]
                    s = f"{s} (ranked {_utils.maybe_int(ranks[metric.name][-1])}/{_utils.maybe_int(np.nanmax(ranks[metric.name]))}, best run: {best_by_metric})"
                click.echo(s)

        click.echo() # newline before leaderboard

    top_k = config.get("top_k", 10)

    shown = []

    # Function to display a run
    def echo_leaderboard_metrics(run: _utils.FinishedRun, rank: int, is_current:bool):
        if run in shown: return
        shown.append(run)

        name = run.info['name']
        if run.status in ("unsubmitted", "current"):
            name = f'{name} (unsubmitted)'

        s = f"{_utils.maybe_int(rank)}. {name}: "
        for metric in run.metrics.values():
            if metric.display_leaderboard:
                s = f"{s}{metric.name}={_utils.format_value(metric.value)}, "

        if mean_ranks is not None:
            mean_rank = mean_ranks[index]
            s = f"{s}mean_rank={_utils.format_value(mean_rank)}, "

        s = s[:-2] # remove ", "

        if run.info["feasible"] is False:
            s = f"{s} (INFEASIBLE)"

        if is_current:
            s = f"{s} <- current run"

        click.echo(s)

    # Display leaderboard (top k)
    click.echo("Current leaderboard:")
    within_top_k = False
    sorted_runs = []
    this_i = 0
    argsort = np.argsort(total_ranks)
    for i, (rank, index) in enumerate(zip(total_ranks[argsort], argsort), start=1):
        run: _utils.FinishedRun = runs[index]
        is_current = (current_run is not None) and (run is current_run)
        if is_current: this_i = i - 1
        sorted_runs.append(run)

        if i <= top_k:
            if is_current: within_top_k = True
            echo_leaderboard_metrics(run, rank, is_current)

    # If current run is not in the leaderboard, show it below next to few neighbours
    if (current_run is not None) and (not within_top_k):
        click.echo("...")

        n_neigbors = config.get("n_neighbors", 2)
        for i, (rank, index) in enumerate(zip(total_ranks[argsort], argsort), start=1):

            if abs(i - this_i) <= n_neigbors:
                run: _utils.FinishedRun = runs[index]
                echo_leaderboard_metrics(run, rank, run is current_run)


def mar_prompt(modifier: prompts.ModifierLiteral | None = None):
    """Prompt for the AI agent."""
    root, config = _utils.get_root_and_config()

    task = _utils.read_text(root / "task.md").strip()

    s = f"{task}\n\n\n{prompts.MAR_INSTRUCTION.strip()}\n\n\n# Runs\n\n{mar_summary()}\n\n{prompts.AFTER_SUMMARY_RUNS}"
    if modifier is not None:
        s = f"{s}\n\n{prompts.MODIFIER_INSTRUCTION.format(modifier_name=modifier, modifier=prompts.MODIFIERS[modifier]).strip()}"

    return s


def _initialize_session(root: Path, config: dict):
    """Moves unsubmitted runs to discarded, copies template over to workdir"""
    # Move old runs from ``unsubmitted`` to ``discarded``
    if len(os.listdir(root / "runs" / "unsubmitted")) > 0:
        # since we don't check for discarded name clashes, each session gets its own subdir
        dt = datetime.fromtimestamp(time.time_ns() / 1e9).strftime('%Y-%m-%d %H-%M-%S')
        discarded_dir = (root / "runs" / "discarded" / dt)
        discarded_dir.mkdir(exist_ok=True)
        shutil.copytree(root / "runs" / "unsubmitted", discarded_dir, dirs_exist_ok=True)
        shutil.rmtree(root / "runs" / "unsubmitted")
        (root / "runs" / "unsubmitted").mkdir()

    # Copy template over to workdir
    work_dir_name: str = config.get("work_dir", "workdir")
    shutil.copytree(root / "templates" / "workdir", root / work_dir_name, dirs_exist_ok=True)

    # Run initialization script
    if (root / "scripts" / "initialize.py").exists():
        subprocess.run([sys.executable, str(root / "scripts" / "initialize.py")], check=True)

def mar_start():
    """Start a new session. Clears current session files and moves unsubmitted runs to discarded."""
    root, config = _utils.get_root_and_config()

    # clear files from previous session
    work_dir_name: str = config.get("work_dir", "workdir")
    for item in (root / work_dir_name).iterdir():
        if item.is_dir(): shutil.rmtree(item)
        else: item.unlink()

    # clear failed runs with no info.json (may happen if agent harness like opencode force terminates evaluation)
    for run in (root / "runs" / "unsubmitted").iterdir():
        if not (run / "info.json").exists():
            shutil.rmtree(run)

    _initialize_session(root, config)

def mar_evaluate(file: str, object: str, name: str, description: str, extra_files: tuple[str, ...] = (), overwrite=False, baseline=False, author=None):
    """
    Evaluates a submission.
    1. Creates a new dir in ``runs/unsubmitted`` folder.
    2. Copies ``file`` to ``run/file``, ``evaluate.py`` to ``run/__myautoresearch_evaluate__.py``, and ``extra_files``.
    3. Runs the experiment in the run folder, capturing outputs to ``console.log``.
    4. Outputs console.log to console.
    5. Saves ``info.json`` and other files.
    6. Outputs leaderboard and other useful information for agent to iterate on.
    """
    if len(name) == 0:
        raise _utils.NoStackTraceException("Passed empty string to --name.")

    if not baseline:
        if len(description) == 0:
            raise _utils.NoStackTraceException("Passed empty string to --description.")

    root, config = _utils.get_root_and_config()
    name = _utils.make_valid_filename(name)

    # Check that `name` is unique
    if overwrite: current_names = mar_list_names("submitted")
    else: current_names = mar_list_names("all")
    if name in current_names:
        raise _utils.NoStackTraceException(f"A run with name '{name}' already exists, make sure the name is unique "
                              f"or use `--overwrite` flag. The following names are used: {current_names}")

    # Create a new dir to run in and copy from eval template
    eval_dir = root / "runs" / "unsubmitted" / name
    if eval_dir.exists(): shutil.rmtree(eval_dir)
    shutil.copytree(root / "templates" / "eval", eval_dir)

    # Copy everything to it
    work_dir_name: str = config.get("work_dir", "workdir")
    if not (root / work_dir_name / file).exists():
        raise _utils.NoStackTraceException(f"File passed to `--file` doesn't exist: {root / work_dir_name / file}")

    shutil.copy2(root / work_dir_name / file, eval_dir / file) # copy the algo
    shutil.copy2(root / "scripts" / "evaluate.py", eval_dir / "__myautoresearch_evaluate__.py") # copy the scoring
    for extra in extra_files: # copy extra files
        if os.path.isfile(extra): shutil.copy2(root / work_dir_name / extra, eval_dir / extra)
        elif os.path.isdir(extra): shutil.copytree(root / work_dir_name / extra, eval_dir / extra)
        else: raise _utils.NoStackTraceException(f"Path passed to `--extra_files` doesn't exist: {extra}")


    # Create a temporary directory for console log, that way it is preserved when evaluator is killed
    (root / "temp").mkdir(exist_ok=True)
    console_log_file = root / "temp" / f"{name}.log"

    def echo_console():
        if console_log_file.exists():
            with open(console_log_file, "r", encoding='utf-8') as f:
                console = f.read()
                if len(console.strip()) > 0: click.echo(console)

    # Run
    timeout = config.get("timeout", None)
    if timeout is not None and timeout <= 0: timeout = None
    start_sec = time.time()
    result = None
    finished = False
    timed_out = False
    should_delete = False

    with open(console_log_file, "a", encoding='utf-8') as console_log:

        current_items = set(os.listdir(eval_dir))

        try:

            result = subprocess.run(
                [
                    sys.executable,
                    "__myautoresearch_evaluate__.py",
                    "--file", file,
                    "--object", f'{object}',
                    "--root", f"{root}",
                    "--name", str(name),
                ],
                cwd = eval_dir,
                check = True,
                text = True,
                stdout = console_log,
                stderr = subprocess.STDOUT,
                timeout = timeout,
            )

            echo_console()
            finished = True

        except subprocess.TimeoutExpired:
            timed_out = True
            should_delete = True
            echo_console()
            click.echo(f"ERROR: script runtime exceeded the timeout of {timeout} sec, "
                        "process has been terminated early.")

        except subprocess.CalledProcessError as e:
            should_delete = True
            echo_console()
            click.echo(f"ERROR: script raised an exception:\n{e}\n")
            if e.returncode == -signal.SIGTERM:
                click.echo("Run was terminated by a SIGTERM likely because another run has started.")
            elif e.returncode == -signal.SIGKILL:
                click.echo("Run was force-killed (SIGKILL) likely because another run has started.")

        except Exception as e:
            should_delete = True
            echo_console()
            click.echo(f"ERROR: execution failed with the following exception:\n{e}\n")

    time_sec = time.time() - start_sec
    click.echo(f"Run finished in {time_sec:.2f} seconds.")

    # Check that eval dir exists
    if not eval_dir.exists():
        raise _utils.NoStackTraceException(f"The evaluation directory {eval_dir} has been deleted. This may happen if you run two evaluations with the same name in parallel. Don't run multiple evaluations in parallel.")

    # copy over files that script has saved back to workdir
    new_items = set(os.listdir(eval_dir)).difference(current_items)
    for item in new_items:

        # Skip system files like __pycache__
        if not ((item.startswith((".", "__"))) or item in ("debug.log", "logger.npz")):

            tgt_path = (root / work_dir_name / item)

            try:
                if tgt_path.exists():
                    if os.path.isfile(item): tgt_path.unlink()
                    else: shutil.rmtree(tgt_path)

                if os.path.isfile(item): shutil.copy2(eval_dir / item, tgt_path)
                elif os.path.isdir(item): shutil.copytree(eval_dir / item, tgt_path)

            except Exception as e:
                warnings.warn(f"Failed to copy {eval_dir / item} to {tgt_path}:\n{e}")

    # copy console log from temp to eval dir and remove temporary directory
    if console_log_file.exists():
        tgt_file = eval_dir / "console.log"
        if tgt_file.exists(): tgt_file.unlink()
        shutil.copy2(console_log_file, tgt_file)

    shutil.rmtree(root / "temp", ignore_errors=True)


    # Copy logger to workdir and display short instruction if enabled in config
    if config.get("copy_logger", False):
        if (eval_dir / "logger.npz").exists():

            try:
                loggers_dir = root / work_dir_name / "loggers"
                if not loggers_dir.exists(): loggers_dir.mkdir()
                shutil.copyfile(eval_dir / "logger.npz", loggers_dir / f"logger_{name}.npz")
                logger = Logger.from_file(loggers_dir / f"logger_{name}.npz")

                click.echo(f'\n`loggers/logger_{name}.npz` was saved to working directory. If you\'d like to inspect it, you can load it by running `from myautoresearch import Logger; logger = Logger.from_file("logger.npz")`. Use `logger.to_numpy(metric_name)` to load metrics for each step as an array. The following metrics were saved:\n{tuple(logger.keys())}\n')

            except Exception as e:
                warnings.warn(f"WARNING: failed to load the logger (this is likely an issue with evaluation script):\n{e}.")

    # Delete run if it failed and display a message
    if should_delete:
        if timed_out: click.echo(f"Run `{name}` was deleted because the execution was timed out.")
        else: click.echo(f"Run `{name}` was deleted because execution failed with an exception.")
        shutil.rmtree(eval_dir)
        return

    # Check if run is feasible and collect unfeasibility reasons
    max_time = config.get("max_time", None)
    if max_time is not None and max_time <= 0: max_time = None
    feasibility = []

    if (eval_dir / "feasibility.json").exists():
        feasibility = _utils.read_json(eval_dir / "feasibility.json")

    if timed_out:
        if max_time is None:
            reason = f"Script runtime exceeded the timeout of {timeout} sec."
        else:
            reason = (
                f"Script runtime exceeded hard timeout of {timeout} sec. Note that maximum "
                f"runtime for a script to be considered feasible is {max_time} sec."
            )
        feasibility.append({"feasible": False, "reason": reason})

    elif max_time is not None and time_sec > max_time:
        feasibility.append({"feasible": False, "reason": f"Script runtime {time_sec:.2f} sec. exceeded maximum allowed runtime of {max_time} sec."})

    elif not finished:
        feasibility.append({"feasible": False, "reason": "Failed to execute the script."})

    infeasibility_reasons = [d["reason"] for d in feasibility if d["feasible"] is False]
    feasible = len(infeasibility_reasons) == 0

    _utils.write_json(feasibility, eval_dir / "feasibility.json")

    # Save info.json to eval dir
    nanos = time.time_ns()
    dt = datetime.fromtimestamp(nanos / 1e9)

    if author is None or author.strip() == "": author = config.get("author", None)
    if baseline: author = None

    info = {
        "file": file,
        "extra_files": extra_files,
        "object_name": object,
        "name": name,
        "description": description,
        "baseline": baseline,
        "feasible": feasible,
        "feasibility": feasibility,
        "author": author,
        "args": result.args if result is not None else None,
        "start_dt": dt.strftime('%Y-%m-%d %H-%M-%S'),
        "start_sec": start_sec,
        "time_sec": time_sec,
        "finished": finished,
        "config": config,
    }

    _utils.write_json(info, eval_dir / "info.json")

    # Display the leaderboard if run finished
    if finished:
        mar_display_leaderboard(current_run_dir=eval_dir)

    if len(infeasibility_reasons) > 0:
        click.echo("\nWARNING: Run is not feasible for the following reasons:")
        for reason in infeasibility_reasons:
            click.echo(reason)

def mar_list_names(status: Literal["unsubmitted", "submitted", "discarded", "all"] = "all"):
    """Displays a list of names with specified status."""
    if status == "all":
        # no discarded here, they are only stored for reference
        return mar_list_names("unsubmitted") + mar_list_names("submitted")

    root, config = _utils.get_root_and_config()
    evals_dir = root / "runs" / status

    all_names = []
    for run in evals_dir.iterdir():
        if (run / "info.json").exists():
            all_names.append(_utils.read_json(run / "info.json")["name"])

    return all_names


def mar_submit(name: str, result: str | None):
    """Moves specified run from ``unsubmitted`` to ``submitted``, adding ``result`` to ``info.json``.
    This run will now appear in run lists and visualizations."""
    root, config = _utils.get_root_and_config()
    name = _utils.make_valid_filename(name)

    evals_dir = root / "runs" / "unsubmitted"
    target_run = None
    all_names = []
    for run in evals_dir.iterdir():
        if not (run / "info.json").exists():
            shutil.rmtree(run)
            continue

        run_name = _utils.read_json(run / "info.json")["name"]
        if run_name == name:
            target_run = run
            break
        all_names.append(run_name)

    if (root / "runs" / "submitted" / name).exists():
        raise _utils.NoStackTraceException(f"You have already submitted run with name '{name}'!")

    if target_run is None:
        raise _utils.NoStackTraceException(f"An unsubmitted run with name '{name}' doesn't exist, "
                                          "make sure 'name' is identical to one passed to `run` command. "
                                          f"The following unsubmitted runs exist: {all_names}")

    new_dir = root / "runs" / "submitted" / name
    shutil.move(target_run, new_dir)

    # Add summary to info
    info = _utils.read_json(new_dir /  "info.json")
    info["result"] = result
    _utils.write_json(info, new_dir /  "info.json")

    click.echo(f'Run "{name}" has been saved to "{new_dir}".')


def mar_discard(*names: str):
    """Discards specified runs"""
    root, config = _utils.get_root_and_config()
    dt = datetime.fromtimestamp(time.time_ns() / 1e9).strftime('%Y-%m-%d %H-%M-%S')
    discarded_dir = root / "runs" / "discarded" / dt
    discarded_dir.mkdir(exist_ok=True)
    for name in names:
        name = _utils.make_valid_filename(name)
        target_run_dir = _utils.find_run_dir_by_name(name, root)
        new_dir = discarded_dir / name
        shutil.move(target_run_dir, new_dir)
        click.echo(f'Run "{name}" has been moved to "{new_dir}".')


def mar_load(name: str):
    """Loads specified run source to workdir, this is meant to be used by AI to give it a way to access existing runs."""
    root, config = _utils.get_root_and_config()
    name = _utils.make_valid_filename(name)

    # need to find name in either submitted or unsubmitted
    target_run_dir = _utils.find_run_dir_by_name(name, root)

    work_dir: Path = root / config.get("work_dir", "workdir")
    (work_dir / "loaded").mkdir(exist_ok=True)
    new_dir = work_dir / "loaded" / name
    shutil.copytree(target_run_dir, new_dir, dirs_exist_ok=True)

    # now we need to delete evaluation code and stuff agents shouldn't be able to access
    def maybe_delete(item: Path):
        if item.exists():
            if item.is_dir(): shutil.rmtree(item)
            else: item.unlink()

    info = _utils.read_json(new_dir / "info.json")
    source_file = info["file"]

    maybe_delete(new_dir / "__myautoresearch_evaluate__.py")
    maybe_delete(new_dir / "info.json")
    maybe_delete(new_dir / "feasibility.json")

    click.echo(f'Source file of run "{name}" has been copied to "{new_dir / source_file}".')

def _is_set(x):
    if x is ...: return False
    if isinstance(x, str) and len(x) == 0: return False
    return True

def mar_config(work_dir, author, max_time, timeout, top_k, n_neighbors, copy_logger):
    """Modify config file through terminal e.g. ``mar config --author=Qwen3.5``"""
    kwargs = locals().copy()

    root, config = _utils.get_root_and_config()

    for k,v in kwargs.items():
        if k in DEFAULT_CONFIG:

            # Add defaults if missing
            if k not in config: config[k] = DEFAULT_CONFIG[k]

            # Skip unset items, we can't have ... as default in click.option
            # So instead defaults are empty strings
            if v is ... or isinstance(v, str) and len(v) == 0: continue

            # click might convert None to string
            if v.strip().lower() == "None": v = None

            # Convert to appropriate type
            if v is not None:
                if k in FLOAT_CONFIG_KEYS: v = float(v)
                elif k in INT_CONFIG_KEYS: v = int(v)
                elif k in BOOL_CONFIG_KEYS:
                    if isinstance(v, str):
                        if v.strip().lower() == "true": v = True
                        elif v.strip().lower() == "false": v = False
                        else: raise RuntimeError(f'Invalid value for boolean parameter "{k}": `{v}`')
                    else:
                        v = bool(v)

            # Set new value
            config[k] = v

    _utils.write_yaml(config, root / "config.yaml", )


def _sort_key(dir: Path):
    if (dir / "info.json").exists():
        info = _utils.read_json(dir / "info.json")
        return info["start_sec"]
    else:
        return 0

def mar_reevaluate():
    """Reevaluates all submitted runs. You can use this if you've updated the evaluation script. Note that unsubmitted and discarded runs will be deleted."""
    root, config = _utils.get_root_and_config()

    work_dir_name = config.get("work_dir", "workdir")
    work_dir = root / work_dir_name

    submitted = (root / "runs" / "submitted")

    dt = datetime.fromtimestamp(time.time_ns() / 1e9).strftime('%Y-%m-%d %H-%M-%S')
    (root / "runs" / "backup").mkdir(exist_ok=True)
    backup_dir = (root / "runs" / "backup" / dt)
    shutil.copytree(root / "runs" / "submitted", backup_dir)
    click.echo(f'Current submitted runs have been backed up to "{backup_dir}".')

    sorted_dirs = list(submitted.iterdir())
    sorted_dirs.sort(key = _sort_key)
    success = True

    for run_dir in sorted_dirs:
        if (run_dir / "info.json").exists():
            click.echo(f"Reevaluating `{run_dir.name}`...")

            info = _utils.read_json(run_dir / "info.json")

            mar_start()

            shutil.copy(run_dir / info["file"], work_dir / info["file"])
            for extra in info["extra_files"]: # copy extra files
                if os.path.isfile(extra): shutil.copy2(run_dir / extra, work_dir / extra)
                elif os.path.isdir(extra): shutil.copytree(run_dir / extra, work_dir / extra)
                else: raise _utils.NoStackTraceException(f"Path passed to `--extra_files` doesn't exist: {extra}")

            with tempfile.TemporaryDirectory() as tmpdir:
                shutil.move(run_dir, os.path.join(tmpdir, run_dir.name))

                try:
                    mar_evaluate(
                        file=info["file"],
                        object=info["object_name"],
                        name=info["name"],
                        description=info["description"],
                        extra_files=info["extra_files"],
                        baseline=info["baseline"],
                    )

                    if info["name"] in os.listdir(root / "runs" / "unsubmitted"):
                        mar_submit(name=info["name"], result=info["result"])

                    else:
                        raise _utils.NoStackTraceException(f"Exception caught while evaluating `{info['name']}`.")

                except Exception as e:
                    if backup_dir.is_dir() and len(os.listdir(backup_dir)) > 0:

                        shutil.rmtree(root / "runs" / "submitted")
                        (root / "runs" / "submitted").mkdir()
                        shutil.move(backup_dir, root / "runs" / "submitted")

                        click.echo(
                            f"Reevaluation of run `{info['name']}` failed:\n{e}\n"
                            f"Reevaluation was cancelled and submitted runs were restored from backup dir."
                        )

                    else:
                        click.echo(
                            f"Reevaluation of run `{info['name']}` failed:"
                            f"\n{e}\nReevaluation was cancelled. "
                            "WARNING: The backup couldn't be restored."
                        )

                    success = False
                    break
        else:
            click.echo(f"`{run_dir.name}` has no info.json, it will be deleted.")
            shutil.rmtree(run_dir)

    if success:
        shutil.rmtree(root / "runs" / "unsubmitted")
        (root / "runs" / "unsubmitted").mkdir()
        shutil.rmtree(root / "runs" / "discarded")
        (root / "runs" / "discarded").mkdir()

def mar_rename(old: str, new: str):
    """Renames a run."""
    root, config = _utils.get_root_and_config()
    old = _utils.make_valid_filename(old)
    new = _utils.make_valid_filename(new)

    # need to find name in either submitted or unsubmitted
    target_run_dir = _utils.find_run_dir_by_name(old, root)
    if (os.path.exists(target_run_dir.parent / new)):
        raise _utils.NoStackTraceException(f"A run with name `{new}` already exists in {target_run_dir.parent}.")

    new_dir = target_run_dir.parent / new
    os.rename(target_run_dir, new_dir)

    info_file = new_dir / "info.json"
    if info_file.exists():
        info = _utils.read_json(info_file)
        info["name"] = new
        _utils.write_json(info, info_file)

    click.echo(f"Run {target_run_dir} was renamed to {new_dir}")
