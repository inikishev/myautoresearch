"""cli"""
import os
from typing import Literal

import click
from contextlib import contextmanager
from . import _utils, commands, prompts

@contextmanager
def no_stack_trace():
    """Removes stack trace from mar exceptions to avoid useless tokens"""
    try:
        yield
    except _utils.NoStackTraceException as e:
        print(f"ERROR: {e}")

@click.group('mar')
def mar():
    pass


@mar.command("init")
@click.argument('work_dir_name', type=str, default="workdir")
def cli_init(work_dir_name: str):
    with no_stack_trace():
        commands.mar_init(work_dir_name)
        click.echo("New project has been initialized.")

@mar.command("start")
@click.argument('modifier', type=str, default=None)
def cli_start(modifier: prompts.ModifierLiteral | None = None):
    with no_stack_trace():
        commands.mar_start()
        click.echo(commands.mar_prompt(modifier))

@mar.command("summary")
def cli_summary():
    with no_stack_trace():
        click.echo(commands.mar_summary())

@mar.command("evaluate")
@click.option('-f', '--file', "file", help='Relative path to the python file to submit for evaluation, e.g. "algorithm.py"', type=str)
@click.option('-o', '--object', "object", help="Name of the item that will be imported from specified file and evaluated.", type=str)
@click.option('-n', '--name', "name", help="Unique name for this run.", type=str)
@click.option('-d', '--description', "description", help="Describe your algorithm. The description should be concise but detailed, it needs to contain all information necessary to recreate the run.", type=str, default="")
@click.option('-e', '--extra-files', "extra_files", help="Include additional files needed for evaluation (e.g., helper modules, data files), this argument can be repeated if multiple items need to be included.", type=str, multiple=True)
@click.option('--overwrite', "overwrite", help="If this flag is specified, if an unsubmitted run with the same name exists, it will be overwritten.", is_flag=True)
# hidden flags - only human can use them
@click.option('-b', '--baseline', "baseline", hidden=True, is_flag=True)
@click.option('-s', '--submit', "submit", hidden=True, is_flag=True)
@click.option('-a', '--author', "author", hidden=True, type=str, default=None)
def evaluate(
    file: str,
    object: str,
    name: str,
    description: str,
    extra_files: tuple[str, ...] = (),
    overwrite: bool = False,
    baseline: bool = False,
    submit: bool = False,
    author: str | None = None
):
    with no_stack_trace():
        commands.mar_evaluate(
            file = file,
            object = object,
            name = name,
            description = description,
            extra_files = extra_files,
            overwrite = overwrite,
            baseline = baseline,
            author = author,
        )

        if submit:
            commands.mar_submit(name=name, result=None)

@mar.command("submit")
@click.option('-n', '--name', "name", help="Name for this algorithm/run, should be the same as one passed to `run` command.", type=str)
@click.option('-r', '--result', "result", help="Describe results of your experiments - what did you try, what worked, what didn't work, did your best attempt beat current leader, can it be improved. This will be shown in previously submitted runs summary next to the description. The summary already shows all metric values, don't duplicate them here.", type=str)
def cli_submit(name: str, result: str | None):
    with no_stack_trace():
        return commands.mar_submit(name=name, result=result)

@mar.command("list")
@click.argument('what', default='all', type=str)
def cli_list(what: Literal["unsubmitted", "submitted", "discarded", "all", "modifiers"]):
    with no_stack_trace():
        if what == "modifiers":
            click.echo(f"{list(prompts.MODIFIERS.keys())}")
        else:
            runs = commands.mar_list_names(what)
            if len(runs) == 0:
                click.echo(f"No runs with status={what}.")
            else:
                inner = '", "'.join(runs)
                click.echo(f'"{inner}"')


@mar.command("leaderboard")
@click.argument('status', default='all', type=str)
def cli_leaderboard(status: Literal["unsubmitted", "submitted", "discarded", "all"]):
    with no_stack_trace():
        commands.mar_display_leaderboard(status)

@mar.command("load")
@click.argument('name', type=str)
def cli_load(name: str):
    with no_stack_trace():
        commands.mar_load(name)


@mar.command("config")
# Some of those options can have None as the value
# so we have to use "" as the default which is not an allowed value
@click.option("--work_dir", default="", type=str)
@click.option("--author", default="", type=str)
@click.option("--max_time", default="")
@click.option("--timeout", default="")
@click.option("--top_k", default="")
@click.option("--n_neighbors", default="")
@click.option("--copy_logger", default="")
def cli_config(
    work_dir=...,
    author=...,
    max_time=...,
    timeout=...,
    top_k=...,
    n_neighbors=...,
    copy_logger=...,
):
    """Edit the config. To pass None, pass "None" string."""
    with no_stack_trace():
        commands.mar_config(
            work_dir=work_dir,
            author=author,
            max_time=max_time,
            timeout=timeout,
            top_k=top_k,
            n_neighbors=n_neighbors,
            copy_logger=copy_logger,
        )


@mar.command("discard")
@click.argument('names', nargs=-1, type=str)
def cli_discard(names: tuple[str]):
    with no_stack_trace():
        commands.mar_discard(*names)


@mar.command("reevaluate")
def cli_reevaluate():
    with no_stack_trace():
        commands.mar_reevaluate()



@mar.command("rename")
@click.argument('old', type=str)
@click.argument('new', type=str)
def cli_rename(old: str, new: str):
    with no_stack_trace():
        commands.mar_rename(old=old, new=new)


@mar.command("cleanup_processes")
def cli_cleanup_processes():
    with no_stack_trace():
        _utils.cleanup_orphans()
