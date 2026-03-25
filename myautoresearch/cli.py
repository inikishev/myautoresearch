"""cli"""
import os
from typing import Literal

import click
from contextlib import contextmanager
from . import commands, utils

@contextmanager
def no_stack_trace():
    try:
        yield
    except utils.NoStackTraceException as e:
        print(f"ERROR: {e}")

@click.group('mar')
def mar():
    pass


@mar.command("init")
@click.argument('work_dir_name', type=str, default="workdir")
def init(work_dir_name: str):
    with no_stack_trace():
        commands.mar_init(work_dir_name)
        click.echo("New project has been initialized.")

@mar.command("start")
@click.argument('modifier', default=None)
def start(modifier: commands.ModifierLiteral | None = None):
    with no_stack_trace():
        commands.mar_clear()
        click.echo(commands.mar_prompt(modifier))

@mar.command("summary")
def summary():
    with no_stack_trace():
        click.echo(commands.mar_summary())

@mar.command("evaluate")
@click.option('-f', '--file', "file", help='Relative path to the python file to submit for evaluation, e.g. "algorithm.py"', type=str)
@click.option('-o', '--object', "object", help="Name of the item that will be imported from specified file and evaluated.", type=str)
@click.option('-n', '--name', "name", help="Unique name for this run.", type=str)
@click.option('-d', '--description', "description", help="Short description of this run in one or a few sentences.", type=str, default="")
@click.option('-e', '--extra-files', "extra_files", help="Relative path to an additional file or folder to include in the submission, this argument can be repeated if multiple items need to be included.", type=str, multiple=True)
@click.option('--overwrite', "overwrite", help="If this flag is specified, if an unsubmitted run with the same name exists, it will be overwritten.", is_flag=True)
@click.option('-b', '--baseline', "baseline", hidden=True, is_flag=True)
@click.option('-s', '--submit', "submit", hidden=True, is_flag=True)
def evaluate(
    file: str,
    object: str,
    name: str,
    description: str,
    extra_files: tuple[str, ...] = (),
    overwrite: bool = False,
    baseline: bool = False,
    submit: bool = False,
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
        )

        if submit:
            commands.mar_submit(name=name, result=None)

@mar.command("submit")
@click.option('-n', '--name', "name", help="Name for this algorithm/run, should be the same as one passed to `run` command.", type=str)
@click.option('-r', '--result', "result", help="Short summary of results of this run in one or a few sentences.", type=str)
def cli_submit(name: str, result: str | None):
    with no_stack_trace():
        return commands.mar_submit(name=name, result=result)

@mar.command("list")
@click.argument('status', default='all')
def cli_list(status: Literal["unsubmitted", "submitted", "discarded", "all"]):
    with no_stack_trace():
        runs = commands.mar_list_names(status)
        if len(runs) == 0:
            click.echo(f"No runs with status={status}.")
        else:
            inner = '", "'.join(runs)
            click.echo(f'"{inner}"')


@mar.command("leaderboard")
@click.argument('status', default='all')
def cli_leaderboard(status: Literal["unsubmitted", "submitted", "discarded", "all"]):
    with no_stack_trace():
        commands.mar_display_leaderboard(status)

@mar.command("clear")
def cli_clear():
    with no_stack_trace():
        commands.mar_clear()
        click.echo("Working directory cleared!")

@mar.command("load")
@click.argument('name', default=None)
def cli_load(name: str):
    with no_stack_trace():
        commands.mar_load(name)

@mar.command("config")
@click.option('--work_dir', default="")
@click.option('--author', default="")
@click.option('--max_time', default="")
@click.option('--timeout', default="")
@click.option('--top_k', default="")
@click.option('--n_neighbors', default="")
def cli_config(
    work_dir=..., author=..., max_time=..., timeout=..., top_k=..., n_neighbors=...
):
    commands.mar_config(
        work_dir=work_dir,
        author=author,
        max_time=max_time,
        timeout=timeout,
        top_k=top_k,
        n_neighbors=n_neighbors,
    )

@mar.command("discard")
@click.argument('names', nargs=-1)
def cli_discard(names: tuple[str]):
    for name in names:
        commands.mar_discard(name)