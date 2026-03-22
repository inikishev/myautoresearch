"""cli"""
import click, os

from . import commands


@click.group('mar')
def mar():
    pass


@mar.command("init")
@click.argument('work_dir_name', type=str, default="workdir")
def init(work_dir_name: str):
    return commands.mar_init(work_dir_name)

@mar.command("start")
def start():
    commands.mar_start()
    click.echo(commands.mar_prompt())

@mar.command("summary")
def summary():
    click.echo(commands.mar_summary())

@mar.command("evaluate")
@click.option('-f', '--file', "file", help='Relative path to the python file to submit for evaluation, e.g. "algorithm.py"', type=str)
@click.option('-o', '--object', "object", help="Name of the item that will be imported from specified file and evaluated.", type=str)
@click.option('-n', '--name', "name", help="Unique name for this run.", type=str)
@click.option('-d', '--description', "description", help="Short description of this run in one or a few sentences.", type=str)
@click.option('-e', '--extra-files', "extra_files", help="Relative path to an additional file or folder to include in the submission, this argument can be repeated if multiple items need to be included.", type=str, multiple=True)
@click.option('-b', '--baseline', "baseline", help="If this flag is passed, this run is considered a baseline run, baseline runs should only be submitted by humans before starting the research loop.", is_flag=True)
def evaluate(
    file: str,
    object: str,
    name: str,
    description: str,
    extra_files: tuple[str, ...] = (),
    baseline: bool = False,
):
    commands.mar_evaluate(
        file = file,
        object = object,
        name = name,
        description = description,
        extra_files = extra_files,
        baseline = baseline,
    )


@mar.command("submit")
@click.option('-n', '--name', "name", help="Name for this algorithm/run, should be the same as one passed to `run` command.", type=str)
@click.option('-r', '--result', "result", help="Short summary of results of this run in one or a few sentences.", type=str)
def cli_submit(name: str, result: str):
    return commands.mar_submit(name=name, result=result)

@mar.group("list")
def mar_list():
    pass

@mar_list.command("current")
def list_current():
    runs = commands.mar_list_names("current")
    if len(runs) == 0:
        click.echo("No current runs.")
    else:
        inner = '", "'.join(runs)
        click.echo(f'"{inner}"')

@mar_list.command("runs")
def list_runs():
    runs = commands.mar_list_names("runs")
    if len(runs) == 0:
        click.echo("No submitted runs.")
    else:
        inner = '", "'.join(runs)
        click.echo(f'"{inner}"')

