import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import psutil


def read_json(file: str | os.PathLike):
    with open(file, "r", encoding='utf-8') as f:
        return json.load(f)

def write_json(obj, file: str | os.PathLike):
    with open(file, "w", encoding='utf-8') as f:
        json.dump(obj, f, sort_keys=False, indent=4, ensure_ascii=False)


def import_object(argv: list[str]):

    # argv will have
    # --file
    # --obj
    # --name
    # --description

    if "__evaluate__.py" in argv: argv.remove("__evaluate__.py")
    parser = argparse.ArgumentParser(description="Evaluates scripts.")
    parser.add_argument('-f', "--file", type=str)
    parser.add_argument('-o', "--object", type=str)
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
    return getattr(module, args.object)


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
    c = ''.join(c for c in s if c.isalnum() or c in (" .,"))[:100]
    while not c[-1].isalnum(): c = c[:-1]
    return c

def get_cwd():
    try:
        return Path(os.getcwd())
    except FileNotFoundError:
        cwd = Path(psutil.Process(os.getpid()).cwd())
        os.chdir(cwd)
        return cwd
