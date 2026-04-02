import importlib
import os
from pathlib import Path

import yaml


def read_yaml(file: str | os.PathLike):
    with open(file, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)


def init_from_template(root: Path):
    template = read_yaml(root / "template.yaml")["template"]
    module = importlib.import_module(f".templates.{template}", "myautoresearch")
    module.create_from_template(root)
