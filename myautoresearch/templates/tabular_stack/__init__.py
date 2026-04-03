import os
import shutil
from pathlib import Path

import click
import polars as pl

from ... import _utils

TEMPLATE_DIR = Path(__file__).parent

EXAMPLE_CLASSIFICATION = """\n```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Example submission (baseline):
estimator = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=0)
)
```\n"""

EXAMPLE_REGRESSION = """\n```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Example submission (baseline):
estimator = make_pipeline(
    StandardScaler(),
    Ridge(random_state=0)
)
```\n"""

def create_from_template(root: Path):
    # Copy template files to new root
    _utils.copy_contents(TEMPLATE_DIR / "root", root)

    # Load template
    template = _utils.read_yaml(root / "template.yaml")

    # Load train data
    df_train = None
    if (root / "train.csv").exists():
        df_train = pl.read_csv(root / "train.csv")

    # ---------------------- Create all strings for task.md ---------------------- #
    # Target column
    target_column = template.get("target_column", None)
    if target_column is None:
        if df_train is None:
            raise _utils.NoStackTraceException("Can't infer target column name. Either specify target_column "
                                               "in template.yaml, or put train.csv to the root.")

        target_column = df_train.columns[-1]
        click.echo(f"Since target_column is not specified, it is assumed to be `{target_column}`")

    # Problem type
    problem_type = template["problem_type"]
    response_method = template.get("response_method", None)
    output_shape = template.get("output_shape", None)
    n_classes = template.get("n_classes", None)
    n_splits = template.get("n_splits", 5)

    if problem_type in ("binary", "multiclass", "classification"):
        fold_type = f"stratified {n_splits}-fold"
        code_example = EXAMPLE_CLASSIFICATION
        if response_method is None: response_method = "predict_proba"
        if n_classes is None:
            if problem_type == "binary": n_classes = 2
            else:
                if df_train is None:
                    raise _utils.NoStackTraceException("Can't infer n_classes. Either specify n_classes "
                                                       "in template.yaml, or put train.csv to the root.")

                n_classes = df_train[target_column].n_unique()

    elif problem_type == "regression":
        code_example = EXAMPLE_REGRESSION
        fold_type = f"{n_splits}-fold"
        if response_method is None: response_method = "predict"

    else:
        code_example = ""
        fold_type = f"{n_splits}-fold"
        if response_method is None: response_method = "predict"

    # Output shape
    if output_shape is None:
        if response_method == "predict_proba":
            output_shape = f"(n_samples, {n_classes})"
        else:
            output_shape = "(n_samples, )"

    # Feature types
    feature_types = template.get("feature_types", "all")
    if feature_types is None: feature_types = "all"
    if not isinstance(feature_types, str):
        feature_types = ", ".join(feature_types)

    # Dropped columns
    drop_cols = template.get("drop_cols", None)
    if drop_cols is None: dropped_info = ""
    else:
        if isinstance(drop_cols, str):
            dropped_info = f" with `{drop_cols}` column dropped"
        else:
            inner = "`, `".join(drop_cols)
            dropped_info = f" with '{inner}' columns dropped"

    # Substitute those into the template and write new task.md
    task_md = _utils.read_text(root / "task.md").format(
        dataset_name = template.get("dataset_name", "this dataset"),
        target_column = target_column,
        problem_type = problem_type,
        metric = template["metric"],
        code_example = code_example,
        response_method = response_method,
        output_shape = output_shape,
        feature_types = feature_types,
        dropped_info = dropped_info,
        timeout = template["timeout"],
        fold_type = fold_type,

    )
    _utils.write_text(task_md, root / "task.md")

    # Subsitute into README.md
    readme_md = _utils.read_text(root / "README.md").format(
        target_column = target_column,
        metric = template["metric"],
        fold_type = fold_type,
    )
    _utils.write_text(readme_md, root / "README.md")

    # Substitute timeout info config
    config = _utils.read_text(root / "config.yaml").format(timeout=template["timeout"])
    _utils.write_text(config, root / "config.yaml")

    # Save updated (inferred) info to template since evaluation reads from it
    template["response_method"] = response_method
    template["target_column"] = target_column
    template["n_classes"] = n_classes
    template["output_shape"] = output_shape
    _utils.write_yaml(template, root / "template.yaml")

    # Move train.csv to all dirs that need it
    shutil.move(root / "train.csv", root / "templates" / "workdir" / "train.csv")
    shutil.copy2(root / "templates" / "workdir" / "train.csv", root / "templates" / "eval" / "train.csv")

    # Add baselines
    if problem_type in ("binary", "multiclass", "classification"):
        baselines = _utils.read_yaml(TEMPLATE_DIR / "baselines" / "baselines_classification.yaml")
    else:
        baselines = _utils.read_yaml(TEMPLATE_DIR / "baselines" / "baselines_regression.yaml")

    click.echo("Modify scripts/evaluate.py with the desired evaluation metric, then run those commands to submit baselines:")
    for baseline in baselines.values():
        shutil.copy2(TEMPLATE_DIR / "baselines" / baseline["file"], root / "workdir" / baseline["file"])

        command = "mar evaluate"
        for k,v in baseline.items():
            command = f'{command} --{k} "{v.replace('"', "'")}"'
        click.echo(f"{command} --baseline --submit --overwrite")
