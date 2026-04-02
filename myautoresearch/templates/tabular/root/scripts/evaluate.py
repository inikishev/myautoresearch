import time
import copy
import os

import numpy as np
import pandas as pd
import yaml
from click import echo
from sklearn.metrics import roc_auc_score
from functools import partial
from sklearn.model_selection import KFold, StratifiedKFold

from myautoresearch import Evaluator, run

DATA_PATH = "train.csv"
RANDOM_SEED = 0
METRIC_FN = partial(roc_auc_score, multi_class="ovr")

def read_yaml(file: str | os.PathLike):
    with open(file, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)

TEMPLATE: dict = read_yaml("../template.yaml")

RESPONSE_METHOD = TEMPLATE["response_method"]
TARGET_COLUMN = TEMPLATE["target_column"]
PROBLEM_TYPE = TEMPLATE["problem_type"]
METRIC_NAME: str = TEMPLATE["metric"]
METRIC_LOWER = METRIC_NAME.lower().replace(" ", "_").replace("-", "_").replace(".", "_")
N_SPLITS = TEMPLATE.get("n_splits", 5)

DROP_COLS = TEMPLATE["drop_cols"]
if DROP_COLS is None: DROP_COLS = []
if isinstance(DROP_COLS, str): DROP_COLS = [DROP_COLS, ]

class EstimatorEvaluator(Evaluator):
    def evaluate(self):
        # self.object is the estimator submitted by the agent

        # Load data
        df = pd.read_csv(DATA_PATH)

        # Separate features and target
        y = df[TARGET_COLUMN]
        if len(DROP_COLS) > 0: X = df.drop(columns=DROP_COLS)
        else: X = df

        # CV with fixed seed
        if PROBLEM_TYPE in ("binary", "multiclass", "classification"):
            kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        else:
            kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

        train_metrics = []
        test_metrics = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            start = time.time()
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Fit the estimator
            estimator = copy.deepcopy(self.object)
            estimator.fit(X_train, y_train)

            # Predict probabilities
            response_method = getattr(estimator, RESPONSE_METHOD)
            y_train_resp = response_method(X_train)
            y_test_resp = response_method(X_test)

            # Compute metric
            train_metric = METRIC_FN(y_train, y_train_resp)
            test_metric = METRIC_FN(y_test, y_test_resp)

            sec = time.time() - start
            echo(f"Fold {fold + 1}/{N_SPLITS}: Train {METRIC_NAME}: {train_metric:.4f}, Test {METRIC_NAME}: {test_metric:.4f}, took {sec:.2f} sec.")

            train_metrics.append(train_metric)
            test_metrics.append(test_metric)

        avg_train = np.mean(train_metrics)
        avg_test = np.mean(test_metrics)

        # Log final metrics
        self.log_final(
            metric=f"train_{METRIC_LOWER}",
            value=avg_train,
            maximize=True,
            is_main=False,
            display_value=True,
            display_rank=False,
            display_leaderboard=True,
            display_summary=True,
        )

        self.log_final(
            metric=f"test_{METRIC_LOWER}",
            value=avg_test,
            maximize=True,
            is_main=True,
            display_value=True,
            display_rank=True,
            display_leaderboard=True,
            display_summary=True,
        )


if __name__ == "__main__":
    run(EstimatorEvaluator())
