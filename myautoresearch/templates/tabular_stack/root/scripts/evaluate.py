import copy
import os
import time
from functools import partial
from pathlib import Path

import myautoml as ma
import numpy as np
import pandas as pd
import yaml
from click import echo
from myautoresearch import Evaluator, NoStackTraceException, run
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.covariance import LedoitWolf
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted

DATA_PATH = "train.csv"
RANDOM_SEED = 0
METRIC_FN = balanced_accuracy_score

def read_yaml(file: str | os.PathLike):
    with open(file, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)

TEMPLATE: dict = read_yaml("../../../template.yaml")

RESPONSE_METHOD = TEMPLATE["response_method"]
TARGET_COLUMN = TEMPLATE["target_column"]
PROBLEM_TYPE = TEMPLATE["problem_type"]
METRIC_NAME: str = TEMPLATE["metric"]
METRIC_LOWER = METRIC_NAME.lower().replace(" ", "_").replace("-", "_").replace(".", "_")
N_SPLITS = TEMPLATE.get("n_splits", 5)

DROP_COLS = TEMPLATE["drop_cols"]
if DROP_COLS is None: DROP_COLS = []
if isinstance(DROP_COLS, str): DROP_COLS = [DROP_COLS, ]


RUNS = Path("../../submitted")
OOFS_LIST = []
for run_dir in RUNS.iterdir():
    oof = np.load(run_dir / "oofs.npz")["data"]
    if oof.ndim == 1: oof = np.expand_dims(oof, -1)
    OOFS_LIST.append(oof)


def is_fitted(estimator):
    try:
        check_is_fitted(estimator)
        return True
    except NotFittedError:
        return False

class EstimatorEvaluator(Evaluator):
    def evaluate(self):
        # self.object is the estimator submitted by the agent

        # Load data
        df = pd.read_csv(DATA_PATH)

        # Separate features and target
        X = df.drop(columns=[TARGET_COLUMN, *DROP_COLS])
        y = df[TARGET_COLUMN]

        # CV with fixed seed
        is_classification = PROBLEM_TYPE in ("binary", "multiclass", "classification")
        if is_classification:
            kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        else:
            kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

        train_metrics = []
        test_metrics = []

        oof = None

        for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            start = time.time()
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Fit the estimator
            estimator = copy.deepcopy(self.object)
            if is_fitted(estimator):
                raise NoStackTraceException("Submitted estimator is already fitted. Please submit an unfitted estimator.")

            estimator.fit(X_train, y_train)

            if not is_fitted(estimator):
                echo("WARNING: Estimator was not marked as fitted (has no attribute ending with underscore)")
                estimator.is_fitted_ = True

            # Predict probabilities
            response_method = getattr(estimator, RESPONSE_METHOD)
            y_train_resp = response_method(X_train)
            y_test_resp = response_method(X_test)

            # Compute metric
            train_metric = METRIC_FN(np.asarray(y_train), np.asarray(y_train_resp))
            test_metric = METRIC_FN(np.asarray(y_test), np.asarray(y_test_resp))

            sec = time.time() - start
            echo(f"Fold {fold + 1}/{N_SPLITS}: Train {METRIC_NAME}: {train_metric:.4f}, Test {METRIC_NAME}: {test_metric:.4f}, took {sec:.2f} sec.")

            train_metrics.append(train_metric)
            test_metrics.append(test_metric)

            if oof is None:
                shape = y_test_resp.shape
                shape[0] = len(y)
                oof = np.zeros(shape)

            oof[test_idx] = np.asarray(y_test)

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

        # Fit linear model on oofs
        assert oof is not None
        if oof.ndim == 1: oof = np.expand_dims(oof, -1)
        all_oofs = np.concatenate(OOFS_LIST + [oof], 1)

        if is_classification:
            estimators = dict(
                LogReg = LogisticRegression(),
                RidgeCV = ma.RegressorAsClassifier(RidgeCV()),
                LassoCV = ma.RegressorAsClassifier(LassoCV()),
                LDA_LW = LinearDiscriminantAnalysis(covariance_estimator=LedoitWolf()),
                QDA_05 = QuadraticDiscriminantAnalysis(reg_param=0.5),
            )
        else:
            estimators = dict(
                RidgeCV = RidgeCV(),
                LassoCV = LassoCV(),
            )

        for name, estimator in estimators.items():
            for fold, (train_idx, test_idx) in enumerate(kfold.split(all_oofs, y)):
                start = time.time()
                X_train, X_test = all_oofs[train_idx], all_oofs[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Fit the estimator
                estimator.fit(X_train, y_train)



if __name__ == "__main__":
    run(EstimatorEvaluator())
