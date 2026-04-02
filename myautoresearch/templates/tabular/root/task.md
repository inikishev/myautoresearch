# Task: Design an Estimator for {dataset_name}

## Task

Your goal is to design an estimator that predicts {target_column} ({problem_type}) from the `train.csv` dataset. The estimator will be evaluated using **{fold_type} cross-validation** with a fixed random seed, and the main metric is {metric} on the test folds.

## API

You must submit an **instantiated unfitted estimator object** (not a class) that follows the sklearn API:
{code_example}
Requirements:
- The estimator must have a `fit(X, y)` method where `X` is a pandas DataFrame and `y` is a pandas Series
- The estimator must have a `{response_method}(X)` method that returns an array of shape {output_shape}
- After fitting, the estimator must have an attribute ending with an underscore (e.g., `is_fitted_ = True`) to indicate it is fitted (sklearn convention)
- The estimator should handle {feature_types} features appropriately

## Evaluation

The evaluation script performs the following:
1. Loads `train.csv` using `pandas.read_csv` and separates features (`X`{dropped_info}) from target (`y = df["{target_column}"]`)
2. Runs stratified 5-fold cross-validation with `random_state=0`
3. For each fold:
   - Fits the estimator on the training split
   - Computes {metric} on both train and test splits
4. Reports average train and test {metric}

The evaluation script automatically times out after {timeout} seconds - if needed, optimize for speed, use CUDA.
