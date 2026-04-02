import myautoml as ma
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Baseline estimator from the 'tabular' template
estimator = make_pipeline(

    # This handles scaling, imputation, one-hot encoding, etc depending on dataset
    ma.pl.ToNumpy(
        label = None,
        scale = True,
        impute = True,
        max_categories = 128,
    ).to_sklearn(),

    LogisticRegression(max_iter=1000, class_weight='balanced'),
)