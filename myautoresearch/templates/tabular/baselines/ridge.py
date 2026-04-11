import mytabular as mt
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

# Baseline estimator from the 'tabular' template
estimator = make_pipeline(

    # This handles scaling, imputation, one-hot encoding, etc depending on dataset
    # and returns a numpy array.
    mt.pl.ToNumpy(
        label = None,
        scale = True,
        impute = True,
        max_categories = 128,
    ).to_sklearn(),

    TransformedTargetRegressor(
        Ridge(),
        transformer = MinMaxScaler(),
    ),
)