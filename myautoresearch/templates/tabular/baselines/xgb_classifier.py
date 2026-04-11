import mytabular as mt
import polars as pl
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

# Baseline estimator from the 'tabular' template
# note: it is recommended that you make a custom estimator
# instead of mytabular to have better understanding and control
estimator = make_pipeline(

    # apply standard scaler, this automatically skips categorical columns
    mt.pl.StandardScaler().to_sklearn(),

    # convert string columns to categorical
    mt.pl.CastCategorical(pl.selectors.string()).to_sklearn(),

    # Run XGBoost with label encoder
    mt.ClassifierWithLabelEncoder(
        XGBClassifier(tree_method="hist", enable_categorical=True, device="cuda", seed=0),
    )
)
