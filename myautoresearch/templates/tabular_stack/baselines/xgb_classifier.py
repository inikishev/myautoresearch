import myautoml as ma
import polars as pl
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

# Baseline estimator from the 'tabular' template
estimator = make_pipeline(

    # apply standard scaler, this automatically skips categorical columns
    ma.pl.StandardScaler().to_sklearn(),

    # convert string columns to categorical (returns polars dataframe)
    ma.pl.CastCategorical(pl.selectors.string()).to_sklearn(),

    # Run XGBoost with label encoder
    ma.ClassifierWithLabelEncoder(
        XGBClassifier(tree_method="hist", enable_categorical=True, device="cuda", seed=0),
    )
)
