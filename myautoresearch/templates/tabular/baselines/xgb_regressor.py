import myautoml as ma
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# Baseline estimator from the 'tabular' template
estimator = make_pipeline(

    # apply standard scaler, this automatically skips categorical columns
    ma.pl.StandardScaler().to_sklearn(),

    # Run XGBoost with scaled targets
    TransformedTargetRegressor(
        XGBRegressor(tree_method="hist", enable_categorical=True, device="cuda", seed=0),
        transformer = MinMaxScaler(),
    ),
)