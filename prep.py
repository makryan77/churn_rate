import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def preprocess(data):
    # Drop unnecessary columns
    cols_to_drop = ["phone number"]
    data.drop(cols_to_drop, axis=1, inplace=True)

    # Compute missing proportions
    #df =df.dropna()
    # Drop cols with >40% missing
#     over_threshold = missing_props[missing_props >= 0.4]
#     df.fillna(over_threshold.index, axis=1, inplace=True)

    # Build feature and target arrays

    X = data.drop("churn", axis=1)
    y = data.churn

    # Build the pipeline for categorical features
    categorical_pipeline = Pipeline(

        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    # Build the pipeline for numerical features
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler())
        ]
    )

    # Extract numeric and categorical col names
    cat_cols = X.select_dtypes(exclude="number").columns
    num_cols = X.select_dtypes(include="number").columns

    # Build the full transformer combining the above pipelines
    full_processor = ColumnTransformer(
         transformers=[
            ("numeric", numeric_pipeline, num_cols),
             ("categorical", categorical_pipeline, cat_cols),
         ]
    )

    # Apply preprocessing
    X_processed = full_processor.fit_transform(X)
    y_processed = y

    return X_processed, y_processed
