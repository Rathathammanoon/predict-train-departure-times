from sklearn.preprocessing import LabelEncoder
import pandas as pd

def preprocess_features(df, features):
    X = df[features].copy()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    datetime_cols = X.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        X[f"{col}_hour"] = X[col].dt.hour
        X[f"{col}_minute"] = X[col].dt.minute
        X.drop(col, axis=1, inplace=True)

    return X, label_encoders
