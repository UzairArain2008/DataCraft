import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_dataset(df: pd.DataFrame):
    """
    Cleans dataset and returns cleaned DataFrame + encoders
    """
    df = df.copy()

    # Drop duplicated rows
    df = df.drop_duplicates()

    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["int64", "float64"]:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical columns
    encoders = {}
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    return df, encoders
