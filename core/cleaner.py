import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_dataset(df: pd.DataFrame, target_col: str):
    df = df.copy()
    df = df.drop_duplicates()

    # Fill nulls
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["int64", "float64"]:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical columns
    encoders = {}
    for col in df.columns:
        if df[col].dtype == "object" and col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    return df, encoders
