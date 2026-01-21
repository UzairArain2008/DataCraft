import pandas as pd

def analyze_dataset(df: pd.DataFrame) -> dict:
    return {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "duplicated_rows": int(df.duplicated().sum()),
        "null_values": df.isnull().sum().to_dict(),
        "column_types": df.dtypes.apply(str).to_dict(),
        "columns": df.columns.tolist()
    }

def suggest_target(df):
    """
    Suggests a target column based on heuristics
    """
    ignore = ["id", "index"]

    candidates = []
    for col in df.columns:
        if col.lower() not in ignore:
            candidates.append(col)

    return candidates[-1] if candidates else None

def suggest_task(df, target_col):
    """
    Suggests ML task based on target column
    """
    unique_vals = df[target_col].nunique()
    dtype = df[target_col].dtype

    # Numeric target
    if dtype in ["int64", "float64"]:
        if unique_vals <= 10:
            return "classification"
        return "regression"

    # Categorical target
    return "classification"