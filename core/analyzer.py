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
