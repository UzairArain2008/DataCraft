import pandas as pd

def analyze_dataset(df: pd.DataFrame) -> dict:
    summary = {}
    
    summary['num_rows'] = df.shape[0]
    summary['num_columns'] = df.shape[1]
    
    summary['duplicated_rows'] = df.duplicated().sum()
    
    summary['null_values'] = df.isnull().sum()
    
    summary['column_types'] = df.dtypes.apply(lambda x: str(x)).to_dict()
    
    summary['columns'] = df.columns.tolist()
    
    return summary
