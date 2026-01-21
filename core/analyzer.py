import pandas as pd

def analyze_dataset(df: pd.DataFrame) -> dict:
    summary = {}
    
    summary['num_rows'] = df.shape[0]
    summary['num_columns'] = df.shape[1]
    
    summary['duplicated_rows'] = df.duplicated().sum()
    
    summary['null_values'] = df.isnull().sum()
    
    summary['column_types'] = df.dtypes.apply(lambda x: str(x)).to_dict()
    
    summary['columns'] = df.columns.tolist()
    
    print("=" * 40)
    print("Dataset Summary:")
    print(f"    Total Number of Rows: {summary['num_rows']}")
    print("-" * 40)
    print(f"    Total number of Columns: {summary['num_columns']}")
    print("-" * 40)
    print(f"    Total number of Duplicated rows: {summary['duplicated_rows']}")
    print("-" * 40)
    print(f"    Total number of Null values per Comlumns: ")
    for col, x in summary['null_values'].items():
        print(F"        {col} = {x}")
    print("-" * 40)
    print(f"Columns: {df.columns.tolist()}")
    
    return summary
