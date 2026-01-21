from core.analyzer import analyze_dataset
import pandas as pd
from pathlib import Path

path = Path(input("Enter dataset path: "))
df = pd.read_csv(path)

summary = analyze_dataset(df)

print("Dataset Loaded")

print("-" * 40)
print("Dataset Summary:")
print(f"Total Number of Rows: {summary['num_rows']}")
print("-" * 40)
print(f"Total number of Columns: {summary['num_columns']}")
print("-" * 40)
print(f"Total number of Duplicated rows: {summary['duplicated_rows']}")
print("-" * 40)
print(f"Total number of Null values per Comlumns: ")
for col, x in summary['null_values'].items():
    print(F"    {col} = {x}")
print("-" * 40)
print(f"Columns are {df.columns.tolist()}")
