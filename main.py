from core.analyzer import analyze_dataset
import pandas as pd
from pathlib import Path

path = Path(input("Enter dataset path: "))
df = pd.read_csv(path)

summary = analyze_dataset(df)

print("Dataset Loaded")

print("Dataset Summary:")
print(f"Total Number of Rows: {summary['num_rows']}")
print(f"Total number of Columns: {summary['num_columns']}")
print(f"Duplicated rows: {summary['duplicated_rows']}")
print("Null values per column:")
for col, nulls in summary['null_values'].items():
    print(f"  {col}: {nulls}")
print("Column types:")
for col, typ in summary['column_types'].items():
    print(f"  {col}: {typ}")
