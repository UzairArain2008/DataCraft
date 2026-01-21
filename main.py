from core.analyzer import analyze_dataset
import pandas as pd
from pathlib import Path

path = Path(input("Enter dataset path: "))
df = pd.read_csv(path)


print("Dataset Loaded")

summary = analyze_dataset(df)

summary = analyze_dataset(df)

print("=" * 40)
print("Dataset Summary:")
print(f"  Rows: {summary['num_rows']}")
print(f"  Columns: {summary['num_columns']}")
print(f"  Duplicated Rows: {summary['duplicated_rows']}")
print("  Null Values:")
for col, val in summary['null_values'].items():
    print(f"    {col}: {val}")
print("  Column Types:")
for col, t in summary['column_types'].items():
    print(f"    {col}: {t}")
