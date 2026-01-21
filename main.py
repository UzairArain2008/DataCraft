from core.analyzer import analyze_dataset
import pandas as pd
from pathlib import Path

path = Path(input("Enter dataset path: "))
df = pd.read_csv(path)


print("Dataset Loaded")

summary = analyze_dataset(df)