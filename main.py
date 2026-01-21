import pandas as pd
from pathlib import Path

from core.analyzer import analyze_dataset, suggest_target, suggest_task
from core.cleaner import clean_dataset
from core.selector import get_available_models, get_model


# 1️⃣ Load dataset
path = Path(input("Enter dataset path: ").strip())

if not path.exists():
    print("❌ File not found")
    exit()

df = pd.read_csv(path)
print("✅ Dataset loaded\n")


# 2️⃣ Analyze dataset
summary = analyze_dataset(df)

print("=" * 40)
print("Dataset Summary")
print(f"Rows: {summary['num_rows']}")
print(f"Columns: {summary['num_columns']}")
print(f"Duplicated Rows: {summary['duplicated_rows']}")
print("Null Values:")
for col, val in summary['null_values'].items():
    print(f"  {col}: {val}")
print("=" * 40)


# 3️⃣ Suggest target
suggested_target = suggest_target(df)
print(f"Suggested target column: {suggested_target}")

accept = input("Do you accept this target? (Y/N): ").lower()
if accept == "y":
    target_col = suggested_target
else:
    target_col = input("Enter target column name: ").strip()


# 4️⃣ Suggest task
task = suggest_task(df, target_col)
print(f"\nSuggested task type: {task}")


# 5️⃣ Clean dataset
clean_df, encoders = clean_dataset(df)
print("✅ Dataset cleaned successfully")


# 6️⃣ Split X & y
X = clean_df.drop(columns=[target_col])
y = clean_df[target_col]

print("Feature shape:", X.shape)
print("Target shape:", y.shape)


# 7️⃣ Model selection
models = get_available_models(task)

print("\nAvailable models:")
for i, model_name in enumerate(models, start=1):
    print(f"{i}. {model_name}")

choice = int(input("Select model number: "))
selected_model_name = models[choice - 1]

model = get_model(task, selected_model_name)
print(f"\n✅ Selected model: {selected_model_name}")
