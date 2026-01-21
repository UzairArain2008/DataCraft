import pandas as pd
from pathlib import Path
import os
import joblib

from core.analyzer import analyze_dataset, suggest_target, suggest_task
from core.cleaner import clean_dataset
from core.selector import get_available_models, get_model
from core.trainer import train_model, save_model

# ----------------------------
# 1️⃣ Load dataset
# ----------------------------
path = Path(input("Enter dataset path: ").strip())
if not path.exists():
    print("❌ File not found")
    exit()

df = pd.read_csv(path)
print("✅ Dataset loaded\n")

# ----------------------------
# 2️⃣ Analyze dataset
# ----------------------------
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

# ----------------------------
# 3️⃣ Suggest target column
# ----------------------------
suggested_target = suggest_target(df)
print(f"Suggested target column: {suggested_target}")
accept = input("Do you accept this target? (Y/N): ").lower()

if accept == "y":
    target_col = suggested_target
else:
    target_col = input("Enter target column name: ").strip()

# ----------------------------
# 4️⃣ Suggest task type
# ----------------------------
task = suggest_task(df, target_col)
print(f"\nSuggested task type: {task}")

# ----------------------------
# 5️⃣ Clean dataset
# ----------------------------
clean_df, encoders = clean_dataset(df, target_col)  # automatically encodes categorical columns
print("✅ Dataset cleaned successfully")

# Save encoders for prediction later
os.makedirs("core/models", exist_ok=True)
joblib.dump(encoders, "core/models/encoders.pkl")

# ----------------------------
# 6️⃣ Split features and target
# ----------------------------
X = clean_df.drop(columns=[target_col])
y = clean_df[target_col].copy()

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# ----------------------------
# 7️⃣ Smart model selection
# ----------------------------
models = get_available_models(task, X)  # pass X for filtering based on dataset

if not models:
    print("❌ No suitable models found for this dataset and task")
    exit()

print("\nSuggested models for your dataset:")
for i, name in enumerate(models, start=1):
    print(f"{i}. {name}")

choice = int(input("Select model number: "))
selected_model_name = models[choice - 1]

model = get_model(task, selected_model_name)
print(f"\n✅ Selected model: {selected_model_name}")

# ----------------------------
# 8️⃣ Train model
# ----------------------------
result = train_model(model, X, y, task)

print("\n🎯 Training complete")
print("Metrics:")
for k, v in result["metrics"].items():
    print(f"  {k}: {v}")

# ----------------------------
# 9️⃣ Save trained model
# ----------------------------
save_model(result["model"], path="core/models/trained_model.pkl")
print("✅ Model saved successfully")
print("\n🎉 Backend CLI is ready. You can now use predictor.py to make predictions.")
