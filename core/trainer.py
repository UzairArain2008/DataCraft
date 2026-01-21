import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

def train_model(model, X, y, task_type, test_size=0.2, random_state=42):
    """
    Trains a model and returns metrics + trained model
    """
    # 1️⃣ Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 1.5️⃣ Optional: scaling for SVM / LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC, SVR

    if isinstance(model, (LogisticRegression, SVC, SVR)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # 2️⃣ Train model
    model.fit(X_train, y_train)

    # 3️⃣ Predict
    y_pred = model.predict(X_test)

    # 4️⃣ Evaluate
    metrics = {}

    if task_type == "classification":
        metrics["accuracy"] = accuracy_score(y_test, y_pred)

    elif task_type == "regression":
        metrics["rmse"] = mean_squared_error(y_test, y_pred, squared=False)
        metrics["r2_score"] = r2_score(y_test, y_pred)

    elif task_type == "clustering":
        try:
            from sklearn.metrics import silhouette_score
            metrics["silhouette"] = silhouette_score(X_test, y_pred)
        except:
            metrics["info"] = "Evaluation not applicable"

    return {
        "model": model,
        "metrics": metrics,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0]
    }

def save_model(model, path="models/trained_model.pkl"):
    """
    Saves trained model to disk
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)  # create folder if not exists
    joblib.dump(model, path)
    print(f"✅ Model saved at {path}")
