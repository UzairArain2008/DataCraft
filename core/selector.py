from core.models.registry import MODEL_REGISTRY

def get_available_models(task_type: str, X):
    """
    Returns filtered models based on task and dataset properties
    """
    models = MODEL_REGISTRY.get(task_type, {}).copy()
    n_samples, n_features = X.shape

    # Example rules:
    filtered = {}

    for name, model_class in models.items():
        # Skip RandomForest if small dataset
        if "random_forest" in name and n_samples < 50:
            continue
        # Skip KNN if too many features
        if "knn" in name and n_features > 50:
            continue
        filtered[name] = model_class

    return list(filtered.keys())


def get_model(task_type: str, model_name: str):
    models = MODEL_REGISTRY.get(task_type)
    if not models or model_name not in models:
        raise ValueError("Invalid model selection")
    return models[model_name]()
