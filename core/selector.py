# core/selector.py

from core.models.registry import MODEL_REGISTRY

def get_available_models(task_type: str) -> list:
    """
    Returns list of model names available for a task
    """
    return list(MODEL_REGISTRY.get(task_type, {}).keys())


def get_model(task_type: str, model_name: str):
    """
    Returns an initialized sklearn model
    """
    task_models = MODEL_REGISTRY.get(task_type)

    if not task_models:
        raise ValueError(f"No models available for task: {task_type}")

    if model_name not in task_models:
        raise ValueError(f"Model '{model_name}' not valid for task '{task_type}'")

    return task_models[model_name]()
