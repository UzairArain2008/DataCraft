# core/models/registry.py

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans

MODEL_REGISTRY = {
    "classification": {
        "logistic_regression": LogisticRegression,
        "knn": KNeighborsClassifier,
        "random_forest": RandomForestClassifier,
    },
    "regression": {
        "linear_regression": LinearRegression,
        "knn": KNeighborsRegressor,
        "random_forest": RandomForestRegressor,
    },
    "clustering": {
        "kmeans": KMeans,
    }
}
